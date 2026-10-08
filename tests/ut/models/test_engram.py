# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Focused tests for the Ascend Engram configuration and storage path."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.deepseek_v41.engram.hash_state import DEAD_ID, AscendNgramHashState
from vllm_ascend.models.deepseek_v41.engram.parallel import resolve_dp_shared_memory


def test_shared_memory_needs_a_local_dp_peer(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import parallel

    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 1)
    assert not resolve_dp_shared_memory(True)
    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 2)
    assert resolve_dp_shared_memory(True)
    assert not resolve_dp_shared_memory(False)


@pytest.mark.parametrize(
    "enabled,mode,expected",
    [(True, "FULL", True), (True, "NONE", True), (False, "FULL", False), (True, "PIECEWISE", False)],
)
def test_preparation_overlap_checks_runtime(monkeypatch, enabled, mode, expected):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import model as model_module

    model = SimpleNamespace(has_engram=True, _engram_overlap_enabled=enabled)
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        model_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode[mode])
    )
    assert model_module.DeepseekV41Model._can_overlap_engram_preparation(model) is expected


def _fake_host_library(device_offset):
    def device_pointer(pointer, out, flags):
        out._obj.value = pointer.value + device_offset
        return 0

    return SimpleNamespace(
        aclrtHostRegisterV2=lambda *args: 0,
        aclrtHostGetDevicePointer=device_pointer,
        aclrtHostUnregister=lambda *args: 0,
    )


@pytest.mark.parametrize("leader_rank", [0, 16])
def test_shared_uva_uses_one_python_shared_memory_segment(monkeypatch, leader_rank):
    name_ready = threading.Event()
    attached = threading.Barrier(2)
    names: list[str] = []
    monkeypatch.setattr(npu, "_host_library", lambda: _fake_host_library(1 << 40))
    monkeypatch.setattr(npu.dist, "get_global_rank", lambda group, rank: leader_rank)
    monkeypatch.setattr(npu.dist, "barrier", lambda group: attached.wait(timeout=10))
    monkeypatch.setattr(npu.dist, "all_gather_object", lambda errors, error, group: None)

    def broadcast(payload, src, group):
        assert src == leader_rank
        if payload[0] is None:
            assert name_ready.wait(timeout=10)
            payload[0] = names[0]
        else:
            names.append(payload[0])
            name_ready.set()

    monkeypatch.setattr(npu.dist, "broadcast_object_list", broadcast)

    def create(rank):
        group = SimpleNamespace(cpu_group=None, rank_in_group=rank, world_size=2)
        return npu.SharedUvaBuffer((8, 32), torch.int8, "cpu", group)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader, follower = list(pool.map(create, (0, 1)))
    leader.tensor.fill_(7)
    assert follower.tensor.tolist() == leader.tensor.tolist()
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=names[0])
    leader.close()
    follower.close()


@pytest.mark.parametrize("num_tokens", [0, 2])
def test_idle_hashes_have_no_valid_rows(num_tokens):
    state = object.__new__(AscendNgramHashState)
    torch.nn.Module.__init__(state)
    state.multipliers = torch.empty(2, 3, dtype=torch.int64)
    state.primes = torch.empty(2, 2, 4, dtype=torch.int64)
    hashes, keep = state.dummy_hashes(torch.zeros(num_tokens, dtype=torch.int64))
    assert hashes.shape == (num_tokens, 2, 8)
    assert hashes.dtype == torch.int32
    assert (hashes == DEAD_ID).all()
    assert keep.shape == (num_tokens,)
    assert keep.dtype == torch.bool
    assert not keep.any()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("remote_group", ["tp", "edp"])
def test_engram_rejects_nonlocal_groups_before_allocation(monkeypatch, shared, remote_group):
    monkeypatch.setattr(embedding_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group="tp"))
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: SimpleNamespace(cpu_group="edp"))
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda pg: [True, pg != remote_group])
    error = "TP ranks" if remote_group == "tp" else "same node and shared-memory namespace"
    with pytest.raises(ValueError, match=error):
        embedding_mod.AscendParallelEngramEmbedding(96, 64, (4,) * 24, 0, dp_shared_memory=shared)


@pytest.mark.parametrize("dp_rank,num_tokens", [(2, 3), (3, 2), (3, 0)])
def test_engram_gather_uses_the_local_edp_token_slice(monkeypatch, dp_rank, num_tokens):
    from vllm_ascend.models.deepseek_v41.engram import parallel as parallel_mod

    edp_group = SimpleNamespace(world_size=2, rank_in_group=dp_rank - 2, all_gather=lambda ids, dim=0: ids.repeat(2, 1))
    monkeypatch.setattr(parallel_mod, "get_engram_dp_group", lambda: edp_group)
    monkeypatch.setattr(parallel_mod, "get_dp_group", lambda: SimpleNamespace(rank_in_group=dp_rank))
    monkeypatch.setattr(parallel_mod, "is_pd_decode_recompute_scheduler_enabled", lambda: False)
    # This EDP starts at global DP rank 2, so its slice is (4, 2) and not the
    # 9 tokens another node is prefilling.
    monkeypatch.setattr(
        parallel_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([1, 9, 4, 2]))),
    )
    ids = torch.full((num_tokens, 5), dp_rank + 10, dtype=torch.int32)
    gathered = embedding_mod.gather_engram_hashes(ids)
    assert gathered.shape == (8, 5)
    for replica in gathered.reshape(2, 4, 5):
        torch.testing.assert_close(replica[:num_tokens], ids)
        assert (replica[num_tokens:] == parallel_mod.DEAD_ID).all()


def test_native_mxfp8_preserves_shard_bits_and_lookup(tmp_path, monkeypatch):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    codes = torch.linspace(-32, 32, 19 * 64).reshape(19, 64).to(torch.float8_e4m3fn)
    scale_bits = (torch.arange(19 * 2).reshape(19, 2) % 7 + 123).to(torch.uint8)
    scales = scale_bits.view(torch.float8_e8m0fnu)
    save_file({key: codes, scale_key: scales}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    assert embedding_mod.engram_storage_dtype(tmp_path, 1) == torch.float8_e4m3fn
    embedding_mod.preflight_engram_checkpoint(tmp_path, [1])
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx, table.vocab_end_idx = 5, 17
    table.block_size, table.dim = 32, 64
    table.head_start, table.part_n_hash_cols = 1, 2
    table.weight = torch.nn.Parameter(torch.empty((12, 64), dtype=torch.float8_e4m3fn), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty((12, 2), dtype=torch.uint8), requires_grad=False)
    monkeypatch.setattr(embedding_mod, "quantize_engram_rows", Mock(side_effect=AssertionError("requantization")))
    table.load_checkpoint(tmp_path, key, chunk_rows=5)
    torch.testing.assert_close(table.weight.view(torch.uint8), codes[5:17].view(torch.uint8))
    torch.testing.assert_close(table.weight_scale_inv, scale_bits[5:17])
    ids = torch.tensor([[0, 5, 16], [0, -1, 19]])
    out = torch.empty((2, 2, 64), dtype=torch.bfloat16)
    embedding_mod._torch_lookup(table, ids, out)
    reference = (codes.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)).flatten(-2).bfloat16()
    torch.testing.assert_close(out[0], reference[[5, 16]], rtol=0, atol=0)
    assert not out[1].any()


def test_native_mxfp8_requires_e8m0_scales_before_allocation(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    save_file(
        {key: torch.zeros(8, 64).to(torch.float8_e4m3fn), scale_key: torch.ones(8, 2)},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="requires E8M0 scales"):
        embedding_mod.preflight_engram_checkpoint(tmp_path, [1])
