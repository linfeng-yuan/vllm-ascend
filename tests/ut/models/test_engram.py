# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize(
    "enabled,shared,tp,mode,expected",
    [
        (True, True, 1, "FULL", True),
        (True, True, 1, "NONE", True),
        (False, True, 1, "FULL", False),
        (True, False, 1, "FULL", False),
        (True, True, 2, "FULL", False),
        (True, True, 1, "PIECEWISE", False),
    ],
)
def test_lookup_overlap_requires_shared_tp1_and_supported_runtime(monkeypatch, enabled, shared, tp, mode, expected):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import model as model_module

    model = SimpleNamespace(_engram_overlap_enabled=enabled, engram_dp_shared_memory=shared)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        model_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode[mode])
    )
    assert model_module.DeepseekV41Model._can_defer_engram_lookup(model) is expected


def test_bf16_gate_without_rotation():
    hidden = torch.ones(2, 4, 64, dtype=torch.bfloat16)
    value = torch.full((2, 64), 0.25, dtype=torch.bfloat16)
    mask = torch.tensor([True, False])
    result = engram_gate(hidden, hidden * 2, value, torch.ones(4, 64), None, mask, 1e-20)
    expected = (1 + 0.25 * torch.sigmoid(torch.tensor(8.0).sqrt())).bfloat16()
    assert torch.all(result[0] == expected)
    assert torch.equal(result[1], hidden[1])


def _fake_host_library(device_offset):
    class Library:
        def aclrtHostRegisterV2(self, pointer, size, flags):
            return 0

        def aclrtHostGetDevicePointer(self, pointer, out, flags):
            out._obj.value = pointer.value + device_offset
            return 0

        def aclrtHostUnregister(self, pointer):
            return 0

    return Library()


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


def _runner(rows, computed, prompt):
    token_ids = np.full((len(rows), 16), -7, dtype=np.int32)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = row
    runner = object.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        num_reqs=len(rows),
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )
    lookback = np.empty((len(rows), 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=lookback,
        copy_to_gpu=lambda: torch.from_numpy(lookback.copy()),
    )
    runner.is_pooling_model = False
    return runner


@pytest.mark.parametrize(
    "computed,num_reqs,expected",
    [
        (4, None, [13, 12, 11]),
        (6, None, [-1, -1, 13]),
        (0, None, [-1, -1, -1]),
        (4, 0, [-1, -1, -1]),
        (4, 1, [13, 12, 11]),
    ],
)
def test_v1_lookback_uses_prompt_tokens_once(computed, num_reqs, expected):
    runner = _runner([[10, 11, 12, 13, -7, -7]], [computed], [4])
    copy = Mock(wraps=runner.lookback_token_ids.copy_to_gpu)
    runner.lookback_token_ids.copy_to_gpu = copy
    kwargs = runner._init_model_kwargs(num_reqs=num_reqs)
    assert kwargs["lookback_token_ids"][0].tolist() == expected
    copy.assert_called_once_with()


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
