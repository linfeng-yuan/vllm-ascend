# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import utils
from vllm_ascend.models.deepseek_v41.engram import parallel


@pytest.fixture
def runtime(monkeypatch):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    ascend = SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=True))
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([7, 7])))
    group = SimpleNamespace(rank_in_group=0, world_size=2)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(parallel, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(parallel, "get_forward_context", lambda: context)
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: 96)
    monkeypatch.setattr(parallel, "get_dp_group", lambda: group)
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: group)
    return config, ascend, context


def test_fixed_slot_ignores_rank_local_metadata(runtime):
    context = runtime[2]
    for local_count in (0, 1, 42, 48, 96):
        context.dp_metadata.num_tokens_across_dp_cpu.fill_(local_count)
        assert parallel.engram_gathered_num_tokens() == 96


def test_synchronized_paths_keep_metadata_slot(runtime):
    config, ascend, context = runtime
    context.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([900, 1024])
    for mode in ("prefill", "recompute_off", "profile", "uniform_warmup"):
        config.kv_transfer_config.is_kv_consumer = mode != "prefill"
        ascend.scheduler_config.recompute_scheduler_enable = mode != "recompute_off"
        context.in_profile_run = mode == "profile"
        context.engram_uniform_dp_warmup = mode == "uniform_warmup"
        assert parallel.engram_gathered_num_tokens() == 1024


def test_fixed_slot_covers_eager_decode_and_graph_padding(runtime, monkeypatch):
    config = runtime[0]
    for batched, seqs, spec, potential, expected in (
        (2048, 128, 5, 512, 768),
        (640, 128, 5, 512, 640),
        (1024, 16, None, 32, 32),
    ):
        config.scheduler_config.max_num_batched_tokens = batched
        config.scheduler_config.max_num_seqs = seqs
        config.speculative_config = None if spec is None else SimpleNamespace(num_speculative_tokens=spec)
        monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda potential=potential: potential)
        assert parallel.engram_gathered_num_tokens() == expected


def test_overflow_fails_before_collective_and_bypasses_do_not_need_metadata(runtime, monkeypatch):
    hashes = torch.empty((97, 2, 24), dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds the DP token slot"):
        parallel.gather_engram_hashes(hashes)
    assert parallel.gather_engram_hashes(hashes, dp_shared_memory=True) is hashes
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: None)
    assert parallel.gather_engram_hashes(hashes) is hashes
