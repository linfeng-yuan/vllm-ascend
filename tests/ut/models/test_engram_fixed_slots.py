# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import parallel


def test_decode_slot_covers_mtp_and_graph_padding(monkeypatch):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=50),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    # Decode metadata may be absent when ranks skip synchronization.
    monkeypatch.setattr(parallel, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None))
    monkeypatch.setattr(parallel, "is_pd_decode_recompute_scheduler_enabled", lambda: True)
    monkeypatch.setattr(parallel, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: 256)
    assert parallel.engram_gathered_num_tokens() == 300
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: 320)
    assert parallel.engram_gathered_num_tokens() == 320


def test_synchronized_slot_uses_local_dp_group(monkeypatch):
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([900, 1024, 4, 2])))
    monkeypatch.setattr(parallel, "get_forward_context", lambda: context)
    monkeypatch.setattr(parallel, "is_pd_decode_recompute_scheduler_enabled", lambda: False)
    monkeypatch.setattr(parallel, "get_dp_group", lambda: SimpleNamespace(rank_in_group=2))
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=2))
    assert parallel.engram_gathered_num_tokens() == 4


def test_overflow_rejected_but_shared_lookup_bypasses_gather(monkeypatch):
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: SimpleNamespace())
    monkeypatch.setattr(parallel, "engram_gathered_num_tokens", lambda: 3)
    hashes = torch.zeros(4, 24, dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds the DP token slot"):
        parallel.gather_engram_hashes(hashes)
    assert parallel.gather_engram_hashes(hashes, dp_shared_memory=True) is hashes
