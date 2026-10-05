# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu import dp_utils as upstream
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2 import dp_utils


@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("ubatching", [False, True])
def test_policy_is_scoped_and_preserves_dbo_coordination(monkeypatch, eligible, ubatching):
    calls = []
    config = object()
    monkeypatch.setattr(dp_utils, "should_skip_allreduce_across_dp_group", lambda cfg: calls.append(cfg) or eligible)
    assert not should_skip_dp_coordination()
    with pytest.raises(ValueError), dp_utils.dp_coordination_context(config, allow_ubatching=ubatching):
        assert should_skip_dp_coordination() == (eligible and not ubatching)
        raise ValueError("forward failure")
    assert not should_skip_dp_coordination()
    assert calls == ([] if ubatching else [config])


@pytest.mark.parametrize("tokens", [0, 1, 6, 384])
@pytest.mark.parametrize("rank", [0, 7])
def test_eligible_local_batches_do_not_call_gloo(monkeypatch, tokens, rank):
    monkeypatch.setattr(dp_utils, "should_skip_allreduce_across_dp_group", lambda _: True)
    monkeypatch.setattr(upstream, "get_dp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(upstream.dist, "all_reduce", lambda *a, **kw: pytest.fail("unexpected Gloo collective"))
    requests = min(tokens, 64)
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=tokens, num_reqs=requests)
    with dp_utils.dp_coordination_context(object(), allow_ubatching=False):
        actual, sync = upstream.sync_cudagraph_and_dp_padding(None, desc, tokens, requests, None, 8, rank)
    assert actual.num_tokens == tokens
    if tokens:
        assert sync.num_tokens_across_dp.tolist() == [tokens] * 8
        assert sync.eager and sync.num_reqs == requests
    else:
        assert sync is None


def test_ineligible_configuration_retains_gloo(monkeypatch):
    calls = []
    group = object()
    monkeypatch.setattr(dp_utils, "should_skip_allreduce_across_dp_group", lambda _: False)
    monkeypatch.setattr(upstream, "get_dp_group", lambda: SimpleNamespace(cpu_group=group))
    monkeypatch.setattr(upstream.dist, "all_reduce", lambda value, *, group: calls.append(group))
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=6, num_reqs=1)
    with dp_utils.dp_coordination_context(object(), allow_ubatching=False):
        upstream.sync_cudagraph_and_dp_padding(None, desc, 6, 1, 6, 8, 0)
    assert calls == [group]


@pytest.mark.parametrize("tokens,padded,requests", [(6, 12, 1), (24, 24, 4), (384, 384, 64)])
def test_local_full_graph_bucket_retains_padding_without_collective(monkeypatch, tokens, padded, requests):
    monkeypatch.setattr(dp_utils, "should_skip_allreduce_across_dp_group", lambda _: True)
    monkeypatch.setattr(upstream, "get_dp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(upstream.dist, "all_reduce", lambda *a, **kw: pytest.fail("unexpected Gloo collective"))
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=padded, num_reqs=padded // 6)
    manager = SimpleNamespace(dispatch=lambda *a, **kw: desc)
    with dp_utils.dp_coordination_context(object(), allow_ubatching=False):
        actual, sync = upstream.sync_cudagraph_and_dp_padding(manager, desc, tokens, requests, 6, 8, 0)
    assert actual is desc
    assert sync.num_tokens_across_dp.tolist() == [padded] * 8
    assert not sync.eager and sync.num_reqs == padded // 6
