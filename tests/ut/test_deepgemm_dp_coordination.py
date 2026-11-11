# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""D-only DeepInfer bypass without changing the communication implementation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend import ascend_forward_context, utils
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.mark.parametrize(
    "deepinfer,consumer,recompute,hierarchy,cann,method,expected",
    [
        (True, True, True, False, False, MoECommType.ALLTOALL, True),
        (True, False, True, False, False, MoECommType.ALLTOALL, False),
        (True, True, False, False, False, MoECommType.ALLTOALL, False),
        (True, True, True, True, False, MoECommType.ALLTOALL, False),
        (True, True, True, False, True, MoECommType.ALLTOALL, False),
        (False, True, True, False, False, MoECommType.ALLTOALL, False),
        (False, True, True, False, False, MoECommType.MC2, True),
    ],
)
@pytest.mark.parametrize("draft", [False, True])
def test_deepinfer_d_skip_scope(monkeypatch, deepinfer, consumer, recompute, hierarchy, cann, method, expected, draft):
    ascend = SimpleNamespace(
        deepgemm_mega_moe=deepinfer,
        get_mc2_comm_alg=lambda: "hierarchy" if hierarchy else "fullmesh",
        scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute),
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=consumer),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(separate_routine=lambda: True)),
    )
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(utils, "is_mega_moe_supported", lambda: cann)
    monkeypatch.setattr(utils, "is_moe_model", lambda _: True)
    monkeypatch.setattr(utils, "is_drafter_moe_model", lambda _: True)
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: 384)
    monkeypatch.setattr(ascend_forward_context, "use_cann_megamoe", lambda _: False)
    selector = Mock(return_value=method)
    monkeypatch.setattr(ascend_forward_context, "select_moe_comm_method", selector)

    assert utils.should_skip_allreduce_across_dp_group(config, is_draft_model=draft) is expected
    if deepinfer:
        selector.assert_not_called()
