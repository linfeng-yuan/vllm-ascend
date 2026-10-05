# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops import rms_norm_cast as dispatch


@pytest.mark.parametrize(
    "invariant,a5,supported,enabled,expected",
    [
        (False, True, False, False, True),
        (True, True, False, False, False),
        (False, False, True, True, True),
        (False, False, True, False, False),
        (False, False, False, True, False),
    ],
)
def test_resolve_during_construction(invariant, a5, supported, enabled, expected):
    with (
        patch.object(dispatch.envs, "VLLM_BATCH_INVARIANT", invariant),
        patch.object(dispatch, "is_950", return_value=a5),
        patch.object(dispatch, "enable_custom_op", return_value=enabled),
        patch.object(dispatch, "get_current_hardware_profile") as profile,
        patch.object(dispatch, "load_custom_op_library") as load,
        patch.object(torch.ops._C_ascend, "npu_rms_norm_cast", create=True) as op,
    ):
        profile.return_value.supports.return_value = supported
        actual = dispatch.get_rms_norm_cast_op()
        assert actual is (op if expected else None)
        assert load.call_count == int(a5 and not invariant)


@pytest.mark.parametrize("version", ["v4", "v41"])
@pytest.mark.parametrize("fused", [False, True])
def test_forward_uses_bound_op_without_loading_or_device_queries(version, fused):
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4DecoderLayer
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer

    cls = DeepseekV4DecoderLayer if version == "v4" else DeepseekV41DecoderLayer
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    rounded = torch.randn_like(x)
    norm = MagicMock(return_value=rounded)
    norm.weight = torch.ones(8, dtype=x.dtype)
    norm.variance_epsilon = 1e-6
    op = MagicMock(return_value=(rounded, rounded.float()))
    layer = SimpleNamespace(post_attention_layernorm=norm, _rms_norm_cast_op=op if fused else None)
    with (
        patch.object(dispatch, "load_custom_op_library", side_effect=AssertionError("forward library load")),
        patch.object(dispatch, "is_950", side_effect=AssertionError("forward device query")),
    ):
        actual, wide = cls.rms_norm_cast(layer, x)
    assert actual is rounded
    torch.testing.assert_close(wide, rounded.float(), rtol=0, atol=0)
    if fused:
        op.assert_called_once_with(x, norm.weight, norm.variance_epsilon)
        norm.assert_not_called()
    else:
        norm.assert_called_once_with(x)
        op.assert_not_called()
