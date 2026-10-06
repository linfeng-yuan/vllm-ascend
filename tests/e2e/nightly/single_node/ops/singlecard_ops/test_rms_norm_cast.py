# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

from vllm_ascend.models.deepseek_v4.model import DeepseekV4DecoderLayer
from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer
from vllm_ascend.ops.rms_norm_cast import _a5_rms_norm_cast


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Independent FP32 reductions can land on adjacent BF16 values.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    return tolerance, tolerance


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("hidden_size", [5120, 7168])
def test_rms_norm_cast(dtype: torch.dtype, num_tokens: int, hidden_size: int):
    torch.manual_seed(7)
    epsilon = 1e-6
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="npu")
    gamma = torch.randn(hidden_size, dtype=dtype, device="npu")

    expected, _ = torch_npu.npu_rms_norm(x, gamma, epsilon)
    actual, actual_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(x, gamma, epsilon)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # Routing must consume the widened, already-rounded RMSNorm result.
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [5120, 7168])
@pytest.mark.parametrize("layer_type", [DeepseekV4DecoderLayer, DeepseekV41DecoderLayer])
@pytest.mark.parametrize("rows", [16, 1024])
def test_rms_norm_cast_npu_graph(dtype: torch.dtype, hidden_size: int, layer_type, rows):
    torch.manual_seed(11)
    x = torch.randn(rows, hidden_size, dtype=dtype, device="npu")
    gamma = torch.randn(hidden_size, dtype=dtype, device="npu")
    layer = SimpleNamespace(
        post_attention_layernorm=SimpleNamespace(weight=gamma, variance_epsilon=1e-6),
        _rms_norm_cast_op=_a5_rms_norm_cast,
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        actual, actual_fp32 = layer_type.rms_norm_cast(layer, x)
    rtol, atol = _tolerances(dtype)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        expected, _ = torch_npu.npu_rms_norm(x, gamma, 1e-6)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)


def test_shape_dispatch_is_fullgraph_traceable():
    compiled = torch.compile(_a5_rms_norm_cast, backend="eager", fullgraph=True, dynamic=True)
    weight = torch.ones(5120, dtype=torch.bfloat16, device="npu")
    for rows in (16, 1024, 16):
        x = torch.randn(rows, 5120, dtype=weight.dtype, device="npu")
        low, wide = compiled(x, weight, 1e-6)
        expected, _ = torch_npu.npu_rms_norm(x, weight, 1e-6)
        torch.testing.assert_close(low, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(wide, low.float(), rtol=0, atol=0)
