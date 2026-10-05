# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MRV2 history gather must preserve request permutation and newest-first order."""

import torch
import torch_npu  # noqa: F401
import vllm.v1.worker.gpu.spec_decode.speculator  # noqa: F401

import vllm_ascend.ops  # noqa: F401

# isort: split

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.model_states.deepseek_v41 import _gather_lookback_kernel


def test_engram_lookback_request_permutation_and_empty_rows():
    torch.npu.set_device(0)
    init_device_properties_triton()
    device = torch.device("npu:0")
    history = torch.arange(40, dtype=torch.int32, device=device).view(4, 10)
    computed = torch.tensor([4, 0, 6, 2], dtype=torch.int32, device=device)
    mapping = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
    window = torch.empty((4, 4), dtype=torch.int32, device=device)
    _gather_lookback_kernel[(4,)](window, mapping, computed, history, history.stride(0), 3, DEPTH=4, BLOCK_DEPTH=4)
    expected = torch.tensor([[25, 24, 23, 22], [3, 2, 1, 0], [31, 30, -1, -1], [-1, -1, -1, -1]])
    torch.testing.assert_close(window.cpu(), expected.to(torch.int32))

    # Reuse the same buffer for a shorter next batch; stale history must not survive.
    mapping[:1] = 1
    _gather_lookback_kernel[(4,)](window, mapping[:1], computed, history, history.stride(0), 1, DEPTH=4, BLOCK_DEPTH=4)
    torch.testing.assert_close(window.cpu(), torch.full((4, 4), -1, dtype=torch.int32))
