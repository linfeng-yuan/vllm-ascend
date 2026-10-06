# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerical comparison of token-tiled and row-wise HOST_UVA FP8 lookups."""

from contextlib import ExitStack

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu

pytestmark = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required")


@pytest.mark.parametrize(
    "num_tokens,local_heads",
    [(1, 24), (17, 24), (128, 24), (128, 3), (384, 3)],
)
def test_host_uva_fp8_token_tiles_match_rowwise_lookup(num_tokens, local_heads):
    width = 256
    table_rows = 64
    vocab_start = 11
    head_start = 2
    pad_heads = local_heads + 2
    device = torch.device("npu:0")
    codes_data = torch.linspace(-2, 2, table_rows * width, dtype=torch.float32).reshape(table_rows, width)
    codes_data = codes_data.to(torch.float8_e4m3fn)
    scale_data = (124 + torch.arange(table_rows * (width // npu.SCALE_GROUP)) % 5).to(torch.uint8)
    scale_data = scale_data.reshape(table_rows, width // npu.SCALE_GROUP)

    with ExitStack() as stack:
        codes = npu.HostUvaBuffer((table_rows, width), torch.float8_e4m3fn, device)
        stack.callback(codes.close)
        scales = npu.HostUvaBuffer(scale_data.shape, torch.uint8, device)
        stack.callback(scales.close)
        codes.tensor.copy_(codes_data)
        scales.tensor.copy_(scale_data)

        ids_data = torch.arange(num_tokens * (local_heads + head_start), dtype=torch.int32)
        ids_data = (ids_data % table_rows + vocab_start).reshape(num_tokens, local_heads + head_start)
        ids_data[0, head_start] = -1
        ids_data[-1, head_start + local_heads - 1] = vocab_start + table_rows
        ids = ids_data.to(device)
        output_shape = (num_tokens * pad_heads, width)
        tiled = torch.full(output_shape, -3, dtype=torch.bfloat16, device=device)
        rowwise = torch.full_like(tiled, -3)

        npu.gather_dequantize_host_uva(
            codes,
            scales,
            ids,
            head_start=head_start,
            local_heads=local_heads,
            pad_heads=pad_heads,
            output=tiled,
            vocab_start=vocab_start,
            vocab_end=vocab_start + table_rows,
        )
        # Launch the original one-row-per-program schedule as the oracle.
        rows = num_tokens * local_heads
        npu._engram_host_uva_gather_dequant_kernel[(rows,)](
            codes.ptrs,
            scales.ptrs,
            ids,
            rowwise,
            rows,
            vocab_start,
            vocab_start + table_rows,
            ids.stride(0),
            CHUNK=npu.CHUNK_ROWS,
            WIDTH=width,
            GROUP=npu.SCALE_GROUP,
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            PAD_HEADS=pad_heads,
            QUANTIZED=True,
            MXFP8=True,
            num_warps=4,
        )
        torch.npu.synchronize()
        assert torch.equal(tiled, rowwise)
