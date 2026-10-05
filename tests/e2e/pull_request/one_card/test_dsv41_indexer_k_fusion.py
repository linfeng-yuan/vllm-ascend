# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with the A5 arena/DSL packages used by DeepSeek V4.1 Flash."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.dsv41_a5.indexer_k import IndexerKFusion
from vllm_ascend.ops.dsv41_a5.rotary import apply_partial_rotary_inplace
from vllm_ascend.ops.dsv41_a5.writers import write_index_cache
from vllm_ascend.ops.triton.fold_indexer_cache import fold_indexer_cache_rows
from vllm_ascend.utils import load_custom_op_library


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("tokens", [1, 16, 64, 384])
@torch.inference_mode()
def test_indexer_k_strided_split_and_folded_graph(page, tokens):
    pytest.importorskip("ops.indexer_prologue_k")
    torch.npu.set_device(0)
    load_custom_op_library()
    torch.manual_seed(1005)
    wk = (torch.randn(128, 512, device="npu") * 0.03).bfloat16()
    gamma = torch.randn(128, device="npu").bfloat16()
    adapter = IndexerKFusion(SimpleNamespace(weight=wk), SimpleNamespace(weight=gamma, variance_epsilon=1e-6), 64)
    latent = torch.randn(tokens, 512, device="npu").bfloat16()
    angle = torch.randn(tokens, 32, device="npu")
    cos, sin = angle.cos().repeat_interleave(2, -1), angle.sin().repeat_interleave(2, -1)
    flat = torch.arange(tokens, device="npu", dtype=torch.int64) + page - 1
    if tokens > 1:
        flat[-1] = -1
    slots = torch.stack((flat // page, flat % page), -1).int()
    if tokens > 1:
        slots[-1] = -1
    blocks, stride = (tokens + page - 1) // page + 2, page * 256
    caches = []
    for _ in range(2):
        backing = torch.full((blocks * stride + 256,), 173, dtype=torch.uint8, device="npu")
        key = backing.as_strided((blocks, page, 1, 64), (stride, 64, 64, 1), 256)
        scale = backing.as_strided((blocks, page, 1, 4), (stride, 4, 4, 1), 256 + page * 64)
        folded = torch.full((blocks, page // 8, 1, 544), 173, dtype=torch.uint8, device="npu")
        caches.append((backing, key, scale, folded))

    def reference():
        key = torch.nn.functional.linear(latent, wk)
        key = torch_npu.npu_rms_norm(key, gamma, epsilon=1e-6)[0].view(tokens, 1, 128)
        apply_partial_rotary_inplace(key, cos.view(tokens, 1, 1, 64), sin.view(tokens, 1, 1, 64), start=64, end=128)
        write_index_cache(caches[0][1:3], slots, key.squeeze(1))
        fold_indexer_cache_rows(caches[0][1:3], caches[0][3], slots)

    def fused():
        adapter(latent, flat, cos, sin, caches[1][1:3])
        fold_indexer_cache_rows(caches[1][1:3], caches[1][3], slots)

    reference()
    fused()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused()
    # Change input contents without changing captured addresses. Compare the
    # whole allocation, not only active rows: padding and skipped slots matter.
    for _ in range(3):
        latent.normal_()
        reference()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(caches[0][0], caches[1][0], rtol=0, atol=0)
        torch.testing.assert_close(caches[0][3], caches[1][3], rtol=0, atol=0)
