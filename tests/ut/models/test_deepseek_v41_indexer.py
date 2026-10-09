# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index selection lifecycle, without executing packaged QLI kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("reuse_output", [False, True])
@pytest.mark.parametrize("tokens", [0, 3])
def test_empty_index_selection_does_not_reuse_stale_results(source, reuse_output, tokens):
    indexer = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_topk = 4
    indexer.packed_cache_ops = Mock()
    indexer.quantize_query = Mock(side_effect=AssertionError("empty selection must not quantize"))
    indices = torch.full((tokens, 4), 99, dtype=torch.int32) if reuse_output else None
    lengths = torch.full((tokens,), 99, dtype=torch.int32)
    candidate_lengths = lengths.clone()
    candidates = torch.full((tokens, 1, 2), 99, dtype=torch.int32)

    selected, blocks = indexer.select_projected(
        torch.empty(tokens, 1, 8),
        torch.empty(tokens, 1),
        torch.arange(tokens),
        None,
        SimpleNamespace(max_cache_seq_len=0),
        is_candidate_source=source,
        uses_candidate_filter=not source,
        candidate_topk_blocks=2,
        candidate_block_size=8,
        candidates=candidates,
        candidate_lengths=candidate_lengths,
        topk_lengths=lengths,
        indices_output=indices,
    )

    indexer.packed_cache_ops.run_a5_indexer.assert_not_called()
    indexer.quantize_query.assert_not_called()
    if reuse_output:
        assert selected is indices
        assert torch.all(selected == -1)
    else:
        assert selected.shape == (tokens, 4 if tokens == 0 else 0)
    assert selected.dtype == torch.int32
    assert torch.all(lengths == 0)
    if source:
        assert blocks.shape == (tokens, 1, 2)
        assert torch.all(blocks == -1)
        assert torch.all(candidate_lengths == 0)
    else:
        assert blocks is candidates
        assert torch.all(blocks == 99)
        assert torch.all(candidate_lengths == 99)


@pytest.mark.parametrize("tuple_output", [False, True])
def test_projection_preserves_heads_and_scales_weights_in_fp32(tuple_output):
    indexer = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_heads, indexer.width, indexer.weights_scale = 2, 4, 0.125
    projected = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    weights = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16)
    indexer.wq_b = Mock(return_value=(projected, None) if tuple_output else projected)
    indexer.weights_proj = Mock(return_value=(weights, None) if tuple_output else weights)
    inputs = torch.ones(3, 8)

    query = indexer.project_query(inputs)
    scores = indexer.project_weights(inputs)

    assert query.shape == (3, 2, 4)
    torch.testing.assert_close(query.flatten(1), projected)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, weights.float() / 8)
    indexer.wq_b.assert_called_once_with(inputs)
    indexer.weights_proj.assert_called_once_with(inputs)
