# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer K projection and fused postscatter into the existing split cache."""

import torch
from torch import nn


class IndexerKFusion(nn.Module):
    def __init__(self, wk, k_norm, rope_width: int, token_counts=None):
        super().__init__()
        # Use the installed arena implementation, not a second generated kernel.
        # Import and static NZ packing occur before warmup / KV memory sizing.
        from ops.indexer_prologue_k import indexer_prologue_k
        from ops.indexer_prologue_qw import to_nz

        if tuple(wk.weight.shape) != (128, 512) or wk.weight.dtype != torch.bfloat16:
            raise ValueError("K fusion requires Flash BF16 wk [128,512]")
        if tuple(k_norm.weight.shape) != (128,) or rope_width != 64:
            raise ValueError("K fusion requires 128-wide RMSNorm and interleaved RoPE width 64")
        self.kernel = indexer_prologue_k
        self.norm_eps = k_norm.variance_epsilon
        self.token_counts = None if token_counts is None else frozenset(token_counts)
        self.register_buffer("wk_nz", to_nz(wk.weight.contiguous()), persistent=False)
        self.register_buffer("norm_weight", k_norm.weight.float().contiguous(), persistent=False)

    def supports_tokens(self, tokens: int) -> bool:
        # Model integration restricts this shape-specialized DSL kernel to
        # configured graph buckets. Arbitrary mixed-prefill shapes must not
        # synchronously compile new binaries in the live request path.
        return self.token_counts is None or tokens in self.token_counts

    def forward(self, latent, flat_slots, cos, sin, cache):
        tokens = latent.shape[0]
        if tokens == 0:
            return
        # Reuse the builder's compression-aware INT64 slots. Reconstructing
        # these from framework block128 would be wrong for R=2 storage page64.
        if flat_slots is None or flat_slots.dtype != torch.int64 or flat_slots.shape != (tokens,):
            raise ValueError("K fusion requires prepared INT64 flat slots [T]")
        key, scale = cache
        self.kernel(
            latent,
            self.wk_nz,
            self.norm_weight,
            sin.reshape(tokens, 64).float().contiguous(),
            cos.reshape(tokens, 64).float().contiguous(),
            key,
            scale,
            cache_index=flat_slots,
            storage_mode=0,
            norm_eps=self.norm_eps,
        )
