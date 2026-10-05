# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.dsv41_a5.indexer_k import IndexerKFusion


@pytest.fixture
def adapter(monkeypatch):
    k_module = ModuleType("ops.indexer_prologue_k")
    qw_module = ModuleType("ops.indexer_prologue_qw")
    calls = []
    k_module.indexer_prologue_k = lambda *args, **kwargs: calls.append((args, kwargs))
    qw_module.to_nz = lambda tensor: tensor.clone()
    monkeypatch.setitem(sys.modules, k_module.__name__, k_module)
    monkeypatch.setitem(sys.modules, qw_module.__name__, qw_module)
    wk = SimpleNamespace(weight=torch.zeros(128, 512, dtype=torch.bfloat16))
    norm = SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16), variance_epsilon=1e-6)
    return IndexerKFusion(wk, norm, 64), calls


@pytest.mark.parametrize("page", [64, 128])
def test_preserves_strided_cache_and_int64_slots(adapter, page):
    fusion, calls = adapter
    backing = torch.zeros(page * 256 * 3 + 256, dtype=torch.uint8)
    key = backing.as_strided((3, page, 1, 64), (page * 256, 64, 64, 1), 256)
    scale = backing.as_strided((3, page, 1, 4), (page * 256, 4, 4, 1), 256 + page * 64)
    slots = torch.tensor([page - 1, page, -1], dtype=torch.int64)
    fusion(
        torch.zeros(3, 512, dtype=torch.bfloat16),
        slots,
        torch.ones(3, 1, 1, 64),
        torch.zeros(3, 1, 1, 64),
        (key, scale),
    )
    args, kwargs = calls[0]
    assert args[5] is key and args[6] is scale
    assert kwargs["cache_index"] is slots
    assert kwargs["storage_mode"] == 0 and kwargs["norm_eps"] == 1e-6
    assert args[2].dtype == torch.float32
    assert args[3].shape == (3, 64)
    assert not fusion.state_dict()


def test_empty_and_missing_or_narrow_slots(adapter):
    fusion, calls = adapter
    fusion(torch.empty(0, 512), None, None, None, None)
    assert not calls
    for slots in (None, torch.zeros(1, dtype=torch.int32), torch.zeros(2, dtype=torch.int64)):
        with pytest.raises(ValueError, match="INT64 flat slots"):
            fusion(torch.empty(1, 512), slots, None, None, None)


def test_large_slot_not_narrowed(adapter):
    fusion, calls = adapter
    slots = torch.tensor([2**31 + 17], dtype=torch.int64)
    fusion(torch.zeros(1, 512, dtype=torch.bfloat16), slots, torch.ones(1, 64), torch.zeros(1, 64), (None, None))
    assert calls[0][1]["cache_index"].item() == 2**31 + 17


def test_source_writer_keeps_folded_twin_and_skips_nonowners(monkeypatch):
    from vllm_ascend.models.deepseek_v41 import indexer

    calls = []
    cache, folded = object(), object()

    class Fused:
        supports_tokens = staticmethod(lambda tokens: tokens == 2)

        def __call__(self, *args):
            calls.append(("fused", args))

    source = SimpleNamespace(
        owns_k=True,
        k_fusion=Fused(),
        k_cache=SimpleNamespace(kv_cache=[cache]),
        k_cache_folded=SimpleNamespace(kv_cache=[folded]),
    )
    monkeypatch.setattr(indexer, "fold_indexer_cache_rows", lambda *args: calls.append(("fold", args)))
    latent, slots, flat = torch.zeros(2, 512), object(), object()
    indexer.DeepseekV41Indexer.update_keys(source, latent, slots, None, None, flat)
    assert calls == [("fused", (latent, flat, None, None, cache)), ("fold", (cache, folded, slots))]
    calls.clear()
    source.owns_k = False
    indexer.DeepseekV41Indexer.prepare_k_fusion(source, [2])
    indexer.DeepseekV41Indexer.update_keys(source, latent, slots, None, None, flat)
    assert not calls
    source.owns_k = True
    indexer.DeepseekV41Indexer.update_keys(source, torch.empty(0, 512), slots, None, None, flat)
    assert not calls


def test_only_configured_graph_token_counts_are_eligible(adapter):
    fusion, _ = adapter
    fusion.token_counts = frozenset([1, 2, 4, 8, 16, 32, 64, 384])
    assert fusion.supports_tokens(64)
    assert fusion.supports_tokens(384)
    assert not fusion.supports_tokens(63)
    assert not fusion.supports_tokens(1024)
    fusion.token_counts = frozenset()
    assert not fusion.supports_tokens(1)


def test_unbucketed_shape_uses_original_writer_without_compiling():
    from vllm_ascend.models.deepseek_v41 import indexer

    writes = []
    key, scale, slots = object(), object(), object()
    source = SimpleNamespace(
        owns_k=True,
        k_fusion=SimpleNamespace(supports_tokens=lambda tokens: False),
        wk=lambda x: torch.zeros(x.shape[0], 128),
        k_norm=lambda x: x,
        width=128,
        rope_width=64,
        k_cache=SimpleNamespace(kv_cache=[(key, scale)]),
        k_cache_folded=None,
        dsv41_backend=SimpleNamespace(
            apply_partial_rotary_inplace=lambda *args, **kwargs: None,
            write_index_cache=lambda *args: writes.append(args),
        ),
    )
    indexer.DeepseekV41Indexer.update_keys(source, torch.empty(7, 512), slots, None, None)
    assert writes[0][0] == (key, scale)
    assert writes[0][1] is slots
    assert writes[0][2].shape == (7, 128)
