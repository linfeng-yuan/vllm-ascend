"""An experiment-only EP8 placement and staging path for target DeepGEMM layers."""

from functools import lru_cache
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def load_placement():
    path = os.environ.get("CODEX_DEEPGEMM_PLACEMENT")
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def place_checkpoint_experts(layer_name, manager):
    """Preserve logical checkpoint IDs while loading the selected local slots."""
    layout = load_placement().get(layer_name)
    if layout is None:
        return None
    assert manager.ep_size == len(layout) == 8
    assert manager.local_num_experts == 48 and manager.global_num_experts == 384
    order = [expert for group in layout for expert in group]
    assert all(len(group) == 48 for group in layout)
    assert sorted(order) == list(range(384))
    existing = manager.expert_map
    assert existing is not None
    local = [-1] * 384
    for slot, expert in enumerate(layout[manager.ep_rank]):
        local[expert] = slot
    existing.copy_(torch.tensor(local, dtype=existing.dtype, device=existing.device))
    physical = [0] * 384
    for slot, logical in enumerate(order):
        physical[logical] = slot
    return physical


@triton.jit(do_not_specialize=["num_routes"])
def _stage_experts(source, mapping, destination, num_routes, TOPK: tl.constexpr,
                   ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = route < num_routes
    logical = tl.load(source + route // TOPK * ROW_STRIDE + route % TOPK * COL_STRIDE,
                      mask=valid, other=-1)
    physical = tl.load(mapping + logical.to(tl.int32), mask=valid & (logical >= 0) & (logical < 384), other=-1)
    tl.store(destination + route, physical, mask=valid)


def stage_experts(destination, logical, mapping):
    num_routes = logical.numel()
    if num_routes:
        _stage_experts[(triton.cdiv(num_routes, 256),)](
            logical, mapping, destination, num_routes, logical.shape[1], *logical.stride(), 256,
        )
