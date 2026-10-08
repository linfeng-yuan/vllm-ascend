# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("vllm.models.deepseek_v41")

from vllm_ascend.models.deepseek_v41.engram import parallel
from vllm_ascend.models.deepseek_v41.engram.elastic import preflight_elastic_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    weight_key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    save_file(
        {
            weight_key: torch.zeros(8, 64).to(torch.float8_e4m3fn),
            scale_key: torch.ones(8, 2).to(torch.float8_e8m0fnu),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    return tmp_path


def test_elastic_checkpoint_accepts_fp8_e8m0(checkpoint):
    preflight_elastic_checkpoint(checkpoint, [1], [8], 64)


def test_elastic_checkpoint_rejects_missing_scale(checkpoint):
    index = checkpoint / "model.safetensors.index.json"
    contents = json.loads(index.read_text())
    del contents["weight_map"]["layers.1.engram.embed.scale"]
    index.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="needs.*scale"):
        preflight_elastic_checkpoint(checkpoint, [1], [8], 64)


def test_elastic_checkpoint_rejects_wrong_scale_dtype(checkpoint):
    save_file(
        {
            "layers.1.engram.embed.weight": torch.zeros(8, 64).to(torch.float8_e4m3fn),
            "layers.1.engram.embed.scale": torch.ones(8, 2),
        },
        checkpoint / "model.safetensors",
    )
    with pytest.raises(ValueError, match="F8_E8M0"):
        preflight_elastic_checkpoint(checkpoint, [1], [8], 64)


def test_shared_memory_requires_local_dp_peers(monkeypatch):
    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 1)
    assert not parallel.resolve_dp_shared_memory(True)
    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 2)
    assert parallel.resolve_dp_shared_memory(True)
    assert not parallel.resolve_dp_shared_memory(False)


def test_overlap_requires_full_graph(monkeypatch):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import model

    target = SimpleNamespace(has_engram=True, _engram_overlap_enabled=True)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    assert model.DeepseekV41Model._can_overlap_engram_preparation(target)
    context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    assert not model.DeepseekV41Model._can_overlap_engram_preparation(target)
