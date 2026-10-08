# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import runpy
from pathlib import Path


def test_engram_patch_is_noop_without_upstream_config(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(root / "vllm_ascend/patch/platform/patch_engram_config.py"))
    assert "verify_model_config" not in namespace


def test_elastic_switch_uses_engram_config():
    import argparse

    import pytest
    from vllm.config import VllmConfig
    from vllm.engine import arg_utils

    from vllm_ascend.patch.platform.patch_engram_config import AscendEngramConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--engram-config", **arg_utils.get_kwargs(VllmConfig)["engram_config"])
    native = parser.parse_args(["--engram-config", "{}"]).engram_config
    elastic = parser.parse_args(["--engram-config", '{"use_elastic_buffer":true}']).engram_config
    assert not native.use_elastic_buffer and elastic.use_elastic_buffer
    assert native.compute_hash() != elastic.compute_hash()
    assert arg_utils.EngineArgs(engram_config={"use_elastic_buffer": True}).engram_config.use_elastic_buffer
    for invalid in ({"use_elastic_buffer": True, "cpu_offload": False}, {"use_elastic_buffer": "bad"}):
        with pytest.raises(ValueError):
            AscendEngramConfig(**invalid)
