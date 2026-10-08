# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A5 V4.1 serving smoke; requires the packaged operators and full checkpoint."""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dspark",
    parallel="DP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="FP8",
    graph_mode="full_decode_only",
)
@pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE),
    reason="Requires the A5 V4.1 packed-cache operators",
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v41_dspark_dp_graph():
    # Two requests per rank exercise both decode graph replay and batch handling.
    prompts = ["The capital of France is", "The sum of two and three is"] * 8
    sampling_params = SamplingParams(max_tokens=32, temperature=0, seed=1024)
    with DPVllmRunner(
        MODEL,
        data_parallel_size=8,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        max_model_len=4096,
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        block_size=128,
        gpu_memory_utilization=0.9,
        quantization="deepseek_v4_fp8",
        tokenizer_mode="deepseek_v41",
        engram_config={"cpu_offload": True, "dp_shared_memory": True},
        limit_mm_per_prompt={"image": 0},
        async_scheduling=True,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": False},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        additional_config={"multistream_engram_overlap": False},
    ) as runner:
        outputs = runner.generate(prompts, sampling_params=sampling_params)
        assert len(outputs) == len(prompts)
        for token_ids, texts in outputs:
            assert len(token_ids) == len(texts) == 1
            assert 0 < len(token_ids[0]) <= sampling_params.max_tokens
            assert texts[0].strip()
