# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse Ascend's MRV1 eligibility policy with MRV2's scoped DP protocol."""

from contextlib import nullcontext

from vllm.config import VllmConfig
from vllm.v1.worker.dp_utils import skip_dp_coordination

from vllm_ascend.utils import should_skip_allreduce_across_dp_group


def dp_coordination_context(vllm_config: VllmConfig, *, allow_ubatching: bool):
    # DBO needs a joint decision about splitting and padding; MRV1's token
    # count predicate alone does not establish that agreement for MRV2.
    if not allow_ubatching and should_skip_allreduce_across_dp_group(vllm_config):
        return skip_dp_coordination()
    return nullcontext()
