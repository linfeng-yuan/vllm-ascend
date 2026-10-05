# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve the fused V4/V4.1 norm before forward, never during graph capture."""

import torch
from vllm import envs

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op, is_950, load_custom_op_library


def get_rms_norm_cast_op():
    # Called by decoder construction after worker device selection. A5 allows
    # this individual vetted operator without enabling every custom-op path.
    if envs.VLLM_BATCH_INVARIANT:
        return None
    if is_950():
        load_custom_op_library()
    elif not (get_current_hardware_profile().supports(HardwareCapability.RMS_NORM_CAST) and enable_custom_op()):
        return None
    return torch.ops._C_ascend.npu_rms_norm_cast
