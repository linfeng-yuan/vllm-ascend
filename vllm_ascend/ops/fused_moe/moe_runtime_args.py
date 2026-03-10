# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import torch

from vllm_ascend.quantization.quant_type import QuantType


@dataclass(frozen=True, slots=True)
class MoEReservedQuantSpec:
    """Reserved quant fields for future rollout.

    These fields are intentionally modeled now and can remain no-op in the
    execution path until the related kernels are fully wired.
    """

    round_mode: str = "rint"
    rollback_quant_config: dict | None = None


@dataclass(frozen=True, slots=True)
class MoEMxfpSpec:
    """MXFP-only precision settings."""

    act_quant_type: torch.dtype | None = None
    weight_quant_type: torch.dtype | None = None
    scale_dtype: torch.dtype | None = None
    per_token_scale_dtype: torch.dtype | None = None
    use_bf16: bool = True


@dataclass(frozen=True, slots=True)
class MoEQuantSpec:
    """Quantization semantic configuration for MoE runtime."""

    quant_type: QuantType = QuantType.NONE
    comm_quant_mode: int | None = None
    mxfp: MoEMxfpSpec | None = None
    reserved: MoEReservedQuantSpec = field(default_factory=MoEReservedQuantSpec)

    @property
    def is_quant(self) -> bool:
        return self.quant_type != QuantType.NONE

    @property
    def is_mxfp(self) -> bool:
        return self.quant_type == QuantType.MXFP8

    @property
    def is_int_quant(self) -> bool:
        return self.quant_type in (QuantType.W8A8, QuantType.W4A8)

    @property
    def dispatch_with_quant(self) -> bool:
        return self.quant_type in (QuantType.W8A8, QuantType.W4A8, QuantType.MXFP8)


@dataclass(frozen=True, slots=True)
class MoEQuantTensors:
    """Quant tensor pack for MoE MLP execution."""

    w1_scale: list[torch.Tensor] | torch.Tensor | None = None
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None
    w1_scale_bias: torch.Tensor | None = None
    w2_scale_bias: torch.Tensor | None = None
    w1_offset: torch.Tensor | None = None
    w2_offset: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEWeightPack:
    """MoE dense/quant weight tensors used in fused experts."""

    w1: torch.Tensor | list[torch.Tensor]
    w2: torch.Tensor | list[torch.Tensor]
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEDispatchSpec:
    """Dispatch stage runtime settings."""

    expert_map: torch.Tensor | None
    global_redundant_expert_num: int
    mc2_mask: torch.Tensor | None
    apply_router_weight_on_input: bool
    dynamic_eplb: bool
    log2phy: torch.Tensor | None = None
    pertoken_scale: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEMlpSpec:
    """MLP stage runtime settings."""

    activation: str = "silu"
    need_trans: bool = False
    dynamic_eplb: bool = False


@dataclass(frozen=True, slots=True)
class MoEMlpKernelSpec:
    """MLP kernel execution settings."""

    fusion: bool
    use_mxfp_quant: bool
    act_quant_type: torch.dtype | None = None
    weight_quant_type: torch.dtype | None = None
    scale_type: torch.dtype | None = None
    per_token_scale_type: torch.dtype | None = None
    use_bf16: bool = True


@dataclass(frozen=True, slots=True)
class FusedExpertsRequest:
    """Unified request for MoE fused experts pipeline."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    weights: MoEWeightPack
    dispatch: MoEDispatchSpec
    mlp: MoEMlpSpec
    quant: MoEQuantSpec
    quant_tensors: MoEQuantTensors = field(default_factory=MoEQuantTensors)


@dataclass(frozen=True, slots=True)
class TokenDispatchRequest:
    """Typed request for token dispatch stage."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    dispatch: MoEDispatchSpec
    quant: MoEQuantSpec


@dataclass(frozen=True, slots=True)
class MlpComputeRequest:
    """Typed request for MLP compute stage."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    dynamic_scale: torch.Tensor | None
    topk_scales: torch.Tensor | None
    weights: MoEWeightPack
    quant: MoEQuantSpec
    quant_tensors: MoEQuantTensors
    mlp: MoEMlpSpec
    kernel: MoEMlpKernelSpec


@dataclass(frozen=True, slots=True)
class MC2CombineContext:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_map: torch.Tensor | None
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expand_scales: torch.Tensor | None
    dispatch_with_quant: bool


@dataclass(frozen=True, slots=True)
class AllGatherCombineContext:
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    restore_shape: torch.Size


@dataclass(frozen=True, slots=True)
class AllToAllCombineContext:
    input_splits: np.ndarray
    output_splits: np.ndarray
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


TCombineContext = TypeVar("TCombineContext")


@dataclass(frozen=True, slots=True)
class TokenDispatchResult(Generic[TCombineContext]):
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_context: TCombineContext
    dynamic_scale: torch.Tensor | None = None
    topk_scales: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TokenCombineResult:
    routed_out: torch.Tensor


@dataclass(frozen=True, slots=True)
class PaddedHiddenStatesPrepareContext:
    padded_hidden_states_shape: torch.Size


@dataclass(frozen=True, slots=True)
class PrepareOutput:
    """Typed output from prepare stage."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    mc2_mask: torch.Tensor | None
    context_metadata: PaddedHiddenStatesPrepareContext | None
    pertoken_scale: torch.Tensor | None = None
