from __future__ import annotations

from typing import TypeVar

import torch

from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    FusedExpertsRequest,
    MlpComputeRequest,
    MoEDispatchSpec,
    MoEMlpKernelSpec,
    MoEMlpSpec,
    MoEMxfpSpec,
    MoEQuantSpec,
    MoEQuantTensors,
    MoEWeightPack,
    TokenDispatchRequest,
    TokenDispatchResult,
)
from vllm_ascend.quantization.quant_type import QuantType

TCombineContext = TypeVar("TCombineContext")


def build_fused_experts_request(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    quant_type: QuantType,
    dynamic_eplb: bool,
    expert_map: torch.Tensor | None = None,
    global_redundant_expert_num: int = 0,
    mc2_mask: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    log2phy: torch.Tensor | None = None,
    pertoken_scale: torch.Tensor | None = None,
    activation: str = "silu",
    need_trans: bool = False,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    comm_quant_mode: int | None = None,
    mxfp: MoEMxfpSpec | None = None,
    w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
) -> FusedExpertsRequest:
    return FusedExpertsRequest(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=MoEWeightPack(
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        ),
        dispatch=MoEDispatchSpec(
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=dynamic_eplb,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
        ),
        mlp=MoEMlpSpec(
            activation=activation,
            need_trans=need_trans,
            dynamic_eplb=dynamic_eplb,
        ),
        quant=MoEQuantSpec(
            quant_type=quant_type,
            comm_quant_mode=comm_quant_mode,
            mxfp=mxfp,
        ),
        quant_tensors=MoEQuantTensors(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
    )


def build_token_dispatch_request(
    *,
    request: FusedExpertsRequest,
    topk_ids: torch.Tensor | None = None,
) -> TokenDispatchRequest:
    return TokenDispatchRequest(
        hidden_states=request.hidden_states,
        topk_weights=request.topk_weights,
        topk_ids=request.topk_ids if topk_ids is None else topk_ids,
        dispatch=request.dispatch,
        quant=request.quant,
    )


def build_mlp_kernel_spec(
    *,
    hidden_states: torch.Tensor,
    quant: MoEQuantSpec,
    use_fusion_ops: bool,
) -> MoEMlpKernelSpec:
    act_quant_type = torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_mxfp_quant = False
    use_bf16 = hidden_states.dtype == torch.bfloat16

    if quant.is_mxfp and quant.mxfp is not None:
        use_mxfp_quant = True
        act_quant_type = quant.mxfp.act_quant_type or act_quant_type
        weight_quant_type = quant.mxfp.weight_quant_type or weight_quant_type
        scale_type = quant.mxfp.scale_dtype
        per_token_scale_type = quant.mxfp.per_token_scale_dtype
        use_bf16 = quant.mxfp.use_bf16

    return MoEMlpKernelSpec(
        fusion=quant.quant_type in (QuantType.W8A8, QuantType.MXFP8) and use_fusion_ops,
        use_mxfp_quant=use_mxfp_quant,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
    )


def build_mlp_compute_request(
    *,
    request: FusedExpertsRequest,
    dispatch_result: TokenDispatchResult[TCombineContext],
    use_fusion_ops: bool,
) -> MlpComputeRequest:
    return MlpComputeRequest(
        hidden_states=dispatch_result.hidden_states,
        group_list=dispatch_result.group_list,
        group_list_type=dispatch_result.group_list_type,
        dynamic_scale=dispatch_result.dynamic_scale,
        topk_scales=dispatch_result.topk_scales,
        weights=request.weights,
        quant=request.quant,
        quant_tensors=request.quant_tensors,
        mlp=request.mlp,
        kernel=build_mlp_kernel_spec(
            hidden_states=dispatch_result.hidden_states,
            quant=request.quant,
            use_fusion_ops=use_fusion_ops,
        ),
    )
