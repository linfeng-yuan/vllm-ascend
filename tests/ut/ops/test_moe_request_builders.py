import unittest

import torch

from vllm_ascend.ops.fused_moe.moe_request_builders import (
    build_fused_experts_request,
    build_mlp_compute_request,
    build_token_dispatch_request,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    AllGatherCombineContext,
    MoEMxfpSpec,
    TokenDispatchResult,
)
from vllm_ascend.quantization.quant_type import QuantType


class TestMoERequestBuilders(unittest.TestCase):
    def test_build_fused_experts_request_preserves_runtime_semantics(self):
        for quant_type in (
            QuantType.NONE,
            QuantType.W4A16,
            QuantType.W4A8,
            QuantType.W8A8,
            QuantType.MXFP8,
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(4, 8)
                topk_weights = torch.randn(4, 2)
                topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32)
                request = build_fused_experts_request(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    w1=torch.randn(2, 8, 16),
                    w2=torch.randn(2, 16, 8),
                    quant_type=quant_type,
                    dynamic_eplb=True,
                    expert_map=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                    global_redundant_expert_num=2,
                    mc2_mask=torch.tensor([True, False, True, False]),
                    apply_router_weight_on_input=True,
                    log2phy=torch.tensor([3, 2, 1, 0], dtype=torch.int32),
                    pertoken_scale=torch.randn(4),
                    activation="gelu",
                )

                self.assertIs(request.hidden_states, hidden_states)
                self.assertIs(request.topk_weights, topk_weights)
                self.assertIs(request.topk_ids, topk_ids)
                self.assertTrue(request.dispatch.dynamic_eplb)
                self.assertTrue(request.dispatch.apply_router_weight_on_input)
                self.assertEqual(request.dispatch.global_redundant_expert_num, 2)
                self.assertEqual(request.mlp.activation, "gelu")
                self.assertEqual(request.quant.quant_type, quant_type)

    def test_build_token_dispatch_request_supports_remapped_topk_ids(self):
        request = build_fused_experts_request(
            hidden_states=torch.randn(2, 4),
            topk_weights=torch.randn(2, 1),
            topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
            w1=torch.randn(1, 4, 8),
            w2=torch.randn(1, 8, 4),
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
        routed_topk_ids = torch.tensor([[3], [2]], dtype=torch.int32)

        dispatch_request = build_token_dispatch_request(
            request=request,
            topk_ids=routed_topk_ids,
        )

        self.assertIs(dispatch_request.hidden_states, request.hidden_states)
        self.assertIs(dispatch_request.topk_weights, request.topk_weights)
        self.assertIs(dispatch_request.dispatch, request.dispatch)
        self.assertIs(dispatch_request.quant, request.quant)
        self.assertIs(dispatch_request.topk_ids, routed_topk_ids)

    def test_build_mlp_compute_request_derives_kernel_spec(self):
        request = build_fused_experts_request(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            topk_weights=torch.randn(2, 2),
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            w1=torch.randn(2, 8, 16),
            w2=torch.randn(2, 16, 8),
            quant_type=QuantType.MXFP8,
            dynamic_eplb=False,
            mxfp=MoEMxfpSpec(
                act_quant_type=torch.float8_e4m3fn,
                weight_quant_type=torch.float8_e4m3fn,
                scale_dtype=torch.float32,
                per_token_scale_dtype=torch.float16,
                use_bf16=False,
            ),
            w1_scale=[torch.randn(1)],
            w2_scale=[torch.randn(1)],
        )
        dispatch_result = TokenDispatchResult(
            hidden_states=torch.randn(4, 8, dtype=torch.bfloat16),
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(4, 1),
            combine_context=AllGatherCombineContext(
                topk_weights=request.topk_weights,
                expanded_row_idx=torch.arange(4, dtype=torch.int32),
                restore_shape=torch.Size([2, 8]),
            ),
        )

        mlp_request = build_mlp_compute_request(
            request=request,
            dispatch_result=dispatch_result,
            use_fusion_ops=True,
        )

        self.assertIs(mlp_request.hidden_states, dispatch_result.hidden_states)
        self.assertIs(mlp_request.weights, request.weights)
        self.assertIs(mlp_request.quant_tensors, request.quant_tensors)
        self.assertTrue(mlp_request.kernel.fusion)
        self.assertTrue(mlp_request.kernel.use_mxfp_quant)
        self.assertEqual(mlp_request.kernel.scale_type, torch.float32)
        self.assertEqual(mlp_request.kernel.per_token_scale_type, torch.float16)
        self.assertFalse(mlp_request.kernel.use_bf16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
