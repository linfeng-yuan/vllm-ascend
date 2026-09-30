#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Select the Ascend config before upstream's CUDA-only Engram validation.

The pinned vLLM resolves Engram before calling the platform config hook.
Keep this single entry-point adapter until upstream provides that hook.
"""

import importlib.util

# Older vLLM builds have no Engram config or CLI entry to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    import argparse
    import json
    from dataclasses import asdict

    import vllm.engine.arg_utils as arg_utils
    from pydantic import model_validator
    from vllm.config.engram import EngramConfig
    from vllm.config.utils import config
    from vllm.config.vllm import VllmConfig

    @config
    class AscendEngramConfig(EngramConfig):
        dp_shared_memory: bool = False

        @model_validator(mode="after")
        def _validate_shared_memory(self):
            if self.dp_shared_memory and not self.cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_shared_memory and self.embedding_across_dp:
                raise ValueError("dp_shared_memory cannot be combined with embedding_across_dp")
            return self

        def verify_model_config(self, model_config) -> None:
            if (
                model_config is None
                or model_config.architecture != "DeepseekV41ForCausalLM"
                or not getattr(model_config.hf_text_config, "engram_layer_ids", None)
            ):
                raise ValueError("Ascend Engram requires DeepSeek V4.1 with non-empty engram_layer_ids.")

        def verify_parallel_config(self, parallel_config) -> None:
            super().verify_parallel_config(parallel_config)
            if self.embedding_across_dp:
                raise ValueError("Ascend Engram does not support embedding_across_dp")
            if self.dp_shared_memory and parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1")
            tp = parallel_config.tensor_parallel_size
            dp = parallel_config.data_parallel_size
            from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

            a5_elastic = (
                self.cpu_offload
                and not self.dp_shared_memory
                and get_current_hardware_profile().device_adaptor_family is DeviceAdaptorFamily.FP8_OPTIMIZED
            )
            if a5_elastic:
                # One process group per physical node serves its local EP
                # ranks. DP32/EP32 on four eight-card nodes therefore stores
                # 1/8 of each FP8 table per card, repeated on every node.
                topology_ok = (
                    tp == 1
                    and dp <= 32
                    and parallel_config.nnodes <= 4
                    and not parallel_config.data_parallel_external_lb
                    and parallel_config.data_parallel_size_local == dp // parallel_config.nnodes
                    and dp % parallel_config.nnodes == 0
                )
            else:
                topology_ok = (
                    tp in (1, 2, 4, 8)
                    and tp * dp <= 16
                    and parallel_config.nnodes == 1
                    and (parallel_config.data_parallel_external_lb or parallel_config.data_parallel_size_local == dp)
                )
            if (
                parallel_config.enable_elastic_ep
                or dp < 1
                or parallel_config.pipeline_parallel_size != 1
                or parallel_config.prefill_context_parallel_size != 1
                or parallel_config.decode_context_parallel_size != 1
                or not topology_ok
            ):
                raise ValueError(
                    "Ascend Engram needs PP=PCP=DCP=1; A5 ElasticBuffer supports "
                    "TP1, internal DP<=32 over up to four nodes with equal local DP, "
                    "while other storage modes require a single node and at most 16 TP x DP ranks."
                )

        def verify_load_config(self, load_config) -> None:
            if self.dp_shared_memory and load_config.load_format not in ("auto", "safetensors"):
                raise ValueError("dp_shared_memory requires load_format auto or safetensors")
            if load_config.load_format not in ("auto", "safetensors", "dummy"):
                raise ValueError("Ascend Engram requires indexed safetensors (auto/safetensors), or dummy weights.")

    # 84030bbe's CLI TypeAdapter is built from the upstream config annotation.
    # Select the Ascend subtype here so its backported field survives parsing.
    arg_utils.EngramConfig = AscendEngramConfig
    _get_kwargs = arg_utils.get_kwargs

    def _get_ascend_kwargs(cls):
        kwargs = _get_kwargs(cls)
        if cls is VllmConfig:

            def parse_engram(value):
                try:
                    return AscendEngramConfig(**json.loads(value))
                except (TypeError, ValueError) as exc:
                    raise argparse.ArgumentTypeError(str(exc)) from exc

            kwargs["engram_config"]["type"] = arg_utils.optional_type(parse_engram)
        return kwargs

    arg_utils.get_kwargs = _get_ascend_kwargs

    _resolve_engram_config = VllmConfig._resolve_and_verify_engram_config

    def _resolve_and_verify_engram_config(self) -> None:
        model_config = self.model_config
        spec = self.speculative_config
        if spec is not None and model_config is spec.draft_model_config:
            model_config = spec.target_model_config
        if model_config is not None and model_config.architecture == "DeepseekV41ForCausalLM":
            if self.engram_config is not None:
                if not isinstance(self.engram_config, AscendEngramConfig):
                    self.engram_config = AscendEngramConfig(**asdict(self.engram_config))
            elif getattr(model_config.hf_text_config, "engram_layer_ids", None):
                self.engram_config = AscendEngramConfig()
        _resolve_engram_config(self)
        if isinstance(self.engram_config, AscendEngramConfig):
            if self.parallel_config.use_ubatching:
                raise ValueError("Ascend Engram does not support DBO or microbatching")
            self.engram_config.verify_load_config(self.load_config)

    VllmConfig._resolve_and_verify_engram_config = _resolve_and_verify_engram_config
