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

"""Allow Engram on Ascend until the vLLM pin includes removal of the CUDA gate."""

import argparse
import importlib.util
import json

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from pydantic import model_validator
    from vllm.config import VllmConfig
    from vllm.config.engram import EngramConfig, model_has_engram_layers
    from vllm.config.utils import config
    from vllm.engine import arg_utils

    def verify_model_config(self, model_config) -> None:
        # Keep upstream's model/layer checks; only its CUDA requirement is lifted.
        if not model_has_engram_layers(model_config):
            raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

    EngramConfig.verify_model_config = verify_model_config

    @config
    class AscendEngramConfig(EngramConfig):
        use_elastic_buffer: bool = False
        """Use A5 node-sharded ElasticBuffer FP8 weights with E8M0 scales in
        HBM. Disabled by default, preserving the native host UVA backend.
        Requires CPU offload, TP1, and indexed FP8/E8M0 safetensors."""

        @model_validator(mode="after")
        def _validate_elastic_buffer(self):
            if self.use_elastic_buffer and (not self.cpu_offload or self.embedding_across_dp):
                raise ValueError("use_elastic_buffer requires cpu_offload=True and embedding_across_dp=False")
            return self

        def verify_model_config(self, model_config) -> None:
            super().verify_model_config(model_config)
            if self.use_elastic_buffer:
                from vllm_ascend.utils import is_950

                if model_config.architecture != "DeepseekV41ForCausalLM" or not is_950():
                    raise ValueError("use_elastic_buffer requires A5 DeepSeek V4.1")

        def verify_parallel_config(self, parallel_config) -> None:
            super().verify_parallel_config(parallel_config)
            if self.use_elastic_buffer and (
                parallel_config.tensor_parallel_size != 1
                or not 1 <= parallel_config.data_parallel_size <= 32
                or parallel_config.pipeline_parallel_size != 1
                or parallel_config.prefill_context_parallel_size != 1
                or parallel_config.decode_context_parallel_size != 1
                or parallel_config.enable_elastic_ep
                or (parallel_config.data_parallel_size > 1 and not parallel_config.enable_expert_parallel)
            ):
                raise ValueError("use_elastic_buffer requires TP=PP=PCP=DCP=1, DP<=32 with EP, and no elastic EP")

        def verify_load_config(self, load_config) -> None:
            super().verify_load_config(load_config)
            if self.use_elastic_buffer and load_config.load_format not in ("auto", "safetensors", "dummy"):
                raise ValueError("use_elastic_buffer requires indexed safetensors (auto/safetensors), or dummy weights")

    # EngineArgs' dict API and CLI must both retain the Ascend-only field.
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
