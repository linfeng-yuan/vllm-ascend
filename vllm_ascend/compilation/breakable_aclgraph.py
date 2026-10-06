#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    weak_ref_workspaces,
)
from vllm_ascend.utils import super_kernel_scope


@contextmanager
def _apply_super_kernel(capture: Any, enabled: bool):
    """Apply Super Kernel to the graphs captured inside the block.

    The breakable runner captures through ``Capture.capture_begin()`` /
    ``capture_end()`` instead of ``torch.cuda.graph()``, so it never goes
    through ``worker.v2.utils.torch_npu_graph_wrapper`` where the Super Kernel
    pass is normally triggered. Without this, enabling ``enable_super_kernel``
    silently produces an unoptimized graph whenever breakable graphs are used.

    ``Capture`` opens a fresh segment (and therefore a fresh graph) at every
    attention break, so every captured graph needs the pass -- not just the
    first. ``Capture`` keeps no handle on the graphs it consumed, only their
    replay closures, so collect them by wrapping ``_begin_segment``.
    """
    captured_graphs: list[Any] = []
    begin_segment = capture._begin_segment

    def record_begin_segment() -> None:
        begin_segment()
        graph = getattr(capture, "_current_graph", None)
        if graph is not None:
            captured_graphs.append(graph)

    try:
        capture._begin_segment = record_begin_segment
        yield
    finally:
        capture._begin_segment = begin_segment
        if enabled:
            for graph in captured_graphs:
                graph.super_kernel_optimize(
                    optimize_options={
                        "dcci_after_kernel_end": [".*"],
                    },
                )


class BreakableACLGraphWrapper(BreakableCUDAGraphWrapper):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ) -> None:
        super().__init__(
            runnable=runnable,
            vllm_config=vllm_config,
        )

        self.use_eagle = use_eagle
        self.enable_enpu = enable_enpu

    def _capture(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        is_full_capture = forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        if is_full_capture:
            # Ascend FULL graph attention creates task groups and records the
            # mutable graph parameters only while this flag is set.
            forward_context.capturing = True

        # The Super Kernel scope must stay open for the whole capture so the
        # fused regions cover the modeled operators, and `super_kernel_optimize`
        # must run once every graph of the capture has been closed.
        enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel
        capture = entry.capture
        with (
            super_kernel_scope("full_model", enable_super_kernel),
            _apply_super_kernel(capture, enable_super_kernel) if capture is not None else nullcontext(),
        ):
            output = super()._capture(entry, args, kwargs)

        if is_full_capture:
            # Keep the same workspace lifetime contract as ACLGraphWrapper.
            weak_ref_workspaces(get_graph_params())
            weak_ref_workspaces(get_draft_graph_params())
            weak_ref_workspaces(get_draft_graph_prefill_params())

        return output

    def _replay(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # Match ACLGraphWrapper's ordering between async attention
            # parameter updates and the previous/current FULL graph replay.
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
            if not self.enable_enpu and not is_draft_eagle:
                torch.npu.current_stream().synchronize()
        super()._replay(entry, args, kwargs)
        return entry.output
