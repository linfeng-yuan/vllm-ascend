# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import vllm.distributed.device_communicators.cuda_communicator as cuda_comm_mod

from vllm_ascend.worker.v2 import utils as v2_utils


def test_v2_utils_context_managers_switch_and_restore():
    fake_cuda = MagicMock()
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    original_comm = cuda_comm_mod.CudaCommunicator
    npu_cls = object()

    with (
        patch.object(v2_utils, "torch", SimpleNamespace(cuda=fake_cuda, npu=fake_npu)),
        patch.object(v2_utils, "breakable_cudagraph", MagicMock()),
        patch.object(v2_utils, "weak_ref_workspaces") as weak_ref,
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
        patch.object(
            v2_utils,
            "get_ascend_config",
            return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=False)),
        ),
        patch.object(v2_utils.logger, "info_once", create=True),
        patch.object(v2_utils.logger, "debug"),
        patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator",
            npu_cls,
        ),
    ):
        with v2_utils.torch_cuda_wrapper():
            assert fake_cuda.Event is fake_npu.Event
            assert fake_cuda.graph is v2_utils.torch_npu_graph_wrapper
            assert v2_utils.breakable_cudagraph.weak_ref_tensor is v2_utils.weak_ref_tensor

        with v2_utils.communicator_switch():
            assert cuda_comm_mod.CudaCommunicator is npu_cls
        assert cuda_comm_mod.CudaCommunicator is original_comm

        with v2_utils.torch_npu_graph_wrapper("capture"):
            pass
        weak_ref.assert_any_call("graph")
        weak_ref.assert_any_call("draft")
        assert weak_ref.call_count == 2


def test_torch_npu_graph_wrapper_skips_super_kernel_when_disabled():
    """MRV2 capture must not touch the super kernel APIs by default."""
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    graph = MagicMock()
    scoped = MagicMock()

    with (
        patch.object(v2_utils, "torch", SimpleNamespace(cuda=MagicMock(), npu=fake_npu)),
        patch.object(v2_utils, "weak_ref_workspaces"),
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
        patch.object(
            v2_utils,
            "get_ascend_config",
            return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=False)),
        ),
        patch.object(v2_utils, "super_kernel_scope", return_value=scoped) as super_scope,
        patch.object(v2_utils.logger, "info_once", create=True) as info_once,
    ):
        with v2_utils.torch_npu_graph_wrapper(graph):
            pass

    # The wrapper always brackets the captured region, but with the feature
    # switched off so the helper itself is a no-op.
    super_scope.assert_called_once_with("full_model", False)
    graph.super_kernel_optimize.assert_not_called()
    info_once.assert_not_called()


def test_torch_npu_graph_wrapper_applies_super_kernel_when_enabled():
    """MRV2 capture applies the Super Kernel scope and optimize call."""
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    graph = MagicMock()
    scoped = MagicMock()

    with (
        patch.object(v2_utils, "torch", SimpleNamespace(cuda=MagicMock(), npu=fake_npu)),
        patch.object(v2_utils, "weak_ref_workspaces"),
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
        patch.object(
            v2_utils,
            "get_ascend_config",
            return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=True)),
        ),
        patch.object(v2_utils, "super_kernel_scope", return_value=scoped) as super_scope,
        patch.object(v2_utils.logger, "info_once", create=True) as info_once,
    ):
        with v2_utils.torch_npu_graph_wrapper(graph):
            pass

    super_scope.assert_called_once_with("full_model", True)
    graph.super_kernel_optimize.assert_called_once_with(
        optimize_options={"dcci_after_kernel_end": [".*"]},
    )
    info_once.assert_called_once_with("Super kernel optimization is enabled for ACL graph capture.")
