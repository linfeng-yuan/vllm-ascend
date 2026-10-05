# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 lifecycle for target metadata produced outside the model graph."""

from contextlib import ExitStack, contextmanager

import torch
from vllm.forward_context import BatchDescriptor

from vllm_ascend.worker.device_metadata import (
    DeviceMetadataExecutor,
    use_device_metadata_executor,
)


class TargetDeviceMetadata:
    """One executor per target model state, including capture-only states.

    Builders are deferred only while preparing the target. The DSpark builder
    remains synchronous and cannot accidentally inherit pending target tasks.
    The main stream waits at first consumers, not at model entry. At the end
    of forward, unused producers are also joined before inputs can be reused
    by the next async-scheduling step.
    """

    def __init__(self):
        self.executor = DeviceMetadataExecutor()
        self._tasks = ()
        self._failed = False

    @contextmanager
    def activate(self):
        with use_device_metadata_executor(self.executor):
            try:
                yield
            finally:
                self.finish()

    def run_build(self, build_fn, **kwargs):
        with self.build(
            kwargs["attn_groups"],
            kwargs["num_tokens"],
            kwargs.get("full_graph_mode", False) or kwargs.get("for_cudagraph_capture", False),
        ):
            return build_fn(**kwargs)

    @contextmanager
    def build(self, attn_groups, num_tokens: int, full_graph: bool):
        if self._failed:
            raise RuntimeError("Metadata producer failed; recreate the target model state before retrying")
        if self.executor.submission_in_flight:
            raise RuntimeError("Target metadata was not retired before rebuilding inputs")
        providers = {
            id(builder): builder
            for groups in attn_groups
            for group in groups
            for builder in (group.get_metadata_builder(0),)
            if hasattr(builder, "defer_device_metadata")
        }
        with ExitStack() as stack:
            for provider in providers.values():
                stack.enter_context(provider.defer_device_metadata())
            try:
                yield
            except BaseException:
                for provider in providers.values():
                    provider.take_device_metadata_tasks()
                raise
            self._tasks = tuple(
                task for provider in providers.values() for task in provider.take_device_metadata_tasks()
            )
        if self._tasks:
            # Capture is prepared with cg_mode=NONE, for_capture=True. Use the
            # same stable key as runtime FULL replay, not that temporary mode.
            descriptor = BatchDescriptor(num_tokens=num_tokens) if full_graph else None
            try:
                self.executor.submit(self._tasks, descriptor)
            except BaseException:
                # A producer may fail before recording its frontier. Waiting
                # on that never-recorded external event during cleanup hangs.
                # Join only work actually submitted, then fail closed: partially
                # recorded external events cannot safely be reused on retry.
                self._failed = True
                if self.executor.submission_in_flight:
                    torch.npu.current_stream().wait_stream(self.executor.stream)
                    self.executor.release()
                self._tasks = ()
                raise

    def finish(self):
        if not self.executor.submission_in_flight:
            return
        # Some cache-only/dummy paths have no consumer. Their producers still
        # read input buffers; join them before the next step can overwrite them.
        for task in self._tasks:
            self.executor.wait(task.stage, task.group_id)
        self.executor.release()
        self._tasks = ()

    def finish_replay(self):
        if not self.executor.submission_in_flight:
            return
        # The captured graph already contains all waits AND resets. Waiting
        # on these external events again on the host would wait for the next
        # iteration's producer, deadlocking this iteration.
        self.executor.release()
        self._tasks = ()
