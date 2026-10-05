# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import vllm_ascend.worker.v2.device_metadata as module
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask


class Executor:
    submission_in_flight = False

    def __init__(self):
        self.calls = []

    def submit(self, tasks, descriptor):
        self.calls.append(("submit", tasks, descriptor))
        self.submission_in_flight = True

    def wait(self, stage, group_id):
        self.calls.append(("wait", stage, group_id))

    def release(self):
        self.calls.append(("release",))
        self.submission_in_flight = False


class Builder:
    enabled = False
    tasks = ()

    @contextmanager
    def defer_device_metadata(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False

    def take_device_metadata_tasks(self):
        result, self.tasks = self.tasks, ()
        return result


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(module, "DeviceMetadataExecutor", Executor)
    return module.TargetDeviceMetadata()


def prepare(state, builder, *, full=False, fail=False):
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    task = DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, 42)
    with state.build(groups, 96, full):
        assert builder.enabled
        builder.tasks = (task,)
        if fail:
            raise ValueError("build failed")
    assert not builder.enabled
    return task


def test_eager_joins_producers_and_restores_builder(state):
    builder = Builder()
    with state.activate():
        prepare(state, builder)
    assert state.executor.calls[-2:] == [("wait", DeviceMetadataStage.ATTENTION, 42), ("release",)]
    assert not state.executor.submission_in_flight
    assert not builder.enabled  # subsequent draft builds remain inline


def test_full_replay_does_not_wait_twice_on_reset_external_events(state):
    with state.activate():
        prepare(state, Builder(), full=True)
        assert state.executor.calls[0][2].num_tokens == 96
        state.finish_replay()
    assert [call[0] for call in state.executor.calls] == ["submit", "release"]


def test_build_failure_drains_tasks_and_restores_provider(state):
    builder = Builder()
    with pytest.raises(ValueError, match="build failed"), state.activate():
        prepare(state, builder, fail=True)
    assert not builder.enabled
    assert builder.tasks == ()
    assert state.executor.calls == []


def test_unretired_submission_cannot_be_overwritten(state):
    with state.activate():
        prepare(state, Builder())
        with pytest.raises(RuntimeError, match="not retired"):
            prepare(state, Builder())


def test_capture_and_replay_use_same_descriptor(state):
    builder = Builder()
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]

    def build(**kwargs):
        builder.tasks = (DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: None, 7),)
        return {"layer": "metadata"}

    with state.activate():
        assert state.run_build(build, attn_groups=groups, num_tokens=96, for_cudagraph_capture=True) == {
            "layer": "metadata"
        }
        capture_descriptor = state.executor.calls[0][2]
        state.finish()
        state.run_build(build, attn_groups=groups, num_tokens=96, full_graph_mode=True)
        assert state.executor.calls[-1][2] == capture_descriptor
        state.finish_replay()


def test_failed_producer_does_not_wait_on_unrecorded_frontier(state, monkeypatch):
    waits = []
    state.executor.stream = object()
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: SimpleNamespace(wait_stream=waits.append))

    def fail_submit(*args):
        state.executor.submission_in_flight = True
        raise ValueError("producer failed")

    state.executor.submit = fail_submit
    with pytest.raises(ValueError, match="producer failed"), state.activate():
        prepare(state, Builder(), full=True)
    assert waits == [state.executor.stream]
    assert state.executor.calls == [("release",)]
    with pytest.raises(RuntimeError, match="recreate"), state.activate():
        prepare(state, Builder(), full=True)
