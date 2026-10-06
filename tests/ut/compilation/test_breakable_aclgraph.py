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
#
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.config import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend.compilation.breakable_aclgraph import (
    BreakableACLGraphWrapper,
    _apply_super_kernel,
)


class _StubCapture:
    """Minimal stand-in for ``BreakableCUDAGraphCapture``.

    Mirrors the upstream segment lifecycle: a fresh graph per segment, and an
    eager break that closes the current segment and opens the next one.
    """

    def __init__(self, segments: int = 1):
        self.remaining = segments
        self.segments = []
        self._current_graph = None

    def __enter__(self):
        self._begin_segment()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end_segment()
        return False

    def _begin_segment(self):
        self._current_graph = MagicMock(name="segment_graph")

    def _end_segment(self):
        if self._current_graph is None:
            return
        self.segments.append(self._current_graph.replay)
        self._current_graph = None

    def add_eager(self, fn):
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        if self.remaining > 1:
            self.remaining -= 1
            self._begin_segment()
        return result


def _super_kernel_config(enabled):
    return SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_super_kernel=enabled),
    )


class TestBreakableACLGraphWrapperCapture(TestBase):
    def _run_capture(self, enable_super_kernel, segments=1):
        """Drive ``_capture`` once, returning the entry and both test doubles."""
        wrapper = object.__new__(BreakableACLGraphWrapper)
        wrapper.use_eagle = False
        wrapper.enable_enpu = False

        entry = SimpleNamespace(capture=_StubCapture(segments=segments), output=None)

        def fake_super_capture(inner_entry, args, kwargs):
            with inner_entry.capture:
                inner_entry.output = "captured"
            return inner_entry.output

        scope = MagicMock()
        apply_scope = MagicMock()

        with (
            patch(
                "vllm_ascend.compilation.breakable_aclgraph.get_forward_context",
                return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
            ),
            patch(
                "vllm_ascend.compilation.breakable_aclgraph.get_ascend_config",
                return_value=_super_kernel_config(enable_super_kernel),
            ),
            patch(
                "vllm_ascend.compilation.breakable_aclgraph.super_kernel_scope",
                return_value=scope,
            ) as super_scope,
            patch(
                "vllm_ascend.compilation.breakable_aclgraph._apply_super_kernel",
                return_value=apply_scope,
            ) as apply_super_kernel,
            patch.object(BreakableACLGraphWrapper.__bases__[0], "_capture", fake_super_capture),
        ):
            result = wrapper._capture(entry, ("arg",), {})

        return result, entry, scope, super_scope, apply_scope, apply_super_kernel

    def test_capture_brackets_body_with_super_kernel_scope_when_enabled(self):
        result, entry, scope, super_scope, apply_scope, apply_super_kernel = self._run_capture(True)

        self.assertEqual(result, "captured")
        super_scope.assert_called_once_with("full_model", True)
        apply_super_kernel.assert_called_once_with(entry.capture, True)
        # The scope must wrap the whole capture, not just the graph call.
        self.assertTrue(scope.__enter__.called)
        self.assertTrue(apply_scope.__enter__.called)

    def test_capture_skips_super_kernel_when_disabled(self):
        result, entry, _scope, super_scope, _apply_scope, apply_super_kernel = self._run_capture(False)

        self.assertEqual(result, "captured")
        super_scope.assert_called_once_with("full_model", False)
        apply_super_kernel.assert_called_once_with(entry.capture, False)


class TestApplySuperKernel(TestBase):
    def _capture_recording_graphs(self, enabled, segments=2):
        """Run the real helper and return the graphs it must have optimized."""
        capture = _StubCapture(segments=segments)
        recorded = []
        original_begin = capture._begin_segment

        def recording_begin():
            original_begin()
            recorded.append(capture._current_graph)

        capture._begin_segment = recording_begin

        with _apply_super_kernel(capture, enabled=enabled):
            with capture:
                capture.add_eager(lambda: None)

        return capture, recorded

    def test_applies_options_to_every_captured_graph(self):
        _capture, recorded = self._capture_recording_graphs(enabled=True)

        # Two segments means two separate graphs; both need the pass, not
        # just the first one.
        self.assertEqual(len(recorded), 2)
        for graph in recorded:
            graph.super_kernel_optimize.assert_called_once_with(
                optimize_options={"dcci_after_kernel_end": [".*"]},
            )

    def test_disabled_leaves_graphs_untouched(self):
        _capture, recorded = self._capture_recording_graphs(enabled=False)

        self.assertEqual(len(recorded), 2)
        for graph in recorded:
            graph.super_kernel_optimize.assert_not_called()

    def test_restores_begin_segment_after_the_block(self):
        capture = _StubCapture()
        # Save the real bound method up front; `tracker` then stands in for a
        # pre-existing override that must survive the helper.
        original_begin = capture._begin_segment

        def tracker():
            original_begin()

        capture._begin_segment = tracker

        with _apply_super_kernel(capture, enabled=True):
            self.assertIsNot(capture._begin_segment, tracker)

        self.assertIs(capture._begin_segment, tracker)

    def test_restores_begin_segment_when_capture_raises(self):
        capture = _StubCapture()
        original_begin = capture._begin_segment

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with _apply_super_kernel(capture, enabled=True):
                raise RuntimeError("capture failed")

        # The patch must be undone even on failure: a leaked tracker would
        # corrupt every later capture on this wrapper.
        self.assertEqual(capture._begin_segment.__func__, original_begin.__func__)
        self.assertIs(capture._begin_segment.__self__, capture)
