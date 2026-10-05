# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("consumer_order", [(0, 1), (1, 0)])
def test_metadata_order_and_buffer_reuse(full_graph, consumer_order):
    torch.npu.set_device(0)
    executor = DeviceMetadataExecutor(capture_producers=True)
    stream = torch.npu.Stream()
    source = torch.zeros(16, device="npu")
    prepared = torch.empty_like(source)
    metadata = [torch.empty_like(source) for _ in range(3)]
    outputs = [torch.empty_like(source) for _ in range(3)]
    calls = []
    snapshots = []
    graph = torch.npu.NPUGraph() if full_graph else None

    def prepare():
        calls.append("prepare")
        prepared.copy_(source)

    def build(index):
        calls.append(index)
        torch.add(prepared, index + 1, out=metadata[index])

    tasks = (
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: build(2), 2),
        DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: build(0), 0),
        DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: build(1), 1),
        DeviceMetadataTask(DeviceMetadataStage.COMPRESSOR, prepare, 0),
    )

    def consume():
        for index in consumer_order:
            executor.wait(DeviceMetadataStage.ATTENTION, index)
            outputs[index].copy_(metadata[index])
        executor.wait(DeviceMetadataStage.INDEXER, 2)
        outputs[2].copy_(metadata[2])

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        executor.submit(tasks)
        assert calls == ["prepare", 0, 1, 2]
        consume()
        executor.release()
        if graph is not None:
            with torch.npu.graph(graph, stream=stream):
                executor.submit(tasks)
                consume()
                executor.release()

        for step in range(20):
            calls.clear()
            source.fill_(step)
            if graph is not None:
                graph.replay()
                assert calls == []  # no Python producer dispatch on replay
            else:
                executor.submit(tasks)
                consume()
                executor.release()
                assert calls == ["prepare", *consumer_order, 2]
            snapshots.append([output.clone() for output in outputs])

    torch.npu.synchronize()
    for step, snapshot in enumerate(snapshots):
        for index, actual in enumerate(snapshot):
            torch.testing.assert_close(actual.cpu(), torch.full((16,), float(step + index + 1)), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_real_a5_metadata_capture_with_dynamic_and_empty_padded_batches(ratio):
    """Actual installed MQ/QLI metadata ops, not a synthetic substitute."""
    from vllm_ascend.ops.pythondsl import ops

    torch.npu.set_device(0)
    batch, query_width = 4, 6
    cu = torch.arange(batch + 1, device="npu", dtype=torch.int32) * query_width
    lengths = torch.full((batch,), 256, device="npu", dtype=torch.int32)
    residual = torch.zeros_like(lengths)
    rows = torch.zeros((batch * query_width, 1), device="npu", dtype=torch.int32)
    buffers = [torch.zeros(1024, dtype=torch.int32, device="npu") for _ in range(2)]
    outputs = [torch.empty_like(b) for b in buffers]
    executor = DeviceMetadataExecutor(capture_producers=True)
    calls = []

    def mq():
        return ops.mixed_quant_sparse_flash_mla_metadata(
            rows,
            rows,
            cu_seqlens_q=cu,
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            quant_mode=1,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=True,
        )

    def qli():
        return ops.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu,
            seqused_k=lengths,
            cmp_residual_k=residual if ratio == 2 else None,
            batch_size=batch,
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
        )

    def build(index, fn):
        calls.append(index)
        buffers[index].copy_(fn())

    tasks = (
        DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: build(0, mq), 0),
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: build(1, qli), 1),
    )

    def forward():
        executor.submit(tasks)
        executor.wait(DeviceMetadataStage.ATTENTION, 0)
        outputs[0].copy_(buffers[0])
        executor.wait(DeviceMetadataStage.INDEXER, 1)
        outputs[1].copy_(buffers[1])
        executor.release()

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        forward()
    snapshots = []
    with torch.npu.stream(stream):
        for step, live in enumerate((4, 1, 0, 3, 2, 0, 4)):
            cu.copy_(
                torch.tensor([min(i, live) * query_width for i in range(batch + 1)], device="npu", dtype=torch.int32)
            )
            lengths.copy_(
                torch.tensor(
                    [512 + step * 17 if i < live else 0 for i in range(batch)], device="npu", dtype=torch.int32
                )
            )
            residual.fill_(step % 2)
            reference = [mq(), qli()]
            calls.clear()
            graph.replay()
            assert calls == []
            snapshots.append(([t.clone() for t in outputs], reference))
    torch.npu.synchronize()
    for actual, reference in snapshots:
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
