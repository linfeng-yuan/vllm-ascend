# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 runner engram overlap: bucket-keyed external events with UVA tables.

Mirrors ``tests/e2e/pull_request/one_card/test_engram_multistream_graph.py``
but drives the V2 entry points: slotless hashing (no block table), events
keyed by padded token count, priming outside any forward context, dummy-step
zeroing, and retire resets after a failed producer.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

# Match worker startup: register ops before importing attention/model modules.
import vllm_ascend.ops  # noqa: F401

# isort: split
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _make_model(codes, scales, vocab, heads, width, capacity):
    host_codes = HostUvaBuffer(codes.shape, codes.dtype, torch.device("npu:0"))
    host_scales = HostUvaBuffer(scales.shape, scales.dtype, torch.device("npu:0"))
    host_codes.tensor.copy_(codes)
    host_scales.tensor.copy_(scales)
    tables = {}
    for layer in (1, 14):
        table = object.__new__(AscendParallelEngramEmbedding)
        torch.nn.Module.__init__(table)
        table.part_n_hash_cols, table.dim, table.head_start = heads, width, 0
        table.n_hash_cols, table.dp_size, table.tp_size = heads, 1, 1
        table.vocab_start_idx, table.vocab_end_idx = 0, vocab
        table._codes_uva, table._scales_uva = host_codes, host_scales
        tables[layer] = table
    model = object.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram = True
    model.engram_dp_shared_memory = True

    # V2 slotless hashing: no slot cache and no block-table rows.
    class HashState:
        lookback_depth = 2
        use_slot_cache = False

        def ensure_cache(self):
            return True

        def __call__(self, input_ids, *args):
            return input_ids[:, None, None].expand(-1, 2, heads).int()

    model.engram_hash = HashState()
    model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=tables.get(layer))) for layer in range(15)]
    model._engram_input_buffers, model._engram_max_tokens = None, capacity
    model._engram_graph_events = {}
    model._engram_prepare_stream = None
    model._engram_overlap_enabled = True
    model.engram_rotation = torch.eye(32, device="npu")
    return model, host_codes, host_scales


@torch.inference_mode()
def test_engram_v2_bucket_events_refresh_rows_dummy_steps_and_failures():
    torch.npu.set_device(0)
    init_device_properties_triton()
    capacity, heads, width, vocab = 192, 3, 256, 101
    codes = ((torch.arange(vocab * width).view(vocab, width) * 7) % 251 - 125).to(torch.int8)
    scales = torch.ones(vocab, width // 32) * 0.25
    model, host_codes, host_scales = _make_model(codes, scales, vocab, heads, width, capacity)
    graphs, outputs = {}, {}
    try:
        with patch("vllm_ascend.models.deepseek_v41.model.gather_engram_hashes", lambda ids, **kwargs: ids):
            # Capture outside any forward context: priming happens on the
            # model's V2 path, keyed by padded token count instead of a batch
            # descriptor.
            for size in (96, 192):
                binding = model.prime_engram_v2_graph_inputs(size)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    AscendParallelEngramEmbedding.wait_lookup(binding["engram_mask_ready_event"], True)
                    output = {}
                    for layer in (1, 14):
                        AscendParallelEngramEmbedding.wait_lookup(binding["engram_pending"][layer], True)
                        output[layer] = torch.where(
                            binding["engram_mask"][:size, None],
                            binding["engram_lookups"][layer][:size],
                            0,
                        )
                graphs[size], outputs[size] = graph, output
            lookback = torch.full((4, 2), -1, dtype=torch.int32, device="npu")
            # Alternate buckets, lengths and padding; a dummy step must zero
            # the rows its replay reads instead of leaking the previous batch.
            plan = ((192, 168, False), (96, 1, False), (192, 0, True), (96, 96, False), (192, 17, False)) * 3
            for phase, (size, count, dummy) in enumerate(plan):
                ids = (torch.arange(max(count, 1)) + phase * 11) % vocab
                query = torch.tensor([0, count] if count else [0], dtype=torch.int32, device="npu")
                model.prepare_engram_inputs(
                    ids[:count].npu(),
                    torch.arange(count, device="npu"),
                    size,
                    lookback,
                    query,
                    cg_mode=CUDAGraphMode.FULL,
                    force_dummy=dummy,
                )
                graphs[size].replay()
                oracle = torch.zeros((size, heads, width), dtype=torch.bfloat16)
                if not dummy:
                    for token, row in enumerate(ids[:count].tolist()):
                        oracle[token] = (codes[row].float() * 0.25).bfloat16()
                for layer in (1, 14):
                    assert torch.equal(outputs[size][layer].cpu(), oracle.flatten(1)), (phase, size, count, layer)
            # A failed producer resets the primed records (via its retire);
            # the next step re-records them and the replay reads fresh rows.
            model.prime_engram_v2_graph_inputs(96)
            with patch.object(model, "prepare_engram", side_effect=RuntimeError("lookup failed")):
                try:
                    model.prepare_engram_inputs(
                        torch.ones(8, dtype=torch.int32, device="npu"),
                        torch.arange(8, device="npu"),
                        96,
                        lookback,
                        torch.tensor([0, 8], dtype=torch.int32, device="npu"),
                        cg_mode=CUDAGraphMode.FULL,
                    )
                    raise AssertionError("producer failure expected")
                except RuntimeError:
                    pass
            ids = (torch.arange(5) + 7) % vocab
            model.prepare_engram_inputs(
                ids.npu(),
                torch.arange(5, device="npu"),
                96,
                lookback,
                torch.tensor([0, 5], dtype=torch.int32, device="npu"),
                cg_mode=CUDAGraphMode.FULL,
            )
            graphs[96].replay()
            oracle = torch.zeros((96, heads, width), dtype=torch.bfloat16)
            for token, row in enumerate(ids.tolist()):
                oracle[token] = (codes[row].float() * 0.25).bfloat16()
            for layer in (1, 14):
                assert torch.equal(outputs[96][layer].cpu(), oracle.flatten(1))
    finally:
        torch.npu.synchronize()
        host_codes.close()
        host_scales.close()
