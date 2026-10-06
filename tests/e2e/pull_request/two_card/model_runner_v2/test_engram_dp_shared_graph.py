# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 runner engram overlap over a node-local DP-shared table.

With ``dp_shared_memory`` every replica hashes its own tokens and looks up
the mapped table directly, so unequal and empty rank batches need no
collectives. The test drives the V2 entry points (bucket-keyed external
events, priming outside any forward context) across two NPU ranks.
"""

import socket
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode
from vllm.distributed import parallel_state

# Match worker startup before importing the model's attention dependencies.
import vllm_ascend.ops  # noqa: F401

# isort: split
from tests.e2e.pull_request.two_card.test_engram_fixed_slots import (
    _CAPACITY,
    _DIM,
    _HEAD_SIZES,
    _ids,
    _reference,
)
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@torch.inference_mode()
def _worker(rank, port):
    torch.npu.set_device(rank)
    init_device_properties_triton()
    world_size = 2
    parallel_state.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="hccl",
    )
    try:
        parallel_state._DP = parallel_state.init_model_parallel_group(
            [[0, 1]], rank, "hccl", group_name="engram_v2_shared_dp"
        )
        # Each replica owns its tokens; the table is shared through host memory.
        parallel_state._ENGRAM_DP = parallel_state._DP
        parallel_state._TP = parallel_state.init_model_parallel_group(
            [[rank]], rank, "hccl", group_name="engram_v2_shared_tp"
        )
        parallel_state.init_model_parallel_group([[0, 1]], rank, "hccl", group_name="engram_v2_shared_ep")
        rows = sum(_HEAD_SIZES)
        codes = ((torch.arange(rows * _DIM).view(rows, _DIM) * 13) % 251 - 125).to(torch.int8)
        scales = torch.pow(2.0, torch.arange(rows * (_DIM // 32)).view(rows, _DIM // 32) % 5 - 3).float()
        tables = {
            layer: AscendParallelEngramEmbedding(rows, _DIM, _HEAD_SIZES, slot, dp_shared_memory=True)
            for slot, layer in enumerate((1, 14))
        }
        for table in tables.values():
            start, stop = table.vocab_start_idx, table.vocab_end_idx
            table.weight.copy_(codes[start:stop].npu())
            table.weight_scale_inv.copy_(scales[start:stop].npu())

        class HashState:
            lookback_depth = 2
            use_slot_cache = False

            def ensure_cache(self):
                return True

            def __call__(self, input_ids, *args):
                return self.ids

        model = object.__new__(DeepseekV41Model)
        torch.nn.Module.__init__(model)
        model.has_engram, model.engram_dp_shared_memory = True, True
        model.engram_hash = HashState()
        model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
        model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=tables.get(layer))) for layer in range(15)]
        model._engram_max_tokens, model._engram_input_buffers = _CAPACITY, None
        model._engram_graph_events = {}
        model._engram_prepare_stream = None
        model._engram_overlap_enabled = True
        model.engram_rotation = torch.eye(32, device="npu")
        bucket = (24, 48)[rank]
        binding = model.prime_engram_v2_graph_inputs(bucket)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            AscendParallelEngramEmbedding.wait_engram_event(binding["engram_mask_ready_event"], True)
            outputs = {}
            for layer in (1, 14):
                AscendParallelEngramEmbedding.wait_engram_event(binding["engram_pending"][layer], True)
                outputs[layer] = torch.where(
                    binding["engram_mask"][:bucket, None], binding["engram_lookups"][layer][:bucket], 0
                )
        # Unequal and empty replica batches: sharing keeps every rank
        # independent, so no rank waits on another's token slot.
        for phase, counts in enumerate(((5, 31), (3, 0), (0, 7)) * 3):
            count = counts[rank]
            ids_cpu = torch.stack([_ids(count, rank, phase + layer) for layer in range(2)], dim=1)
            model.engram_hash.ids = ids_cpu.npu()
            model.prepare_engram_inputs(
                torch.ones(count, dtype=torch.int32, device="npu"),
                torch.arange(count, device="npu"),
                bucket,
                torch.full((4, 2), -1, dtype=torch.int32, device="npu"),
                torch.tensor([0, count] if count else [0], dtype=torch.int32, device="npu"),
                cg_mode=CUDAGraphMode.FULL,
            )
            graph.replay()
            for slot, layer in enumerate((1, 14)):
                expected = torch.zeros((bucket, len(_HEAD_SIZES), _DIM), dtype=torch.bfloat16)
                expected[:count] = _reference(ids_cpu[:, slot], codes, scales)
                assert torch.equal(outputs[layer].cpu(), expected.flatten(1)), (rank, phase, layer)
        print(f"ENGRAM_V2_DP_SHARED_GRAPH_9_PASSED rank={rank}", flush=True)
    finally:
        torch.npu.synchronize()
        parallel_state._ENGRAM_DP = None
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def test_engram_v2_dp_shared_graph_overlap_with_unequal_and_empty_rank_batches():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    test_engram_v2_dp_shared_graph_overlap_with_unequal_and_empty_rank_batches()
