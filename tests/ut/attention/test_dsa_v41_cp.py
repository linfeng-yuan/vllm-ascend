# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 CP contracts between the runner, cache metadata and attention.

Metadata tests use real builders for all four cache planes. Forward tests
exercise the inherited orchestration with CPU projections and cache writes;
only distributed communication and device operators are replaced. Graph
tests check persistent addresses and state controls, not NPU graph replay.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_cp, dsa_v41_cp
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.ops import rope_dsv4
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext

LAYER = "model.layers.0.attn"
PREFIXES = {
    "swa": "model.layers.0.swa_cache",
    "long_kv": "model.layers.0.long_kv_cache",
    "index_k": "model.layers.0.indexer.k_cache",
    "compressor_state": "model.layers.0.compressor.state_cache",
}
# Scheduler token IDs owned by each rank, in DualChunkSwap order. Repeated
# decode rows are added separately; the oracle does not call PCP helpers.
PREFILL_ROWS = {
    2: [[0, 1, 6, 7, 8, 9, 14, 15], [2, 3, 4, 5, 10, 11, 12, 13]],
    4: [[0, 7, 8, 15], [1, 6, 9, 14], [2, 5, 10, 13], [3, 4, 11, 12]],
}


@pytest.fixture
def builders(monkeypatch):
    """Allocate real CPU builders with visible RoPE values and small caches."""
    full_cos = torch.arange(128, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 1, 1, 2).contiguous()
    full_sin = -full_cos
    rope_state = rope_dsv4.RopeGlobalState()
    rope_state.full_rope_cache["cp-test"] = (full_cos, full_sin)
    rope_state.registry_summary["cp-test"] = {"default"}
    rope_state.layer_info[LAYER] = ("cp-test", ["default"])
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", rope_state)
    monkeypatch.setattr(dsa_v41.DeviceOperator, "get_deepseek_v41_backend", lambda: None)
    for module in (dsa_v41, dsa_v41_cp):
        monkeypatch.setattr(module, "get_full_cos_and_sin_dsa_for_layer", lambda name: (full_cos, full_sin))

    def make(world=2, rank=0, *, max_seqs=2, graph_sizes=(12, 24), dcp=1, legacy=False):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=world if legacy else 1,
                prefill_context_parallel_size=1 if legacy else world,
                decode_context_parallel_size=dcp,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=64),
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=graph_sizes),
            cache_config=SimpleNamespace(block_size=8),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    sliding_window=4,
                    num_attention_heads=8 if legacy else 2,
                    head_dim=2,
                    qk_rope_head_dim=2,
                    index_topk=3,
                )
            ),
        )
        common = dict(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32)
        specs = {
            "swa": AscendSlidingWindowMLASpec(**common, sliding_window=4),
            "long_kv": AscendMLAAttentionSpec(**common, tokens_per_state=2),
            "index_k": AscendMLAAttentionSpec(**common, tokens_per_state=2, scale_dim=1),
            "compressor_state": CircularBufferSpec(**common, tokens_per_state=2),
        }
        monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=rank))
        monkeypatch.setattr(dsa_cp, "get_tp_group", lambda: SimpleNamespace(world_size=world, rank_in_group=rank))
        builder_class = (
            dsa_v41_cp.AscendDSAV41CPMetadataBuilder if legacy else dsa_v41_cp.AscendDSAV41PCPMetadataBuilder
        )
        result = {
            kind: builder_class(spec, [PREFIXES[kind]], config, torch.device("cpu")) for kind, spec in specs.items()
        }
        for builder in result.values():
            builder.prepare_source_rope()
        return result

    return make


def _batch(world, rank, scenario):
    """The runner's local view and independent canonical global context."""
    if scenario == "prefill":
        rows = PREFILL_ROWS[world]
        lengths, computed, flags = [8, 8], [4, 8], [True, True]
        local_lengths = [8 // (2 * world)] * 4
    elif scenario == "mixed":
        rows = [row[: len(row) // 2] + list(range(8, 14)) for row in PREFILL_ROWS[world]]
        lengths, computed, flags = [8, 6], [4, 8], [True, False]
        local_lengths = [8 // (2 * world)] * 2 + [6]
    elif scenario == "empty":
        rows = [[0]] + [[] for _ in range(world - 1)]
        lengths, computed, flags = [1], [3], [True]
        local_lengths = [1, 0] if rank == 0 else [0, 0]
    else:
        assert scenario in {"decode", "dummy"}
        rows = [list(range(6)) for _ in range(world)]
        lengths, computed, flags = [6], [8], [False]
        local_lengths = [6, 0, 0, 0]

    graph = scenario in {"decode", "dummy"}
    positions = torch.cat([torch.arange(start, start + length) for start, length in zip(computed, lengths)])
    num_tokens = len(positions)
    num_reqs = len(lengths)
    global_reqs = 4 if graph else num_reqs
    global_padded = 24 if graph else num_tokens + 2
    local_padded = 24 if graph else max(map(len, rows)) + 2
    offsets = np.array([0, *np.cumsum(lengths), *([num_tokens] * (global_reqs - num_reqs))], dtype=np.int32)
    seq_lens = torch.tensor(
        [start + length for start, length in zip(computed, lengths)] + [0] * (global_reqs - num_reqs), dtype=torch.int32
    )
    padded_positions = torch.cat([positions, torch.zeros(global_padded - num_tokens, dtype=torch.int64)])
    tables = torch.tensor([[2 + 2 * req, 3 + 2 * req] for req in range(global_reqs)], dtype=torch.int32)
    slots = torch.cat(
        [
            16 * (req + 1) + torch.arange(start, start + length)
            for req, (start, length) in enumerate(zip(computed, lengths))
        ]
    )
    global_slots = torch.cat([slots, torch.full((global_padded - num_tokens,), -1)])
    global_batch = SimpleNamespace(
        num_tokens=num_tokens,
        num_tokens_after_padding=global_padded,
        num_reqs=num_reqs,
        num_reqs_after_padding=global_reqs,
        query_start_loc=torch.from_numpy(offsets.copy()),
        query_start_loc_np=offsets.copy(),
        seq_lens=seq_lens,
        seq_lens_np=seq_lens.numpy().copy(),
        seq_lens_cpu_upper_bound=seq_lens + 1000,
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
        num_scheduled_tokens=np.array(lengths, dtype=np.int32),
        is_prefilling_np=np.array(flags),
        positions=padded_positions,
        dcp_local_seq_lens=None,
        attn_state=scenario,
        is_dummy=scenario == "dummy",
    )
    # First occurrence selects a single copy of replicated decode tokens.
    gathered_rows = torch.full((world, local_padded), -1, dtype=torch.int64)
    for peer, ids in enumerate(rows):
        gathered_rows[peer, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
    restore = torch.tensor(
        [int((gathered_rows.flatten() == token).nonzero()[0]) for token in range(num_tokens)], dtype=torch.int64
    )
    restore = torch.cat([restore, torch.full((global_padded - num_tokens,), -99)])
    gathered_slots = torch.full_like(gathered_rows, -1)
    for peer, ids in enumerate(rows):
        gathered_slots[peer, : len(ids)] = slots[ids]
    context = AscendPCPAttentionContext(
        global_batch=global_batch,
        global_block_tables=(tables, tables + 16),
        global_slot_mappings=torch.stack([global_slots, torch.where(global_slots >= 0, global_slots + 128, -1)]),
        hidden_restore_idx=restore,
    )
    local_ids = rows[rank]
    local_positions = torch.cat([positions[local_ids], torch.zeros(local_padded - len(local_ids), dtype=torch.int64)])
    local_offsets = torch.tensor([0, *np.cumsum(local_lengths)], dtype=torch.int32)
    local_seqs = torch.tensor(
        [
            int(local_positions[end - 1]) + 1 if end > start else 0
            for start, end in zip(local_offsets[:-1], local_offsets[1:])
        ],
        dtype=torch.int32,
    )
    local_flags = [False] if graph else ([True, True, False] if scenario == "mixed" else [True] * len(local_lengths))
    local = AscendCommonAttentionMetadata(
        query_start_loc=local_offsets,
        query_start_loc_cpu=local_offsets.clone(),
        seq_lens=local_seqs,
        seq_lens_cpu=local_seqs.clone(),
        seq_lens_cpu_upper_bound=local_seqs + 1000,
        num_reqs=len(local_lengths),
        num_actual_tokens=len(local_ids),
        num_input_tokens=local_padded,
        max_query_len=max(local_lengths),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=tables.repeat_interleave(2, dim=0)[: len(local_lengths)],
        slot_mapping=gathered_slots.flatten().clone(),
        positions=local_positions,
        is_prefilling=torch.tensor(local_flags),
        attn_state=scenario,
    )
    return SimpleNamespace(
        local=local,
        context=context,
        rows=rows,
        positions=positions,
        slots=slots,
        local_ids=local_ids,
        graph=graph,
        lengths=lengths,
        computed=computed,
    )


def _build_planes(builders, batch):
    # Like the runner: batch metadata spans cache groups, physical mappings do
    # not. Global and local namespaces must stay separate within both scopes.
    shared_batch, result = {}, {}
    for kind, builder in builders.items():
        group = int(kind in {"long_kv", "index_k"})
        slots = batch.local.slot_mapping.clone()
        if group:
            slots = torch.where(slots >= 0, slots + 128, -1)
        result[PREFIXES[kind]] = builder.build(
            0,
            batch.local.replace(slot_mapping=slots),
            pcp_context=batch.context,
            pcp_cache_group_idx=group,
            num_actual_reqs=1 if batch.graph else batch.local.num_reqs,
            full_graph_mode=batch.graph,
            common_v41_metadata={},
            common_v41_batch_metadata=shared_batch,
        )
    # The target adapter must ignore independent draft cache metadata.
    result["draft.attn"] = object()
    return result


class TestLegacyMetadata:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("last_rank", [False, True])
    @pytest.mark.parametrize("causal", [False, True])
    def test_query_intersection_uses_device_lengths_and_global_rope(
        self, builders, monkeypatch, world, last_rank, causal
    ):
        rank = world - 1 if last_rank else 0
        batch = _batch(world, rank, "prefill")
        global_batch = batch.context.global_batch
        # Only pinned staging is device-specific. The legacy metadata builder
        # and its request-intersection calculation execute unmocked.
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
        common = AscendCommonAttentionMetadata(
            query_start_loc=global_batch.query_start_loc,
            query_start_loc_cpu=global_batch.query_start_loc.clone(),
            seq_lens=global_batch.seq_lens,
            seq_lens_cpu=global_batch.seq_lens + 1000,
            num_reqs=2,
            num_actual_tokens=16,
            num_input_tokens=18,
            max_query_len=8,
            max_seq_len=16,
            block_table_tensor=batch.context.global_block_tables[0],
            slot_mapping=batch.context.global_slot_mappings[0],
            positions=global_batch.positions,
            is_prefilling=torch.tensor([True, True]),
            causal=causal,
        )
        local = builders(world, rank, legacy=True)["swa"].build(
            0, common, common_v41_metadata={}, common_v41_batch_metadata={}
        )
        # Independent examples include partial requests and rank-tail padding.
        expected = {
            (2, 0): ([0, 8, 9], [12, 9], 0, 9),
            (2, 1): ([0, 0, 7], [0, 16], 9, 16),
            (4, 0): ([0, 5, 5], [9, 0], 0, 5),
            (4, 3): ([0, 0, 1], [0, 16], 15, 16),
        }
        offsets, lengths, start, end = expected[world, rank]
        if not causal:
            lengths = [
                length if right > left else 0 for length, left, right in zip([12, 16], offsets[:-1], offsets[1:])
            ]
        torch.testing.assert_close(local.query_start_loc, torch.tensor(offsets, dtype=torch.int32))
        torch.testing.assert_close(local.seq_lens, torch.tensor(lengths, dtype=torch.int32))
        assert local.num_actual_tokens == end - start
        assert local.seq_lens.data_ptr() != local.global_metadata.seq_lens.data_ptr()
        torch.testing.assert_close(local.global_metadata.seq_lens, torch.tensor([12, 16], dtype=torch.int32))
        torch.testing.assert_close(local.positions, batch.positions[start:end])
        torch.testing.assert_close(local.cos[LAYER], local.global_metadata.cos[LAYER][start:end])
        torch.testing.assert_close(local.global_metadata.cos[LAYER][:16, 0, 0, 0], batch.positions.float())


class TestPCPMetadata:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("last_rank", [False, True])
    @pytest.mark.parametrize("scenario", ["prefill", "mixed", "decode", "empty", "dummy"])
    def test_local_queries_and_global_cache_planes(self, builders, world, last_rank, scenario):
        rank = world - 1 if last_rank else 0
        batch = _batch(world, rank, scenario)
        metadata = _build_planes(builders(world, rank), batch)
        for kind, prefix in PREFIXES.items():
            local = metadata[prefix]
            global_meta = local.global_metadata
            assert isinstance(local, dsa_v41_cp.AscendDSAV41PCPMetadata)
            assert local.num_actual_tokens == len(batch.local_ids)
            assert local.local_num_tokens_after_padding == batch.local.num_input_tokens
            assert global_meta.num_actual_tokens == len(batch.positions)
            assert global_meta.num_actual_reqs == len(batch.lengths)
            torch.testing.assert_close(local.positions[: len(batch.local_ids)], batch.positions[batch.local_ids])
            torch.testing.assert_close(global_meta.positions[: len(batch.positions)], batch.positions)
            assert torch.all(local.hidden_restore_idx[len(batch.positions) :] == 0)
            if batch.graph:
                expected_offsets = torch.arange(0, 25, 6, dtype=torch.int32)
                torch.testing.assert_close(local.query_start_loc, expected_offsets)
                torch.testing.assert_close(global_meta.query_start_loc, expected_offsets)
                assert torch.count_nonzero(local.seq_lens[1:]) == 0
                assert torch.count_nonzero(global_meta.seq_lens[1:]) == 0
            if kind == "compressor_state":
                assert local.c2_ring_metadata is None
                used = torch.zeros_like(global_meta.c2_ring_metadata[1])
                if scenario != "dummy":
                    used[: len(batch.lengths)] = torch.tensor(batch.lengths, dtype=used.dtype)
                torch.testing.assert_close(global_meta.c2_ring_metadata[1], used)
                complete = torch.zeros_like(global_meta.c2_complete_mask)
                if scenario != "dummy":
                    complete[: len(batch.positions)] = batch.positions.remainder(2) == 1
                torch.testing.assert_close(global_meta.c2_complete_mask, complete)
                assert torch.all(global_meta.slot_mapping == -1)
                continue
            expected = batch.slots + (128 if kind in {"long_kv", "index_k"} else 0)
            if kind in {"long_kv", "index_k"}:
                expected = torch.tensor([slot // 2 if int(slot) % 2 else -1 for slot in expected])
            if scenario == "dummy":
                expected.fill_(-1)
            torch.testing.assert_close(global_meta.flat_slot_mapping[: len(expected)], expected)
            assert torch.all(global_meta.flat_slot_mapping[len(expected) :] == -1)
            torch.testing.assert_close(local.flat_slot_mapping[: len(batch.local_ids)], expected[batch.local_ids])
            assert torch.all(local.flat_slot_mapping[len(batch.local_ids) :] == -1)
            if kind == "swa" and batch.local_ids:
                assert local.cos[LAYER].data_ptr() != global_meta.cos[LAYER].data_ptr()
                torch.testing.assert_close(
                    local.cos[LAYER][: len(batch.local_ids), 0, 0, 0], batch.positions[batch.local_ids].float()
                )
                torch.testing.assert_close(
                    global_meta.cos[LAYER][: len(batch.positions), 0, 0, 0], batch.positions.float()
                )

    @pytest.mark.parametrize("max_seqs,graph_sizes", [(2, (12, 24)), (16, ())])
    def test_capacity_and_buffer_lifetime_across_batches(self, builders, max_seqs, graph_sizes):
        group = builders(max_seqs=max_seqs, graph_sizes=graph_sizes)
        for builder in group.values():
            assert builder._seq_lens.numel() >= max(2 * max_seqs, max(graph_sizes, default=0)) + 1
            assert builder._global_builder._seq_lens.numel() >= max(max_seqs, max(graph_sizes, default=0)) + 1
        addresses = None
        for scenario in ("prefill", "decode", "dummy", "empty"):
            batch = _batch(2, 1, scenario)
            metadata = _build_planes(group, batch)
            swa = metadata[PREFIXES["swa"]]
            state = metadata[PREFIXES["compressor_state"]].global_metadata
            current = (
                swa.hidden_restore_idx.data_ptr(),
                swa.seq_lens.data_ptr(),
                swa.global_metadata.cos[LAYER].data_ptr(),
                state.c2_complete_mask.data_ptr(),
            )
            if addresses is not None:
                assert current == addresses
            addresses = current
            assert torch.all(swa.hidden_restore_idx[len(batch.positions) :] == 0)
            # Empty ranks use no local RoPE, but cannot destroy global views.
            if scenario != "empty":
                local_rope = swa.cos[LAYER].clone()
                group["swa"]._build_pcp_rope_views(batch.local, global_cache=True)
                torch.testing.assert_close(swa.cos[LAYER], local_rope)

    @pytest.mark.parametrize("in_graph", [False, True])
    @pytest.mark.parametrize("raises", [False, True])
    def test_metadata_scope_restores_both_builders_on_exit(self, builders, in_graph, raises):
        builder = builders()["swa"]
        # Device metadata guards are enabled only for the packaged A5 path.
        builder._uses_a5_packed_cache = builder._global_builder._uses_a5_packed_cache = True
        scope = pytest.raises(RuntimeError, match="abort") if raises else nullcontext()
        with scope, builder.defer_device_metadata(in_graph=in_graph):
            for owner in (builder, builder._global_builder):
                assert owner._device_metadata_enabled
                assert owner._device_metadata_in_graph == in_graph
            if raises:
                raise RuntimeError("abort")
        for owner in (builder, builder._global_builder):
            assert not owner._device_metadata_enabled
            assert not owner._device_metadata_in_graph

    def test_rejects_unsupported_context(self, builders):
        with pytest.raises(NotImplementedError, match="DCP=1"):
            builders(dcp=2)
        builder = builders()["swa"]
        with pytest.raises(AssertionError, match="global batch context"):
            builder.build(0, _batch(2, 0, "prefill").local)


class TestForwardContract:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("scenario", ["prefill", "mixed", "decode", "empty", "dummy"])
    @pytest.mark.parametrize("kv_source", [False, True])
    @pytest.mark.parametrize("index_role", ["reuse", "index_source", "candidate_source", "filtered_source"])
    def test_pcp_updates_global_caches_once_and_keeps_queries_local(
        self, builders, monkeypatch, world, scenario, kv_source, index_role
    ):
        rank = world - 1
        batch = _batch(world, rank, scenario)
        metadata = _build_planes(builders(world, rank), batch)
        impl = dsa_v41_cp.AscendDSAV41PCPImpl.__new__(dsa_v41_cp.AscendDSAV41PCPImpl)
        index_source = index_role != "reuse"
        candidate_source = index_role == "candidate_source"
        filtered_source = index_role == "filtered_source"
        impl.role = SimpleNamespace(
            is_kv_source=kv_source,
            has_long_context=True,
            is_index_source=index_source,
            is_candidate_source=candidate_source,
            uses_candidate_filter=filtered_source,
        )
        impl.topology = SimpleNamespace(candidate_topk_blocks=2, candidate_block_size=8)
        impl.swa_prefix, impl.long_kv_source_prefix, impl.index_k_source_prefix, impl.compressor_state_prefix = (
            PREFIXES.values()
        )
        events = []
        canonical = torch.arange(len(batch.positions) * 4, dtype=torch.float32).view(-1, 4) + 1
        peers = []
        for ids in batch.rows:
            peer = torch.full((batch.local.num_input_tokens, 4), 9999.0)
            peer[: len(ids)] = canonical[ids]
            peers.append(peer)
        hidden = peers[rank]

        def gather(value, dim):
            events.append("gather")
            torch.testing.assert_close(value, hidden)
            return torch.cat(peers, dim=dim)

        cache = torch.full((256, 2), -1.0)

        def write(cache_tensor, slots, values, *, kind):
            events.append("swa")
            torch.testing.assert_close(values, canonical[:, :2])
            expected_slots = torch.full_like(batch.slots, -1) if scenario == "dummy" else batch.slots
            torch.testing.assert_close(slots, expected_slots)
            valid = slots >= 0
            cache_tensor[slots[valid]] = values[valid]

        def rotary(value, cos, sin, **kwargs):
            events.append("rotary")
            expected = batch.positions if value.shape[0] == len(batch.positions) else batch.positions[batch.local_ids]
            torch.testing.assert_close(cos[:, 0, 0, 0], expected.float())
            return value

        def compressed(attn, values, positions, cos, sin, meta, **kwargs):
            events.append("compressed")
            torch.testing.assert_close(values, canonical)
            assert meta.swa is metadata[PREFIXES["swa"]].global_metadata
            assert meta.compressor.state is metadata[PREFIXES["compressor_state"]].global_metadata
            if scenario == "dummy":
                assert torch.count_nonzero(meta.compressor.state.c2_ring_metadata[1]) == 0
                assert torch.count_nonzero(meta.compressor.state.c2_complete_mask) == 0

        def attention(attn, q, meta, indices):
            events.append("attention")
            assert events.count("swa") == 1
            expected_cache = torch.full_like(canonical[:, :2], -1) if scenario == "dummy" else canonical[:, :2]
            torch.testing.assert_close(cache[batch.slots], expected_cache)
            torch.testing.assert_close(indices, attn.shared_state.topk_indices[: len(batch.local_ids)])
            return q.clone()

        def fused_query_weights(values, qr, cos, sin):
            events.append("index_query")
            assert events.count("swa") == 1
            expected = canonical[batch.local_ids]
            torch.testing.assert_close(values, expected)
            torch.testing.assert_close(qr, expected)
            torch.testing.assert_close(cos[:, 0, 0, 0], batch.positions[batch.local_ids].float())
            return values[:, :2].to(torch.uint8), torch.ones_like(values[:, :1]), torch.ones_like(values[:, :1])

        index_cache, folded_cache = (object(), object()), object()

        def select(query, weights, positions, source_cache, cache_metadata, **kwargs):
            events.append("select")
            assert query is None  # The fused device boundary already quantized Q.
            torch.testing.assert_close(kwargs["quantized_query"], canonical[batch.local_ids, :2].to(torch.uint8))
            torch.testing.assert_close(positions, batch.positions[batch.local_ids])
            assert weights.shape[0] == len(batch.local_ids)
            assert cache_metadata is metadata[PREFIXES["index_k"]]
            assert source_cache == ((*index_cache, folded_cache) if filtered_source else index_cache)
            assert kwargs["is_candidate_source"] == candidate_source
            assert kwargs["uses_candidate_filter"] == filtered_source
            torch.testing.assert_close(kwargs["candidates"], candidates_before[: len(batch.local_ids)])
            assert kwargs["indices_output"].data_ptr() == topk.data_ptr()
            selected = torch.full_like(kwargs["indices_output"], 11)
            candidates = torch.full_like(kwargs["candidates"], 23) if candidate_source else kwargs["candidates"]
            return selected, candidates

        def project(value, output):
            events.append("project")
            output.copy_(value.flatten(1))

        kv = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            kv.weight.copy_(torch.eye(4)[:2])
        topk = torch.arange(batch.local.num_input_tokens * 3, dtype=torch.int32).view(-1, 3)
        candidates = torch.full((batch.local.num_input_tokens, 1, 2), 7, dtype=torch.int32)
        attn = SimpleNamespace(
            rotary_emb=SimpleNamespace(layername=LAYER),
            n_heads=2,
            n_local_heads=2,
            head_dim=2,
            nope_head_dim=0,
            wq_a=torch.nn.Identity(),
            q_norm=torch.nn.Identity(),
            wq_b=torch.nn.Identity(),
            wkv=kv,
            kv_norm=torch.nn.Identity(),
            dsv41_backend=SimpleNamespace(write_attention_cache=write, apply_partial_rotary_inplace=rotary),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[cache]),
                dsa_attn=SimpleNamespace(impl=SimpleNamespace(_forward_o_proj=project)),
            ),
            indexer=SimpleNamespace(qw_fusion=fused_query_weights, select_projected=select, dsv41_backend=object()),
            shared_state=SimpleNamespace(
                topk_indices=topk, candidates=candidates, candidate_lengths=None, topk_lengths=None
            ),
        )
        monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather))
        monkeypatch.setattr(
            dsa_v41,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=metadata,
                no_compile_layers={
                    PREFIXES["index_k"]: SimpleNamespace(kv_cache=[index_cache]),
                    PREFIXES["index_k"] + "_folded": SimpleNamespace(kv_cache=[folded_cache]),
                },
            ),
        )
        monkeypatch.setattr(impl, "_write_compressed_source", compressed)
        monkeypatch.setattr(impl, "_forward_attention", attention)
        output = torch.full_like(hidden, -777.0)
        topk_before = topk.clone()
        candidates_before = candidates.clone()
        assert impl.forward(attn, None, hidden, output) is output
        expected = torch.zeros_like(hidden)
        expected[: len(batch.local_ids)] = canonical[batch.local_ids]
        torch.testing.assert_close(output, expected)
        if index_source:
            topk_before[: len(batch.local_ids)] = 11
        if candidate_source:
            candidates_before[: len(batch.local_ids)] = 23
        torch.testing.assert_close(topk, topk_before)
        torch.testing.assert_close(candidates, candidates_before)
        assert events.count("gather") == events.count("swa") == events.count("project") == 1
        assert events.count("compressed") == int(kv_source)
        assert events.count("attention") == int(bool(batch.local_ids))
        assert events.count("index_query") == events.count("select") == int(index_source and bool(batch.local_ids))
        if scenario == "dummy":
            assert torch.all(cache == -1)

    def test_legacy_cp_compressor_uses_global_rope_and_query_slice(self, monkeypatch):
        impl = dsa_v41_cp.AscendDSAV41CPImpl.__new__(dsa_v41_cp.AscendDSAV41CPImpl)
        hidden = torch.arange(24).view(6, 4)
        local = SimpleNamespace(swa=SimpleNamespace(cp_token_range=(2, 4, 2, 0), num_actual_tokens=2))
        cos, sin = torch.ones(4, 2), torch.zeros(4, 2)
        global_metadata = SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=4),
            positions=torch.arange(4),
            rope=lambda name, count: (cos[:count], sin[:count]),
        )
        calls = []
        monkeypatch.setattr(impl, "_global_layer_metadata", lambda metadata: global_metadata)
        monkeypatch.setattr(dsa_v41_cp, "get_forward_context", lambda: SimpleNamespace(attn_metadata=object()))
        monkeypatch.setattr(impl, "_write_compressed_source", lambda *args, **kwargs: calls.append(args))
        torch.testing.assert_close(impl._indexer_hidden_states(hidden, local), hidden[2:4])
        impl._write_forward_compressed_source(
            SimpleNamespace(rotary_emb=SimpleNamespace(layername=LAYER)), hidden, None, -cos, -sin, local, None
        )
        assert len(calls) == 1
        torch.testing.assert_close(calls[0][1], hidden[:4])
        assert calls[0][3].data_ptr() == cos.data_ptr()
        assert calls[0][4].data_ptr() == sin.data_ptr()
        assert calls[0][5] is global_metadata


class TestProjectionLayout:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("rank", [0, -1])
    @pytest.mark.parametrize("live_tokens,extra_padding", [(5, 0), (5, 1), (5, 3), (0, 2)])
    def test_legacy_cp_preserves_tp_heads_and_flashcomm_output_extent(
        self, monkeypatch, world, rank, live_tokens, extra_padding
    ):
        rank %= world
        per_rank = (live_tokens + world - 1) // world
        token_extent = world * (per_rank + extra_padding)
        heads, width = 2 * world, 2
        canonical = torch.arange(live_tokens * heads * width, dtype=torch.float32).view(live_tokens, heads, width) + 1
        peers = torch.zeros(world * per_rank, heads, width)
        peers[:live_tokens] = canonical
        local = peers[rank * per_rank : min((rank + 1) * per_rank, live_tokens)]
        group = SimpleNamespace(world_size=world, device_group=object())

        def exchange(recv, send, *, group):
            # Emulate peers' all-to-all sends, while running restore_tp_heads
            # itself unmocked to test the token/head coordinate change.
            expected_send = peers[rank * per_rank : (rank + 1) * per_rank]
            expected_send = torch.cat(expected_send.split(2, dim=1), dim=0)
            torch.testing.assert_close(send, expected_send)
            recv.copy_(peers[:, rank * 2 : (rank + 1) * 2])

        def project(value, output):
            expected = torch.zeros(token_extent, 2, width)
            expected[:live_tokens] = canonical[:, rank * 2 : (rank + 1) * 2]
            torch.testing.assert_close(value, expected)
            # FlashComm reduce-scatter keeps a TP-local token buffer.
            output.copy_(value.flatten(1).chunk(world, dim=0)[rank])

        monkeypatch.setattr(dsa_v41_cp, "get_tp_group", lambda: group)
        monkeypatch.setattr(dsa_cp.dist, "all_to_all_single", exchange)
        impl = dsa_v41_cp.AscendDSAV41CPImpl.__new__(dsa_v41_cp.AscendDSAV41CPImpl)
        attn = SimpleNamespace(
            dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(_forward_o_proj=project)))
        )
        output = torch.full((token_extent // world, 4), -777.0)
        metadata = SimpleNamespace(swa=SimpleNamespace(cp_token_range=(0, 0, per_rank, 0)))
        assert impl._project_output(attn, local, torch.empty(token_extent, 4), metadata, projected=output) is output
        full = torch.zeros(token_extent, 4)
        full[:live_tokens] = canonical[:, rank * 2 : (rank + 1) * 2].flatten(1)
        torch.testing.assert_close(output, full.chunk(world)[rank])

    @pytest.mark.parametrize("impl_class", [dsa_v41.AscendDSAV41Impl, dsa_v41_cp.AscendDSAV41PCPImpl])
    @pytest.mark.parametrize("live,padded", [(4, 4), (3, 8), (0, 8)])
    def test_ordinary_and_pcp_preserve_local_output_buffer(self, impl_class, live, padded):
        values = torch.arange(live * 4, dtype=torch.float32).view(live, 2, 2) + 1
        output = torch.full((padded, 4), -777.0)
        attn = SimpleNamespace(
            dsa_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    impl=SimpleNamespace(_forward_o_proj=lambda value, out: out.copy_(value.flatten(1)))
                )
            )
        )
        impl = impl_class.__new__(impl_class)
        assert impl._project_output(attn, values, torch.empty(padded, 4), None, projected=output) is output
        expected = torch.zeros_like(output)
        expected[:live] = values.flatten(1)
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("legacy,pcp", [(False, False), (True, False), (False, True), (True, True)])
def test_backend_selection_follows_current_target_or_draft_config(monkeypatch, legacy, pcp):
    config = {"legacy": legacy, "pcp": pcp}
    monkeypatch.setattr(dsa_v41_cp, "enable_dsa_cp", lambda: config["legacy"])
    monkeypatch.setattr(dsa_v41_cp, "enable_pcp", lambda: config["pcp"])
    if legacy and pcp:
        with pytest.raises(ValueError, match="cannot be enabled"):
            dsa_v41_cp.get_v41_cp_classes()
        return
    expected = (
        (dsa_v41_cp.AscendDSAV41CPMetadataBuilder, dsa_v41_cp.AscendDSAV41CPImpl)
        if legacy
        else (dsa_v41_cp.AscendDSAV41PCPMetadataBuilder, dsa_v41_cp.AscendDSAV41PCPImpl)
        if pcp
        else (dsa_v41.AscendDSAV41MetadataBuilder, dsa_v41.AscendDSAV41Impl)
    )
    assert dsa_v41_cp.get_v41_cp_classes() == expected
    # DSpark constructs its replicated draft under PCP1 in the same process.
    config.update(legacy=False, pcp=False)
    assert dsa_v41_cp.get_v41_cp_classes() == (dsa_v41.AscendDSAV41MetadataBuilder, dsa_v41.AscendDSAV41Impl)
