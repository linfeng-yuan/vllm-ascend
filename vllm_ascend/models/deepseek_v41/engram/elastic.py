# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 Engram: node-sharded checkpoint FP8 rows served by ElasticBuffer."""

import socket
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.distributed import get_ep_group, get_tp_group
from vllm.logger import logger
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ops.dsv41_a5.package_loader import import_packaged_a5_module

from .embedding import AscendParallelEngramEmbedding

SCALE_GROUP = 32
E8M0_ONE_BITS = 127
MAX_LOCAL_RANKS = 8
MAX_NODES = 4


def _elastic_buffer_cls():
    # The public package initializer discovers and JIT-builds unrelated ops.
    return import_packaged_a5_module("cann_ops_transformer.ops.mc2.common.elastic_buffer").ElasticBuffer


class EngramElasticGroup:
    """One independent HCCL context per table and per node."""

    def __init__(self, device_group, tp_group, tp_source):
        self.device_group = device_group
        self.tp_group = tp_group
        self.tp_source = tp_source
        self.rank = dist.get_rank(device_group)
        self.size = dist.get_world_size(device_group)
        self.is_source = dist.get_rank() == tp_source

    @classmethod
    def from_vllm(cls):
        ep, tp = get_ep_group(), get_tp_group()
        if tp.world_size != 1:
            raise ValueError("Elastic Engram requires TP1")
        hosts = [None] * ep.world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=ep.cpu_group)
        node_groups = [[ep.ranks[i] for i, host in enumerate(hosts) if host == name] for name in dict.fromkeys(hosts)]
        sizes = {len(ranks) for ranks in node_groups}
        if len(node_groups) > MAX_NODES or len(sizes) != 1 or max(sizes) > MAX_LOCAL_RANKS:
            raise ValueError("Elastic Engram supports up to four nodes with equal groups of at most eight ranks")
        selected = None
        # Every EP rank creates the same groups in the same order. Each call
        # creates new physical groups: two tables cannot share an HCCL context.
        for ranks in node_groups:
            dist.new_group(ranks, backend="gloo")
            device = dist.new_group(ranks, backend=dist.get_backend(ep.device_group))
            if dist.get_rank() in ranks:
                selected = cls(device, tp.device_group, tp.ranks[0])
        if selected is None:
            raise RuntimeError("Engram rank is absent from its EP group")
        return selected


class ElasticEngramEmbedding(nn.Module):
    """Full hash-head range, with FP8 rows split over local EP ranks.

    Each physical node owns a complete FP8 table, split into
    ceil(rows / node_group_size) rows per rank. The E8M0 scale table stays
    replicated in HBM, following ElasticBuffer's inference ABI.
    """

    @classmethod
    def _checkpoint_index(cls, root, key):
        return AscendParallelEngramEmbedding._checkpoint_index(root, key)

    def __init__(self, rows: int, dim: int, head_sizes: tuple[int, ...], group: EngramElasticGroup):
        super().__init__()
        if dim % SCALE_GROUP or (dim // SCALE_GROUP) % 2:
            raise ValueError("Elastic Engram width requires an even number of group32 scales")
        if rows <= 0 or not head_sizes or sum(head_sizes) > rows:
            raise ValueError("Invalid Elastic Engram table layout")
        self.rows = rows
        self.dim = dim
        self.n_hash_cols = len(head_sizes)
        self.group = group
        self.tp_size = 1
        self.dp_size = 1
        self._pending_wait = None
        self.shard_rows = (rows + group.size - 1) // group.size
        self.padded_rows = self.shard_rows * group.size
        self.start = group.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, rows)
        if self.start >= rows:
            raise ValueError("Elastic Engram needs at least one row per rank")
        # The real CPU shard is temporary during checkpoint loading. Keeping
        # only a parameter placeholder avoids two extra full-table allocations
        # while vLLM constructs the model and profiles with dummy weights.
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn, device="cpu"), requires_grad=False)
        self.weight_scale_inv = nn.Parameter(
            torch.empty((self.padded_rows, dim // SCALE_GROUP), dtype=torch.float8_e8m0fnu, device="npu"),
            requires_grad=False,
        )
        self.weight_scale_inv.data.view(torch.uint8).fill_(E8M0_ONE_BITS)
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(param, {"weight_loader": self._weight_loader})
        self.elastic_buffer = None

    def bind_checkpoint(self, path, key):
        self._checkpoint_path = path
        self._checkpoint_key = key

    def _weight_loader(self, param, loaded_weight):
        # The indexed reader loads both tensors in one pass. The separate
        # scale callback must not overwrite the replicated E8M0 device table.
        if param is self.weight:
            self.load_checkpoint(self._checkpoint_path, self._checkpoint_key)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        if self.elastic_buffer is not None:
            raise RuntimeError("Elastic Engram checkpoint is already loaded")
        root = Path(model_path)
        scale_key = key.removesuffix(".weight") + ".scale"
        index = self._checkpoint_index(root, key)
        if scale_key not in index:
            raise ValueError(f"{key}: FP8 ElasticBuffer needs {scale_key}")
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            source = file.get_slice(key)
            if source.get_shape() != [self.rows, self.dim] or source.get_dtype() not in ("F8_E4M3", "F8_E4M3FN"):
                raise ValueError(f"{key}: expected FP8 [{self.rows}, {self.dim}]")
            with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                scales = sf.get_slice(scale_key)
                if scales.get_shape() != [self.rows, self.dim // SCALE_GROUP] or scales.get_dtype() != "F8_E8M0":
                    raise ValueError(f"{scale_key}: expected E8M0 [{self.rows}, {self.dim // SCALE_GROUP}]")
                weight = torch.empty((self.shard_rows, self.dim), dtype=torch.float8_e4m3fn, device="cpu")
                weight_bits = weight.view(torch.uint8)
                weight_bits.zero_()
                for start in range(self.start, self.end, chunk_rows):
                    stop = min(start + chunk_rows, self.end)
                    weight_bits[start - self.start : stop - self.start].copy_(source[start:stop].view(torch.uint8))
                scale_bits = self.weight_scale_inv.data.view(torch.uint8)
                for start in range(0, self.rows, chunk_rows):
                    stop = min(start + chunk_rows, self.rows)
                    scale_bits[start:stop].copy_(scales[start:stop].view(torch.uint8))
        buffer_cls = _elastic_buffer_cls()
        size = buffer_cls.get_engram_storage_size_hint(self.shard_rows, self.dim, torch.float8_e4m3fn)
        buffer = buffer_cls(self.group.device_group, num_cpu_bytes=size)
        try:
            buffer.engram_write(weight, self.weight_scale_inv)
        except Exception:
            buffer.destroy()
            raise
        self.elastic_buffer = buffer
        logger.info(
            "Elastic Engram %s: node group %d, CPU FP8/rank %.2f GiB, device E8M0/rank %.2f GiB",
            key,
            self.group.size,
            self.shard_rows * self.dim / 1024**3,
            self.padded_rows * (self.dim // SCALE_GROUP) / 1024**3,
        )

    def begin_lookup(self, ids: torch.Tensor):
        if self.elastic_buffer is None:
            # vLLM profiles with dummy weights before checkpoint loading.
            return ids.shape, None, None
        valid = (ids >= 0) & (ids < self.rows)
        flat = torch.where(valid, ids, 0).reshape(-1) if self.group.is_source else ids.new_empty(0)
        indices = flat.to(device=self.weight_scale_inv.device, dtype=torch.int32)
        wait = self.elastic_buffer.engram_fetch(indices)
        self._pending_wait = wait

        def complete():
            try:
                return wait()
            finally:
                self._pending_wait = None

        return ids.shape, valid, complete

    def retire_lookup(self):
        # Consume outstanding callbacks if another table's launch/finish fails.
        wait, self._pending_wait = self._pending_wait, None
        if wait is not None:
            wait()

    def finish_lookup(self, pending):
        shape, valid, wait = pending
        count = shape[0] * shape[1]
        if wait is None:
            return torch.zeros((*shape, self.dim), dtype=torch.bfloat16, device=self.weight_scale_inv.device)
        import torch_npu

        fetched, scales = wait()
        if self.group.is_source and count:
            result = torch_npu.npu_anti_mx_quant(
                fetched,
                scales.unflatten(-1, (-1, 2)),
                axis=-1,
                dst_type=torch.bfloat16,
                src_type=torch.float8_e4m3fn,
            ).reshape(count, self.dim)
        else:
            result = torch.empty((count, self.dim), dtype=torch.bfloat16, device=self.weight_scale_inv.device)
        result = result.view(*shape, self.dim)
        return torch.where(valid.unsqueeze(-1), result, 0) if valid is not None else result

    def forward(self, ids):
        return self.finish_lookup(self.begin_lookup(ids))

    def destroy(self):
        if self.elastic_buffer is not None:
            self.elastic_buffer.destroy()
            self.elastic_buffer = None


def preflight_elastic_checkpoint(root, layer_ids, row_counts, dim):
    """Reject incompatible checkpoints before allocating host and device tables."""
    root = Path(root)
    for layer_id, rows in zip(layer_ids, row_counts):
        key = f"layers.{layer_id}.engram.embed.weight"
        scale_key = key.removesuffix(".weight") + ".scale"
        index = ElasticEngramEmbedding._checkpoint_index(root, key)
        if scale_key not in index:
            raise ValueError(f"{key}: FP8 ElasticBuffer needs {scale_key}")
        for name, shape, dtype in (
            (key, [rows, dim], ("F8_E4M3", "F8_E4M3FN")),
            (scale_key, [rows, dim // SCALE_GROUP], ("F8_E8M0",)),
        ):
            shard = root / index[name]
            if not shard.is_file():
                raise ValueError(f"{name}: checkpoint shard is absent: {shard}")
            with safe_open(shard, framework="pt", device="cpu") as file:
                if name not in file.keys():  # noqa: SIM118 - safe_open is not a mapping
                    raise ValueError(f"{name}: tensor is absent from {shard}")
                source = file.get_slice(name)
                if source.get_shape() != shape or source.get_dtype() not in dtype:
                    raise ValueError(f"{name}: expected {dtype} {shape}")


def prepare_elastic_lookups(tables, hashes, output_buffers=None, *, output_tokens=None, ready_events=None):
    """Launch both one-sided fetches before waiting on the current producer.

    The model owns the preparation stream and mask event. Each table event
    follows its fixed-address output and zeroed padding, using the same
    eager/FULL graph protocol as native UVA. No per-step DP gather is needed.
    """
    lookups = {}
    try:
        launched = [
            (layer, table, table.begin_lookup(hashes[:, slot])) for slot, (layer, table) in enumerate(tables.items())
        ]
        for layer, table, pending in launched:
            values = table.finish_lookup(pending).flatten(1)
            if output_buffers is None:
                lookups[layer] = values
            else:
                target = output_buffers[layer]
                target[: values.shape[0]].copy_(values)
                target[values.shape[0] : output_tokens].zero_()
                lookups[layer] = target
            if ready_events is not None:
                ready_events[layer].record(torch.npu.current_stream())
    except Exception:
        # Consume callbacks even when a later table launch/dequantization fails.
        for table in tables.values():
            try:
                table.retire_lookup()
            except Exception:
                logger.exception("Failed to retire Elastic Engram fetch")
        raise
    return lookups
