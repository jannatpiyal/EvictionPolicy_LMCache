"""
Queued transfer scheduler for batched KV movement across entries.

This sits above `KVEntry` and batches promotions/demotions for multiple cache
entries so the cache can make room or promote hits with fewer synchronization
points and less per-entry transfer overhead.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

import torch

from cache.kv_entry import KVEntry


class TransferScheduler:
    def __init__(self, layer_batch_size: int = 4):
        self.layer_batch_size = max(1, int(layer_batch_size))
        self._promotion_queue: list[KVEntry] = []
        self._cpu_demotion_queue: list[KVEntry] = []
        self._disk_spill_queue: list[KVEntry] = []

    def queue_promotion(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._promotion_queue.extend(self._normalize_entries(entries))

    def queue_cpu_demotion(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._cpu_demotion_queue.extend(self._normalize_entries(entries))

    def queue_disk_spill(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._disk_spill_queue.extend(self._normalize_entries(entries))

    def flush_promotions(self, device: str) -> float:
        entries = self._drain_queue(self._promotion_queue)
        return self.promote_to_gpu(entries, device)

    def flush_cpu_demotions(self) -> float:
        entries = self._drain_queue(self._cpu_demotion_queue)
        return self.demote_to_cpu(entries)

    def flush_disk_spills(self, disk_dir: str) -> float:
        entries = self._drain_queue(self._disk_spill_queue)
        return self.spill_to_disk(entries, disk_dir)

    def promote_to_gpu(self, entries: list[KVEntry], device: str) -> float:
        if not entries:
            return 0.0

        start = time.perf_counter()
        self._prefetch_disk_entries(entries)
        self._ensure_cpu_resident(entries)

        if not torch.cuda.is_available():
            for entry in entries:
                if entry.past_key_values is None:
                    continue
                entry.past_key_values = tuple(
                    (k.to(device), v.to(device)) for k, v in entry.past_key_values
                )
                entry.tier = "gpu"
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._distribute_elapsed(entries, elapsed_ms)
            return elapsed_ms

        stream = torch.cuda.Stream(device=device)
        moved: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {id(entry): [] for entry in entries}
        with torch.cuda.stream(stream):
            for entry in entries:
                kv_tuple = entry.past_key_values or ()
                for batch in KVEntry._layer_batches(kv_tuple, self.layer_batch_size):
                    for key, value in batch:
                        key_src = key.pin_memory() if key.device.type == "cpu" and not key.is_pinned() else key
                        value_src = value.pin_memory() if value.device.type == "cpu" and not value.is_pinned() else value
                        moved[id(entry)].append((
                            key_src.to(device, non_blocking=True),
                            value_src.to(device, non_blocking=True),
                        ))
        stream.synchronize()

        for entry in entries:
            entry.past_key_values = tuple(moved[id(entry)])
            entry.tier = "gpu"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._distribute_elapsed(entries, elapsed_ms)
        return elapsed_ms

    def demote_to_cpu(self, entries: list[KVEntry]) -> float:
        if not entries:
            return 0.0

        start = time.perf_counter()
        self._prefetch_disk_entries(entries)
        self._ensure_cpu_resident(entries, include_gpu=False)

        gpu_entries = [
            entry for entry in entries
            if entry.past_key_values is not None and any(k.is_cuda or v.is_cuda for k, v in entry.past_key_values)
        ]
        if gpu_entries and torch.cuda.is_available():
            stream_device = next(
                (tensor.device for entry in gpu_entries for key, value in entry.past_key_values or () for tensor in (key, value) if tensor.is_cuda),
                torch.device("cuda:0"),
            )
            stream = torch.cuda.Stream(device=stream_device)
            moved: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {id(entry): [] for entry in gpu_entries}
            with torch.cuda.stream(stream):
                for entry in gpu_entries:
                    kv_tuple = entry.past_key_values or ()
                    for batch in KVEntry._layer_batches(kv_tuple, self.layer_batch_size):
                        for key, value in batch:
                            cpu_key = torch.empty_like(key, device="cpu", pin_memory=True)
                            cpu_value = torch.empty_like(value, device="cpu", pin_memory=True)
                            cpu_key.copy_(key, non_blocking=True)
                            cpu_value.copy_(value, non_blocking=True)
                            moved[id(entry)].append((cpu_key, cpu_value))
            stream.synchronize()
            for entry in gpu_entries:
                entry.past_key_values = tuple(moved[id(entry)])

        for entry in entries:
            entry.tier = "cpu"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._distribute_elapsed(entries, elapsed_ms)
        return elapsed_ms

    def spill_to_disk(self, entries: list[KVEntry], disk_dir: str) -> float:
        if not entries:
            return 0.0

        start = time.perf_counter()
        self.demote_to_cpu(entries)
        os.makedirs(disk_dir, exist_ok=True)
        for entry in entries:
            entry.disk_path = os.path.join(disk_dir, f"{entry.prefix_hash}.pt")
            if entry.past_key_values is not None:
                torch.save(entry.past_key_values, entry.disk_path)
                entry.past_key_values = None
            entry.tier = "disk"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._distribute_elapsed(entries, elapsed_ms)
        return elapsed_ms

    @staticmethod
    def _normalize_entries(entries: Iterable[KVEntry] | KVEntry) -> list[KVEntry]:
        if isinstance(entries, KVEntry):
            return [entries]
        return [entry for entry in entries if entry is not None]

    @staticmethod
    def _drain_queue(queue: list[KVEntry]) -> list[KVEntry]:
        entries = list(queue)
        queue.clear()
        return entries

    @staticmethod
    def _prefetch_disk_entries(entries: list[KVEntry]) -> None:
        for entry in entries:
            if entry.tier == "disk":
                entry.start_disk_prefetch()

    @staticmethod
    def _ensure_cpu_resident(entries: list[KVEntry], include_gpu: bool = True) -> None:
        for entry in entries:
            if entry.past_key_values is None and entry.tier == "disk":
                entry._load_from_disk()
            if include_gpu:
                continue
            if entry.past_key_values is None:
                continue

    @staticmethod
    def _distribute_elapsed(entries: list[KVEntry], elapsed_ms: float) -> None:
        total_bytes = sum(max(entry.size_bytes, 1) for entry in entries)
        if total_bytes <= 0:
            for entry in entries:
                entry.last_transfer_ms = elapsed_ms
            return

        for entry in entries:
            weight = max(entry.size_bytes, 1) / total_bytes
            entry.last_transfer_ms = elapsed_ms * weight
