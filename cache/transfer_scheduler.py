"""
Unified transfer scheduler for batched KV movement across entries.

This version keeps the old queue/flush API for compatibility, but internally it
uses explicit transfer jobs and separate executors per link type so heterogeneous
transfers can be submitted and run in parallel.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, Optional

import torch

from cache.kv_entry import KVEntry
from cache.transfer_job import TransferJob, TransferKind


class TransferScheduler:
    def __init__(
        self,
        layer_batch_size: int = 4,
        cpu_gpu_workers: int = 1,
        gpu_cpu_workers: int = 1,
        cpu_disk_workers: int = 2,
    ):
        self.layer_batch_size = max(1, int(layer_batch_size))
        self._promotion_queue: list[KVEntry] = []
        self._cpu_demotion_queue: list[KVEntry] = []
        self._disk_spill_queue: list[KVEntry] = []

        self._executors = {
            TransferKind.CPU_TO_GPU: ThreadPoolExecutor(max_workers=max(1, int(cpu_gpu_workers))),
            TransferKind.GPU_TO_CPU: ThreadPoolExecutor(max_workers=max(1, int(gpu_cpu_workers))),
            TransferKind.CPU_TO_DISK: ThreadPoolExecutor(max_workers=max(1, int(cpu_disk_workers))),
        }
        self._active_jobs: dict[str, tuple[TransferJob, Future]] = {}
        self._completed_jobs: dict[str, float] = {}

    def queue_promotion(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._promotion_queue.extend(self._normalize_entries(entries))

    def queue_cpu_demotion(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._cpu_demotion_queue.extend(self._normalize_entries(entries))

    def queue_disk_spill(self, entries: Iterable[KVEntry] | KVEntry) -> None:
        self._disk_spill_queue.extend(self._normalize_entries(entries))

    def submit_job(self, job: TransferJob) -> str:
        executor = self._executors[job.kind]
        future = executor.submit(self._run_job, job)
        self._active_jobs[job.job_id] = (job, future)
        return job.job_id

    def submit_promotion(self, entries: Iterable[KVEntry] | KVEntry, device: str) -> str:
        job = TransferJob(
            kind=TransferKind.CPU_TO_GPU,
            entries=self._normalize_entries(entries),
            device=device,
        )
        return self.submit_job(job)

    def submit_cpu_demotion(self, entries: Iterable[KVEntry] | KVEntry) -> str:
        job = TransferJob(
            kind=TransferKind.GPU_TO_CPU,
            entries=self._normalize_entries(entries),
        )
        return self.submit_job(job)

    def submit_disk_spill(self, entries: Iterable[KVEntry] | KVEntry, disk_dir: str) -> str:
        job = TransferJob(
            kind=TransferKind.CPU_TO_DISK,
            entries=self._normalize_entries(entries),
            disk_dir=disk_dir,
        )
        return self.submit_job(job)

    def wait(self, job_ids: Iterable[str]) -> float:
        total_ms = 0.0
        for job_id in job_ids:
            completed = self._completed_jobs.get(job_id)
            if completed is not None:
                total_ms += completed
                continue
            job_tuple = self._active_jobs.get(job_id)
            if job_tuple is None:
                continue
            _, future = job_tuple
            elapsed_ms = float(future.result())
            total_ms += elapsed_ms
            self._completed_jobs[job_id] = elapsed_ms
            self._active_jobs.pop(job_id, None)
        return total_ms

    def wait_for_job(self, job_id: str) -> float:
        return self.wait([job_id])

    def submit_queued(self, device: Optional[str] = None, disk_dir: Optional[str] = None) -> list[str]:
        job_ids: list[str] = []
        promotions = self._drain_queue(self._promotion_queue)
        if promotions:
            if device is None:
                raise ValueError("device is required to submit queued promotions")
            job_ids.append(self.submit_promotion(promotions, device))
        demotions = self._drain_queue(self._cpu_demotion_queue)
        if demotions:
            job_ids.append(self.submit_cpu_demotion(demotions))
        spills = self._drain_queue(self._disk_spill_queue)
        if spills:
            if disk_dir is None:
                raise ValueError("disk_dir is required to submit queued disk spills")
            job_ids.append(self.submit_disk_spill(spills, disk_dir))
        return job_ids

    def flush_all(self, device: Optional[str] = None, disk_dir: Optional[str] = None) -> float:
        return self.wait(self.submit_queued(device=device, disk_dir=disk_dir))

    def flush_promotions(self, device: str) -> float:
        entries = self._drain_queue(self._promotion_queue)
        if not entries:
            return 0.0
        return self.wait_for_job(self.submit_promotion(entries, device))

    def flush_cpu_demotions(self) -> float:
        entries = self._drain_queue(self._cpu_demotion_queue)
        if not entries:
            return 0.0
        return self.wait_for_job(self.submit_cpu_demotion(entries))

    def flush_disk_spills(self, disk_dir: str) -> float:
        entries = self._drain_queue(self._disk_spill_queue)
        if not entries:
            return 0.0
        return self.wait_for_job(self.submit_disk_spill(entries, disk_dir))

    def _run_job(self, job: TransferJob) -> float:
        if job.kind == TransferKind.CPU_TO_GPU:
            if job.device is None:
                raise ValueError("CPU_TO_GPU job requires device")
            return self.promote_to_gpu(job.entries, job.device)
        if job.kind == TransferKind.GPU_TO_CPU:
            return self.demote_to_cpu(job.entries)
        if job.kind == TransferKind.CPU_TO_DISK:
            if job.disk_dir is None:
                raise ValueError("CPU_TO_DISK job requires disk_dir")
            return self.spill_to_disk(job.entries, job.disk_dir)
        raise ValueError(f"Unsupported transfer kind: {job.kind}")

    def promote_to_gpu(self, entries: list[KVEntry], device: str) -> float:
        if not entries:
            return 0.0

        pipelined_entries = [
            entry for entry in entries
            if entry.uses_chunk_storage() or entry.enable_layerwise_pipeline
        ]
        regular_entries = [
            entry for entry in entries
            if not entry.uses_chunk_storage() and not entry.enable_layerwise_pipeline
        ]
        chunked_elapsed = 0.0
        if pipelined_entries:
            chunked_start = time.perf_counter()
            for entry in pipelined_entries:
                entry.move_to_gpu(device)
            chunked_elapsed = (time.perf_counter() - chunked_start) * 1000

        if not regular_entries:
            self._distribute_elapsed(pipelined_entries, chunked_elapsed)
            return chunked_elapsed

        start = time.perf_counter()
        self._prefetch_disk_entries(regular_entries)
        self._ensure_cpu_resident(regular_entries)

        if not torch.cuda.is_available():
            for entry in regular_entries:
                if entry.past_key_values is None:
                    continue
                entry.past_key_values = tuple(
                    (k.to(device), v.to(device)) for k, v in entry.past_key_values
                )
                entry.tier = "gpu"
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._distribute_elapsed(regular_entries, elapsed_ms)
            self._distribute_elapsed(pipelined_entries, chunked_elapsed)
            return elapsed_ms + chunked_elapsed

        stream = torch.cuda.Stream(device=device)
        moved: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {id(entry): [] for entry in regular_entries}
        with torch.cuda.stream(stream):
            for entry in regular_entries:
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

        for entry in regular_entries:
            entry.past_key_values = tuple(moved[id(entry)])
            entry.tier = "gpu"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._distribute_elapsed(regular_entries, elapsed_ms)
        self._distribute_elapsed(pipelined_entries, chunked_elapsed)
        return elapsed_ms + chunked_elapsed

    def demote_to_cpu(self, entries: list[KVEntry]) -> float:
        if not entries:
            return 0.0

        pipelined_entries = [
            entry for entry in entries
            if entry.uses_chunk_storage() or entry.enable_layerwise_pipeline
        ]
        regular_entries = [
            entry for entry in entries
            if not entry.uses_chunk_storage() and not entry.enable_layerwise_pipeline
        ]
        pipelined_elapsed = 0.0
        if not regular_entries:
            for entry in pipelined_entries:
                entry.tier = "cpu"
            return 0.0

        if pipelined_entries:
            pipelined_start = time.perf_counter()
            for entry in pipelined_entries:
                entry.move_to_cpu()
            pipelined_elapsed = (time.perf_counter() - pipelined_start) * 1000
            self._distribute_elapsed(pipelined_entries, pipelined_elapsed)

        start = time.perf_counter()
        self._prefetch_disk_entries(regular_entries)
        self._ensure_cpu_resident(regular_entries, include_gpu=False)

        gpu_entries = [
            entry for entry in regular_entries
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

        for entry in regular_entries + pipelined_entries:
            entry.tier = "cpu"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._distribute_elapsed(regular_entries, elapsed_ms)
        return elapsed_ms + pipelined_elapsed

    def spill_to_disk(self, entries: list[KVEntry], disk_dir: str) -> float:
        if not entries:
            return 0.0

        start = time.perf_counter()
        self.demote_to_cpu(entries)
        os.makedirs(disk_dir, exist_ok=True)
        for entry in entries:
            entry.disk_path = os.path.join(disk_dir, f"{entry.prefix_hash}.pt")
            if entry.past_key_values is not None or entry.kv_chunks is not None:
                torch.save(
                    {
                        "past_key_values": entry.past_key_values,
                        "kv_chunks": entry.kv_chunks,
                        "chunk_size_tokens": entry.chunk_size_tokens,
                    },
                    entry.disk_path,
                )
                entry.past_key_values = None
                entry.kv_chunks = None
                entry._release_mapped_regions()
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

    def shutdown(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
