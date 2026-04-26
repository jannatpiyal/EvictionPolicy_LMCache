from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DynamicOffloadWindow:
    """
    Chunk-level approximation of LMCache dynamic offloading.

    We do not have direct access to a vLLM-style page allocator in this repo, so
    this manager models the same start/current/end-pointer idea over GPU-resident
    chunk entries. Chunks between `start_idx` and `current_idx` have already
    been duplicated to CPU, while chunks between `current_idx` and `end_idx` are
    the duplication window that query allocation may need to wait on.
    """

    enabled: bool = False
    window_factor: float = 1.0
    queue: list[str] = field(default_factory=list)
    duplicated_keys: set[str] = field(default_factory=set)
    start_idx: int = 0
    current_idx: int = 0
    end_idx: int = 0
    total_duplicate_ms: float = 0.0
    total_stalls: int = 0
    total_duplicated_bytes: int = 0
    total_reclaimed_bytes: int = 0

    def register_gpu_chunk(self, chunk_key: str) -> None:
        if not self.enabled:
            return
        if chunk_key in self.queue:
            self.queue = [key for key in self.queue if key != chunk_key]
            self.start_idx = min(self.start_idx, len(self.queue))
            self.current_idx = min(self.current_idx, len(self.queue))
            self.end_idx = min(self.end_idx, len(self.queue))
        self.queue.append(chunk_key)

    def unregister_gpu_chunk(self, chunk_key: str) -> None:
        if not self.enabled:
            return
        if chunk_key in self.duplicated_keys:
            self.duplicated_keys.discard(chunk_key)

    def compact(self, gpu_keys: set[str]) -> None:
        if not self.enabled:
            return
        self.queue = [key for key in self.queue if key in gpu_keys]
        self.duplicated_keys.intersection_update(gpu_keys)
        n = len(self.queue)
        self.start_idx = min(self.start_idx, n)
        self.current_idx = min(self.current_idx, n)
        self.end_idx = min(self.end_idx, n)
        while self.start_idx < len(self.queue) and self.queue[self.start_idx] not in self.duplicated_keys:
            self.start_idx += 1

    def plan_window(self, required_bytes: int, gpu_entries_by_key: dict[str, object]) -> list[str]:
        if not self.enabled or required_bytes <= 0:
            return []
        target_bytes = required_bytes * max(self.window_factor, 0.0)
        planned_bytes = 0
        idx = self.end_idx
        while idx < len(self.queue) and planned_bytes < target_bytes:
            key = self.queue[idx]
            entry = gpu_entries_by_key.get(key)
            if entry is not None and key not in self.duplicated_keys:
                planned_bytes += entry.size_bytes
            idx += 1
        self.end_idx = idx
        return [
            key
            for key in self.queue[self.current_idx:self.end_idx]
            if key not in self.duplicated_keys and key in gpu_entries_by_key
        ]

    def mark_duplicated(self, chunk_keys: list[str], duplicated_bytes: int, duplicate_ms: float) -> None:
        if not self.enabled:
            return
        for key in chunk_keys:
            self.duplicated_keys.add(key)
        self.current_idx = max(self.current_idx, self.end_idx)
        self.total_duplicate_ms += duplicate_ms
        self.total_duplicated_bytes += duplicated_bytes

    def reclaimable_keys(self, required_bytes: int, gpu_entries_by_key: dict[str, object]) -> list[str]:
        if not self.enabled or required_bytes <= 0:
            return []
        reclaimed: list[str] = []
        reclaimed_bytes = 0
        idx = self.start_idx
        while idx < self.current_idx and reclaimed_bytes < required_bytes:
            key = self.queue[idx]
            entry = gpu_entries_by_key.get(key)
            if entry is not None and key in self.duplicated_keys:
                reclaimed.append(key)
                reclaimed_bytes += entry.size_bytes
            idx += 1
        return reclaimed

    def mark_reclaimed(self, reclaimed_keys: list[str], reclaimed_bytes: int) -> None:
        if not self.enabled:
            return
        reclaimed_set = set(reclaimed_keys)
        self.duplicated_keys.difference_update(reclaimed_set)
        self.total_reclaimed_bytes += reclaimed_bytes
        while self.start_idx < len(self.queue):
            key = self.queue[self.start_idx]
            if key in reclaimed_set:
                self.start_idx += 1
                continue
            if key not in self.duplicated_keys:
                self.start_idx += 1
                continue
            break

    def note_stall(self) -> None:
        if self.enabled:
            self.total_stalls += 1

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "start_pointer": self.start_idx,
            "current_pointer": self.current_idx,
            "end_pointer": self.end_idx,
            "queued_gpu_chunks": len(self.queue),
            "duplicated_gpu_chunks": len(self.duplicated_keys),
            "total_duplicate_ms": self.total_duplicate_ms,
            "total_stalls": self.total_stalls,
            "total_duplicated_bytes": self.total_duplicated_bytes,
            "total_reclaimed_bytes": self.total_reclaimed_bytes,
        }
