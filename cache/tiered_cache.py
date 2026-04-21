"""
TieredCache: Manages real KV tensors across GPU, CPU RAM, and Disk.

Tensors are physically transferred between tiers using PyTorch .to() and torch.save/load.
Tracks real transfer latencies and memory usage.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import torch

from config import TierConfig, StorageTier, WorkerConfig
from cache.kv_entry import KVEntry
from cache.eviction import EvictionPolicy

logger = logging.getLogger(__name__)


@dataclass
class TierState:
    """Runtime state of a single storage tier."""
    config: TierConfig
    entries: dict = field(default_factory=dict)  # prefix_hash -> KVEntry
    used_bytes: int = 0

    @property
    def free_bytes(self):
        return self.config.capacity_bytes - self.used_bytes

    @property
    def utilization(self):
        if self.config.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.config.capacity_bytes

    def has_space(self, size_bytes):
        return self.free_bytes >= size_bytes

    def add(self, entry):
        if entry.prefix_hash in self.entries:
            raise ValueError(f"Entry {entry.prefix_hash} already in {self.config.tier.value}")
        self.entries[entry.prefix_hash] = entry
        self.used_bytes += entry.size_bytes

    def remove(self, prefix_hash):
        entry = self.entries.pop(prefix_hash, None)
        if entry:
            self.used_bytes -= entry.size_bytes
        return entry

    def get(self, prefix_hash):
        return self.entries.get(prefix_hash)


@dataclass
class AccessEvent:
    """Record of a cache access with real timing."""
    timestamp: float
    prefix_hash: str
    hit: bool
    tier: Optional[str] = None
    transfer_ms: float = 0.0        # Real measured transfer time
    eviction_triggered: bool = False


class TieredCache:
    """
    Multi-tier KV cache with real tensor movement.

    GPU (hot) -> CPU (warm) -> Disk (cold)

    On hit in lower tier: tensors are transferred to GPU for inference.
    On eviction: tensors are demoted down the tier hierarchy.
    All transfers are real and timed.
    """

    TIER_ORDER = [StorageTier.GPU, StorageTier.CPU, StorageTier.DISK]

    def __init__(self, worker_config: WorkerConfig, eviction_policy: EvictionPolicy,
                 disk_dir: str = "/tmp/kv_cache", device: str = "cuda"):
        if eviction_policy is None:
            raise ValueError("eviction_policy cannot be None")

        self.worker_id = worker_config.worker_id
        self.eviction_policy = eviction_policy
        self.disk_dir = disk_dir
        self.device = device

        self.tiers = {
            StorageTier.GPU: TierState(config=worker_config.gpu_tier),
            StorageTier.CPU: TierState(config=worker_config.cpu_tier),
            StorageTier.DISK: TierState(config=worker_config.disk_tier),
        }

        # Metrics
        self.access_log: list[AccessEvent] = []
        self.total_hits = 0
        self.total_misses = 0
        self.tier_hits = {"gpu": 0, "cpu": 0, "disk": 0}
        self.total_evictions = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_transfer_ms = 0.0    # Real measured transfer time

    def get(self, prefix_hash: str) -> Optional[KVEntry]:
        """
        Look up a KV entry. On hit in lower tier, promote to GPU.
        Returns the entry with tensors on GPU, or None.
        """
        start = time.perf_counter()

        for tier_enum in self.TIER_ORDER:
            tier_state = self.tiers[tier_enum]
            entry = tier_state.get(prefix_hash)

            if entry is not None:
                entry.last_hit_tier = tier_enum.value
                entry.record_access()
                self.eviction_policy.on_access(entry)
                self.total_hits += 1
                self.tier_hits[tier_enum.value] += 1

                transfer_ms = 0.0

                # Promote to GPU if in lower tier
                if tier_enum != StorageTier.GPU:
                    transfer_ms = self._promote_to_gpu(entry, from_tier=tier_enum)
                    self.total_transfer_ms += transfer_ms

                self.access_log.append(AccessEvent(
                    timestamp=start,
                    prefix_hash=prefix_hash,
                    hit=True,
                    tier=tier_enum.value,
                    transfer_ms=transfer_ms,
                ))
                return entry

        # Miss
        self.total_misses += 1
        self.access_log.append(AccessEvent(
            timestamp=start, prefix_hash=prefix_hash, hit=False,
        ))
        return None

    def put_cpu(self, entry: KVEntry) -> bool:
        """
        Insert an entry into the CPU tier. Entry tensors must be on CPU.

        This is used for central-store fetches (shared KV) where the payload
        arrives in CPU memory and should only be promoted to GPU on demand.
        """
        # De-dupe: if already exists anywhere, treat as success and update policy.
        for tier_state in self.tiers.values():
            existing = tier_state.get(entry.prefix_hash)
            if existing is not None:
                existing.record_access()
                self.eviction_policy.on_access(existing)
                return True

        cpu_tier = self.tiers[StorageTier.CPU]

        while not cpu_tier.has_space(entry.size_bytes):
            evicted = self._evict_from_tier(StorageTier.CPU)
            if not evicted:
                logger.warning(
                    f"Cannot fit entry {entry.prefix_hash} ({entry.size_bytes} bytes) in CPU"
                )
                return False

        entry.tier = "cpu"
        entry.worker_id = self.worker_id
        cpu_tier.add(entry)
        self.eviction_policy.on_insert(entry)
        return True

    def put(self, entry: KVEntry) -> bool:
        """
        Insert a new KV entry into GPU tier. Evicts if needed.
        Entry tensors should already be on GPU when calling this.
        """
        # Check if exists
        for tier_state in self.tiers.values():
            existing = tier_state.get(entry.prefix_hash)
            if existing is not None:
                existing.record_access()
                self.eviction_policy.on_access(existing)
                return True

        gpu_tier = self.tiers[StorageTier.GPU]

        # Make room in GPU
        while not gpu_tier.has_space(entry.size_bytes):
            evicted = self._evict_from_tier(StorageTier.GPU)
            if not evicted:
                logger.warning(f"Cannot fit entry {entry.prefix_hash} ({entry.size_bytes} bytes) in GPU")
                return False

        entry.tier = "gpu"
        entry.worker_id = self.worker_id
        gpu_tier.add(entry)
        self.eviction_policy.on_insert(entry)
        return True

    def _evict_from_tier(self, tier: StorageTier) -> bool:
        """
        Evict one entry from tier. Demote to next tier with REAL transfer.
        """
        tier_state = self.tiers[tier]
        entries = list(tier_state.entries.values())

        if not entries:
            return False

        victim = self.eviction_policy.select_victim(entries)
        if victim is None:
            return False

        # Remove from current tier
        tier_state.remove(victim.prefix_hash)
        self.total_evictions += 1
        self.eviction_policy.on_evict(victim)

        # Demote to next tier
        tier_idx = self.TIER_ORDER.index(tier)
        if tier_idx < len(self.TIER_ORDER) - 1:
            next_tier = self.TIER_ORDER[tier_idx + 1]
            next_state = self.tiers[next_tier]

            # Make room in next tier if needed
            while not next_state.has_space(victim.size_bytes):
                if not self._evict_from_tier(next_tier):
                    break

            if next_state.has_space(victim.size_bytes):
                # REAL transfer
                if next_tier == StorageTier.CPU:
                    transfer_ms = victim.move_to_cpu()
                elif next_tier == StorageTier.DISK:
                    transfer_ms = victim.move_to_disk(self.disk_dir)
                else:
                    transfer_ms = 0.0

                next_state.add(victim)
                self.total_demotions += 1
                self.total_transfer_ms += transfer_ms
                logger.debug(
                    f"Demoted {victim.prefix_hash[:8]} "
                    f"{tier.value} -> {next_tier.value} "
                    f"({transfer_ms:.2f}ms)"
                )
                return True

        # Fully evicted — free memory
        victim.free_memory()
        logger.debug(f"Fully evicted {victim.prefix_hash[:8]} from {tier.value}")
        return True

    def _promote_to_gpu(self, entry: KVEntry, from_tier: StorageTier) -> float:
        """
        Promote entry to GPU with REAL tensor transfer.
        Returns transfer time in ms.
        """
        # Remove from current tier
        self.tiers[from_tier].remove(entry.prefix_hash)

        gpu_tier = self.tiers[StorageTier.GPU]

        # Make room
        while not gpu_tier.has_space(entry.size_bytes):
            if not self._evict_from_tier(StorageTier.GPU):
                # Can't promote — put back
                self.tiers[from_tier].add(entry)
                return 0.0

        # REAL transfer to GPU
        transfer_ms = entry.move_to_gpu(self.device)
        gpu_tier.add(entry)
        self.total_promotions += 1
        return transfer_ms

    def contains(self, prefix_hash: str) -> bool:
        return any(ts.get(prefix_hash) is not None for ts in self.tiers.values())

    @property
    def total_entries(self):
        return sum(len(ts.entries) for ts in self.tiers.values())

    @property
    def hit_rate(self):
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "policy": self.eviction_policy.name,
            "total_entries": self.total_entries,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.hit_rate,
            "tier_hits": dict(self.tier_hits),
            "total_evictions": self.total_evictions,
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_transfer_ms": self.total_transfer_ms,
            "tier_utilization": {
                t.value: self.tiers[t].utilization for t in self.TIER_ORDER
            },
            "tier_entries": {
                t.value: len(self.tiers[t].entries) for t in self.TIER_ORDER
            },
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        }

    def reset(self) -> None:
        """Free all entries and reset."""
        for tier_state in self.tiers.values():
            for entry in list(tier_state.entries.values()):
                entry.free_memory()
            tier_state.entries.clear()
            tier_state.used_bytes = 0
        self.access_log.clear()
        self.total_hits = 0
        self.total_misses = 0
        self.tier_hits = {"gpu": 0, "cpu": 0, "disk": 0}
        self.total_evictions = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_transfer_ms = 0.0
        self.eviction_policy.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
