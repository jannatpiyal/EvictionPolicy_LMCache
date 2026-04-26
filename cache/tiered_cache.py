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
from cache.dynamic_offload import DynamicOffloadWindow
from cache.eviction import EvictionPolicy
from cache.transfer_scheduler import TransferScheduler

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
        key = entry.cache_key()
        if key in self.entries:
            raise ValueError(f"Entry {key} already in {self.config.tier.value}")
        self.entries[key] = entry
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


@dataclass
class PrefixManifest:
    prefix_hash: str
    prefix_tokens: list[int]
    prompt_text: str
    num_tokens: int
    chunk_keys: list[str]
    chunk_size_tokens: int = 0
    embedding: Optional[list] = None


class TieredCache:
    """
    Multi-tier KV cache with real tensor movement.

    GPU (hot) -> CPU (warm) -> Disk (cold)

    On hit in lower tier: tensors are transferred to GPU for inference.
    On eviction: tensors are demoted down the tier hierarchy.
    All transfers are real and timed.
    """

    TIER_ORDER = [StorageTier.GPU, StorageTier.CPU, StorageTier.DISK]

    def __init__(
        self,
        worker_config: WorkerConfig,
        eviction_policy: EvictionPolicy,
        disk_dir: str = "/tmp/kv_cache",
        device: str = "cuda",
        log_evictions: bool = False,
        enable_dynamic_offload: bool = False,
        dynamic_offload_window_factor: float = 1.0,
    ):
        if eviction_policy is None:
            raise ValueError("eviction_policy cannot be None")

        self.worker_id = worker_config.worker_id
        self.eviction_policy = eviction_policy
        self.disk_dir = disk_dir
        self.device = device
        self.log_evictions = log_evictions

        self.tiers = {
            StorageTier.GPU: TierState(config=worker_config.gpu_tier),
            StorageTier.CPU: TierState(config=worker_config.cpu_tier),
            StorageTier.DISK: TierState(config=worker_config.disk_tier),
        }
        self.transfer_scheduler = TransferScheduler()
        self.prefix_manifests: dict[str, PrefixManifest] = {}
        self.dynamic_offload = DynamicOffloadWindow(
            enabled=enable_dynamic_offload,
            window_factor=float(dynamic_offload_window_factor),
        )

        # Metrics
        self.access_log: list[AccessEvent] = []
        self.total_hits = 0
        self.total_misses = 0
        self.tier_hits = {"gpu": 0, "cpu": 0, "disk": 0}
        self.total_evictions = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_transfer_ms = 0.0    # Real measured transfer time

    def _log_eviction_details(self, victims: list[KVEntry], from_tier: StorageTier, to_tier: Optional[StorageTier]) -> None:
        if not self.log_evictions or not victims:
            return
        destination = to_tier.value if to_tier is not None else "dropped"
        for victim in victims:
            logger.info(
                "Evict worker=%s prefix=%s from=%s to=%s size_mb=%.1f accesses=%s last_hit=%s",
                self.worker_id,
                victim.root_prefix_hash()[:8],
                from_tier.value,
                destination,
                victim.size_bytes / 1024 / 1024,
                victim.access_count,
                victim.last_hit_tier,
            )

    def _find_chunk(self, chunk_key: str) -> tuple[Optional[StorageTier], Optional[KVEntry]]:
        for tier_enum in self.TIER_ORDER:
            entry = self.tiers[tier_enum].get(chunk_key)
            if entry is not None:
                return tier_enum, entry
        return None, None

    def _register_manifest(self, entry: KVEntry, chunk_entries: list[KVEntry]) -> PrefixManifest:
        root_prefix_hash = entry.root_prefix_hash()
        manifest = PrefixManifest(
            prefix_hash=root_prefix_hash,
            prefix_tokens=entry.prefix_tokens,
            prompt_text=entry.prompt_text,
            num_tokens=entry.num_tokens,
            chunk_keys=[chunk.cache_key() for chunk in chunk_entries],
            chunk_size_tokens=entry.chunk_size_tokens,
            embedding=entry.embedding,
        )
        self.prefix_manifests[root_prefix_hash] = manifest
        return manifest

    def _drop_prefix(self, prefix_hash: str, free_memory: bool = True) -> None:
        manifest = self.prefix_manifests.pop(prefix_hash, None)
        if manifest is None:
            return
        for chunk_key in manifest.chunk_keys:
            self.dynamic_offload.unregister_gpu_chunk(chunk_key)
            for tier_state in self.tiers.values():
                entry = tier_state.remove(chunk_key)
                if entry is not None and free_memory:
                    entry.free_memory()

    def _build_aggregate_entry(self, manifest: PrefixManifest, chunk_entries: list[KVEntry], tier: str) -> KVEntry:
        ordered = sorted(chunk_entries, key=lambda entry: entry.chunk_index)
        kv_tuple = KVEntry.merge_kv_chunks([chunk.past_key_values for chunk in ordered if chunk.past_key_values is not None])
        aggregate = KVEntry(
            prefix_hash=manifest.prefix_hash,
            prefix_tokens=manifest.prefix_tokens,
            prompt_text=manifest.prompt_text,
            num_tokens=manifest.num_tokens,
            past_key_values=kv_tuple,
            size_bytes=sum(chunk.size_bytes for chunk in ordered),
            tier=tier,
            worker_id=self.worker_id,
            chunk_size_tokens=manifest.chunk_size_tokens,
            embedding=manifest.embedding,
        )
        aggregate.last_hit_tier = tier
        aggregate.access_count = max((chunk.access_count for chunk in ordered), default=0)
        aggregate.last_accessed_at = max((chunk.last_accessed_at for chunk in ordered), default=time.time())
        return aggregate

    def _promote_chunks_to_gpu(self, entries_by_tier: dict[StorageTier, list[KVEntry]]) -> float:
        transfer_ms = 0.0
        chunks_to_promote = [entry for entries in entries_by_tier.values() for entry in entries]
        if not chunks_to_promote:
            return 0.0

        total_bytes = sum(entry.size_bytes for entry in chunks_to_promote)
        for tier_enum, entries in entries_by_tier.items():
            if tier_enum == StorageTier.DISK:
                for entry in entries:
                    entry.start_disk_prefetch()
            for entry in entries:
                self.tiers[tier_enum].remove(entry.cache_key())

        if not self._ensure_space_in_tier(StorageTier.GPU, total_bytes):
            for tier_enum, entries in entries_by_tier.items():
                for entry in entries:
                    self.tiers[tier_enum].add(entry)
            return 0.0

        self.transfer_scheduler.queue_promotion(chunks_to_promote)
        transfer_ms += self.transfer_scheduler.flush_promotions(self.device)
        for entry in chunks_to_promote:
            self.tiers[StorageTier.GPU].add(entry)
            self.dynamic_offload.register_gpu_chunk(entry.cache_key())
        self.total_promotions += len(chunks_to_promote)
        return transfer_ms

    def _gpu_entries_by_key(self) -> dict[str, KVEntry]:
        return dict(self.tiers[StorageTier.GPU].entries)

    def _duplicate_gpu_window_to_cpu(self, required_bytes: int) -> float:
        if not self.dynamic_offload.enabled or required_bytes <= 0:
            return 0.0
        gpu_entries = self._gpu_entries_by_key()
        self.dynamic_offload.compact(set(gpu_entries.keys()))
        planned_keys = self.dynamic_offload.plan_window(required_bytes, gpu_entries)
        if not planned_keys:
            return 0.0

        self.dynamic_offload.note_stall()
        duplicate_ms = 0.0
        duplicated_bytes = 0
        duplicated_keys: list[str] = []
        cpu_tier = self.tiers[StorageTier.CPU]

        for key in planned_keys:
            gpu_entry = gpu_entries.get(key)
            if gpu_entry is None or cpu_tier.get(key) is not None:
                duplicated_keys.append(key)
                continue

            if not self._ensure_space_in_tier(StorageTier.CPU, gpu_entry.size_bytes):
                break

            start = time.perf_counter()
            cpu_duplicate = gpu_entry.clone_to_cpu_duplicate()
            cpu_duplicate.worker_id = self.worker_id
            cpu_duplicate.tier = "cpu"
            cpu_tier.add(cpu_duplicate)
            elapsed_ms = (time.perf_counter() - start) * 1000
            duplicate_ms += elapsed_ms
            duplicated_bytes += gpu_entry.size_bytes
            duplicated_keys.append(key)

        if duplicated_keys:
            self.dynamic_offload.mark_duplicated(duplicated_keys, duplicated_bytes, duplicate_ms)
        return duplicate_ms

    def _reclaim_preduplicated_gpu_chunks(self, required_bytes: int) -> int:
        if not self.dynamic_offload.enabled or required_bytes <= 0:
            return 0
        gpu_entries = self._gpu_entries_by_key()
        self.dynamic_offload.compact(set(gpu_entries.keys()))
        reclaim_keys = self.dynamic_offload.reclaimable_keys(required_bytes, gpu_entries)
        reclaimed_bytes = 0
        for key in reclaim_keys:
            gpu_entry = self.tiers[StorageTier.GPU].remove(key)
            if gpu_entry is None:
                continue
            self.total_evictions += 1
            self.eviction_policy.on_evict(gpu_entry)
            reclaimed_bytes += gpu_entry.size_bytes
            self.dynamic_offload.unregister_gpu_chunk(key)
        if reclaim_keys:
            self.dynamic_offload.mark_reclaimed(reclaim_keys, reclaimed_bytes)
        return reclaimed_bytes

    def get(self, prefix_hash: str) -> Optional[KVEntry]:
        """
        Look up a KV entry. On hit in lower tier, promote to GPU.
        Returns the entry with tensors on GPU, or None.
        """
        start = time.perf_counter()

        manifest = self.prefix_manifests.get(prefix_hash)
        if manifest is not None:
            chunk_locations: list[tuple[StorageTier, KVEntry]] = []
            highest_tier = StorageTier.GPU
            tier_rank = {StorageTier.GPU: 0, StorageTier.CPU: 1, StorageTier.DISK: 2}
            for chunk_key in manifest.chunk_keys:
                tier_enum, entry = self._find_chunk(chunk_key)
                if entry is None or tier_enum is None:
                    self._drop_prefix(prefix_hash, free_memory=True)
                    manifest = None
                    break
                chunk_locations.append((tier_enum, entry))
                if tier_rank[tier_enum] > tier_rank[highest_tier]:
                    highest_tier = tier_enum

            if manifest is not None:
                for tier_enum, entry in chunk_locations:
                    entry.last_hit_tier = highest_tier.value
                    entry.record_access()
                    self.eviction_policy.on_access(entry)
                self.total_hits += 1
                self.tier_hits[highest_tier.value] += 1

                transfer_ms = 0.0
                lower_tier_chunks: dict[StorageTier, list[KVEntry]] = {}
                for tier_enum, entry in chunk_locations:
                    if tier_enum != StorageTier.GPU:
                        lower_tier_chunks.setdefault(tier_enum, []).append(entry)
                if lower_tier_chunks:
                    transfer_ms = self._promote_chunks_to_gpu(lower_tier_chunks)
                    self.total_transfer_ms += transfer_ms

                gpu_chunks = []
                for chunk_key in manifest.chunk_keys:
                    entry = self.tiers[StorageTier.GPU].get(chunk_key)
                    if entry is None:
                        self._drop_prefix(prefix_hash, free_memory=True)
                        gpu_chunks = []
                        break
                    gpu_chunks.append(entry)
                if gpu_chunks:
                    aggregate = self._build_aggregate_entry(manifest, gpu_chunks, tier="gpu")
                    self.access_log.append(AccessEvent(
                        timestamp=start,
                        prefix_hash=prefix_hash,
                        hit=True,
                        tier=highest_tier.value,
                        transfer_ms=transfer_ms,
                    ))
                    return aggregate

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
        root_prefix_hash = entry.root_prefix_hash()
        if root_prefix_hash in self.prefix_manifests:
            if self.contains(root_prefix_hash):
                return True
            self._drop_prefix(root_prefix_hash, free_memory=True)

        chunk_entries = entry.split_for_cache()
        self._register_manifest(entry, chunk_entries)
        cpu_tier = self.tiers[StorageTier.CPU]

        for chunk_entry in chunk_entries:
            if not self._ensure_space_in_tier(StorageTier.CPU, chunk_entry.size_bytes):
                logger.warning(
                    f"Cannot fit entry {chunk_entry.root_prefix_hash()} chunk {chunk_entry.chunk_index} ({chunk_entry.size_bytes} bytes) in CPU"
                )
                self._drop_prefix(root_prefix_hash, free_memory=True)
                return False
            chunk_entry.tier = "cpu"
            chunk_entry.worker_id = self.worker_id
            cpu_tier.add(chunk_entry)
            self.eviction_policy.on_insert(chunk_entry)
        return True

    def put(self, entry: KVEntry) -> bool:
        """
        Insert a new KV entry into GPU tier. Evicts if needed.
        Entry tensors should already be on GPU when calling this.
        """
        root_prefix_hash = entry.root_prefix_hash()
        if root_prefix_hash in self.prefix_manifests:
            if self.contains(root_prefix_hash):
                return True
            self._drop_prefix(root_prefix_hash, free_memory=True)

        chunk_entries = entry.split_for_cache()
        self._register_manifest(entry, chunk_entries)
        gpu_tier = self.tiers[StorageTier.GPU]

        for chunk_entry in chunk_entries:
            if not self._ensure_space_in_tier(StorageTier.GPU, chunk_entry.size_bytes):
                logger.warning(f"Cannot fit entry {chunk_entry.root_prefix_hash()} chunk {chunk_entry.chunk_index} ({chunk_entry.size_bytes} bytes) in GPU")
                self._drop_prefix(root_prefix_hash, free_memory=True)
                return False
            chunk_entry.tier = "gpu"
            chunk_entry.worker_id = self.worker_id
            gpu_tier.add(chunk_entry)
            self.dynamic_offload.register_gpu_chunk(chunk_entry.cache_key())
            self.eviction_policy.on_insert(chunk_entry)
        return True

    def _collect_victims(self, tier: StorageTier, required_bytes: int) -> list[KVEntry]:
        tier_state = self.tiers[tier]
        candidates = list(tier_state.entries.values())
        victims: list[KVEntry] = []
        freed_bytes = 0

        while tier_state.free_bytes + freed_bytes < required_bytes:
            if not candidates:
                return []

            victim = self.eviction_policy.select_victim(candidates)
            if victim is None:
                return []

            candidates.remove(victim)
            tier_state.remove(victim.cache_key())
            self.total_evictions += 1
            self.eviction_policy.on_evict(victim)
            victims.append(victim)
            freed_bytes += victim.size_bytes

        return victims

    def _ensure_space_in_tier(self, tier: StorageTier, required_bytes: int) -> bool:
        ok, _, transfer_ms = self._ensure_space_in_tier_internal(tier, required_bytes, defer_flush=False)
        self.total_transfer_ms += transfer_ms
        return ok

    def _ensure_space_in_tier_internal(
        self,
        tier: StorageTier,
        required_bytes: int,
        defer_flush: bool,
    ) -> tuple[bool, list[tuple[StorageTier, StorageTier, list[KVEntry]]], float]:
        tier_state = self.tiers[tier]
        extra_transfer_ms = 0.0
        if tier_state.has_space(required_bytes):
            return True, [], 0.0

        if tier == StorageTier.GPU and self.dynamic_offload.enabled:
            extra_transfer_ms += self._duplicate_gpu_window_to_cpu(required_bytes)
            reclaimed_bytes = self._reclaim_preduplicated_gpu_chunks(required_bytes)
            if reclaimed_bytes > 0 and tier_state.has_space(required_bytes):
                return True, [], extra_transfer_ms

        victims = self._collect_victims(tier, required_bytes)
        if not victims:
            return False, [], extra_transfer_ms

        tier_idx = self.TIER_ORDER.index(tier)
        if tier_idx >= len(self.TIER_ORDER) - 1:
            self._log_eviction_details(victims, tier, None)
            for victim in victims:
                self._drop_prefix(victim.root_prefix_hash(), free_memory=True)
            return tier_state.has_space(required_bytes), [], extra_transfer_ms

        next_tier = self.TIER_ORDER[tier_idx + 1]
        total_bytes = sum(victim.size_bytes for victim in victims)

        child_ok, child_movements, child_transfer_ms = self._ensure_space_in_tier_internal(
            next_tier,
            total_bytes,
            defer_flush=True,
        )
        if not child_ok:
            self._log_eviction_details(victims, tier, None)
            for victim in victims:
                self._drop_prefix(victim.root_prefix_hash(), free_memory=True)
            return tier_state.has_space(required_bytes), [], child_transfer_ms + extra_transfer_ms

        self._queue_tier_movement(next_tier, victims)
        movements = child_movements + [(tier, next_tier, victims)]

        if defer_flush:
            return tier_state.has_space(required_bytes), movements, child_transfer_ms + extra_transfer_ms

        transfer_ms = child_transfer_ms + extra_transfer_ms + self.transfer_scheduler.flush_all(
            device=self.device,
            disk_dir=self.disk_dir,
        )
        self._commit_tier_movements(movements)
        return tier_state.has_space(required_bytes), movements, transfer_ms

    def _queue_tier_movement(self, next_tier: StorageTier, victims: list[KVEntry]) -> None:
        if next_tier == StorageTier.CPU:
            self.transfer_scheduler.queue_cpu_demotion(victims)
        elif next_tier == StorageTier.DISK:
            self.transfer_scheduler.queue_disk_spill(victims)

    def _commit_tier_movements(self, movements: list[tuple[StorageTier, StorageTier, list[KVEntry]]]) -> None:
        for source_tier, next_tier, victims in movements:
            next_state = self.tiers[next_tier]
            for victim in victims:
                next_state.add(victim)
            self.total_demotions += len(victims)
            self._log_eviction_details(victims, source_tier, next_tier)
            logger.debug(
                "Batch demoted %d entries %s -> %s",
                len(victims),
                source_tier.value,
                next_tier.value,
            )

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
        tier_state.remove(victim.cache_key())
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
        self._drop_prefix(victim.root_prefix_hash(), free_memory=True)
        logger.debug(f"Fully evicted {victim.prefix_hash[:8]} from {tier.value}")
        return True

    def _promote_to_gpu(self, entry: KVEntry, from_tier: StorageTier) -> float:
        """
        Promote entry to GPU with REAL tensor transfer.
        Returns transfer time in ms.
        """
        if from_tier == StorageTier.DISK:
            entry.start_disk_prefetch()

        # Remove from current tier
        self.tiers[from_tier].remove(entry.cache_key())
        if from_tier == StorageTier.GPU:
            self.dynamic_offload.unregister_gpu_chunk(entry.cache_key())

        gpu_tier = self.tiers[StorageTier.GPU]

        if not self._ensure_space_in_tier(StorageTier.GPU, entry.size_bytes):
            self.tiers[from_tier].add(entry)
            return 0.0

        transfer_ms = self.transfer_scheduler.flush_promotions(self.device)
        gpu_tier.add(entry)
        self.dynamic_offload.register_gpu_chunk(entry.cache_key())
        self.total_promotions += 1
        return transfer_ms

    def contains(self, prefix_hash: str) -> bool:
        manifest = self.prefix_manifests.get(prefix_hash)
        if manifest is None:
            return False
        return all(self._find_chunk(chunk_key)[1] is not None for chunk_key in manifest.chunk_keys)

    def list_prefixes(self) -> list[str]:
        return [prefix for prefix in self.prefix_manifests if self.contains(prefix)]

    @property
    def total_entries(self):
        return len(self.list_prefixes())

    @property
    def total_chunks(self):
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
            "total_chunks": self.total_chunks,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.hit_rate,
            "tier_hits": dict(self.tier_hits),
            "total_evictions": self.total_evictions,
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_transfer_ms": self.total_transfer_ms,
            "dynamic_offload": self.dynamic_offload.stats(),
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
        self.prefix_manifests.clear()
        self.access_log.clear()
        self.total_hits = 0
        self.total_misses = 0
        self.tier_hits = {"gpu": 0, "cpu": 0, "disk": 0}
        self.total_evictions = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_transfer_ms = 0.0
        self.dynamic_offload = DynamicOffloadWindow(
            enabled=self.dynamic_offload.enabled,
            window_factor=self.dynamic_offload.window_factor,
        )
        self.eviction_policy.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
