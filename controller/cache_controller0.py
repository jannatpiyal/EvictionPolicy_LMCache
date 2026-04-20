"""
Cache Controller: Coordinates multiple GPU inference workers.

Manages a global index of cached prefixes across all workers.
Routes requests to workers that already have the prefix cached,
falling back to round-robin for load balancing on cache misses.

Each worker runs its own Llama 8B instance on a separate GPU.
"""

import time
import logging
from typing import Optional

import torch

from config import FrameworkConfig, ControllerConfig, EvictionPolicyType
from cache.kv_entry import KVEntry
from cache.eviction import create_policy
from worker.inference_worker import InferenceWorker
from workload.loader import InferenceRequest

logger = logging.getLogger(__name__)


class CacheController:
    """
    Central controller coordinating KV caching across multiple GPU workers.

    Architecture:
        [Requests] → [Controller] → routes to → [Worker 0 on cuda:0]
                                                 [Worker 1 on cuda:1]
                                                 [Worker 2 on cuda:2]

    Routing strategy:
        1. Check global index for which worker has the prefix cached
        2. If found → route to that worker (cache-aware)
        3. If not found → round-robin across workers (load balance)

    Each worker independently manages its own tiered cache (GPU/CPU/Disk).
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.controller_config = config.controller
        self.eviction_policy_type = config.controller.eviction_policy

        # Global prefix index: prefix_hash -> worker_id
        self.prefix_index: dict[str, int] = {}

        # Routing stats
        self.total_requests = 0
        self.cache_routed = 0       # Routed to worker with cached prefix
        self.rr_routed = 0          # Round-robin routed (no cache match)
        self._rr_counter = 0

        # Workers (initialized by create_workers or set externally)
        self.workers: dict[int, InferenceWorker] = {}

    def create_workers(self) -> None:
        """
        Create inference workers, each on its own GPU.

        Worker 0 → cuda:0, Worker 1 → cuda:1, etc.
        Each loads its own copy of the model.
        """
        num_gpus = torch.cuda.device_count()
        num_workers = len(self.config.workers)

        if num_workers > num_gpus:
            logger.warning(
                f"Requested {num_workers} workers but only {num_gpus} GPUs available. "
                f"Using {num_gpus} workers."
            )
            num_workers = num_gpus

        logger.info(f"Creating {num_workers} workers across {num_gpus} GPUs")

        for i in range(num_workers):
            worker_config = self.config.workers[i]
            policy = create_policy(self.eviction_policy_type)

            worker = InferenceWorker(
                worker_config=worker_config,
                model_config=self.config.model,
                eviction_policy=policy,
                disk_dir=f"/tmp/kv_cache_worker_{i}",
            )
            self.workers[worker_config.worker_id] = worker

            logger.info(
                f"  Worker {i}: {worker.device}, "
                f"GPU mem: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
            )

    def route_request(self, prefix_hash: str) -> int:
        """
        Route a request to a worker.

        Strategy:
        1. Cache-aware: if a worker has this prefix cached, route there.
        2. Round-robin: distribute evenly across workers.
        """
        self.total_requests += 1

        # Check global index
        cached_worker_id = self.prefix_index.get(prefix_hash)
        if cached_worker_id is not None and cached_worker_id in self.workers:
            worker = self.workers[cached_worker_id]
            if worker.cache.contains(prefix_hash):
                self.cache_routed += 1
                return cached_worker_id

        # Round-robin fallback
        worker_ids = sorted(self.workers.keys())
        worker_id = worker_ids[self._rr_counter % len(worker_ids)]
        self._rr_counter += 1
        self.rr_routed += 1
        return worker_id

    def process_request(self, request: InferenceRequest) -> dict:
        """
        Process a single request: route to worker, execute inference.

        Returns:
            Dict with full result including worker assignment and cache info.
        """
        # Tokenize to get prefix hash (use any worker's tokenizer — they're identical)
        any_worker = next(iter(self.workers.values()))
        prefix_tokens = any_worker.tokenize(request.system_prompt)
        prefix_hash = KVEntry.compute_prefix_hash(prefix_tokens)

        # Route
        worker_id = self.route_request(prefix_hash)
        worker = self.workers[worker_id]

        # Execute real inference
        start = time.perf_counter()
        result = worker.process_request(
            system_prompt=request.system_prompt,
            user_query=request.user_query,
            max_new_tokens=request.max_new_tokens,
        )
        result["routing_ms"] = 0.0  # In-process routing is negligible
        result["worker_id"] = worker_id
        result["request_id"] = request.request_id

        # Update global index
        if not result["cache_hit"]:
            # New prefix cached on this worker
            self.prefix_index[prefix_hash] = worker_id

        return result

    def get_global_stats(self) -> dict:
        """Aggregate stats from all workers."""
        worker_stats = {}
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        total_transfer_ms = 0.0
        total_saved_ms = 0.0

        for wid, worker in self.workers.items():
            ws = worker.get_stats()
            worker_stats[wid] = ws
            total_hits += ws["total_hits"]
            total_misses += ws["total_misses"]
            total_evictions += ws["total_evictions"]
            total_transfer_ms += ws["total_transfer_ms"]
            total_saved_ms += ws.get("total_cache_saved_ms", 0)

        total_accesses = total_hits + total_misses

        return {
            "policy": self.eviction_policy_type.value,
            "num_workers": len(self.workers),
            "total_requests": self.total_requests,
            "cache_routed": self.cache_routed,
            "rr_routed": self.rr_routed,
            "routing_efficiency": self.cache_routed / self.total_requests if self.total_requests > 0 else 0,
            "global_hit_rate": total_hits / total_accesses if total_accesses > 0 else 0,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_evictions": total_evictions,
            "total_transfer_ms": total_transfer_ms,
            "total_saved_ms": total_saved_ms,
            "worker_stats": worker_stats,
        }

    def reset(self) -> None:
        """Reset all workers and routing state for a new trial."""
        for worker in self.workers.values():
            worker.reset()
        self.prefix_index.clear()
        self.total_requests = 0
        self.cache_routed = 0
        self.rr_routed = 0
        self._rr_counter = 0

    def set_policy(self, policy_type: EvictionPolicyType) -> None:
        """Switch eviction policy on all workers."""
        self.eviction_policy_type = policy_type
        for worker in self.workers.values():
            new_policy = create_policy(policy_type)
            worker.cache.eviction_policy = new_policy
        self.reset()