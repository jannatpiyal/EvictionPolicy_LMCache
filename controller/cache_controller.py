"""
Cache Controller: Coordinates multiple GPU inference workers.

NOW supports:
- Legacy workload (system_prompt + user_query)
- LMCache-style workload (prompt with long document)

Routes based on prefix hash.
"""

import time
import logging

import torch

from config import FrameworkConfig, EvictionPolicyType
from cache.kv_entry import KVEntry
from cache.eviction import create_policy
from worker.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)


class CacheController:
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.eviction_policy_type = config.controller.eviction_policy

        self.prefix_index: dict[str, int] = {}

        self.total_requests = 0
        self.cache_routed = 0
        self.rr_routed = 0
        self._rr_counter = 0

        self.workers: dict[int, InferenceWorker] = {}

    # -----------------------------
    # WORKER CREATION
    # -----------------------------
    def create_workers(self):
        num_gpus = torch.cuda.device_count()
        num_workers = len(self.config.workers)

        if num_workers > num_gpus:
            logger.warning(f"Using only {num_gpus} GPUs")
            num_workers = num_gpus

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

    # -----------------------------
    # PREFIX EXTRACTION (KEY CHANGE)
    # -----------------------------
    def _extract_prefix_tokens(self, request):
        worker = next(iter(self.workers.values()))

        # LMCache-style
        if hasattr(request, "prompt") and request.prompt is not None:
            prompt = request.prompt

            if "Question:" in prompt:
                prefix_text = prompt.split("Question:")[0]
            else:
                # fallback split
                split_idx = max(len(prompt) - 200, 0)
                prefix_text = prompt[:split_idx]

            return worker.tokenize(prefix_text)

        # Legacy
        return worker.tokenize(request.system_prompt)

    # -----------------------------
    # ROUTING
    # -----------------------------
    def route_request(self, prefix_hash: str) -> int:
        self.total_requests += 1

        cached_worker_id = self.prefix_index.get(prefix_hash)
        if cached_worker_id is not None:
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

    # -----------------------------
    # MAIN PROCESS
    # -----------------------------
    def process_request(self, request) -> dict:
        # Extract prefix tokens (FIXED)
        prefix_tokens = self._extract_prefix_tokens(request)
        prefix_hash = KVEntry.compute_prefix_hash(prefix_tokens)

        # Route
        worker_id = self.route_request(prefix_hash)
        worker = self.workers[worker_id]

        # Execute
        if hasattr(request, "prompt") and request.prompt is not None:
            result = worker.process_request(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
            )
        else:
            result = worker.process_request(
                system_prompt=request.system_prompt,
                user_query=request.user_query,
                max_new_tokens=request.max_new_tokens,
            )

        result["worker_id"] = worker_id
        result["request_id"] = request.request_id

        # Update index
        if not result["cache_hit"]:
            self.prefix_index[prefix_hash] = worker_id

        return result

    # -----------------------------
    # STATS
    # -----------------------------
    def get_global_stats(self):
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

    def reset(self):
        for worker in self.workers.values():
            worker.reset()

        self.prefix_index.clear()
        self.total_requests = 0
        self.cache_routed = 0
        self.rr_routed = 0
        self._rr_counter = 0

    def set_policy(self, policy_type: EvictionPolicyType):
        self.eviction_policy_type = policy_type
        for worker in self.workers.values():
            worker.cache.eviction_policy = create_policy(policy_type)
        self.reset()