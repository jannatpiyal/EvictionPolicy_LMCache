"""
Benchmark Harness: Runs real inference experiments across eviction policies.

Supports both single-worker and multi-worker (multi-GPU) modes.
In multi-worker mode, uses CacheController for request routing.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from config import FrameworkConfig, ControllerConfig, EvictionPolicyType
from cache.eviction import create_policy
from controller.cache_controller import CacheController
from worker.inference_worker import InferenceWorker
from workload.loader import WorkloadGenerator, InferenceRequest

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    policy: str
    trial_id: int
    num_requests: int
    num_workers: int
    hit_rate: float
    gpu_hits: int
    cpu_hits: int
    disk_hits: int
    total_misses: int
    avg_prefill_ms: float
    avg_decode_ms: float
    avg_total_ms: float
    p50_total_ms: float
    p95_total_ms: float
    p99_total_ms: float
    avg_prefill_on_hit_ms: float
    avg_prefill_on_miss_ms: float
    prefill_speedup: float
    total_evictions: int
    total_promotions: int
    total_demotions: int
    total_transfer_ms: float
    total_cache_saved_ms: float
    routing_efficiency: float
    cache_routed: int
    rr_routed: int
    worker_stats: dict = field(default_factory=dict)
    request_details: list = field(default_factory=list)


@dataclass
class BenchmarkResults:
    trials: list = field(default_factory=list)
    workload_stats: dict = field(default_factory=dict)
    config_summary: dict = field(default_factory=dict)

    def get_comparison_table(self):
        policies = sorted(set(t.policy for t in self.trials))
        table = []
        for policy in policies:
            pts = [t for t in self.trials if t.policy == policy]
            table.append({
                "policy": policy,
                "num_workers": pts[0].num_workers,
                "hit_rate": np.mean([t.hit_rate for t in pts]),
                "avg_prefill_ms": np.mean([t.avg_prefill_ms for t in pts]),
                "avg_total_ms": np.mean([t.avg_total_ms for t in pts]),
                "avg_p95_ms": np.mean([t.p95_total_ms for t in pts]),
                "avg_prefill_hit_ms": np.mean([t.avg_prefill_on_hit_ms for t in pts]),
                "avg_prefill_miss_ms": np.mean([t.avg_prefill_on_miss_ms for t in pts]),
                "speedup": np.mean([t.prefill_speedup for t in pts]),
                "evictions": np.mean([t.total_evictions for t in pts]),
                "transfer_ms": np.mean([t.total_transfer_ms for t in pts]),
                "saved_ms": np.mean([t.total_cache_saved_ms for t in pts]),
                "routing_efficiency": np.mean([t.routing_efficiency for t in pts]),
            })
        return table


class BenchmarkHarness:
    def __init__(self, config: FrameworkConfig):
        self.config = config

    def run(self) -> BenchmarkResults:
        results = BenchmarkResults()
        num_workers = len(self.config.workers)

        results.config_summary = {
            "model": self.config.model.model_path,
            "num_workers": num_workers,
            "num_gpus_available": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_cache_mb": self.config.workers[0].gpu_tier.capacity_mb,
            "cpu_cache_mb": self.config.workers[0].cpu_tier.capacity_mb,
            "disk_cache_mb": self.config.workers[0].disk_tier.capacity_mb,
            "num_requests": self.config.workload.num_requests,
            "num_prefixes": self.config.workload.num_unique_prefixes,
            "reuse_ratio": self.config.workload.prefix_reuse_ratio,
        }

        workload_gen = WorkloadGenerator(self.config.workload)
        workload = workload_gen.generate()
        results.workload_stats = workload_gen.get_reuse_stats(workload)
        logger.info(f"Workload: {results.workload_stats}")

        for policy_type in self.config.benchmark.policies_to_evaluate:
            logger.info(f"\n{'='*60}")
            logger.info(f"POLICY: {policy_type.value} ({num_workers} worker(s))")
            logger.info(f"{'='*60}")

            for trial_id in range(self.config.benchmark.num_trials):
                logger.info(f"  Trial {trial_id + 1}/{self.config.benchmark.num_trials}")

                if num_workers > 1:
                    trial = self._run_multi_worker_trial(policy_type, workload, trial_id)
                else:
                    trial = self._run_single_worker_trial(policy_type, workload, trial_id)

                results.trials.append(trial)
                logger.info(
                    f"    Hit rate: {trial.hit_rate:.1%}, "
                    f"Speedup: {trial.prefill_speedup:.1f}x, "
                    f"Evictions: {trial.total_evictions}, "
                    f"Routing eff: {trial.routing_efficiency:.1%}"
                )

        return results

    def _run_multi_worker_trial(self, policy_type, workload, trial_id):
        controller_config = ControllerConfig(
            num_workers=len(self.config.workers),
            eviction_policy=policy_type,
        )
        trial_config = FrameworkConfig(
            model=self.config.model,
            workers=self.config.workers,
            controller=controller_config,
            workload=self.config.workload,
            benchmark=self.config.benchmark,
        )

        controller = CacheController(trial_config)
        controller.create_workers()

        warmup_n = self.config.benchmark.warmup_requests
        request_details = []

        for i, request in enumerate(workload):
            result = controller.process_request(request)
            is_warmup = i < warmup_n
            detail = {
                "request_id": request.request_id,
                "prefix_group": request.prefix_group,
                "worker_id": result["worker_id"],
                "cache_hit": result["cache_hit"],
                "tier_hit": result.get("tier_hit"),
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "total_ms": result["total_ms"],
                "is_warmup": is_warmup,
            }
            request_details.append(detail)

            status = "HIT " if result["cache_hit"] else "MISS"
            tier = f"({result.get('tier_hit', '')})" if result["cache_hit"] else ""
            logger.info(
                f"    [{status}]{tier} W{result['worker_id']} Req {i:3d}: "
                f"prefill={result['prefill_ms']:6.1f}ms "
                f"decode={result['decode_ms']:6.1f}ms "
                f"| {request.user_query[:50]}"
            )

        global_stats = controller.get_global_stats()
        trial_result = self._compute_metrics(
            policy_type, trial_id, request_details, warmup_n,
            global_stats, len(self.config.workers),
        )
        controller.reset()
        torch.cuda.empty_cache()
        return trial_result

    def _run_single_worker_trial(self, policy_type, workload, trial_id):
        policy = create_policy(policy_type)
        worker = InferenceWorker(
            worker_config=self.config.workers[0],
            model_config=self.config.model,
            eviction_policy=policy,
            disk_dir=f"/tmp/kv_cache_trial_{trial_id}",
        )

        warmup_n = self.config.benchmark.warmup_requests
        request_details = []

        for i, request in enumerate(workload):
            result = worker.process_request(
                system_prompt=request.system_prompt,
                user_query=request.user_query,
                max_new_tokens=request.max_new_tokens,
            )
            is_warmup = i < warmup_n
            detail = {
                "request_id": request.request_id,
                "prefix_group": request.prefix_group,
                "worker_id": 0,
                "cache_hit": result["cache_hit"],
                "tier_hit": result.get("tier_hit"),
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "total_ms": result["total_ms"],
                "is_warmup": is_warmup,
            }
            request_details.append(detail)

            status = "HIT " if result["cache_hit"] else "MISS"
            tier = f"({result.get('tier_hit', '')})" if result["cache_hit"] else ""
            logger.info(
                f"    [{status}]{tier} Req {i:3d}: "
                f"prefill={result['prefill_ms']:6.1f}ms "
                f"decode={result['decode_ms']:6.1f}ms "
                f"| {request.user_query[:50]}"
            )

        ws = worker.get_stats()
        global_stats = {
            "policy": policy_type.value, "num_workers": 1,
            "total_requests": len(workload),
            "cache_routed": 0, "rr_routed": len(workload),
            "routing_efficiency": 0.0,
            "global_hit_rate": ws["hit_rate"],
            "total_hits": ws["total_hits"], "total_misses": ws["total_misses"],
            "total_evictions": ws["total_evictions"],
            "total_transfer_ms": ws["total_transfer_ms"],
            "total_saved_ms": ws.get("total_cache_saved_ms", 0),
            "worker_stats": {0: ws},
        }

        trial_result = self._compute_metrics(
            policy_type, trial_id, request_details, warmup_n, global_stats, 1,
        )
        worker.reset()
        torch.cuda.empty_cache()
        return trial_result

    def _compute_metrics(self, policy_type, trial_id, request_details,
                         warmup_n, global_stats, num_workers):
        measured = [d for d in request_details if not d["is_warmup"]]
        hits = [d for d in measured if d["cache_hit"]]
        misses = [d for d in measured if not d["cache_hit"]]

        total_ms = [d["total_ms"] for d in measured] or [0]
        hit_pf = [d["prefill_ms"] for d in hits] or [0]
        miss_pf = [d["prefill_ms"] for d in misses] or [0]

        avg_hit = float(np.mean(hit_pf))
        avg_miss = float(np.mean(miss_pf))
        speedup = avg_miss / avg_hit if avg_hit > 0 else 0

        total_promos = sum(ws.get("total_promotions", 0) for ws in global_stats.get("worker_stats", {}).values())
        total_demos = sum(ws.get("total_demotions", 0) for ws in global_stats.get("worker_stats", {}).values())

        return TrialResult(
            policy=policy_type.value, trial_id=trial_id,
            num_requests=len(measured), num_workers=num_workers,
            hit_rate=len(hits) / len(measured) if measured else 0,
            gpu_hits=sum(1 for d in hits if d.get("tier_hit") == "gpu"),
            cpu_hits=sum(1 for d in hits if d.get("tier_hit") == "cpu"),
            disk_hits=sum(1 for d in hits if d.get("tier_hit") == "disk"),
            total_misses=len(misses),
            avg_prefill_ms=float(np.mean([d["prefill_ms"] for d in measured])) if measured else 0,
            avg_decode_ms=float(np.mean([d["decode_ms"] for d in measured])) if measured else 0,
            avg_total_ms=float(np.mean(total_ms)),
            p50_total_ms=float(np.percentile(total_ms, 50)),
            p95_total_ms=float(np.percentile(total_ms, 95)),
            p99_total_ms=float(np.percentile(total_ms, 99)),
            avg_prefill_on_hit_ms=avg_hit, avg_prefill_on_miss_ms=avg_miss,
            prefill_speedup=speedup,
            total_evictions=global_stats.get("total_evictions", 0),
            total_promotions=total_promos, total_demotions=total_demos,
            total_transfer_ms=global_stats.get("total_transfer_ms", 0),
            total_cache_saved_ms=global_stats.get("total_saved_ms", 0),
            routing_efficiency=global_stats.get("routing_efficiency", 0),
            cache_routed=global_stats.get("cache_routed", 0),
            rr_routed=global_stats.get("rr_routed", 0),
            worker_stats=global_stats.get("worker_stats", {}),
            request_details=request_details,
        )

    @staticmethod
    def save_results(results, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        comparison = results.get_comparison_table()
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        summary = {
            "config": results.config_summary,
            "workload_stats": results.workload_stats,
            "comparison": comparison,
            "trials": [{
                "policy": t.policy, "trial_id": t.trial_id,
                "num_workers": t.num_workers, "hit_rate": t.hit_rate,
                "gpu_hits": t.gpu_hits, "cpu_hits": t.cpu_hits, "disk_hits": t.disk_hits,
                "avg_prefill_ms": t.avg_prefill_ms, "avg_total_ms": t.avg_total_ms,
                "p95_total_ms": t.p95_total_ms, "prefill_speedup": t.prefill_speedup,
                "total_evictions": t.total_evictions, "total_transfer_ms": t.total_transfer_ms,
                "routing_efficiency": t.routing_efficiency,
                "request_details": t.request_details,
            } for t in results.trials],
        }
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved to {output_dir}/")