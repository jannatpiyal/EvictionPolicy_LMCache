"""
Benchmark Harness: Runs real inference experiments across eviction policies.

Supports both single-worker and multi-worker modes.
Adds LMCache-style evaluation metrics:
- TTFT
- ITL
- Request throughput
- Output token throughput
- Trial wall time

Fixes:
- speedup is only computed when both hits and misses are present
- debug counts added for measured hits/misses
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from config import FrameworkConfig, ControllerConfig, EvictionPolicyType
from cache.eviction import create_policy
from controller.cache_controller import CacheController
from worker.inference_worker import InferenceWorker
from workload.loader import WorkloadGenerator

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
    prefill_speedup: float | None

    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    avg_itl_ms: float

    trial_wall_time_s: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    total_generated_tokens: int

    total_evictions: int
    total_promotions: int
    total_demotions: int
    total_transfer_ms: float
    total_cache_saved_ms: float

    routing_efficiency: float
    cache_routed: int
    rr_routed: int

    num_measured_hits: int = 0
    num_measured_misses: int = 0

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

            valid_speedups = [t.prefill_speedup for t in pts if t.prefill_speedup is not None]
            avg_speedup = float(np.mean(valid_speedups)) if valid_speedups else None

            table.append({
                "policy": policy,
                "num_workers": pts[0].num_workers,
                "hit_rate": np.mean([t.hit_rate for t in pts]),
                "avg_prefill_ms": np.mean([t.avg_prefill_ms for t in pts]),
                "avg_total_ms": np.mean([t.avg_total_ms for t in pts]),
                "avg_p95_ms": np.mean([t.p95_total_ms for t in pts]),
                "avg_prefill_hit_ms": np.mean([t.avg_prefill_on_hit_ms for t in pts]),
                "avg_prefill_miss_ms": np.mean([t.avg_prefill_on_miss_ms for t in pts]),
                "speedup": avg_speedup,
                "evictions": np.mean([t.total_evictions for t in pts]),
                "transfer_ms": np.mean([t.total_transfer_ms for t in pts]),
                "saved_ms": np.mean([t.total_cache_saved_ms for t in pts]),
                "routing_efficiency": np.mean([t.routing_efficiency for t in pts]),
                "avg_ttft_ms": np.mean([t.avg_ttft_ms for t in pts]),
                "p95_ttft_ms": np.mean([t.p95_ttft_ms for t in pts]),
                "avg_itl_ms": np.mean([t.avg_itl_ms for t in pts]),
                "request_throughput_rps": np.mean([t.request_throughput_rps for t in pts]),
                "output_token_throughput_tps": np.mean([t.output_token_throughput_tps for t in pts]),
                "trial_wall_time_s": np.mean([t.trial_wall_time_s for t in pts]),
                "num_measured_hits": np.mean([t.num_measured_hits for t in pts]),
                "num_measured_misses": np.mean([t.num_measured_misses for t in pts]),
            })
        return table


class BenchmarkHarness:
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self._shared_controller = None
        self._shared_single_worker = None

    def _build_config_summary(self, num_workers: int) -> dict:
        summary = {
            "model": self.config.model.model_path,
            "num_workers": num_workers,
            "num_gpus_available": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_cache_mb": self.config.workers[0].gpu_tier.capacity_mb,
            "cpu_cache_mb": self.config.workers[0].cpu_tier.capacity_mb,
            "disk_cache_mb": self.config.workers[0].disk_tier.capacity_mb,
        }

        workload_cfg = self.config.workload
        if hasattr(workload_cfg, "num_documents"):
            summary.update({
                "workload_type": "long_document_multi_round_qa",
                "num_documents": workload_cfg.num_documents,
                "document_length_tokens": getattr(workload_cfg, "document_length_tokens", None),
                "num_rounds": getattr(workload_cfg, "num_rounds", None),
                "hit_ratio": getattr(workload_cfg, "hit_ratio", None),
                "initial_concurrency": getattr(workload_cfg, "initial_concurrency", None),
                "arrival_mode": getattr(workload_cfg, "arrival_mode", None),
                "interarrival_mean_sec": getattr(workload_cfg, "interarrival_mean_sec", None),
                "max_new_tokens": getattr(workload_cfg, "max_new_tokens", None),
                "num_requests": getattr(workload_cfg, "num_requests", None),
                "total_requests": getattr(workload_cfg, "total_requests", None),
            })
        else:
            summary.update({
                "workload_type": "shared_prefix_queries",
                "num_requests": workload_cfg.num_requests,
                "num_prefixes": workload_cfg.num_unique_prefixes,
                "reuse_ratio": workload_cfg.prefix_reuse_ratio,
                "max_new_tokens": workload_cfg.max_new_tokens,
            })
        return summary

    def _request_preview(self, request) -> str:
        if hasattr(request, "user_query") and request.user_query is not None:
            return request.user_query[:50]
        if hasattr(request, "prompt") and request.prompt is not None:
            return request.prompt[:50].replace("\n", " ")
        return "<no preview>"

    def _request_meta(self, request) -> dict:
        return {
            "request_id": getattr(request, "request_id", None),
            "prefix_group": getattr(request, "prefix_group", None),
            "round_id": getattr(request, "round_id", None),
            "document_id": getattr(request, "document_id", None),
            "question_id": getattr(request, "question_id", None),
        }

    def _process_single_worker_request(self, worker, request):
        if hasattr(request, "prompt") and request.prompt is not None:
            try:
                return worker.process_request(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                )
            except TypeError:
                return worker.process_request(
                    system_prompt="",
                    user_query=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                )

        return worker.process_request(
            system_prompt=request.system_prompt,
            user_query=request.user_query,
            max_new_tokens=request.max_new_tokens,
        )

    def run(self) -> BenchmarkResults:
        results = BenchmarkResults()
        num_workers = len(self.config.workers)

        results.config_summary = self._build_config_summary(num_workers)

        workload_gen = WorkloadGenerator(self.config.workload)
        workload = workload_gen.generate()
        results.workload_stats = workload_gen.get_reuse_stats(workload)
        logger.info("Workload: %s", results.workload_stats)

        if num_workers > 1:
            initial_policy = self.config.benchmark.policies_to_evaluate[0]
            controller_config = replace(
                self.config.controller,
                num_workers=len(self.config.workers),
                eviction_policy=initial_policy,
            )
            shared_config = FrameworkConfig(
                model=self.config.model,
                workers=self.config.workers,
                controller=controller_config,
                workload=self.config.workload,
                benchmark=self.config.benchmark,
            )
            self._shared_controller = CacheController(shared_config)
            self._shared_controller.create_workers()
        else:
            initial_policy = self.config.benchmark.policies_to_evaluate[0]
            self._shared_single_worker = InferenceWorker(
                worker_config=self.config.workers[0],
                model_config=self.config.model,
                eviction_policy=create_policy(initial_policy),
                disk_dir="/tmp/kv_cache_trial_shared",
            )

        for policy_type in self.config.benchmark.policies_to_evaluate:
            logger.info(f"\n{'='*60}")
            logger.info(f"POLICY: {policy_type.value} ({num_workers} worker(s))")
            logger.info(f"{'='*60}")

            for trial_id in range(self.config.benchmark.num_trials):
                logger.info(
                    "  Trial %d/%d",
                    trial_id + 1,
                    self.config.benchmark.num_trials,
                )

                if num_workers > 1:
                    trial = self._run_multi_worker_trial(policy_type, workload, trial_id)
                else:
                    trial = self._run_single_worker_trial(policy_type, workload, trial_id)

                results.trials.append(trial)
                speedup_str = f"{trial.prefill_speedup:.2f}x" if trial.prefill_speedup is not None else "N/A"
                logger.info(
                    "    Hit rate: %.1f%%, TTFT: %.1f ms, ITL: %.1f ms, RPS: %.2f, Speedup: %s, Measured hits=%d, misses=%d",
                    trial.hit_rate * 100,
                    trial.avg_ttft_ms,
                    trial.avg_itl_ms,
                    trial.request_throughput_rps,
                    speedup_str,
                    trial.num_measured_hits,
                    trial.num_measured_misses,
                )

        return results

    def _run_multi_worker_trial(self, policy_type, workload, trial_id):
        controller = self._shared_controller
        if controller is None:
            raise RuntimeError("Shared controller was not initialized")
        controller.set_policy(policy_type)
        controller.clear_shared_state()

        warmup_n = self.config.benchmark.warmup_requests
        request_details = []
        trial_start = time.perf_counter()

        for i, request in enumerate(workload):
            result = controller.process_request(request)
            is_warmup = i < warmup_n
            meta = self._request_meta(request)

            detail = {
                **meta,
                "worker_id": result["worker_id"],
                "cache_hit": result["cache_hit"],
                "tier_hit": result.get("tier_hit"),
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "total_ms": result["total_ms"],
                "ttft_ms": result.get("ttft_ms", result["prefill_ms"]),
                "avg_itl_ms": result.get("avg_itl_ms", 0.0),
                "generated_tokens": result.get("generated_tokens", 0),
                "output_tokens_per_s": result.get("output_tokens_per_s", 0.0),
                "is_warmup": is_warmup,
            }
            request_details.append(detail)

            status = "HIT " if result["cache_hit"] else "MISS"
            tier = f"({result.get('tier_hit', '')})" if result["cache_hit"] else ""
            logger.info(
                f"    [{status}]{tier} W{result['worker_id']} Req {i:3d}: "
                f"ttft={detail['ttft_ms']:6.1f}ms "
                f"itl={detail['avg_itl_ms']:6.1f}ms "
                f"| {self._request_preview(request)}"
            )

        trial_wall_time_s = time.perf_counter() - trial_start
        global_stats = controller.get_global_stats()
        trial_result = self._compute_metrics(
            policy_type,
            trial_id,
            request_details,
            global_stats,
            len(self.config.workers),
            trial_wall_time_s,
        )
        controller.reset()
        torch.cuda.empty_cache()
        return trial_result

    def _run_single_worker_trial(self, policy_type, workload, trial_id):
        worker = self._shared_single_worker
        if worker is None:
            raise RuntimeError("Shared worker was not initialized")
        worker.cache.eviction_policy = create_policy(policy_type)
        worker.reset()

        warmup_n = self.config.benchmark.warmup_requests
        request_details = []
        trial_start = time.perf_counter()

        for i, request in enumerate(workload):
            result = self._process_single_worker_request(worker, request)
            is_warmup = i < warmup_n
            meta = self._request_meta(request)

            detail = {
                **meta,
                "worker_id": 0,
                "cache_hit": result["cache_hit"],
                "tier_hit": result.get("tier_hit"),
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "total_ms": result["total_ms"],
                "ttft_ms": result.get("ttft_ms", result["prefill_ms"]),
                "avg_itl_ms": result.get("avg_itl_ms", 0.0),
                "generated_tokens": result.get("generated_tokens", 0),
                "output_tokens_per_s": result.get("output_tokens_per_s", 0.0),
                "is_warmup": is_warmup,
            }
            request_details.append(detail)

            status = "HIT " if result["cache_hit"] else "MISS"
            tier = f"({result.get('tier_hit', '')})" if result["cache_hit"] else ""
            logger.info(
                f"    [{status}]{tier} Req {i:3d}: "
                f"ttft={detail['ttft_ms']:6.1f}ms "
                f"itl={detail['avg_itl_ms']:6.1f}ms "
                f"| {self._request_preview(request)}"
            )

        trial_wall_time_s = time.perf_counter() - trial_start
        ws = worker.get_stats()
        global_stats = {
            "policy": policy_type.value,
            "num_workers": 1,
            "total_requests": len(workload),
            "cache_routed": 0,
            "rr_routed": len(workload),
            "routing_efficiency": 0.0,
            "global_hit_rate": ws["hit_rate"],
            "total_hits": ws["total_hits"],
            "total_misses": ws["total_misses"],
            "total_evictions": ws["total_evictions"],
            "total_transfer_ms": ws["total_transfer_ms"],
            "total_saved_ms": ws.get("total_cache_saved_ms", 0),
            "worker_stats": {0: ws},
        }

        trial_result = self._compute_metrics(
            policy_type,
            trial_id,
            request_details,
            global_stats,
            1,
            trial_wall_time_s,
        )
        worker.reset()
        torch.cuda.empty_cache()
        return trial_result

    def _compute_metrics(self, policy_type, trial_id, request_details,
                         global_stats, num_workers, trial_wall_time_s):
        measured = [d for d in request_details if not d["is_warmup"]]
        hits = [d for d in measured if d["cache_hit"]]
        misses = [d for d in measured if not d["cache_hit"]]

        total_ms = [d["total_ms"] for d in measured] or [0.0]
        ttft_ms = [d["ttft_ms"] for d in measured] or [0.0]
        itl_ms = [d["avg_itl_ms"] for d in measured if d["generated_tokens"] > 1] or [0.0]

        hit_pf = [d["prefill_ms"] for d in hits]
        miss_pf = [d["prefill_ms"] for d in misses]

        avg_hit = float(np.mean(hit_pf)) if hit_pf else 0.0
        avg_miss = float(np.mean(miss_pf)) if miss_pf else 0.0

        if hit_pf and miss_pf and avg_hit > 0:
            speedup = avg_miss / avg_hit
        else:
            speedup = None

        total_promos = sum(
            ws.get("total_promotions", 0)
            for ws in global_stats.get("worker_stats", {}).values()
        )
        total_demos = sum(
            ws.get("total_demotions", 0)
            for ws in global_stats.get("worker_stats", {}).values()
        )

        total_generated_tokens = int(sum(d.get("generated_tokens", 0) for d in measured))
        request_throughput_rps = (len(measured) / trial_wall_time_s) if trial_wall_time_s > 0 else 0.0
        output_token_throughput_tps = (total_generated_tokens / trial_wall_time_s) if trial_wall_time_s > 0 else 0.0

        return TrialResult(
            policy=policy_type.value,
            trial_id=trial_id,
            num_requests=len(measured),
            num_workers=num_workers,

            hit_rate=len(hits) / len(measured) if measured else 0.0,
            gpu_hits=sum(1 for d in hits if d.get("tier_hit") == "gpu"),
            cpu_hits=sum(1 for d in hits if d.get("tier_hit") == "cpu"),
            disk_hits=sum(1 for d in hits if d.get("tier_hit") == "disk"),
            total_misses=len(misses),

            avg_prefill_ms=float(np.mean([d["prefill_ms"] for d in measured])) if measured else 0.0,
            avg_decode_ms=float(np.mean([d["decode_ms"] for d in measured])) if measured else 0.0,
            avg_total_ms=float(np.mean(total_ms)),
            p50_total_ms=float(np.percentile(total_ms, 50)),
            p95_total_ms=float(np.percentile(total_ms, 95)),
            p99_total_ms=float(np.percentile(total_ms, 99)),

            avg_prefill_on_hit_ms=avg_hit,
            avg_prefill_on_miss_ms=avg_miss,
            prefill_speedup=speedup,

            avg_ttft_ms=float(np.mean(ttft_ms)),
            p50_ttft_ms=float(np.percentile(ttft_ms, 50)),
            p95_ttft_ms=float(np.percentile(ttft_ms, 95)),
            avg_itl_ms=float(np.mean(itl_ms)),

            trial_wall_time_s=float(trial_wall_time_s),
            request_throughput_rps=float(request_throughput_rps),
            output_token_throughput_tps=float(output_token_throughput_tps),
            total_generated_tokens=total_generated_tokens,

            total_evictions=global_stats.get("total_evictions", 0),
            total_promotions=total_promos,
            total_demotions=total_demos,
            total_transfer_ms=global_stats.get("total_transfer_ms", 0),
            total_cache_saved_ms=global_stats.get("total_saved_ms", 0),

            routing_efficiency=global_stats.get("routing_efficiency", 0),
            cache_routed=global_stats.get("cache_routed", 0),
            rr_routed=global_stats.get("rr_routed", 0),

            num_measured_hits=len(hits),
            num_measured_misses=len(misses),

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
                "policy": t.policy,
                "trial_id": t.trial_id,
                "num_workers": t.num_workers,
                "hit_rate": t.hit_rate,
                "gpu_hits": t.gpu_hits,
                "cpu_hits": t.cpu_hits,
                "disk_hits": t.disk_hits,
                "avg_prefill_ms": t.avg_prefill_ms,
                "avg_total_ms": t.avg_total_ms,
                "p95_total_ms": t.p95_total_ms,
                "avg_ttft_ms": t.avg_ttft_ms,
                "p95_ttft_ms": t.p95_ttft_ms,
                "avg_itl_ms": t.avg_itl_ms,
                "request_throughput_rps": t.request_throughput_rps,
                "output_token_throughput_tps": t.output_token_throughput_tps,
                "prefill_speedup": t.prefill_speedup,
                "total_evictions": t.total_evictions,
                "total_transfer_ms": t.total_transfer_ms,
                "routing_efficiency": t.routing_efficiency,
                "num_measured_hits": t.num_measured_hits,
                "num_measured_misses": t.num_measured_misses,
                "request_details": t.request_details,
            } for t in results.trials],
        }

        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Results saved to %s/", output_dir)
