#!/usr/bin/env python3
"""
KV-Cache Eviction Policy Benchmark — Real Inference with Llama 3.1-8B

Runs actual LLM inference with real KV tensor caching and tiered storage.
Configured for LMCache-style long-document multi-round QA workloads.

This version prints both:
- your existing metrics (hit rate, prefill hit/miss, evictions, transfer)
- LMCache-style metrics (TTFT, ITL, throughput)
"""

import argparse
import logging
import sys
import time

import torch

from config import FrameworkConfig, EvictionPolicyType
from evaluation.benchmark import BenchmarkHarness
from evaluation.visualize import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POLICY_MAP = {
    "lru": EvictionPolicyType.LRU,
    "lfu": EvictionPolicyType.LFU,
    "semantic": EvictionPolicyType.SEMANTIC,
    "learned": EvictionPolicyType.LEARNED,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Real KV-Cache Eviction Benchmark with Llama 3.1-8B"
    )

    p.add_argument("--model", type=str, default=None,
                   help="Model path")
    p.add_argument("--policies", nargs="+", choices=list(POLICY_MAP.keys()), default=None,
                   help="Policies to evaluate")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs/workers")
    p.add_argument("--gpu-mb", type=float, default=None,
                   help="GPU KV cache capacity in MB per worker")
    p.add_argument("--cpu-mb", type=float, default=None,
                   help="CPU KV cache capacity in MB per worker")
    p.add_argument("--disk-mb", type=float, default=None,
                   help="Disk KV cache capacity in MB per worker")
    p.add_argument("--kv-chunk-size", type=int, default=None,
                   help="Token chunk size for LMCache-style shared-store KV chunking")
    p.add_argument("--layerwise-kv-pipeline", action="store_true",
                   help="Enable staged layer-wise KV transfer pipelining during CPU/GPU movement")
    p.add_argument("--pipeline-stage-layers", type=int, default=None,
                   help="How many layers to move per pipeline stage when layer-wise transfer is enabled")
    p.add_argument("--dynamic-offload", action="store_true",
                   help="Enable chunk-level dynamic GPU->CPU offloading with a duplication window")
    p.add_argument("--dynamic-offload-window-factor", type=float, default=None,
                   help="How much GPU allocation demand to pre-duplicate to CPU before reclaiming GPU chunks")

    p.add_argument("--central-store", choices=["none", "filesystem", "redis"], default="none",
                   help="Enable a shared central KV store across workers")
    p.add_argument("--central-dir", type=str, default="/tmp/lmcache_central_kv",
                   help="Central KV store directory when using filesystem backend")
    p.add_argument("--redis-url", type=str, default="redis://localhost:6379/0",
                   help="Redis URL when using redis backend")

    p.add_argument("--metadata-registry", choices=["none", "redis"], default="none",
                   help="Enable metadata registry for fault tolerance (replica leases)")
    p.add_argument("--metadata-redis-url", type=str, default="redis://localhost:6379/0",
                   help="Redis URL for metadata registry")
    p.add_argument("--lease-ttl", type=int, default=30,
                   help="TTL in seconds for worker heartbeats and replica leases")
    p.add_argument("--pd-disaggregation", action="store_true",
                   help="Split requests into prefill and decode stages across workers when possible")
    p.add_argument("--log-evictions", action="store_true",
                   help="Emit detailed eviction and tier-movement logs during the run")

    p.add_argument("--documents", type=int, default=None,
                   help="Number of long shared documents")
    p.add_argument("--document-length", type=int, default=None,
                   help="Approximate document length in tokens")
    p.add_argument("--rounds", type=int, default=None,
                   help="Number of rounds")
    p.add_argument("--hit-ratio", type=float, default=None,
                   help="Fraction of reused documents in rounds > 0")
    p.add_argument("--initial-concurrency", type=int, default=None,
                   help="Initial concurrent requests/users")
    p.add_argument("--arrival-mode", choices=["bursty", "poisson"], default=None,
                   help="Traffic pattern")
    p.add_argument("--interarrival", type=float, default=None,
                   help="Mean interarrival time in seconds")
    p.add_argument("--repeat-mode", choices=["tile", "random", "interleave"], default=None,
                   help="How reused documents are ordered across rounds")
    p.add_argument("--question-style", choices=["document_qa", "summarization", "mixed"], default=None,
                   help="Question style for generated requests")
    p.add_argument("--questions-per-document", type=int, default=None,
                   help="How many unique questions each document can serve across rounds")
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Max new tokens per request")

    p.add_argument("--requests", type=int, default=None,
                   help="Optional hard cap on total requests")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory")
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def build_config(args) -> FrameworkConfig:
    num_gpus = args.num_gpus or 1
    gpu_mb = args.gpu_mb or 2048
    cpu_mb = args.cpu_mb or 8192
    disk_mb = args.disk_mb or 32768

    if num_gpus > 1:
        config = FrameworkConfig.make_multi_gpu(num_gpus, gpu_mb, cpu_mb, disk_mb)
    else:
        config = FrameworkConfig()
        if args.gpu_mb is not None:
            config.workers[0].gpu_tier.capacity_mb = gpu_mb
        if args.cpu_mb is not None:
            config.workers[0].cpu_tier.capacity_mb = cpu_mb
        if args.disk_mb is not None:
            config.workers[0].disk_tier.capacity_mb = disk_mb

    if args.model:
        config.model.model_path = args.model
    config.model.max_new_tokens = args.max_tokens
    if args.kv_chunk_size is not None:
        config.model.kv_chunk_size_tokens = args.kv_chunk_size
    if args.layerwise_kv_pipeline:
        config.model.enable_layerwise_kv_pipeline = True
    if args.pipeline_stage_layers is not None:
        config.model.layerwise_pipeline_stage_layers = args.pipeline_stage_layers

    if args.quick:
        config.workload.num_documents = 4
        config.workload.document_length_tokens = 2000
        config.workload.num_rounds = 2
        config.workload.hit_ratio = 1.0
        config.workload.initial_concurrency = 4
        config.workload.arrival_mode = "bursty"
        config.workload.interarrival_mean_sec = 0.1
        config.workload.max_new_tokens = min(args.max_tokens, 32)

        config.benchmark.num_trials = 1
        config.benchmark.warmup_requests = 4
        config.benchmark.policies_to_evaluate = [EvictionPolicyType.LRU]

        for w in config.workers:
            w.gpu_tier.capacity_mb = min(w.gpu_tier.capacity_mb, 256)

    if args.documents is not None:
        config.workload.num_documents = args.documents
    if args.document_length is not None:
        config.workload.document_length_tokens = args.document_length
    if args.rounds is not None:
        config.workload.num_rounds = args.rounds
    if args.hit_ratio is not None:
        config.workload.hit_ratio = args.hit_ratio
    if args.initial_concurrency is not None:
        config.workload.initial_concurrency = args.initial_concurrency
    if args.arrival_mode is not None:
        config.workload.arrival_mode = args.arrival_mode
    if args.interarrival is not None:
        config.workload.interarrival_mean_sec = args.interarrival
    if args.repeat_mode is not None:
        config.workload.repeat_mode = args.repeat_mode
    if args.question_style is not None:
        config.workload.question_style = args.question_style
    if args.questions_per_document is not None:
        config.workload.num_questions_per_document = args.questions_per_document
    if args.requests is not None:
        config.workload.num_requests = args.requests

    config.workload.max_new_tokens = args.max_tokens
    config.workload.seed = args.seed

    if args.policies is not None:
        config.benchmark.policies_to_evaluate = [POLICY_MAP[p] for p in args.policies]

    # Central KV store
    if args.central_store and args.central_store != "none":
        config.controller.enable_central_store = True
        config.controller.central_store_backend = args.central_store
        config.controller.central_store_dir = args.central_dir
        config.controller.redis_url = args.redis_url

    # Metadata registry (fault tolerance)
    if args.metadata_registry and args.metadata_registry != "none":
        config.controller.enable_metadata_registry = True
        config.controller.metadata_backend = args.metadata_registry
        config.controller.metadata_redis_url = args.metadata_redis_url
        config.controller.worker_lease_ttl_s = args.lease_ttl

    if args.pd_disaggregation:
        config.controller.enable_pd_disaggregation = True
    if args.log_evictions:
        config.controller.log_evictions = True
    if args.dynamic_offload:
        config.controller.enable_dynamic_offload = True
    if args.dynamic_offload_window_factor is not None:
        config.controller.dynamic_offload_window_factor = args.dynamic_offload_window_factor

    config.benchmark.output_dir = args.output

    if config.benchmark.warmup_requests > config.workload.total_requests:
        config.benchmark.warmup_requests = min(5, config.workload.total_requests)

    return config


def log_setup(config: FrameworkConfig) -> None:
    logger.info("=" * 72)
    logger.info("KV-Cache Eviction Benchmark — REAL INFERENCE")
    logger.info("=" * 72)
    logger.info("Model: %s", config.model.model_path)
    logger.info("Device: %s", config.model.device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )

    logger.info(
        "KV Cache tiers: GPU=%sMB, CPU=%sMB, Disk=%sMB",
        config.workers[0].gpu_tier.capacity_mb,
        config.workers[0].cpu_tier.capacity_mb,
        config.workers[0].disk_tier.capacity_mb,
    )

    logger.info("Workload type: LMCache-style long-document multi-round QA")
    logger.info("Documents: %s", config.workload.num_documents)
    logger.info("Document length: ~%s tokens", config.workload.document_length_tokens)
    logger.info("Rounds: %s", config.workload.num_rounds)
    logger.info("Hit ratio: %.2f", config.workload.hit_ratio)
    logger.info("Initial concurrency: %s", config.workload.initial_concurrency)
    logger.info("Arrival mode: %s", config.workload.arrival_mode)
    logger.info("Interarrival mean: %.3fs", config.workload.interarrival_mean_sec)
    logger.info("Repeat mode: %s", config.workload.repeat_mode)
    logger.info("Questions per document: %s", config.workload.num_questions_per_document)
    logger.info("KV chunk size: %s tokens", config.model.kv_chunk_size_tokens)
    logger.info("Layer-wise KV pipeline: %s", config.model.enable_layerwise_kv_pipeline)
    if config.model.enable_layerwise_kv_pipeline:
        logger.info("Pipeline stage layers: %s", config.model.layerwise_pipeline_stage_layers)
    logger.info("Dynamic offload: %s", config.controller.enable_dynamic_offload)
    if config.controller.enable_dynamic_offload:
        logger.info("Dynamic offload window factor: %.2f", config.controller.dynamic_offload_window_factor)
    logger.info("Max new tokens: %s", config.workload.max_new_tokens)
    logger.info("Estimated total requests: %s", config.workload.total_requests)
    logger.info("Policies: %s", [p.value for p in config.benchmark.policies_to_evaluate])
    logger.info("PD disaggregation: %s", config.controller.enable_pd_disaggregation)
    logger.info("=" * 72)


def print_summary(results, elapsed: float) -> None:
    comparison = results.get_comparison_table()

    logger.info("\n" + "=" * 120)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 120)

    header = (
        f"{'Policy':<10} {'HitRate':>8} {'TTFT':>10} {'ITL':>10} "
        f"{'Req/s':>10} {'Tok/s':>10} {'PF(HIT)':>11} {'PF(MISS)':>12} "
        f"{'Speedup':>8} {'Evict':>8} {'Xfer(ms)':>10} "
        f"{'Hits':>6} {'Misses':>8}"
    )
    logger.info(header)
    logger.info("-" * 120)

    for row in comparison:
        speedup_str = f"{row['speedup']:.2f}" if row["speedup"] is not None else "N/A"

        logger.info(
            f"{row['policy']:<10} "
            f"{row['hit_rate']:>7.1%} "
            f"{row['avg_ttft_ms']:>9.1f} "
            f"{row['avg_itl_ms']:>9.1f} "
            f"{row['request_throughput_rps']:>9.2f} "
            f"{row['output_token_throughput_tps']:>9.2f} "
            f"{row['avg_prefill_hit_ms']:>10.1f} "
            f"{row['avg_prefill_miss_ms']:>11.1f} "
            f"{speedup_str:>8} "
            f"{row['evictions']:>8.0f} "
            f"{row['transfer_ms']:>10.1f} "
            f"{row.get('num_measured_hits', 0):>6.0f} "
            f"{row.get('num_measured_misses', 0):>8.0f}"
        )

    logger.info("=" * 120)
    logger.info("Benchmark completed in %.1fs", elapsed)


def main():
    args = parse_args()
    config = build_config(args)

    log_setup(config)

    start = time.time()
    harness = BenchmarkHarness(config)
    results = harness.run()
    elapsed = time.time() - start

    print_summary(results, elapsed)

    BenchmarkHarness.save_results(results, config.benchmark.output_dir)

    if not args.no_plots:
        try:
            generate_all_plots(results, config.benchmark.output_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
