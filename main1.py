#!/usr/bin/env python3
"""
KV-Cache Eviction Policy Benchmark — Real Inference with Llama 3.1-8B

Runs actual LLM inference with real KV tensor caching and tiered storage.
Configured for LMCache-style long-document multi-round QA workloads.

Usage:
    python main.py
    python main.py --policies lru lfu
    python main.py --documents 40 --document-length 10000 --rounds 2
    python main.py --hit-ratio 1.0 --initial-concurrency 40
    python main.py --quick
"""

import argparse
import logging
import sys
import time

import torch

from config import (
    FrameworkConfig,
    EvictionPolicyType,
)
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

    # Model / system
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)",
    )
    p.add_argument(
        "--policies",
        nargs="+",
        choices=list(POLICY_MAP.keys()),
        default=None,
        help="Policies to evaluate (default: lru, lfu)",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs/workers (default: 1)",
    )
    p.add_argument(
        "--gpu-mb",
        type=float,
        default=None,
        help="GPU KV cache capacity in MB per worker (default: 2048)",
    )
    p.add_argument(
        "--cpu-mb",
        type=float,
        default=None,
        help="CPU KV cache capacity in MB per worker (default: 8192)",
    )
    p.add_argument(
        "--disk-mb",
        type=float,
        default=None,
        help="Disk KV cache capacity in MB per worker (default: 32768)",
    )

    # LMCache-style workload
    p.add_argument(
        "--documents",
        type=int,
        default=None,
        help="Number of long shared documents (default: 40)",
    )
    p.add_argument(
        "--document-length",
        type=int,
        default=None,
        help="Approximate document length in tokens (default: 10000)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of rounds (default: 2; warmup + reuse)",
    )
    p.add_argument(
        "--hit-ratio",
        type=float,
        default=None,
        help="Fraction of reused documents in rounds > 0 (default: 1.0)",
    )
    p.add_argument(
        "--initial-concurrency",
        type=int,
        default=None,
        help="Initial concurrent requests/users (default: 40)",
    )
    p.add_argument(
        "--arrival-mode",
        choices=["bursty", "poisson"],
        default=None,
        help="Traffic pattern (default: bursty)",
    )
    p.add_argument(
        "--interarrival",
        type=float,
        default=None,
        help="Mean interarrival time in seconds (default: 0.2)",
    )
    p.add_argument(
        "--question-style",
        choices=["document_qa", "summarization", "mixed"],
        default=None,
        help="Question style for generated requests (default: document_qa)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max new tokens per request (default: 100)",
    )

    # Misc
    p.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Optional backward-compatible legacy args
    p.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Optional hard cap on total requests",
    )

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

    # Model
    if args.model:
        config.model.model_path = args.model
    config.model.max_new_tokens = args.max_tokens

    # Quick mode
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

    # CLI overrides: workload
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
    if args.question_style is not None:
        config.workload.question_style = args.question_style
    if args.requests is not None:
        config.workload.num_requests = args.requests

    config.workload.max_new_tokens = args.max_tokens
    config.workload.seed = args.seed

    # CLI overrides: benchmark / controller
    if args.policies is not None:
        config.benchmark.policies_to_evaluate = [POLICY_MAP[p] for p in args.policies]

    config.benchmark.output_dir = args.output

    # Keep benchmark warmup aligned with workload if still tiny/quick
    if config.benchmark.warmup_requests > config.workload.total_requests:
        config.benchmark.warmup_requests = min(5, config.workload.total_requests)

    return config


def log_setup(config: FrameworkConfig) -> None:
    logger.info("=" * 60)
    logger.info("KV-Cache Eviction Benchmark — REAL INFERENCE")
    logger.info("=" * 60)
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
    logger.info("Max new tokens: %s", config.workload.max_new_tokens)
    logger.info("Estimated total requests: %s", config.workload.total_requests)
    logger.info("Policies: %s", [p.value for p in config.benchmark.policies_to_evaluate])
    logger.info("=" * 60)


def main():
    args = parse_args()
    config = build_config(args)

    log_setup(config)

    start = time.time()
    harness = BenchmarkHarness(config)
    results = harness.run()
    elapsed = time.time() - start

    comparison = results.get_comparison_table()

    logger.info("\n" + "=" * 90)
    logger.info("RESULTS SUMMARY (REAL INFERENCE)")
    logger.info("=" * 90)

    header = (
        f"{'Policy':<10} {'Hit Rate':>9} {'Prefill(HIT)':>13} "
        f"{'Prefill(MISS)':>14} {'Speedup':>8} {'Evictions':>10} "
        f"{'Transfer(ms)':>13}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for row in comparison:
        logger.info(
            f"{row['policy']:<10} "
            f"{row['hit_rate']:>8.1%} "
            f"{row['avg_prefill_hit_ms']:>12.1f}ms "
            f"{row['avg_prefill_miss_ms']:>13.1f}ms "
            f"{row['speedup']:>7.1f}x "
            f"{row['evictions']:>10.0f} "
            f"{row['transfer_ms']:>12.1f}ms"
        )

    logger.info("=" * 90)
    logger.info("Benchmark completed in %.1fs", elapsed)

    BenchmarkHarness.save_results(results, config.benchmark.output_dir)

    if not args.no_plots:
        try:
            generate_all_plots(results, config.benchmark.output_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())