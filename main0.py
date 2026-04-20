#!/usr/bin/env python3
"""
KV-Cache Eviction Policy Benchmark — Real Inference with Llama 3.1-8B

Runs actual LLM inference with real KV tensor caching and tiered storage.
All tensor transfers between GPU, CPU, and disk are real and timed.

Usage:
    python main.py                                    # Default: LRU+LFU, 100 requests
    python main.py --policies lru lfu learned          # Specific policies
    python main.py --requests 200 --gpu-mb 512         # More requests, smaller GPU cache
    python main.py --model /path/to/local/model        # Local model path
    python main.py --quick                             # Quick test: 20 requests, LRU only
"""

import argparse
import logging
import sys
import time

import torch

from config import (
    FrameworkConfig, ModelConfig, WorkerConfig, TierConfig,
    ControllerConfig, WorkloadConfig, BenchmarkConfig,
    EvictionPolicyType, StorageTier,
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
    p = argparse.ArgumentParser(description="Real KV-Cache Eviction Benchmark with Llama 8B")
    p.add_argument("--model", type=str, default=None,
                   help="Model path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--policies", nargs="+", choices=list(POLICY_MAP.keys()), default=None,
                   help="Policies to evaluate (default: lru, lfu)")
    p.add_argument("--requests", type=int, default=None,
                   help="Number of requests (default: 100)")
    p.add_argument("--prefixes", type=int, default=None,
                   help="Number of unique prefixes (default: 10)")
    p.add_argument("--reuse-ratio", type=float, default=None,
                   help="Prefix reuse ratio (default: 0.7)")
    p.add_argument("--gpu-mb", type=float, default=None,
                   help="GPU KV cache capacity in MB per worker (default: 2048)")
    p.add_argument("--cpu-mb", type=float, default=None,
                   help="CPU KV cache capacity in MB per worker (default: 8192)")
    p.add_argument("--disk-mb", type=float, default=None,
                   help="Disk KV cache capacity in MB per worker (default: 32768)")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs/workers (default: 1, auto-detects available)")
    p.add_argument("--max-tokens", type=int, default=50,
                   help="Max new tokens per request (default: 50)")
    p.add_argument("--output", type=str, default="results",
                   help="Output directory (default: results)")
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 20 requests, LRU only, small cache")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_config(args) -> FrameworkConfig:
    # Determine number of GPUs
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

    # Quick mode
    if args.quick:
        config.workload.num_requests = 20
        config.workload.num_unique_prefixes = 3
        config.benchmark.num_trials = 1
        config.benchmark.warmup_requests = 2
        config.benchmark.policies_to_evaluate = [EvictionPolicyType.LRU]
        for w in config.workers:
            w.gpu_tier.capacity_mb = 256

    # CLI overrides
    if args.requests is not None:
        config.workload.num_requests = args.requests
    if args.prefixes is not None:
        config.workload.num_unique_prefixes = args.prefixes
    if args.reuse_ratio is not None:
        config.workload.prefix_reuse_ratio = args.reuse_ratio
    if args.policies is not None:
        config.benchmark.policies_to_evaluate = [POLICY_MAP[p] for p in args.policies]

    config.workload.max_new_tokens = args.max_tokens
    config.workload.seed = args.seed
    config.benchmark.output_dir = args.output

    return config


def main():
    args = parse_args()
    config = build_config(args)

    # Print setup
    logger.info("=" * 60)
    logger.info("KV-Cache Eviction Benchmark — REAL INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.model_path}")
    logger.info(f"Device: {config.model.device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"KV Cache tiers: GPU={config.workers[0].gpu_tier.capacity_mb}MB, "
                f"CPU={config.workers[0].cpu_tier.capacity_mb}MB, "
                f"Disk={config.workers[0].disk_tier.capacity_mb}MB")
    logger.info(f"Requests: {config.workload.num_requests}")
    logger.info(f"Unique prefixes: {config.workload.num_unique_prefixes}")
    logger.info(f"Reuse ratio: {config.workload.prefix_reuse_ratio}")
    logger.info(f"Policies: {[p.value for p in config.benchmark.policies_to_evaluate]}")
    logger.info("=" * 60)

    # Run
    start = time.time()
    harness = BenchmarkHarness(config)
    results = harness.run()
    elapsed = time.time() - start

    # Print summary
    comparison = results.get_comparison_table()
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY (REAL INFERENCE)")
    logger.info("=" * 80)
    header = (f"{'Policy':<10} {'Hit Rate':>9} {'Prefill(HIT)':>13} "
              f"{'Prefill(MISS)':>14} {'Speedup':>8} {'Evictions':>10} "
              f"{'Transfer(ms)':>13}")
    logger.info(header)
    logger.info("-" * 80)
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
    logger.info("=" * 80)
    logger.info(f"Benchmark completed in {elapsed:.1f}s")

    # Save
    BenchmarkHarness.save_results(results, config.benchmark.output_dir)

    if not args.no_plots:
        try:
            generate_all_plots(results, config.benchmark.output_dir)
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())