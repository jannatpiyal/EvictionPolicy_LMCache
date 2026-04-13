"""
Visualization for real inference benchmark results.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_hit_rate(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()
    policies = [r["policy"] for r in comp]
    rates = [r["hit_rate"] for r in comp]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(policies, rates, color=colors[:len(policies)])
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Cache Hit Rate by Eviction Policy (Real Llama 8B)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontweight="bold")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "hit_rate.png"), dpi=150)
    plt.close()


def plot_prefill_speedup(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()
    policies = [r["policy"] for r in comp]
    hit_ms = [r["avg_prefill_hit_ms"] for r in comp]
    miss_ms = [r["avg_prefill_miss_ms"] for r in comp]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, miss_ms, width, label="Cache MISS (full prefill)", color="#F44336")
    ax.bar(x + width/2, hit_ms, width, label="Cache HIT (partial prefill)", color="#4CAF50")
    ax.set_ylabel("Avg Prefill Latency (ms)")
    ax.set_title("Prefill Latency: Cache Hit vs Miss (Real Llama 8B)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "prefill_speedup.png"), dpi=150)
    plt.close()


def plot_latency_timeline(results, output_dir="results"):
    """Plot per-request latency over time, colored by hit/miss."""
    if not HAS_MPL:
        return

    for trial in results.trials:
        fig, ax = plt.subplots(figsize=(14, 5))
        measured = [d for d in trial.request_details if not d["is_warmup"]]

        hits_x = [i for i, d in enumerate(measured) if d["cache_hit"]]
        hits_y = [d["prefill_ms"] for d in measured if d["cache_hit"]]
        misses_x = [i for i, d in enumerate(measured) if not d["cache_hit"]]
        misses_y = [d["prefill_ms"] for d in measured if not d["cache_hit"]]

        ax.scatter(misses_x, misses_y, c="#F44336", alpha=0.6, s=20, label="Miss")
        ax.scatter(hits_x, hits_y, c="#4CAF50", alpha=0.6, s=20, label="Hit")
        ax.set_xlabel("Request #")
        ax.set_ylabel("Prefill Latency (ms)")
        ax.set_title(f"Prefill Latency Timeline — {trial.policy.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"timeline_{trial.policy}.png"), dpi=150)
        plt.close()


def plot_tier_hits(results, output_dir="results"):
    if not HAS_MPL:
        return

    seen = {}
    for trial in results.trials:
        if trial.policy not in seen:
            seen[trial.policy] = trial

    policies = list(seen.keys())
    gpu = [seen[p].gpu_hits for p in policies]
    cpu = [seen[p].cpu_hits for p in policies]
    disk = [seen[p].disk_hits for p in policies]
    misses = [seen[p].total_misses for p in policies]

    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, gpu, label="GPU Hit", color="#4CAF50")
    ax.bar(x, cpu, bottom=gpu, label="CPU Hit", color="#FF9800")
    ax.bar(x, disk, bottom=[g+c for g, c in zip(gpu, cpu)], label="Disk Hit", color="#2196F3")
    ax.bar(x, misses, bottom=[g+c+d for g, c, d in zip(gpu, cpu, disk)],
           label="Miss", color="#F44336", alpha=0.5)
    ax.set_ylabel("Request Count")
    ax.set_title("Cache Tier Hit Distribution (Real Transfers)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tier_hits.png"), dpi=150)
    plt.close()


def generate_all_plots(results, output_dir="results"):
    if not HAS_MPL:
        logger.error("matplotlib required. pip install matplotlib")
        return
    plot_hit_rate(results, output_dir)
    plot_prefill_speedup(results, output_dir)
    plot_latency_timeline(results, output_dir)
    plot_tier_hits(results, output_dir)
    logger.info(f"All plots saved to {output_dir}/")
