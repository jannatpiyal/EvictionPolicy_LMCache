"""
Visualization for real inference benchmark results.

Keeps current plots and adds LMCache-style ones:
- TTFT by policy
- ITL by policy
- Throughput by policy
- TTFT vs throughput scatter
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
    bars = ax.bar(policies, rates)
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Cache Hit Rate by Eviction Policy")
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
    ax.bar(x - width/2, miss_ms, width, label="Cache MISS (full prefill)")
    ax.bar(x + width/2, hit_ms, width, label="Cache HIT (partial prefill)")
    ax.set_ylabel("Avg Prefill Latency (ms)")
    ax.set_title("Prefill Latency: Cache Hit vs Miss")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "prefill_speedup.png"), dpi=150)
    plt.close()


def plot_latency_timeline(results, output_dir="results"):
    if not HAS_MPL:
        return

    for trial in results.trials:
        fig, ax = plt.subplots(figsize=(14, 5))
        measured = [d for d in trial.request_details if not d["is_warmup"]]

        hits_x = [i for i, d in enumerate(measured) if d["cache_hit"]]
        hits_y = [d["prefill_ms"] for d in measured if d["cache_hit"]]
        misses_x = [i for i, d in enumerate(measured) if not d["cache_hit"]]
        misses_y = [d["prefill_ms"] for d in measured if not d["cache_hit"]]

        ax.scatter(misses_x, misses_y, alpha=0.6, s=20, label="Miss")
        ax.scatter(hits_x, hits_y, alpha=0.6, s=20, label="Hit")
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
    ax.bar(x, gpu, label="GPU Hit")
    ax.bar(x, cpu, bottom=gpu, label="CPU Hit")
    ax.bar(x, disk, bottom=[g+c for g, c in zip(gpu, cpu)], label="Disk Hit")
    ax.bar(x, misses, bottom=[g+c+d for g, c, d in zip(gpu, cpu, disk)],
           label="Miss", alpha=0.5)
    ax.set_ylabel("Request Count")
    ax.set_title("Cache Tier Hit Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tier_hits.png"), dpi=150)
    plt.close()


def plot_ttft(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()
    policies = [r["policy"] for r in comp]
    avg_ttft = [r["avg_ttft_ms"] for r in comp]
    p95_ttft = [r["p95_ttft_ms"] for r in comp]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_ttft, width, label="Avg TTFT")
    ax.bar(x + width/2, p95_ttft, width, label="P95 TTFT")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ttft.png"), dpi=150)
    plt.close()


def plot_itl(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()
    policies = [r["policy"] for r in comp]
    avg_itl = [r["avg_itl_ms"] for r in comp]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policies, avg_itl)
    ax.set_ylabel("Average ITL (ms)")
    ax.set_title("Inter-Token Latency by Policy")
    for bar, val in zip(bars, avg_itl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_itl) * 0.02 if avg_itl else 0.02,
                f"{val:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "itl.png"), dpi=150)
    plt.close()


def plot_throughput(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()
    policies = [r["policy"] for r in comp]
    rps = [r["request_throughput_rps"] for r in comp]
    tps = [r["output_token_throughput_tps"] for r in comp]

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, rps, width, label="Requests/s")
    ax.bar(x + width/2, tps, width, label="Output tokens/s")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "throughput.png"), dpi=150)
    plt.close()


def plot_ttft_vs_throughput(results, output_dir="results"):
    if not HAS_MPL:
        return
    comp = results.get_comparison_table()

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in comp:
        ax.scatter(row["avg_ttft_ms"], row["request_throughput_rps"], s=80)
        ax.annotate(row["policy"], (row["avg_ttft_ms"], row["request_throughput_rps"]))

    ax.set_xlabel("Average TTFT (ms)")
    ax.set_ylabel("Request Throughput (requests/s)")
    ax.set_title("TTFT vs Throughput")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ttft_vs_throughput.png"), dpi=150)
    plt.close()


def generate_all_plots(results, output_dir="results"):
    if not HAS_MPL:
        logger.error("matplotlib required. pip install matplotlib")
        return
    plot_hit_rate(results, output_dir)
    plot_prefill_speedup(results, output_dir)
    plot_latency_timeline(results, output_dir)
    plot_tier_hits(results, output_dir)
    plot_ttft(results, output_dir)
    plot_itl(results, output_dir)
    plot_throughput(results, output_dir)
    plot_ttft_vs_throughput(results, output_dir)
    logger.info(f"All plots saved to {output_dir}/")