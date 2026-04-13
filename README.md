# KV-Cache Eviction Policy Framework — Real Inference

Benchmarks KV-cache eviction policies using **real Llama 3.1-8B inference** with actual tensor storage and transfer across GPU, CPU, and disk tiers.

## What's Real

- **Real model inference**: Llama 3.1-8B runs actual prefill and decode
- **Real KV tensors**: `past_key_values` extracted from model forward pass
- **Real tensor transfers**: GPU↔CPU via `tensor.to()`, CPU↔Disk via `torch.save/load`
- **Real prefix reuse**: Cached KV passed via `past_key_values` parameter to skip prefill
- **Real timing**: All latencies measured with `torch.cuda.synchronize()` + `time.perf_counter()`

## Project Structure

```
kv-cache-framework-real/
├── main.py                     # Entry point
├── config.py                   # Configuration
├── run_benchmark.slurm         # SLURM job for Alpine
├── cache/
│   ├── kv_entry.py             # KVEntry with real tensor transfer methods
│   ├── tiered_cache.py         # TieredCache with real GPU/CPU/disk movement
│   └── eviction.py             # All eviction policies + factory
├── worker/
│   └── inference_worker.py     # Real Llama 8B inference with KV reuse
├── workload/
│   └── loader.py               # Request generation with shared prefixes
└── evaluation/
    ├── benchmark.py            # Benchmark harness
    └── visualize.py            # Result plots
```

## How Tensor Tiers Work

```
GPU (Hot)  ←→  CPU (Warm)  ←→  Disk (Cold)
  tensor.to("cuda")   tensor.to("cpu")   torch.save/load
```

- **GPU → CPU**: `tensor.to("cpu")` — moves tensor from VRAM to system RAM
- **CPU → Disk**: `torch.save(tensor, path)` — serializes to disk, frees RAM
- **Disk → CPU**: `torch.load(path, map_location="cpu")` — deserializes
- **CPU → GPU**: `tensor.to("cuda")` — transfers back to VRAM for inference

## Usage on Alpine

```bash
# Quick test (20 requests, LRU only)
sbatch run_benchmark.slurm

# Or run interactively on a GPU node:
sinteractive --partition=blanca-clearlab1 --gpus=1 --mem=64G --time=02:00:00
conda activate jana7431-gpt
python main.py --quick --model /scratch/alpine/jana7431/Intune/models/Meta-Llama-3.1-8B-Instruct

# Full benchmark
python main.py \
    --model /scratch/alpine/jana7431/Intune/models/Meta-Llama-3.1-8B-Instruct \
    --policies lru lfu \
    --requests 100 --gpu-mb 512
```

## Expected Output

```
[MISS]     Req   0: prefill= 42.3ms decode= 35.1ms total= 77.4ms | What is a linked list...
[MISS]     Req   1: prefill= 38.7ms decode= 33.2ms total= 71.9ms | How does quicksort...
[HIT](gpu) Req   2: prefill=  6.2ms decode= 34.8ms total= 41.0ms | Explain Big O notation...
[HIT](cpu) Req   3: prefill= 12.4ms decode= 33.9ms total= 46.3ms | What is the CAP theorem...
```

## Key Differences from Simulation Framework

| Aspect | Simulation | Real |
|--------|-----------|------|
| Model | None | Llama 3.1-8B |
| KV tensors | Metadata only | Real PyTorch tensors |
| Transfers | Simulated latency | Actual `.to()` / `torch.save` |
| Prefill | Estimated ms/token | Measured with CUDA sync |
| Runs on | CPU (any machine) | GPU (Alpine/Blanca) |
