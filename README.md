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

## Multi-Node (Fault Tolerance Prototype)

This repo now includes a simple HTTP-based worker service and router to support
multi-node experiments with:

- Central KV blob storage (filesystem or Redis)
- Redis-backed metadata registry (worker heartbeat + per-prefix replica leases)
- Fault tolerance via lease expiry and rehydration from the central store

### Start Redis

Run Redis in a place reachable by all nodes (VM, service, or head node).

#### Local Redis (No Docker)

macOS (Homebrew):

```bash
brew update
brew install redis
brew services start redis
redis-cli ping  # expect PONG
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping  # expect PONG
```

Redis URL for local use:

- `redis://localhost:6379/0`

If you need other machines to connect to your Redis, you must bind it to a
non-loopback interface and secure it (firewall + password). Do not expose an
unauthenticated Redis instance to the internet.

### Start Workers (on each GPU node)

Each worker runs a local HTTP server:

```bash
python worker/http_service.py \
  --port 8001 --worker-id 0 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --central-store redis --redis-url redis://<redis-host>:6379/0 \
  --metadata-redis-url redis://<redis-host>:6379/0 \
  --lease-ttl 30
```

Repeat per worker with a unique `--port` and `--worker-id`.

### Start Router (any node)

```bash
python controller/http_service.py \
  --port 9000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --metadata-redis-url redis://<redis-host>:6379/0 \
  --workers http://<worker0-host>:8001 http://<worker1-host>:8002
```

### Send A Request

```bash
python - <<'PY'
import json, urllib.request
payload = {"prompt": "You are a helpful assistant.\\n\\nDocument:\\nHello\\n\\nQuestion:\\nWhat is this?\\n\\nAnswer:", "max_new_tokens": 16}
req = urllib.request.Request("http://localhost:9000/process", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
```

### Run The PD Smoke Test

This lightweight smoke test boots two worker HTTP servers plus the router using
test doubles and validates the `/prefill -> /decode` handoff path.

```bash
./run_pd_smoke_test.sh
```

It requires `python3` and `torch` in the current environment.

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
