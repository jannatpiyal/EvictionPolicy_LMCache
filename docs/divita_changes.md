# Divita Changes (Central KV Store + Shared Backends)

This document summarizes all code changes implemented so far for Divita’s scope:

- Central KV store abstraction
- Filesystem “object store” backend
- Redis backend
- Wiring into controller/worker for cross-worker KV reuse

## Motivation

The original codebase maintained KV cache entries **only locally per worker**. A request could only be a cache hit if routed to the same worker that originally prefetched the prefix. This limited cross-worker reuse and made fault tolerance / multi-node scaling impossible.

The changes below introduce a shared storage layer for KV blobs (KV tensors), enabling:

- Cross-worker prefix reuse (a worker can fetch KV created by a different worker)
- A foundation for multi-node coordination (shared storage + metadata/leases as a next step)

## What Was Added

### 1) Central KV Store Abstraction

New module: `store/central_kv_store.py`

- `CentralKVStore` interface:
  - `get(prefix_hash) -> CentralKVRecord | None`
  - `put(prefix_hash, kv_tuple) -> None`
  - `contains(prefix_hash) -> bool`
  - `delete(prefix_hash) -> None`
- `CentralKVRecord` stores:
  - `prefix_hash`
  - `kv_tuple` (typically a tuple of per-layer `(K, V)` tensors on CPU)
  - `stored_at`

New exports: `store/__init__.py`

### 2) Filesystem Backend (“Object Store” Stand-in)

Implementation: `FileSystemCentralKVStore` in `store/central_kv_store.py`

- Persists KV blobs as:
  - `<root_dir>/<prefix_hash>.pt` via `torch.save`
  - `<root_dir>/<prefix_hash>.json` for small metadata
- Uses temp file + `os.replace` to avoid partially written blobs.

This backend behaves like an object store in the sense that it is **shared blob storage**. In multi-node deployments it only works if `root_dir` is on a **shared filesystem** (NFS/WEKA/etc).

### 3) Redis Backend

Implementation: `RedisCentralKVStore` in `store/central_kv_store.py`

- Stores `torch.save(kv_tuple)` bytes in Redis under keys like `lmcache:kv:<prefix_hash>` (configurable prefix).
- Imports `redis` lazily; if the package isn’t installed it raises a clear error at runtime.

## How It Was Wired Into The System

### Controller: Central Store Construction + Worker Injection

File: `controller/cache_controller.py`

- If enabled by config, the controller constructs:
  - `FileSystemCentralKVStore` or
  - `RedisCentralKVStore`
- The controller passes `central_store=...` into each `InferenceWorker`.

### Worker: Central Store Read-Through + Write-Through

File: `worker/inference_worker.py`

New behavior:

1. Compute `prefix_hash` from the system/prefix tokens.
2. Attempt **local cache** lookup (`TieredCache.get`).
3. On local miss and if `central_store` is enabled:
   - Fetch `kv_tuple` from central store in CPU memory
   - Insert it into the local **CPU tier** (`TieredCache.put_cpu`)
   - Re-run `cache.get(prefix_hash)` so the normal promote-to-GPU path is used
   - Mark result as `cache_hit=True` and `central_hit=True`
4. On miss after full prefill and local insert:
   - Best-effort **write-through**: copy the locally cached GPU KV to CPU and `central_store.put(prefix_hash, cpu_kv)`

Returned metrics:

- `central_hit`: `True` if the cache hit came from central store
- `prefix_hash`: included in the result so the controller can index it

### Tiered Cache: CPU Insert Path

File: `cache/tiered_cache.py`

Added `put_cpu(entry)` which:

- Inserts an entry directly into the CPU tier (KV must already be CPU-resident)
- Evicts from CPU tier if necessary (demoting to Disk if configured)

### Tier Hit Accounting Fix

Problem: on a CPU/Disk hit, the cache promotes the entry to GPU before the worker reports `tier_hit`, so `tier_hit` could incorrectly appear as `"gpu"`.

Fix:

- `cache/kv_entry.py` adds `last_hit_tier`
- `cache/tiered_cache.py` sets `entry.last_hit_tier = <tier>` when it is found
- `worker/inference_worker.py` reports `tier_hit` using `last_hit_tier` if present

## Config / CLI

File: `config.py`

Added controller options:

- `enable_central_store`
- `central_store_backend` (`filesystem` or `redis`)
- `central_store_dir`
- `redis_url`

File: `main.py`

Added flags:

- `--central-store none|filesystem|redis`
- `--central-dir <path>`
- `--redis-url redis://...`

## Test

File: `tests/test_central_store_filesystem.py`

- Minimal round-trip test for filesystem backend (skips if `torch` is not installed in the current environment).

## Known Limitations (Intentional in This Slice)

- This central store is a **shared blob store only**; it does not implement:
  - authoritative metadata / leases
  - replication or replica selection
  - fault tolerance
  - multi-node controller/worker RPC
- Current implementation is a good foundation for the next step: a metadata service (e.g., Redis) that tracks replicas with TTL/leases and enables fault-tolerant routing.

## Follow-up: Metadata Registry (Fault Tolerance Foundation)

New module: `metadata/registry.py`

Adds a Redis-backed metadata/lease registry that can be used in multi-node settings to:

- Track worker liveness via heartbeat TTL
- Track prefix replicas via per-prefix replica set + per-replica lease TTL
- Route to a live replica when available

Wiring:

- `controller/cache_controller.py` optionally creates `RedisMetadataRegistry` and uses it to route requests to a live replica before falling back to the in-memory `prefix_index`.
- `worker/inference_worker.py` best-effort registers/heartbeats the worker and claims a replica lease whenever it has the KV for a prefix (local hit, central hit, or miss after caching).

CLI:

- `main.py` adds `--metadata-registry none|redis`, `--metadata-redis-url`, and `--lease-ttl`.

## Follow-up: “Real” Fault Tolerance Prototype (Multi-Node)

This repo now includes a minimal multi-node prototype where:

- **KV bytes** live in a shared central store (filesystem or Redis).
- **Coordination state** lives in a Redis metadata registry (heartbeats + per-prefix leases).
- A router selects a **live replica** if one exists; otherwise it routes to any worker, which
  **rehydrates** the KV from the central store and then claims a new lease.

### Components

- Worker HTTP service: `worker/http_service.py`
  - `POST /process`: runs inference using local tiered cache, falling back to central store on miss.
  - Background renewal thread:
    - refreshes worker heartbeat TTL
    - refreshes replica leases for all locally cached prefixes
  - `POST /prefetch`: optional helper to pull a specific `prefix_hash` from central store into CPU tier.

- Router HTTP service: `controller/http_service.py`
  - `POST /process`:
    - computes `prefix_hash` from the prefix text using a tokenizer
    - queries Redis metadata for live replicas for that `prefix_hash`
    - forwards the request to a live replica if available
    - otherwise picks a worker and relies on rehydration at the worker

### Why This Improves Fault Tolerance

- If a worker crashes, its heartbeat and leases expire (TTL).
- The router stops choosing it as a replica.
- Another worker can rehydrate from the central store and take over.

### How To Run

The runnable commands are documented in:

- Multi-node workflow: `docs/fault_tolerance_ops.md`
- Redis install (no Docker): `README.md`
