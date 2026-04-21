# Fault Tolerance (Multi-Node) Ops Notes

This repo’s fault-tolerance prototype uses:

- Central KV blob store: filesystem or Redis (KV bytes)
- Metadata registry: Redis (worker heartbeats + replica leases)
- Router: HTTP service that routes by live replicas
- Workers: HTTP services that run inference and periodically renew leases

## Components

### Worker Service

File: `worker/http_service.py`

Endpoints:

- `POST /process`: run inference using local cache + central store fallback
- `POST /prefetch`: prefetch a prefix KV from central store into local CPU tier

Lease renewal:

- Background thread heartbeats the worker and refreshes leases for all locally
  cached prefixes every `max(1, lease_ttl/3)` seconds.

### Router Service

File: `controller/http_service.py`

Endpoint:

- `POST /process`: computes `prefix_hash` using tokenizer, queries metadata for
  live replicas, and forwards to a worker. If none are live, chooses a worker
  at random (rehydration happens at worker).

## Expected Fault-Tolerance Behavior

- If a worker crashes, its heartbeat TTL expires.
- Its replica leases expire.
- The router stops routing to it.
- Requests for those prefixes will route elsewhere; the new worker will fetch
  from the central KV blob store and then claim a replica lease.

## Recommended First Deployment

- Redis reachable by all nodes.
- Central KV store = Redis (simplest for multi-node).
- Lease TTL = 30 seconds.

