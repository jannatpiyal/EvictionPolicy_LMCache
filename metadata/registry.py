from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: str
    address: str


class MetadataRegistry(ABC):
    """
    Fault-tolerance / coordination metadata registry.

    Stores:
    - worker liveness (heartbeat with TTL)
    - replica leases for prefix_hash (prefix replica with TTL)
    - worker addresses for routing/replication
    """

    @abstractmethod
    def register_worker(self, worker_id: str, address: str, ttl_s: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def heartbeat_worker(self, worker_id: str, ttl_s: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def claim_replica(self, prefix_hash: str, worker_id: str, ttl_s: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_live_replicas(self, prefix_hash: str) -> list[WorkerInfo]:
        raise NotImplementedError


class RedisMetadataRegistry(MetadataRegistry):
    """
    Redis implementation using simple keys + TTL.

    Key schema (prefix = lmcache:meta by default):
    - <p>:worker_addr:<worker_id> = "<host:port or node id>"
    - <p>:worker_hb:<worker_id>  (string, TTL) indicates liveness
    - <p>:replicas:<prefix_hash> (set) all workers that ever claimed replica
    - <p>:lease:<prefix_hash>:<worker_id> (string, TTL) indicates replica lease
    """

    def __init__(self, redis_url: str, key_prefix: str = "lmcache:meta:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._client = None

    def _redis(self):
        if self._client is None:
            try:
                import redis  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "RedisMetadataRegistry requires `redis` package. Install with `pip install redis`."
                ) from e
            self._client = redis.Redis.from_url(self.redis_url)
        return self._client

    def _k(self, suffix: str) -> str:
        return f"{self.key_prefix}{suffix}"

    def register_worker(self, worker_id: str, address: str, ttl_s: int) -> None:
        r = self._redis()
        r.set(self._k(f"worker_addr:{worker_id}"), address)
        self.heartbeat_worker(worker_id, ttl_s=ttl_s)

    def heartbeat_worker(self, worker_id: str, ttl_s: int) -> None:
        # Store any value; TTL is the liveness signal.
        self._redis().set(self._k(f"worker_hb:{worker_id}"), "1", ex=int(ttl_s))

    def claim_replica(self, prefix_hash: str, worker_id: str, ttl_s: int) -> None:
        r = self._redis()
        r.sadd(self._k(f"replicas:{prefix_hash}"), worker_id)
        r.set(self._k(f"lease:{prefix_hash}:{worker_id}"), "1", ex=int(ttl_s))

    def list_live_replicas(self, prefix_hash: str) -> list[WorkerInfo]:
        r = self._redis()
        replica_set_key = self._k(f"replicas:{prefix_hash}")
        worker_ids = [wid.decode("utf-8") for wid in r.smembers(replica_set_key)]
        if not worker_ids:
            return []

        live: list[WorkerInfo] = []
        for wid in worker_ids:
            # Replica must have both an unexpired lease and a live worker heartbeat.
            has_lease = r.exists(self._k(f"lease:{prefix_hash}:{wid}")) == 1
            is_alive = r.exists(self._k(f"worker_hb:{wid}")) == 1
            if not (has_lease and is_alive):
                continue
            addr = r.get(self._k(f"worker_addr:{wid}"))
            if addr is None:
                continue
            live.append(WorkerInfo(worker_id=wid, address=addr.decode("utf-8")))
        return live


class InMemoryMetadataRegistry(MetadataRegistry):
    """
    Dependency-free registry used for tests and local smoke checks.
    """

    def __init__(self):
        self._worker_addr: dict[str, str] = {}
        self._worker_hb_exp: dict[str, float] = {}
        self._replicas: dict[str, set[str]] = {}
        self._lease_exp: dict[tuple[str, str], float] = {}

    def register_worker(self, worker_id: str, address: str, ttl_s: int) -> None:
        self._worker_addr[worker_id] = address
        self.heartbeat_worker(worker_id, ttl_s=ttl_s)

    def heartbeat_worker(self, worker_id: str, ttl_s: int) -> None:
        self._worker_hb_exp[worker_id] = time.time() + float(ttl_s)

    def claim_replica(self, prefix_hash: str, worker_id: str, ttl_s: int) -> None:
        self._replicas.setdefault(prefix_hash, set()).add(worker_id)
        self._lease_exp[(prefix_hash, worker_id)] = time.time() + float(ttl_s)

    def list_live_replicas(self, prefix_hash: str) -> list[WorkerInfo]:
        now = time.time()
        worker_ids = list(self._replicas.get(prefix_hash, set()))
        live: list[WorkerInfo] = []
        for wid in worker_ids:
            if self._lease_exp.get((prefix_hash, wid), 0.0) <= now:
                continue
            if self._worker_hb_exp.get(wid, 0.0) <= now:
                continue
            addr = self._worker_addr.get(wid)
            if not addr:
                continue
            live.append(WorkerInfo(worker_id=wid, address=addr))
        return live
