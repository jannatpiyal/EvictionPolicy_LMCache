from __future__ import annotations

import io
import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentralKVRecord:
    prefix_hash: str
    kv_tuple: tuple
    stored_at: float


class CentralKVStore(ABC):
    """
    Minimal interface for a shared KV store.

    Values are stored as serialized torch objects (typically a tuple of per-layer
    (K,V) tensors on CPU).
    """

    @abstractmethod
    def get(self, prefix_hash: str) -> Optional[CentralKVRecord]:
        raise NotImplementedError

    @abstractmethod
    def put(self, prefix_hash: str, kv_tuple: tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, prefix_hash: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, prefix_hash: str) -> None:
        raise NotImplementedError


class FileSystemCentralKVStore(CentralKVStore):
    """
    Simple "object store" backed by a shared filesystem directory.

    Stores:
      - <root>/<prefix_hash>.pt (torch.save payload)
      - <root>/<prefix_hash>.json (small metadata)
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _pt_path(self, prefix_hash: str) -> str:
        return os.path.join(self.root_dir, f"{prefix_hash}.pt")

    def _meta_path(self, prefix_hash: str) -> str:
        return os.path.join(self.root_dir, f"{prefix_hash}.json")

    def get(self, prefix_hash: str) -> Optional[CentralKVRecord]:
        pt_path = self._pt_path(prefix_hash)
        if not os.path.exists(pt_path):
            return None

        # Late import so non-GPU dev environments can still import the package.
        import torch  # type: ignore

        try:
            kv_tuple = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Central store load failed for %s: %s", prefix_hash, e)
            return None

        stored_at = time.time()
        meta_path = self._meta_path(prefix_hash)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                stored_at = float(meta.get("stored_at", stored_at))
            except Exception:
                pass

        return CentralKVRecord(prefix_hash=prefix_hash, kv_tuple=kv_tuple, stored_at=stored_at)

    def put(self, prefix_hash: str, kv_tuple: tuple) -> None:
        import torch  # type: ignore

        pt_path = self._pt_path(prefix_hash)
        meta_path = self._meta_path(prefix_hash)

        # Atomic-ish write: write temp then rename.
        tmp_path = pt_path + ".tmp"
        torch.save(kv_tuple, tmp_path)
        os.replace(tmp_path, pt_path)

        meta = {"prefix_hash": prefix_hash, "stored_at": time.time()}
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_meta, meta_path)

    def contains(self, prefix_hash: str) -> bool:
        return os.path.exists(self._pt_path(prefix_hash))

    def delete(self, prefix_hash: str) -> None:
        for path in (self._pt_path(prefix_hash), self._meta_path(prefix_hash)):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


class RedisCentralKVStore(CentralKVStore):
    """
    Redis-backed central KV store.

    Requires `redis` Python package at runtime. Payload is `torch.save` bytes.
    """

    def __init__(self, redis_url: str, key_prefix: str = "lmcache:kv:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._client = None

    def _redis(self):
        if self._client is None:
            try:
                import redis  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "RedisCentralKVStore requires `redis` package. Install with `pip install redis`."
                ) from e
            self._client = redis.Redis.from_url(self.redis_url)
        return self._client

    def _key(self, prefix_hash: str) -> str:
        return f"{self.key_prefix}{prefix_hash}"

    def get(self, prefix_hash: str) -> Optional[CentralKVRecord]:
        import torch  # type: ignore

        data = self._redis().get(self._key(prefix_hash))
        if data is None:
            return None

        try:
            buf = io.BytesIO(data)
            kv_tuple = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Redis central store decode failed for %s: %s", prefix_hash, e)
            return None
        return CentralKVRecord(prefix_hash=prefix_hash, kv_tuple=kv_tuple, stored_at=time.time())

    def put(self, prefix_hash: str, kv_tuple: tuple) -> None:
        import torch  # type: ignore

        buf = io.BytesIO()
        torch.save(kv_tuple, buf)
        self._redis().set(self._key(prefix_hash), buf.getvalue())

    def contains(self, prefix_hash: str) -> bool:
        return self._redis().exists(self._key(prefix_hash)) == 1

    def delete(self, prefix_hash: str) -> None:
        self._redis().delete(self._key(prefix_hash))

