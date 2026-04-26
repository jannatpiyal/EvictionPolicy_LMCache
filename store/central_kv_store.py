from __future__ import annotations

import io
import mmap
import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from cache.kv_entry import KVEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentralKVRecord:
    prefix_hash: str
    kv_tuple: Optional[tuple]
    kv_chunks: Optional[list[tuple]]
    stored_at: float
    chunk_size_tokens: int = 0
    mapped_regions: tuple[Any, ...] = ()


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
    def put(self, prefix_hash: str, kv_tuple: tuple, chunk_size_tokens: int = 0) -> None:
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
      - <root>/<prefix_hash>/manifest.json
      - <root>/<prefix_hash>/chunk_XXXX_layer_YY_[kv].bin

    Each prefix is stored as token chunks. Chunk payloads are raw tensor bytes so
    a worker can memory-map them and build CPU tensor views without an extra
    deserialize-and-copy step.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    _DTYPE_TO_NAME = {
        "torch.float16": "float16",
        "torch.float32": "float32",
        "torch.bfloat16": "bfloat16",
        "torch.int64": "int64",
        "torch.int32": "int32",
    }
    _NAME_TO_DTYPE = {
        "float16": None,
        "float32": None,
        "bfloat16": None,
        "int64": None,
        "int32": None,
    }

    def _prefix_dir(self, prefix_hash: str) -> str:
        return os.path.join(self.root_dir, prefix_hash)

    def _manifest_path(self, prefix_hash: str) -> str:
        return os.path.join(self._prefix_dir(prefix_hash), "manifest.json")

    def _chunk_tensor_path(self, prefix_hash: str, chunk_idx: int, layer_idx: int, tensor_kind: str) -> str:
        return os.path.join(
            self._prefix_dir(prefix_hash),
            f"chunk_{chunk_idx:04d}_layer_{layer_idx:03d}_{tensor_kind}.bin",
        )

    @classmethod
    def _dtype_name(cls, dtype) -> str:
        if cls._NAME_TO_DTYPE["float16"] is None:
            import torch  # type: ignore
            cls._NAME_TO_DTYPE["float16"] = torch.float16
            cls._NAME_TO_DTYPE["float32"] = torch.float32
            cls._NAME_TO_DTYPE["bfloat16"] = torch.bfloat16
            cls._NAME_TO_DTYPE["int64"] = torch.int64
            cls._NAME_TO_DTYPE["int32"] = torch.int32
        return cls._DTYPE_TO_NAME[str(dtype)]

    @classmethod
    def _dtype_from_name(cls, name: str):
        if cls._NAME_TO_DTYPE["float16"] is None:
            import torch  # type: ignore
            cls._NAME_TO_DTYPE["float16"] = torch.float16
            cls._NAME_TO_DTYPE["float32"] = torch.float32
            cls._NAME_TO_DTYPE["bfloat16"] = torch.bfloat16
            cls._NAME_TO_DTYPE["int64"] = torch.int64
            cls._NAME_TO_DTYPE["int32"] = torch.int32
        return cls._NAME_TO_DTYPE[name]

    def get(self, prefix_hash: str) -> Optional[CentralKVRecord]:
        manifest_path = self._manifest_path(prefix_hash)
        if not os.path.exists(manifest_path):
            return None

        # Late import so non-GPU dev environments can still import the package.
        import torch  # type: ignore

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("Central store manifest load failed for %s: %s", prefix_hash, e)
            return None

        chunk_size_tokens = int(manifest.get("chunk_size_tokens", 0))
        chunk_count = int(manifest.get("chunk_count", 0))
        chunk_manifests = manifest.get("chunks", [])
        mapped_regions: list[Any] = []
        kv_chunks: list[tuple] = []

        try:
            for chunk_idx in range(chunk_count):
                chunk_layers = []
                for layer_meta in chunk_manifests[chunk_idx]:
                    key_path = self._chunk_tensor_path(prefix_hash, chunk_idx, int(layer_meta["layer_idx"]), "k")
                    value_path = self._chunk_tensor_path(prefix_hash, chunk_idx, int(layer_meta["layer_idx"]), "v")
                    key_region = self._map_tensor_file(key_path)
                    value_region = self._map_tensor_file(value_path)
                    mapped_regions.extend([key_region["mmap"], value_region["mmap"]])
                    key = torch.frombuffer(
                        key_region["mmap"],
                        dtype=self._dtype_from_name(layer_meta["key_dtype"]),
                    ).reshape(layer_meta["key_shape"])
                    value = torch.frombuffer(
                        value_region["mmap"],
                        dtype=self._dtype_from_name(layer_meta["value_dtype"]),
                    ).reshape(layer_meta["value_shape"])
                    chunk_layers.append((key, value))
                kv_chunks.append(tuple(chunk_layers))
        except Exception as e:
            logger.warning("Central store chunk mmap failed for %s: %s", prefix_hash, e)
            for region in mapped_regions:
                try:
                    region.close()
                except Exception:
                    pass
            return None

        return CentralKVRecord(
            prefix_hash=prefix_hash,
            kv_tuple=None,
            kv_chunks=kv_chunks,
            stored_at=float(manifest.get("stored_at", time.time())),
            chunk_size_tokens=chunk_size_tokens,
            mapped_regions=tuple(mapped_regions),
        )

    def put(self, prefix_hash: str, kv_tuple: tuple, chunk_size_tokens: int = 0) -> None:
        prefix_dir = self._prefix_dir(prefix_hash)
        os.makedirs(prefix_dir, exist_ok=True)

        chunk_size_tokens = int(chunk_size_tokens or 0)
        kv_chunks = KVEntry.split_kv_into_chunks(kv_tuple, chunk_size_tokens)
        chunk_manifests = []

        for chunk_idx, chunk in enumerate(kv_chunks):
            layer_entries = []
            for layer_idx, (key, value) in enumerate(chunk):
                key_cpu = key.detach().contiguous().to("cpu")
                value_cpu = value.detach().contiguous().to("cpu")
                key_path = self._chunk_tensor_path(prefix_hash, chunk_idx, layer_idx, "k")
                value_path = self._chunk_tensor_path(prefix_hash, chunk_idx, layer_idx, "v")
                with open(key_path + ".tmp", "wb") as f:
                    f.write(memoryview(key_cpu.numpy()))
                os.replace(key_path + ".tmp", key_path)
                with open(value_path + ".tmp", "wb") as f:
                    f.write(memoryview(value_cpu.numpy()))
                os.replace(value_path + ".tmp", value_path)
                layer_entries.append({
                    "layer_idx": layer_idx,
                    "key_shape": list(key_cpu.shape),
                    "value_shape": list(value_cpu.shape),
                    "key_dtype": self._dtype_name(key_cpu.dtype),
                    "value_dtype": self._dtype_name(value_cpu.dtype),
                })
            chunk_manifests.append(layer_entries)

        manifest = {
            "prefix_hash": prefix_hash,
            "stored_at": time.time(),
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_count": len(kv_chunks),
            "chunks": chunk_manifests,
        }
        manifest_path = self._manifest_path(prefix_hash)
        with open(manifest_path + ".tmp", "w") as f:
            json.dump(manifest, f)
        os.replace(manifest_path + ".tmp", manifest_path)

    def contains(self, prefix_hash: str) -> bool:
        return os.path.exists(self._manifest_path(prefix_hash))

    def delete(self, prefix_hash: str) -> None:
        prefix_dir = self._prefix_dir(prefix_hash)
        for path in (self._manifest_path(prefix_hash),):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        if os.path.isdir(prefix_dir):
            for filename in os.listdir(prefix_dir):
                try:
                    os.remove(os.path.join(prefix_dir, filename))
                except Exception:
                    pass
            try:
                os.rmdir(prefix_dir)
            except Exception:
                pass

    @staticmethod
    def _map_tensor_file(path: str) -> dict[str, Any]:
        fd = os.open(path, os.O_RDONLY)
        try:
            region = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        return {"mmap": region}


class RedisCentralKVStore(CentralKVStore):
    """
    Redis-backed central KV store.

    Requires `redis` Python package at runtime. Payload is `torch.save` bytes.
    """

    def __init__(self, redis_url: str, key_prefix: str = "lmcache:kv:", chunk_bytes: int = 64 * 1024 * 1024):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.chunk_bytes = int(chunk_bytes)
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

    def _meta_key(self, prefix_hash: str) -> str:
        return f"{self._key(prefix_hash)}:meta"

    def _chunk_key(self, prefix_hash: str, chunk_idx: int) -> str:
        return f"{self._key(prefix_hash)}:chunk:{chunk_idx}"

    def get(self, prefix_hash: str) -> Optional[CentralKVRecord]:
        import torch  # type: ignore

        r = self._redis()
        chunk_size_tokens = 0
        meta_raw = r.get(self._meta_key(prefix_hash))
        if meta_raw is not None:
            try:
                meta = json.loads(meta_raw)
                num_chunks = int(meta["num_chunks"])
                chunk_size_tokens = int(meta.get("chunk_size_tokens", 0))
            except Exception as e:
                logger.warning("Redis central store metadata decode failed for %s: %s", prefix_hash, e)
                return None

            chunks = []
            for idx in range(num_chunks):
                chunk = r.get(self._chunk_key(prefix_hash, idx))
                if chunk is None:
                    logger.warning("Redis central store missing chunk %d for %s", idx, prefix_hash)
                    return None
                chunks.append(chunk)
            data = b"".join(chunks)
        else:
            data = r.get(self._key(prefix_hash))
            if data is None:
                return None

        try:
            buf = io.BytesIO(data)
            kv_tuple = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Redis central store decode failed for %s: %s", prefix_hash, e)
            return None
        return CentralKVRecord(
            prefix_hash=prefix_hash,
            kv_tuple=kv_tuple,
            kv_chunks=None,
            stored_at=time.time(),
            chunk_size_tokens=chunk_size_tokens,
        )

    def put(self, prefix_hash: str, kv_tuple: tuple, chunk_size_tokens: int = 0) -> None:
        import torch  # type: ignore

        buf = io.BytesIO()
        torch.save(kv_tuple, buf)
        data = buf.getvalue()
        r = self._redis()

        if len(data) <= self.chunk_bytes:
            r.set(self._key(prefix_hash), data)
            r.delete(self._meta_key(prefix_hash))
            return

        num_chunks = (len(data) + self.chunk_bytes - 1) // self.chunk_bytes
        pipe = r.pipeline()
        pipe.delete(self._key(prefix_hash))
        for idx in range(num_chunks):
            start = idx * self.chunk_bytes
            end = start + self.chunk_bytes
            pipe.set(self._chunk_key(prefix_hash, idx), data[start:end])
        pipe.set(
            self._meta_key(prefix_hash),
            json.dumps({
                "num_chunks": num_chunks,
                "stored_at": time.time(),
                "chunk_size_tokens": int(chunk_size_tokens or 0),
            }),
        )
        pipe.execute()

    def contains(self, prefix_hash: str) -> bool:
        r = self._redis()
        return r.exists(self._key(prefix_hash)) == 1 or r.exists(self._meta_key(prefix_hash)) == 1

    def delete(self, prefix_hash: str) -> None:
        r = self._redis()
        meta_raw = r.get(self._meta_key(prefix_hash))
        keys = [self._key(prefix_hash), self._meta_key(prefix_hash)]
        if meta_raw is not None:
            try:
                meta = json.loads(meta_raw)
                num_chunks = int(meta.get("num_chunks", 0))
                keys.extend(self._chunk_key(prefix_hash, idx) for idx in range(num_chunks))
            except Exception:
                pass
        if keys:
            r.delete(*keys)
