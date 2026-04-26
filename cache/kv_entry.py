"""
KVEntry: Holds real KV cache tensors with actual GPU/CPU/Disk transfers.

Each entry stores the KV state for one reusable prefix. GPU execution still
materializes full `past_key_values`, but CPU/shared-store payloads may be kept
as fixed-token chunks so workers do not have to deserialize one giant blob per
prefix.
"""

import os
import time
import hashlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Any

import torch

logger = logging.getLogger(__name__)

_DISK_IO_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 4)))
_TRANSFER_BATCH_SIZE = 4


@dataclass
class KVEntry:
    """
    A KV cache entry holding real tensors.

    The past_key_values is a tuple of (key_tensor, value_tensor) per layer,
    extracted from model(..., use_cache=True).past_key_values.
    """

    # --- Identity ---
    prefix_hash: str
    prefix_tokens: list[int]
    prompt_text: str = ""
    num_tokens: int = 0
    parent_prefix_hash: Optional[str] = None
    chunk_index: int = 0
    chunk_count: int = 1

    # --- Real tensors ---
    # When on GPU: tensors live on CUDA
    # When on CPU: tensors live on CPU RAM
    # When on disk: tensors are None, saved to disk_path
    past_key_values: Optional[tuple] = None
    kv_chunks: Optional[list[tuple]] = None
    disk_path: Optional[str] = None
    chunk_size_tokens: int = 0
    mapped_regions: list[Any] = field(default_factory=list, repr=False, compare=False)
    enable_layerwise_pipeline: bool = False
    pipeline_stage_layers: int = 1

    # --- Size ---
    size_bytes: int = 0

    # --- Location ---
    tier: str = "gpu"  # "gpu", "cpu", "disk"
    worker_id: int = 0

    # --- Access metadata ---
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_reuse_gap: float = 0.0
    last_hit_tier: Optional[str] = None  # "gpu" | "cpu" | "disk"

    # --- Semantic metadata ---
    embedding: Optional[list] = None

    # --- Transfer timing ---
    last_transfer_ms: float = 0.0
    _disk_load_future: Optional[Future] = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def compute_prefix_hash(tokens: list[int]) -> str:
        token_bytes = b"".join(t.to_bytes(4, "big") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def cache_key(self) -> str:
        if self.parent_prefix_hash is None:
            return self.prefix_hash
        return f"{self.parent_prefix_hash}::chunk:{self.chunk_index}"

    def root_prefix_hash(self) -> str:
        return self.parent_prefix_hash or self.prefix_hash

    @staticmethod
    def _iter_kv_layers(past_key_values):
        """
        Normalize different Hugging Face cache containers into an iterator of
        `(key, value)` pairs.

        Newer transformers versions may return `DynamicCache` instead of the
        older tuple-of-tuples format. Prefer the official legacy conversion path
        when it exists, then fall back to common cache internals.
        """
        if past_key_values is None:
            return ()

        if isinstance(past_key_values, tuple):
            return past_key_values

        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if callable(to_legacy):
            legacy = to_legacy()
            if isinstance(legacy, tuple):
                return legacy

        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return tuple(zip(key_cache, value_cache))

        layers = getattr(past_key_values, "layers", None)
        if layers is not None:
            normalized = []
            for layer in layers:
                if isinstance(layer, tuple) and len(layer) >= 2:
                    normalized.append((layer[0], layer[1]))
                    continue
                layer_keys = getattr(layer, "keys", None)
                layer_values = getattr(layer, "values", None)
                if layer_keys is not None and layer_values is not None:
                    normalized.append((layer_keys, layer_values))
                    continue
                raise TypeError(f"Unsupported cache layer type: {type(layer)}")
            return tuple(normalized)

        try:
            normalized = []
            for layer in past_key_values:
                if isinstance(layer, tuple) and len(layer) >= 2:
                    normalized.append((layer[0], layer[1]))
                else:
                    raise TypeError(f"Unsupported cache entry type: {type(layer)}")
            return tuple(normalized)
        except TypeError:
            raise
        except Exception as e:
            raise TypeError(f"Unsupported cache container type: {type(past_key_values)}") from e

    @staticmethod
    def _to_tuple(past_key_values, clone: bool = True) -> tuple:
        """Convert past_key_values to tuple format, whether DynamicCache or tuple."""
        try:
            layers = KVEntry._iter_kv_layers(past_key_values)
            return tuple(
                (k.clone(), v.clone()) if clone else (k, v)
                for k, v in layers
            )
        except Exception as e:
            logger.warning(f"Cannot convert past_key_values of type {type(past_key_values)}: {e}")
            raise

    @staticmethod
    def _to_cache(kv_tuple, device=None):
        """Convert tuple of (key, value) back to DynamicCache for model input."""
        try:
            from transformers import DynamicCache
            cache = DynamicCache()
            for key, value in kv_tuple:
                if device is not None:
                    key = key.to(device)
                    value = value.to(device)
                cache.update(key, value, layer_idx=len(cache))
            return cache
        except ImportError:
            # Old transformers without DynamicCache — return tuple directly
            if device is not None:
                return tuple((k.to(device), v.to(device)) for k, v in kv_tuple)
            return kv_tuple

    @staticmethod
    def measure_kv_size(past_key_values) -> int:
        """Measure actual byte size of KV tensors. Handles both Cache and chunk lists."""
        total = 0
        if isinstance(past_key_values, list):
            for chunk in past_key_values:
                total += KVEntry.measure_kv_size(chunk)
            return total
        for key, value in KVEntry._iter_kv_layers(past_key_values):
            total += key.nelement() * key.element_size()
            total += value.nelement() * value.element_size()
        return total

    @staticmethod
    def clone_kv(past_key_values) -> tuple:
        """Deep clone KV tensors into tuple format for storage."""
        return KVEntry._to_tuple(past_key_values, clone=True)

    @staticmethod
    def capture_kv(past_key_values) -> tuple:
        """Capture model KV without eagerly deep-cloning it."""
        return KVEntry._to_tuple(past_key_values, clone=False)

    @staticmethod
    def _sequence_dim(tensor: torch.Tensor) -> int:
        return tensor.ndim - 2

    @staticmethod
    def split_kv_into_chunks(kv_tuple: tuple, chunk_size_tokens: int) -> list[tuple]:
        if not kv_tuple or chunk_size_tokens <= 0:
            return [kv_tuple]

        seq_len = kv_tuple[0][0].shape[KVEntry._sequence_dim(kv_tuple[0][0])]
        if seq_len <= chunk_size_tokens:
            return [kv_tuple]

        chunks: list[tuple] = []
        for start in range(0, seq_len, chunk_size_tokens):
            end = min(start + chunk_size_tokens, seq_len)
            chunk_layers = []
            for key, value in kv_tuple:
                dim = KVEntry._sequence_dim(key)
                chunk_layers.append((
                    key.narrow(dim, start, end - start),
                    value.narrow(dim, start, end - start),
                ))
            chunks.append(tuple(chunk_layers))
        return chunks

    @staticmethod
    def merge_kv_chunks(kv_chunks: list[tuple], device: Optional[str] = None) -> tuple:
        if not kv_chunks:
            return ()
        if len(kv_chunks) == 1:
            merged = kv_chunks[0]
            if device is None:
                return merged
            return tuple((k.to(device), v.to(device)) for k, v in merged)

        num_layers = len(kv_chunks[0])
        merged_layers = []
        for layer_idx in range(num_layers):
            keys = []
            values = []
            for chunk in kv_chunks:
                key, value = chunk[layer_idx]
                if device is not None:
                    key = key.to(device, non_blocking=True)
                    value = value.to(device, non_blocking=True)
                keys.append(key)
                values.append(value)
            seq_dim = KVEntry._sequence_dim(keys[0])
            merged_layers.append((
                torch.cat(keys, dim=seq_dim),
                torch.cat(values, dim=seq_dim),
            ))
        return tuple(merged_layers)

    def uses_chunk_storage(self) -> bool:
        return bool(self.kv_chunks) and self.past_key_values is None

    def split_for_cache(self) -> list["KVEntry"]:
        if self.kv_chunks is not None:
            chunks = self.kv_chunks
        elif self.past_key_values is not None and self.chunk_size_tokens > 0:
            chunks = KVEntry.split_kv_into_chunks(self.past_key_values, self.chunk_size_tokens)
        else:
            return [self]

        if len(chunks) <= 1:
            return [self]

        root_prefix_hash = self.root_prefix_hash()
        entries: list[KVEntry] = []
        for idx, chunk in enumerate(chunks):
            entries.append(
                KVEntry(
                    prefix_hash=f"{root_prefix_hash}::chunk:{idx}",
                    prefix_tokens=self.prefix_tokens,
                    prompt_text=self.prompt_text,
                    num_tokens=self.num_tokens,
                    parent_prefix_hash=root_prefix_hash,
                    chunk_index=idx,
                    chunk_count=len(chunks),
                    past_key_values=chunk if self.past_key_values is not None else None,
                    kv_chunks=[chunk] if self.kv_chunks is not None else None,
                    size_bytes=KVEntry.measure_kv_size(chunk),
                    disk_path=None,
                    chunk_size_tokens=self.chunk_size_tokens,
                    tier=self.tier,
                    worker_id=self.worker_id,
                    embedding=self.embedding,
                    enable_layerwise_pipeline=self.enable_layerwise_pipeline,
                    pipeline_stage_layers=self.pipeline_stage_layers,
                )
            )
        return entries

    def clone_to_cpu_duplicate(self) -> "KVEntry":
        if self.past_key_values is None:
            raise ValueError("Cannot duplicate KVEntry without resident past_key_values")

        if any(k.is_cuda or v.is_cuda for k, v in self.past_key_values):
            if self.enable_layerwise_pipeline:
                cpu_kv = self._layerwise_gpu_to_cpu_pipeline(
                    self.past_key_values,
                    self.pipeline_stage_layers,
                )
            else:
                cpu_kv = self._batched_gpu_to_cpu(self.past_key_values)
        else:
            cpu_kv = tuple((k.clone(), v.clone()) for k, v in self.past_key_values)

        duplicate = KVEntry(
            prefix_hash=self.prefix_hash,
            prefix_tokens=self.prefix_tokens,
            prompt_text=self.prompt_text,
            num_tokens=self.num_tokens,
            parent_prefix_hash=self.parent_prefix_hash,
            chunk_index=self.chunk_index,
            chunk_count=self.chunk_count,
            past_key_values=cpu_kv,
            size_bytes=self.size_bytes,
            tier="cpu",
            worker_id=self.worker_id,
            chunk_size_tokens=self.chunk_size_tokens,
            embedding=self.embedding,
            enable_layerwise_pipeline=self.enable_layerwise_pipeline,
            pipeline_stage_layers=self.pipeline_stage_layers,
        )
        duplicate.created_at = self.created_at
        duplicate.last_accessed_at = self.last_accessed_at
        duplicate.access_count = self.access_count
        duplicate.last_reuse_gap = self.last_reuse_gap
        duplicate.last_hit_tier = self.last_hit_tier
        return duplicate

    def _release_mapped_regions(self) -> None:
        for region in self.mapped_regions:
            try:
                region.close()
            except Exception:
                pass
        self.mapped_regions.clear()

    @staticmethod
    def _layer_batches(kv_tuple: tuple, batch_size: int = _TRANSFER_BATCH_SIZE):
        for i in range(0, len(kv_tuple), batch_size):
            yield kv_tuple[i:i + batch_size]

    @staticmethod
    def _layerwise_cpu_to_gpu_pipeline(kv_tuple: tuple, device: str, stage_layers: int) -> tuple:
        if not kv_tuple:
            return kv_tuple
        if not torch.cuda.is_available():
            return tuple((k.to(device), v.to(device)) for k, v in kv_tuple)

        stage_layers = max(1, int(stage_layers))
        streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
        pending: list[tuple[torch.cuda.Stream, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        moved_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for idx, batch in enumerate(KVEntry._layer_batches(kv_tuple, stage_layers)):
            stream = streams[idx % len(streams)]
            moved_batch: list[tuple[torch.Tensor, torch.Tensor]] = []
            with torch.cuda.stream(stream):
                for key, value in batch:
                    key_src = key.pin_memory() if key.device.type == "cpu" and not key.is_pinned() else key
                    value_src = value.pin_memory() if value.device.type == "cpu" and not value.is_pinned() else value
                    moved_batch.append((
                        key_src.to(device, non_blocking=True),
                        value_src.to(device, non_blocking=True),
                    ))
            pending.append((stream, moved_batch))
            if len(pending) > 1:
                prev_stream, prev_batch = pending.pop(0)
                prev_stream.synchronize()
                moved_layers.extend(prev_batch)

        for stream, moved_batch in pending:
            stream.synchronize()
            moved_layers.extend(moved_batch)
        return tuple(moved_layers)

    @staticmethod
    def _layerwise_gpu_to_cpu_pipeline(kv_tuple: tuple, stage_layers: int) -> tuple:
        if not kv_tuple:
            return kv_tuple
        if not torch.cuda.is_available() or not any(k.is_cuda or v.is_cuda for k, v in kv_tuple):
            return tuple((k.to("cpu"), v.to("cpu")) for k, v in kv_tuple)

        stage_layers = max(1, int(stage_layers))
        stream_device = next(
            (tensor.device for key, value in kv_tuple for tensor in (key, value) if tensor.is_cuda),
            torch.device("cuda:0"),
        )
        streams = [torch.cuda.Stream(device=stream_device), torch.cuda.Stream(device=stream_device)]
        pending: list[tuple[torch.cuda.Stream, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        moved_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for idx, batch in enumerate(KVEntry._layer_batches(kv_tuple, stage_layers)):
            stream = streams[idx % len(streams)]
            moved_batch: list[tuple[torch.Tensor, torch.Tensor]] = []
            with torch.cuda.stream(stream):
                for key, value in batch:
                    cpu_key = torch.empty_like(key, device="cpu", pin_memory=True)
                    cpu_value = torch.empty_like(value, device="cpu", pin_memory=True)
                    cpu_key.copy_(key, non_blocking=True)
                    cpu_value.copy_(value, non_blocking=True)
                    moved_batch.append((cpu_key, cpu_value))
            pending.append((stream, moved_batch))
            if len(pending) > 1:
                prev_stream, prev_batch = pending.pop(0)
                prev_stream.synchronize()
                moved_layers.extend(prev_batch)

        for stream, moved_batch in pending:
            stream.synchronize()
            moved_layers.extend(moved_batch)
        return tuple(moved_layers)

    @staticmethod
    def _batched_cpu_to_gpu(kv_tuple: tuple, device: str) -> tuple:
        """
        Move KV tensors from CPU to GPU in batches.

        Uses pinned host memory plus a dedicated CUDA stream so we only
        synchronize once per batched transfer rather than once per layer.
        """
        if not kv_tuple:
            return kv_tuple

        if not torch.cuda.is_available():
            return tuple((k.to(device), v.to(device)) for k, v in kv_tuple)

        stream = torch.cuda.Stream(device=device)
        moved_layers = []
        with torch.cuda.stream(stream):
            for batch in KVEntry._layer_batches(kv_tuple):
                for key, value in batch:
                    key_src = key.pin_memory() if not key.is_pinned() else key
                    value_src = value.pin_memory() if not value.is_pinned() else value
                    moved_layers.append((
                        key_src.to(device, non_blocking=True),
                        value_src.to(device, non_blocking=True),
                    ))
        stream.synchronize()
        return tuple(moved_layers)

    @staticmethod
    def _batched_gpu_to_cpu(kv_tuple: tuple) -> tuple:
        """
        Move KV tensors from GPU to CPU in batches using pinned host buffers.
        """
        if not kv_tuple:
            return kv_tuple

        if not torch.cuda.is_available() or not any(k.is_cuda or v.is_cuda for k, v in kv_tuple):
            return tuple((k.to("cpu"), v.to("cpu")) for k, v in kv_tuple)

        stream_device = next(
            (tensor.device for key, value in kv_tuple for tensor in (key, value) if tensor.is_cuda),
            torch.device("cuda:0"),
        )
        stream = torch.cuda.Stream(device=stream_device)
        moved_layers = []
        with torch.cuda.stream(stream):
            for batch in KVEntry._layer_batches(kv_tuple):
                for key, value in batch:
                    cpu_key = torch.empty_like(key, device="cpu", pin_memory=True)
                    cpu_value = torch.empty_like(value, device="cpu", pin_memory=True)
                    cpu_key.copy_(key, non_blocking=True)
                    cpu_value.copy_(value, non_blocking=True)
                    moved_layers.append((cpu_key, cpu_value))
        stream.synchronize()
        return tuple(moved_layers)

    def start_disk_prefetch(self) -> Optional[Future]:
        """
        Start loading a disk-resident KV entry into CPU memory in the background.

        This lets the cache overlap disk I/O with the promotion bookkeeping that
        happens before the tensors are needed on GPU.
        """
        if self.tier != "disk" or self.past_key_values is not None:
            return None
        if not self.disk_path or not os.path.exists(self.disk_path):
            return None
        if self._disk_load_future is not None and not self._disk_load_future.done():
            return self._disk_load_future

        path = self.disk_path

        def _load():
            return torch.load(path, map_location="cpu", weights_only=False)

        self._disk_load_future = _DISK_IO_EXECUTOR.submit(_load)
        return self._disk_load_future

    def record_access(self) -> None:
        now = time.time()
        self.last_reuse_gap = now - self.last_accessed_at
        self.last_accessed_at = now
        self.access_count += 1

    def time_since_last_access(self) -> float:
        return time.time() - self.last_accessed_at

    def age(self) -> float:
        return time.time() - self.created_at

    # ==========================================================
    # Real tier transfer methods
    # ==========================================================

    def move_to_gpu(self, device: str = "cuda") -> float:
        """
        Move KV tensors to GPU. Returns transfer time in ms.
        Loads from disk if currently on disk.
        """
        start = time.perf_counter()

        if self.tier == "gpu":
            return 0.0

        if self.tier == "disk":
            self.start_disk_prefetch()
            self._load_from_disk()

        if self.kv_chunks is not None and self.past_key_values is None:
            mover = self._layerwise_cpu_to_gpu_pipeline if self.enable_layerwise_pipeline else self._batched_cpu_to_gpu
            gpu_chunks = [mover(chunk, device, self.pipeline_stage_layers) if self.enable_layerwise_pipeline else mover(chunk, device) for chunk in self.kv_chunks]
            self.past_key_values = self.merge_kv_chunks(gpu_chunks)
            self.kv_chunks = None
            self._release_mapped_regions()
        elif self.past_key_values is not None:
            # Transfer CPU -> GPU
            if self.enable_layerwise_pipeline:
                self.past_key_values = self._layerwise_cpu_to_gpu_pipeline(
                    self.past_key_values,
                    device,
                    self.pipeline_stage_layers,
                )
            else:
                self.past_key_values = self._batched_cpu_to_gpu(self.past_key_values, device)

        self.tier = "gpu"
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_transfer_ms = elapsed_ms
        logger.debug(f"  [{self.prefix_hash[:8]}] -> GPU in {elapsed_ms:.2f}ms ({self.size_bytes / 1024:.0f}KB)")
        return elapsed_ms

    def move_to_cpu(self) -> float:
        """
        Move KV tensors to CPU RAM. Returns transfer time in ms.
        """
        start = time.perf_counter()

        if self.tier == "cpu":
            return 0.0

        if self.tier == "disk":
            self._load_from_disk()
        elif self.tier == "gpu":
            # GPU -> CPU
            if self.past_key_values is not None:
                if self.enable_layerwise_pipeline:
                    self.past_key_values = self._layerwise_gpu_to_cpu_pipeline(
                        self.past_key_values,
                        self.pipeline_stage_layers,
                    )
                else:
                    self.past_key_values = self._batched_gpu_to_cpu(self.past_key_values)

        self.tier = "cpu"
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_transfer_ms = elapsed_ms
        logger.debug(f"  [{self.prefix_hash[:8]}] -> CPU in {elapsed_ms:.2f}ms ({self.size_bytes / 1024:.0f}KB)")
        return elapsed_ms

    def move_to_disk(self, disk_dir: str = "/tmp/kv_cache") -> float:
        """
        Serialize KV tensors to disk. Frees GPU/CPU memory.
        Returns transfer time in ms.
        """
        start = time.perf_counter()

        if self.tier == "disk":
            return 0.0

        os.makedirs(disk_dir, exist_ok=True)
        self.disk_path = os.path.join(disk_dir, f"{self.prefix_hash}.pt")

        if self.past_key_values is not None:
            # Move to CPU first if on GPU
            if any(k.is_cuda or v.is_cuda for k, v in self.past_key_values):
                if self.enable_layerwise_pipeline:
                    cpu_kv = self._layerwise_gpu_to_cpu_pipeline(
                        self.past_key_values,
                        self.pipeline_stage_layers,
                    )
                else:
                    cpu_kv = self._batched_gpu_to_cpu(self.past_key_values)
            else:
                cpu_kv = self.past_key_values
            # Save to disk
            payload = {
                "past_key_values": cpu_kv,
                "kv_chunks": self.kv_chunks,
                "chunk_size_tokens": self.chunk_size_tokens,
            }
            torch.save(payload, self.disk_path)
            # Free memory
            del self.past_key_values
            del cpu_kv
            self.past_key_values = None
            self.kv_chunks = None
            self._release_mapped_regions()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tier = "disk"
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_transfer_ms = elapsed_ms
        logger.debug(f"  [{self.prefix_hash[:8]}] -> DISK in {elapsed_ms:.2f}ms ({self.size_bytes / 1024:.0f}KB)")
        return elapsed_ms

    def _load_from_disk(self) -> None:
        """Load KV tensors from disk into CPU memory."""
        if self.past_key_values is not None:
            return

        if self._disk_load_future is not None:
            try:
                self.past_key_values = self._disk_load_future.result()
            except Exception as e:
                logger.warning("Disk prefetch failed for %s: %s", self.prefix_hash, e)
                self.past_key_values = None
            finally:
                self._disk_load_future = None
            return

        if self.disk_path and os.path.exists(self.disk_path):
            payload = torch.load(self.disk_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                self.past_key_values = payload.get("past_key_values")
                self.kv_chunks = payload.get("kv_chunks")
                self.chunk_size_tokens = int(payload.get("chunk_size_tokens", self.chunk_size_tokens or 0))
            else:
                self.past_key_values = payload
        else:
            logger.warning(f"Disk file missing for {self.prefix_hash}: {self.disk_path}")
            self.past_key_values = None

    def delete_from_disk(self) -> None:
        """Remove disk file if it exists."""
        if self.disk_path and os.path.exists(self.disk_path):
            os.remove(self.disk_path)
            self.disk_path = None

    def free_memory(self) -> None:
        """Free all memory held by this entry."""
        if self._disk_load_future is not None:
            try:
                self._disk_load_future.cancel()
            except Exception:
                pass
            self._disk_load_future = None
        if self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None
        if self.kv_chunks is not None:
            del self.kv_chunks
            self.kv_chunks = None
        self._release_mapped_regions()
        self.delete_from_disk()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_kv_on_device(self, device: str = "cuda"):
        """
        Get past_key_values as a DynamicCache on the requested device.
        Loads from disk if needed. Does NOT change self.tier.
        """
        if self.past_key_values is None and self.kv_chunks is None:
            if self.tier == "disk":
                self.start_disk_prefetch()
                self._load_from_disk()
            if self.past_key_values is None and self.kv_chunks is None:
                return None

        if self.past_key_values is None and self.kv_chunks is not None:
            mover = self._layerwise_cpu_to_gpu_pipeline if self.enable_layerwise_pipeline else self._batched_cpu_to_gpu
            gpu_chunks = [mover(chunk, device, self.pipeline_stage_layers) if self.enable_layerwise_pipeline else mover(chunk, device) for chunk in self.kv_chunks]
            self.past_key_values = self.merge_kv_chunks(gpu_chunks)
            self.kv_chunks = None
            self._release_mapped_regions()

        # Convert stored tuple to DynamicCache on the target device
        return KVEntry._to_cache(self.past_key_values, device=device)

    def __repr__(self) -> str:
        return (
            f"KVEntry(hash={self.prefix_hash[:8]}..., tokens={self.num_tokens}, "
            f"tier={self.tier}, accesses={self.access_count}, "
            f"size={self.size_bytes / 1024:.1f}KB)"
        )
