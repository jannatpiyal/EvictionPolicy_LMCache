"""
KVEntry: Holds real KV cache tensors with actual GPU/CPU/Disk transfers.

Each entry stores the full past_key_values from a HuggingFace model forward pass.
Tensors are physically moved between GPU memory, CPU RAM, and disk.
"""

import os
import time
import hashlib
import pickle
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


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

    # --- Real tensors ---
    # When on GPU: tensors live on CUDA
    # When on CPU: tensors live on CPU RAM
    # When on disk: tensors are None, saved to disk_path
    past_key_values: Optional[tuple] = None
    disk_path: Optional[str] = None

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

    # --- Semantic metadata ---
    embedding: Optional[list] = None

    # --- Transfer timing ---
    last_transfer_ms: float = 0.0

    @staticmethod
    def compute_prefix_hash(tokens: list[int]) -> str:
        token_bytes = b"".join(t.to_bytes(4, "big") for t in tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    @staticmethod
    def _to_tuple(past_key_values) -> tuple:
        """Convert past_key_values to tuple format, whether DynamicCache or tuple."""
        # If it's already a tuple of tuples, return as-is
        if isinstance(past_key_values, tuple):
            return past_key_values
        # DynamicCache or similar Cache object — extract tensors
        try:
            # DynamicCache stores key_cache and value_cache as lists
            if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                return tuple(
                    (k.clone(), v.clone())
                    for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
                )
            # Fallback: try iterating
            return tuple((k.clone(), v.clone()) for k, v in past_key_values)
        except Exception as e:
            logger.warning(f"Cannot convert past_key_values of type {type(past_key_values)}: {e}")
            return past_key_values

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
        """Measure actual byte size of KV tensors. Handles both Cache and tuple."""
        total = 0
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        else:
            for key, value in past_key_values:
                total += key.nelement() * key.element_size()
                total += value.nelement() * value.element_size()
        return total

    @staticmethod
    def clone_kv(past_key_values) -> tuple:
        """Deep clone KV tensors into tuple format for storage."""
        return KVEntry._to_tuple(past_key_values)

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
            # Load from disk first
            self._load_from_disk()

        # Transfer CPU -> GPU
        if self.past_key_values is not None:
            self.past_key_values = tuple(
                (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
                for k, v in self.past_key_values
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

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
                self.past_key_values = tuple(
                    (k.to("cpu"), v.to("cpu"))
                    for k, v in self.past_key_values
                )

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
            cpu_kv = tuple(
                (k.to("cpu"), v.to("cpu"))
                for k, v in self.past_key_values
            )
            # Save to disk
            torch.save(cpu_kv, self.disk_path)
            # Free memory
            del self.past_key_values
            del cpu_kv
            self.past_key_values = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tier = "disk"
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_transfer_ms = elapsed_ms
        logger.debug(f"  [{self.prefix_hash[:8]}] -> DISK in {elapsed_ms:.2f}ms ({self.size_bytes / 1024:.0f}KB)")
        return elapsed_ms

    def _load_from_disk(self) -> None:
        """Load KV tensors from disk into CPU memory."""
        if self.disk_path and os.path.exists(self.disk_path):
            self.past_key_values = torch.load(self.disk_path, map_location="cpu", weights_only=False)
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
        if self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None
        self.delete_from_disk()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_kv_on_device(self, device: str = "cuda"):
        """
        Get past_key_values as a DynamicCache on the requested device.
        Loads from disk if needed. Does NOT change self.tier.
        """
        if self.past_key_values is None:
            if self.tier == "disk":
                self._load_from_disk()
            if self.past_key_values is None:
                return None

        # Convert stored tuple to DynamicCache on the target device
        return KVEntry._to_cache(self.past_key_values, device=device)

    def __repr__(self) -> str:
        return (
            f"KVEntry(hash={self.prefix_hash[:8]}..., tokens={self.num_tokens}, "
            f"tier={self.tier}, accesses={self.access_count}, "
            f"size={self.size_bytes / 1024:.1f}KB)"
        )