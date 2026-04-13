"""
Configuration for KV-Cache Eviction Policy Framework.
Real inference mode with Llama 3.1-8B and actual tensor storage.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvictionPolicyType(Enum):
    LRU = "lru"
    LFU = "lfu"
    SEMANTIC = "semantic"
    LEARNED = "learned"


class StorageTier(Enum):
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class TierConfig:
    """Configuration for a single storage tier."""
    tier: StorageTier
    capacity_mb: float

    @property
    def capacity_bytes(self) -> int:
        return int(self.capacity_mb * 1024 * 1024)


@dataclass
class WorkerConfig:
    """Configuration for a single inference worker."""
    worker_id: int
    gpu_tier: TierConfig = field(default_factory=lambda: TierConfig(
        tier=StorageTier.GPU, capacity_mb=2048,  # 2GB for KV cache on GPU
    ))
    cpu_tier: TierConfig = field(default_factory=lambda: TierConfig(
        tier=StorageTier.CPU, capacity_mb=8192,  # 8GB in RAM
    ))
    disk_tier: TierConfig = field(default_factory=lambda: TierConfig(
        tier=StorageTier.DISK, capacity_mb=32768,  # 32GB on disk
    ))


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    torch_dtype: str = "float16"
    max_new_tokens: int = 50
    device: str = "cuda"


@dataclass
class ControllerConfig:
    """Configuration for the central cache controller."""
    num_workers: int = 1  # Number of GPU workers (1 per GPU)
    eviction_policy: EvictionPolicyType = EvictionPolicyType.LRU
    # For semantic policy
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    num_requests: int = 100
    num_unique_prefixes: int = 10
    prefix_reuse_ratio: float = 0.7
    zipf_alpha: float = 1.2
    max_new_tokens: int = 50
    seed: int = 42


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    policies_to_evaluate: list = field(default_factory=lambda: [
        EvictionPolicyType.LRU,
        EvictionPolicyType.LFU,
    ])
    num_trials: int = 1
    warmup_requests: int = 5
    output_dir: str = "results"


@dataclass
class FrameworkConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    workers: list = field(default_factory=lambda: [WorkerConfig(worker_id=0)])
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @staticmethod
    def make_multi_gpu(num_gpus: int, gpu_cache_mb: float = 2048,
                       cpu_cache_mb: float = 8192, disk_cache_mb: float = 32768):
        """Helper to create a multi-GPU config."""
        workers = [
            WorkerConfig(
                worker_id=i,
                gpu_tier=TierConfig(tier=StorageTier.GPU, capacity_mb=gpu_cache_mb),
                cpu_tier=TierConfig(tier=StorageTier.CPU, capacity_mb=cpu_cache_mb),
                disk_tier=TierConfig(tier=StorageTier.DISK, capacity_mb=disk_cache_mb),
            )
            for i in range(num_gpus)
        ]
        return FrameworkConfig(
            workers=workers,
            controller=ControllerConfig(num_workers=num_gpus),
        )