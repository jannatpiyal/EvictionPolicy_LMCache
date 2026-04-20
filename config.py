"""
Configuration for KV-Cache Eviction Policy Framework.
LMCache-style long-document multi-round QA workload with real inference mode.
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
        tier=StorageTier.GPU, capacity_mb=2048,   # 2 GB KV cache on GPU
    ))
    cpu_tier: TierConfig = field(default_factory=lambda: TierConfig(
        tier=StorageTier.CPU, capacity_mb=8192,   # 8 GB RAM cache
    ))
    disk_tier: TierConfig = field(default_factory=lambda: TierConfig(
        tier=StorageTier.DISK, capacity_mb=32768, # 32 GB disk cache
    ))


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    torch_dtype: str = "float16"
    max_new_tokens: int = 100
    device: str = "cuda"

    # Optional but useful for long-context experiments
    max_context_tokens: int = 20000


@dataclass
class ControllerConfig:
    """Configuration for the central cache controller."""
    num_workers: int = 1
    eviction_policy: EvictionPolicyType = EvictionPolicyType.LRU

    # Semantic policy knobs
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7

    # Optional cache-management behavior
    pin_warmup_prefixes: bool = False
    enable_cross_tier_promotion: bool = True
    enable_cross_tier_demotion: bool = True


@dataclass
class WorkloadConfig:
    """
    Configuration for LMCache-style workload generation.

    This replaces generic 'num_unique_prefixes/prefix_reuse_ratio'
    with document-centric multi-round QA parameters.
    """

    seed: int = 42

    # Core LMCache-style workload shape
    num_documents: int = 40
    document_length_tokens: int = 10_000
    num_rounds: int = 2                 # round 0 = warmup, round 1 = reuse
    hit_ratio: float = 1.0              # 1.0 => all docs reused in later rounds
    max_new_tokens: int = 100

    # Traffic pattern
    initial_concurrency: int = 40
    arrival_mode: str = "bursty"        # "bursty" or "poisson"
    interarrival_mean_sec: float = 0.2

    # Prompt composition
    include_system_instruction: bool = True
    question_style: str = "document_qa" # document_qa, summarization, mixed

    # Synthetic corpus controls
    num_questions_per_document: int = 4
    allow_partial_reuse: bool = False   # later useful for sensitivity studies

    # Backward compatibility with older code paths
    num_requests: Optional[int] = None
    num_unique_prefixes: int = 40
    prefix_reuse_ratio: float = 1.0
    zipf_alpha: float = 1.0

    @property
    def total_requests(self) -> int:
        if self.num_requests is not None:
            return self.num_requests
        return self.num_documents * self.num_rounds


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    policies_to_evaluate: list = field(default_factory=lambda: [
        EvictionPolicyType.LRU,
        EvictionPolicyType.LFU,
    ])
    num_trials: int = 1

    # Warmup should be larger for long-doc prefix caching
    warmup_requests: int = 40

    output_dir: str = "results"

    # Useful benchmark metrics for LMCache-style evaluation
    measure_ttft: bool = True
    measure_itl: bool = True
    measure_cache_hit_rate: bool = True
    measure_evictions: bool = True
    measure_tier_utilization: bool = True


@dataclass
class FrameworkConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    workers: list = field(default_factory=lambda: [WorkerConfig(worker_id=0)])
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @staticmethod
    def make_multi_gpu(
        num_gpus: int,
        gpu_cache_mb: float = 2048,
        cpu_cache_mb: float = 8192,
        disk_cache_mb: float = 32768,
    ):
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