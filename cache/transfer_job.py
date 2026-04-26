from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TransferKind(Enum):
    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_CPU = "gpu_to_cpu"
    CPU_TO_DISK = "cpu_to_disk"


@dataclass
class TransferJob:
    """
    Unified transfer descriptor for chunk/prefix movement across tiers.

    Jobs are intentionally coarse enough to group multiple entries that share a
    source/destination link, so the scheduler can batch them or run different
    link types in parallel.
    """

    kind: TransferKind
    entries: list
    device: Optional[str] = None
    disk_dir: Optional[str] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
