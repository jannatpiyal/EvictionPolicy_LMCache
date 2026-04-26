import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace


class TestTransferScheduler(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="lmcache-transfer-scheduler-")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_queued_disk_spill_moves_multiple_entries(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from cache.kv_entry import KVEntry
        from cache.transfer_scheduler import TransferScheduler

        scheduler = TransferScheduler()
        entries = [
            KVEntry(
                prefix_hash=f"entry{i:02d}",
                prefix_tokens=[i],
                past_key_values=(
                    (torch.full((1, 2), i, dtype=torch.float32), torch.full((1, 2), i + 1, dtype=torch.float32)),
                ),
                size_bytes=16,
                tier="cpu",
            )
            for i in range(2)
        ]

        scheduler.queue_disk_spill(entries)
        elapsed_ms = scheduler.flush_disk_spills(self._tmpdir)

        self.assertGreaterEqual(elapsed_ms, 0.0)
        for entry in entries:
            self.assertEqual(entry.tier, "disk")
            self.assertIsNone(entry.past_key_values)
            self.assertTrue(os.path.exists(entry.disk_path))

    def test_async_job_submission_runs_disk_spill(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from cache.kv_entry import KVEntry
        from cache.transfer_scheduler import TransferScheduler

        scheduler = TransferScheduler()
        entries = [
            KVEntry(
                prefix_hash=f"async{i:02d}",
                prefix_tokens=[i],
                past_key_values=(
                    (torch.full((1, 2), i, dtype=torch.float32), torch.full((1, 2), i + 1, dtype=torch.float32)),
                ),
                size_bytes=16,
                tier="cpu",
            )
            for i in range(2)
        ]

        job_id = scheduler.submit_disk_spill(entries, self._tmpdir)
        elapsed_ms = scheduler.wait_for_job(job_id)

        self.assertGreaterEqual(elapsed_ms, 0.0)
        for entry in entries:
            self.assertEqual(entry.tier, "disk")
            self.assertTrue(os.path.exists(entry.disk_path))

    def test_collect_victims_frees_full_required_capacity(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")

        from cache.eviction import create_policy
        from cache.kv_entry import KVEntry
        from cache.tiered_cache import TieredCache
        from config import EvictionPolicyType, StorageTier, TierConfig, WorkerConfig

        worker_config = WorkerConfig(
            worker_id=0,
            gpu_tier=TierConfig(tier=StorageTier.GPU, capacity_mb=2048),
            cpu_tier=TierConfig(tier=StorageTier.CPU, capacity_mb=8192),
            disk_tier=TierConfig(tier=StorageTier.DISK, capacity_mb=8192),
        )
        cache = TieredCache(
            worker_config=worker_config,
            eviction_policy=create_policy(EvictionPolicyType.LRU),
            disk_dir=self._tmpdir,
            device="cpu",
        )

        chunk_size_bytes = 32 * 1024 * 1024
        for idx in range(65):
            entry = KVEntry(
                prefix_hash=f"prefix::{idx}",
                prefix_tokens=[idx],
                parent_prefix_hash=f"prefix::{idx // 32}",
                chunk_index=idx,
                chunk_count=65,
                size_bytes=chunk_size_bytes,
                tier="gpu",
            )
            cache.tiers[StorageTier.GPU].add(entry)
            cache.eviction_policy.on_insert(entry)

        required_bytes = 1011.9 * 1024 * 1024
        victims = cache._collect_victims(StorageTier.GPU, required_bytes)

        self.assertGreaterEqual(cache.tiers[StorageTier.GPU].free_bytes, required_bytes)
        self.assertGreaterEqual(sum(v.size_bytes for v in victims), required_bytes - (7.1 * 1024 * 1024))


if __name__ == "__main__":
    unittest.main()
