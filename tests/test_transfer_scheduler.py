import os
import shutil
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
