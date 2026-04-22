import shutil
import tempfile
import unittest


class TestKVEntryPipeline(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="lmcache-kv-entry-")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_disk_prefetch_roundtrip(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from cache.kv_entry import KVEntry

        entry = KVEntry(
            prefix_hash="feedfacefeedface",
            prefix_tokens=[1, 2, 3],
            past_key_values=(
                (torch.zeros((1, 2), dtype=torch.float32), torch.ones((1, 2), dtype=torch.float32)),
            ),
            size_bytes=16,
            tier="cpu",
        )

        entry.move_to_disk(self._tmpdir)
        self.assertEqual(entry.tier, "disk")
        self.assertIsNone(entry.past_key_values)

        future = entry.start_disk_prefetch()
        self.assertIsNotNone(future)
        entry._load_from_disk()

        self.assertIsNotNone(entry.past_key_values)
        loaded_key, loaded_value = entry.past_key_values[0]
        self.assertTrue(torch.allclose(loaded_key, torch.zeros((1, 2), dtype=torch.float32)))
        self.assertTrue(torch.allclose(loaded_value, torch.ones((1, 2), dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
