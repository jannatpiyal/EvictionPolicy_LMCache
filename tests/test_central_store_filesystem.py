import os
import shutil
import tempfile
import unittest


class TestFileSystemCentralKVStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="lmcache-central-test-")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_roundtrip(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from store.central_kv_store import FileSystemCentralKVStore

        store = FileSystemCentralKVStore(self._tmpdir)
        prefix_hash = "deadbeefdeadbeef"

        kv_tuple = (
            (torch.zeros((1, 2), dtype=torch.float32), torch.ones((1, 2), dtype=torch.float32)),
        )

        self.assertFalse(store.contains(prefix_hash))
        self.assertIsNone(store.get(prefix_hash))

        store.put(prefix_hash, kv_tuple)
        self.assertTrue(store.contains(prefix_hash))

        rec = store.get(prefix_hash)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.prefix_hash, prefix_hash)

        loaded = rec.kv_tuple
        self.assertEqual(len(loaded), len(kv_tuple))
        self.assertTrue(torch.allclose(loaded[0][0], kv_tuple[0][0]))
        self.assertTrue(torch.allclose(loaded[0][1], kv_tuple[0][1]))

        # Files exist on disk
        self.assertTrue(os.path.exists(os.path.join(self._tmpdir, f"{prefix_hash}.pt")))

        store.delete(prefix_hash)
        self.assertFalse(store.contains(prefix_hash))


if __name__ == "__main__":
    unittest.main()

