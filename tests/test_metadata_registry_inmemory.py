import time
import unittest


class TestInMemoryMetadataRegistry(unittest.TestCase):
    def test_ttl_expiry(self):
        from metadata.registry import InMemoryMetadataRegistry

        reg = InMemoryMetadataRegistry()
        reg.register_worker("w1", "http://w1:8000", ttl_s=1)
        reg.claim_replica("abc", "w1", ttl_s=1)

        self.assertEqual(len(reg.list_live_replicas("abc")), 1)
        time.sleep(1.2)
        self.assertEqual(len(reg.list_live_replicas("abc")), 0)


if __name__ == "__main__":
    unittest.main()

