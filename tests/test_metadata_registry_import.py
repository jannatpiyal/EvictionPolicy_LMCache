import unittest


class TestMetadataRegistryImport(unittest.TestCase):
    def test_import(self):
        # Should import without requiring redis installed (lazy import).
        from metadata.registry import RedisMetadataRegistry  # noqa: F401


if __name__ == "__main__":
    unittest.main()

