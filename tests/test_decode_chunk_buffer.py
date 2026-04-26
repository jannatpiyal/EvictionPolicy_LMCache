import unittest


class TestDecodeChunkBuffer(unittest.TestCase):
    def test_flush_threshold_and_pending(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        from cache.decode_chunk_buffer import DecodeChunkBuffer

        buf = DecodeChunkBuffer(chunk_size_tokens=4)
        self.assertFalse(buf.should_flush(3))
        self.assertTrue(buf.should_flush(4))
        buf.mark_flushed(4)
        self.assertFalse(buf.should_flush(7))
        self.assertTrue(buf.has_pending(5))
        self.assertTrue(buf.should_flush(8))


if __name__ == "__main__":
    unittest.main()
