import importlib.util
import pathlib
import sys
import unittest


class _Entry:
    def __init__(self, size_bytes):
        self.size_bytes = size_bytes


class TestDynamicOffloadWindow(unittest.TestCase):
    def test_pointer_progression(self):
        module_path = pathlib.Path(__file__).resolve().parents[1] / "cache" / "dynamic_offload.py"
        spec = importlib.util.spec_from_file_location("dynamic_offload_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        DynamicOffloadWindow = module.DynamicOffloadWindow

        window = DynamicOffloadWindow(enabled=True, window_factor=1.0)
        for key in ("a", "b", "c"):
            window.register_gpu_chunk(key)

        planned = window.plan_window(10, {"a": _Entry(4), "b": _Entry(4), "c": _Entry(4)})
        self.assertEqual(planned, ["a", "b", "c"])
        window.mark_duplicated(planned[:2], duplicated_bytes=8, duplicate_ms=1.0)
        self.assertEqual(window.current_idx, 3)
        reclaim = window.reclaimable_keys(4, {"a": _Entry(4), "b": _Entry(4), "c": _Entry(4)})
        self.assertEqual(reclaim, ["a"])


if __name__ == "__main__":
    unittest.main()
