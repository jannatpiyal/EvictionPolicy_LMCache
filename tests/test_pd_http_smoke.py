import json
import importlib.util
import threading
import unittest
import urllib.request
from unittest.mock import patch


class _FakeWorker:
    def __init__(self, worker_id: int, shared_store: dict[str, dict]):
        self.worker_id = worker_id
        self.shared_store = shared_store
        self.central_store = object()
        self.metadata_registry = None
        self.metadata_worker_id = str(worker_id)
        self.lease_ttl_s = 30
        self.prefill_calls = 0
        self.decode_calls = 0

    def prepare_prefix_kv(self, system_prompt=None, user_query=None, prompt=None):
        self.prefill_calls += 1
        prefix_hash = f"prefix-{abs(hash(prompt or system_prompt or '')) % 100000}"
        self.shared_store[prefix_hash] = {
            "producer": self.worker_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "user_query": user_query,
        }
        return {
            "ok": True,
            "prefix_hash": prefix_hash,
            "prefix_len": 4,
            "num_prefix_tokens": 4,
            "prefix_only_ms": 1.0,
            "stored_in_central_store": True,
            "cache_hit": False,
            "central_hit": False,
            "worker_id": self.worker_id,
        }

    def decode_request(
        self,
        system_prompt=None,
        user_query=None,
        prompt=None,
        max_new_tokens=50,
        prefix_hash=None,
        require_cached_prefix=False,
    ):
        self.decode_calls += 1
        if require_cached_prefix and prefix_hash not in self.shared_store:
            raise KeyError(f"Missing prefix {prefix_hash}")
        return {
            "prefill_ms": 0.5,
            "decode_ms": 1.5,
            "total_ms": 2.0,
            "ttft_ms": 0.8,
            "avg_itl_ms": 0.2,
            "generated_text": "ok",
            "generated_tokens": min(2, max_new_tokens),
            "output_tokens_per_s": 100.0,
            "cache_hit": True,
            "tier_hit": "cpu",
            "central_hit": True,
            "savings_ms": 1.0,
            "prefix_hash": prefix_hash,
            "worker_id": self.worker_id,
        }

    def process_request(self, system_prompt=None, user_query=None, prompt=None, max_new_tokens=50):
        raise AssertionError("Smoke test should exercise /prefill -> /decode, not /process")


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [len(text), 7, 11]


class _FakeRegistry:
    def list_live_replicas(self, prefix_hash):
        return []


class TestPDHTTPSmoke(unittest.TestCase):
    def test_router_prefill_decode_handoff(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not available")

        from worker.http_service import _WorkerServer
        from controller.http_service import _RouterServer

        shared_store: dict[str, dict] = {}
        worker_a = _FakeWorker(worker_id=0, shared_store=shared_store)
        worker_b = _FakeWorker(worker_id=1, shared_store=shared_store)

        worker_server_a = _WorkerServer("127.0.0.1", 0, worker=worker_a)
        worker_server_b = _WorkerServer("127.0.0.1", 0, worker=worker_b)

        worker_thread_a = threading.Thread(target=worker_server_a.serve_forever, daemon=True)
        worker_thread_b = threading.Thread(target=worker_server_b.serve_forever, daemon=True)
        worker_thread_a.start()
        worker_thread_b.start()

        worker_url_a = f"http://127.0.0.1:{worker_server_a.server_address[1]}"
        worker_url_b = f"http://127.0.0.1:{worker_server_b.server_address[1]}"

        router_server = _RouterServer(
            "127.0.0.1",
            0,
            tokenizer=_FakeTokenizer(),
            registry=_FakeRegistry(),
            worker_urls=[worker_url_a, worker_url_b],
        )
        router_thread = threading.Thread(target=router_server.serve_forever, daemon=True)
        router_thread.start()
        router_url = f"http://127.0.0.1:{router_server.server_address[1]}"

        payload = {
            "prompt": "Document: hello world\nQuestion: what is this?",
            "max_new_tokens": 4,
        }

        try:
            with patch("controller.http_service.random.choice", side_effect=[worker_url_a, worker_url_b]):
                req = urllib.request.Request(
                    url=router_url + "/process",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
        finally:
            router_server.shutdown()
            router_server.server_close()
            worker_server_a.shutdown()
            worker_server_a.server_close()
            worker_server_b.shutdown()
            worker_server_b.server_close()

        self.assertTrue(body["ok"])
        self.assertEqual(body["mode"], "pd_disaggregated")
        self.assertEqual(body["prefill_worker"], worker_url_a)
        self.assertEqual(body["decode_worker"], worker_url_b)

        prefill_result = body["prefill"]["result"]
        decode_result = body["result"]["result"]

        self.assertTrue(prefill_result["stored_in_central_store"])
        self.assertEqual(prefill_result["worker_id"], 0)
        self.assertEqual(decode_result["prefix_hash"], prefill_result["prefix_hash"])
        self.assertTrue(decode_result["cache_hit"])
        self.assertTrue(decode_result["central_hit"])
        self.assertEqual(worker_a.prefill_calls, 1)
        self.assertEqual(worker_a.decode_calls, 0)
        self.assertEqual(worker_b.decode_calls, 1)


if __name__ == "__main__":
    unittest.main()
