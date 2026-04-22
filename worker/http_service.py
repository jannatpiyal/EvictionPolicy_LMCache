from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from config import FrameworkConfig
from cache.eviction import create_policy
from store import FileSystemCentralKVStore, RedisCentralKVStore
from metadata import RedisMetadataRegistry
from worker.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)


class _WorkerHandler(BaseHTTPRequestHandler):
    server: "_WorkerServer"  # type: ignore[assignment]

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(body.decode("utf-8"))

    def _write_json(self, code: int, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/prefill":
                req = self._read_json()
                res = self.server.worker.prepare_prefix_kv(
                    system_prompt=req.get("system_prompt"),
                    user_query=req.get("user_query"),
                    prompt=req.get("prompt"),
                )
                self._write_json(200, {"ok": True, "result": res})
                return
            if self.path == "/decode":
                req = self._read_json()
                res = self.server.worker.decode_request(
                    system_prompt=req.get("system_prompt"),
                    user_query=req.get("user_query"),
                    prompt=req.get("prompt"),
                    max_new_tokens=int(req.get("max_new_tokens", 50)),
                    prefix_hash=req.get("prefix_hash"),
                    require_cached_prefix=bool(req.get("require_cached_prefix", True)),
                )
                self._write_json(200, {"ok": True, "result": res})
                return
            if self.path == "/process":
                req = self._read_json()
                res = self.server.worker.process_request(
                    system_prompt=req.get("system_prompt"),
                    user_query=req.get("user_query"),
                    prompt=req.get("prompt"),
                    max_new_tokens=int(req.get("max_new_tokens", 50)),
                )
                self._write_json(200, {"ok": True, "result": res})
                return
            if self.path == "/prefetch":
                req = self._read_json()
                prefix_hash = req.get("prefix_hash")
                if not prefix_hash:
                    self._write_json(400, {"ok": False, "error": "missing prefix_hash"})
                    return
                # Trigger a central-store fetch into local CPU tier by making a synthetic request.
                # The worker-side path uses prefix_hash derived from tokenization, so we directly
                # call the central store and insert without any model compute.
                if self.server.worker.central_store is None:
                    self._write_json(400, {"ok": False, "error": "central_store disabled"})
                    return
                rec = self.server.worker.central_store.get(prefix_hash)
                if rec is None:
                    self._write_json(404, {"ok": False, "error": "not found"})
                    return
                from cache.kv_entry import KVEntry
                kv_tuple = rec.kv_tuple
                kv_bytes = KVEntry.measure_kv_size(kv_tuple)
                entry = KVEntry(
                    prefix_hash=prefix_hash,
                    prefix_tokens=[],
                    prompt_text="",
                    num_tokens=0,
                    past_key_values=kv_tuple,
                    size_bytes=kv_bytes,
                    worker_id=self.server.worker.worker_id,
                    tier="cpu",
                )
                self.server.worker.cache.put_cpu(entry)
                if self.server.worker.metadata_registry is not None:
                    try:
                        self.server.worker.metadata_registry.claim_replica(
                            prefix_hash=prefix_hash,
                            worker_id=self.server.worker.metadata_worker_id,
                            ttl_s=self.server.worker.lease_ttl_s,
                        )
                    except Exception:
                        pass
                self._write_json(200, {"ok": True})
                return

            self._write_json(404, {"ok": False, "error": "unknown path"})
        except Exception as e:
            logger.exception("Worker handler error")
            self._write_json(500, {"ok": False, "error": str(e)})

    def log_message(self, fmt: str, *args):  # silence default HTTP logs
        return


class _WorkerServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, worker: InferenceWorker):
        super().__init__((host, port), _WorkerHandler)
        self.worker = worker


def _renewal_loop(worker: InferenceWorker, interval_s: float):
    while True:
        time.sleep(interval_s)
        if worker.metadata_registry is None:
            continue
        try:
            worker.metadata_registry.heartbeat_worker(worker.metadata_worker_id, ttl_s=worker.lease_ttl_s)
        except Exception:
            pass
        try:
            prefixes = worker.cache.list_prefixes()
        except Exception:
            prefixes = []
        for ph in prefixes:
            try:
                worker.metadata_registry.claim_replica(ph, worker.metadata_worker_id, ttl_s=worker.lease_ttl_s)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description="Inference worker HTTP service")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--central-store", choices=["none", "filesystem", "redis"], default="none")
    p.add_argument("--central-dir", type=str, default="/tmp/lmcache_central_kv")
    p.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")

    p.add_argument("--metadata-redis-url", type=str, default=None)
    p.add_argument("--lease-ttl", type=int, default=30)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = FrameworkConfig()
    cfg.model.model_path = args.model
    cfg.model.device = args.device
    cfg.workers[0].worker_id = args.worker_id

    central_store = None
    if args.central_store == "filesystem":
        central_store = FileSystemCentralKVStore(args.central_dir)
    elif args.central_store == "redis":
        central_store = RedisCentralKVStore(args.redis_url)

    metadata_registry = None
    if args.metadata_redis_url:
        metadata_registry = RedisMetadataRegistry(args.metadata_redis_url)

    policy = create_policy(cfg.controller.eviction_policy)
    worker = InferenceWorker(
        worker_config=cfg.workers[0],
        model_config=cfg.model,
        eviction_policy=policy,
        central_store=central_store,
        metadata_registry=metadata_registry,
        metadata_worker_id=str(args.worker_id),
        metadata_worker_addr=f"http://{args.host}:{args.port}",
        lease_ttl_s=args.lease_ttl,
    )

    # Start background renewal thread for true fault tolerance.
    interval = max(1.0, float(args.lease_ttl) / 3.0)
    t = threading.Thread(target=_renewal_loop, args=(worker, interval), daemon=True)
    t.start()

    server = _WorkerServer(args.host, args.port, worker=worker)
    logger.info("Worker %s listening on %s:%s", args.worker_id, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
