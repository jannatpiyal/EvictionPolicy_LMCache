from __future__ import annotations

import argparse
import json
import logging
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from transformers import AutoTokenizer

from cache.kv_entry import KVEntry
from metadata import RedisMetadataRegistry

logger = logging.getLogger(__name__)


def _extract_prefix_text(prompt: str) -> str:
    if "Question:" in prompt:
        return prompt.split("Question:")[0]
    split_idx = max(len(prompt) - 200, 0)
    return prompt[:split_idx]


def _post_json(url: str, path: str, payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


class _RouterHandler(BaseHTTPRequestHandler):
    server: "_RouterServer"  # type: ignore[assignment]

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
            if self.path != "/process":
                self._write_json(404, {"ok": False, "error": "unknown path"})
                return

            req = self._read_json()
            prompt = req.get("prompt")
            system_prompt = req.get("system_prompt")
            user_query = req.get("user_query")
            max_new_tokens = int(req.get("max_new_tokens", 50))

            if prompt is None:
                # legacy mode: prefix = system_prompt
                prefix_text = system_prompt or ""
            else:
                prefix_text = _extract_prefix_text(prompt)

            tokens = self.server.tokenizer.encode(prefix_text, add_special_tokens=True)
            prefix_hash = KVEntry.compute_prefix_hash(tokens)

            worker_url = None
            try:
                replicas = self.server.registry.list_live_replicas(prefix_hash)
                if replicas:
                    worker_url = replicas[0].address
            except Exception:
                worker_url = None

            if worker_url is None:
                # No live replicas: pick a random worker (rehydration happens at worker).
                worker_url = random.choice(self.server.worker_urls)

            res = _post_json(
                worker_url,
                "/process",
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "user_query": user_query,
                    "max_new_tokens": max_new_tokens,
                },
            )
            self._write_json(200, {"ok": True, "worker": worker_url, "result": res})
        except Exception as e:
            logger.exception("Router handler error")
            self._write_json(500, {"ok": False, "error": str(e)})

    def log_message(self, fmt: str, *args):
        return


class _RouterServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, tokenizer, registry, worker_urls: list[str]):
        super().__init__((host, port), _RouterHandler)
        self.tokenizer = tokenizer
        self.registry = registry
        self.worker_urls = worker_urls


def main():
    p = argparse.ArgumentParser(description="Cache-aware router/controller HTTP service")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", type=str, required=True, help="Tokenizer model path/id")
    p.add_argument("--metadata-redis-url", type=str, required=True)
    p.add_argument("--workers", nargs="+", required=True, help="Worker base URLs (e.g., http://host:8001)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    registry = RedisMetadataRegistry(args.metadata_redis_url)

    server = _RouterServer(args.host, args.port, tokenizer=tokenizer, registry=registry, worker_urls=args.workers)
    logger.info("Router listening on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()

