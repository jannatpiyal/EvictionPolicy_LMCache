"""
Real Inference Worker: Runs Llama 3.1-8B with actual KV tensor caching.

On cache miss: full prefill, extract real KV tensors, store in tiered cache.
On cache hit: retrieve cached KV, only process new tokens via past_key_values.
All tensor transfers between GPU/CPU/disk are real and timed.
"""

import time
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import WorkerConfig, ModelConfig, FrameworkConfig
from cache.kv_entry import KVEntry
from cache.tiered_cache import TieredCache
from cache.eviction import EvictionPolicy

logger = logging.getLogger(__name__)


class InferenceWorker:
    """
    GPU inference worker running real Llama 3.1-8B.

    Manages a TieredCache of real KV tensors and performs
    actual prefix-aware inference.
    """

    def __init__(
        self,
        worker_config: WorkerConfig,
        model_config: ModelConfig,
        eviction_policy: EvictionPolicy,
        disk_dir: str = "/tmp/kv_cache",
    ):
        self.worker_id = worker_config.worker_id
        self.model_config = model_config

        # Assign specific GPU: "cuda:0", "cuda:1", etc.
        if model_config.device.startswith("cuda") and ":" not in model_config.device:
            self.device = f"cuda:{worker_config.worker_id}"
        else:
            self.device = model_config.device

        # Load real model onto this worker's GPU
        logger.info(f"Worker {self.worker_id}: Loading {model_config.model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            torch_dtype=getattr(torch, model_config.torch_dtype),
            device_map=self.device,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"Worker {self.worker_id}: Model loaded. "
            f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        # Tiered cache with real tensor storage
        self.cache = TieredCache(
            worker_config=worker_config,
            eviction_policy=eviction_policy,
            disk_dir=disk_dir,
            device=self.device,
        )

        # Metrics
        self.requests_processed = 0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_cache_saved_ms = 0.0

        # Track real miss prefill times to compute savings (no estimation)
        self._miss_prefill_times: dict[str, float] = {}  # prefix_hash -> measured full prefill ms

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def process_request(
        self,
        system_prompt: str,
        user_query: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """
        Process an inference request with real KV cache reuse.

        1. Tokenize system prompt (prefix) and full prompt
        2. Hash prefix tokens -> check cache
        3. On HIT: load cached KV tensors, only process new user tokens
        4. On MISS: full prefill, cache prefix KV tensors
        5. Autoregressive decode
        """
        self.requests_processed += 1

        # Tokenize
        prefix_tokens = self.tokenize(system_prompt)
        full_text = system_prompt + "\n" + user_query
        full_tokens = self.tokenize(full_text)
        new_tokens = full_tokens[len(prefix_tokens):]

        if not new_tokens:
            new_tokens = [self.tokenizer.eos_token_id]

        prefix_hash = KVEntry.compute_prefix_hash(prefix_tokens)

        # --- Cache lookup ---
        cached_entry = self.cache.get(prefix_hash)

        if cached_entry is not None:
            # ============ CACHE HIT ============
            result = self._process_with_cached_kv(
                cached_entry, new_tokens, len(prefix_tokens), max_new_tokens
            )
            result["cache_hit"] = True
            result["tier_hit"] = cached_entry.tier
            result["prefix_tokens"] = len(prefix_tokens)
            result["new_tokens"] = len(new_tokens)
            result["prefix_hash"] = prefix_hash

            # Compute real savings using measured miss time for this prefix
            measured_miss_ms = self._miss_prefill_times.get(prefix_hash)
            if measured_miss_ms is not None:
                result["savings_ms"] = measured_miss_ms - result["prefill_ms"]
                self.total_cache_saved_ms += result["savings_ms"]
            else:
                result["savings_ms"] = 0.0  # No baseline yet for this prefix

        else:
            # ============ CACHE MISS ============
            result = self._process_full_and_cache(
                prefix_tokens, new_tokens, prefix_hash,
                system_prompt, full_tokens, max_new_tokens
            )
            result["cache_hit"] = False
            result["tier_hit"] = None
            result["prefix_tokens"] = len(prefix_tokens)
            result["new_tokens"] = len(new_tokens)
            result["prefix_hash"] = prefix_hash
            result["savings_ms"] = 0.0

            # Record real prefill time for this prefix as baseline
            self._miss_prefill_times[prefix_hash] = result["prefill_ms"]

        self.total_prefill_ms += result["prefill_ms"]
        self.total_decode_ms += result["decode_ms"]

        return result

    def _process_with_cached_kv(
        self, entry: KVEntry, new_tokens: list[int],
        prefix_len: int, max_new_tokens: int,
    ) -> dict:
        """Process request using cached KV tensors."""

        # Get KV on GPU (may transfer from CPU/disk)
        kv_on_gpu = entry.get_kv_on_device(self.device)
        if kv_on_gpu is None:
            logger.warning(f"Cached KV is None for {entry.prefix_hash}, falling back to full")
            return self._process_without_cache(
                self.tokenize(entry.prompt_text) + new_tokens, max_new_tokens
            )

        new_input_ids = torch.tensor([new_tokens], device=self.device)
        start_pos = prefix_len
        position_ids = torch.arange(
            start_pos, start_pos + len(new_tokens), device=self.device
        ).unsqueeze(0)

        # Prefill only new tokens
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=kv_on_gpu,
                position_ids=position_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - start) * 1000

        # Decode
        start = time.perf_counter()
        generated = self._generate(
            outputs.past_key_values, outputs.logits,
            max_new_tokens, start_pos + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": len(generated),
        }

    def _process_full_and_cache(
        self, prefix_tokens: list[int], new_tokens: list[int],
        prefix_hash: str, prompt_text: str,
        full_tokens: list[int], max_new_tokens: int,
    ) -> dict:
        """Full prefill, cache prefix KV, then process remaining tokens."""

        prefix_input_ids = torch.tensor([prefix_tokens], device=self.device)

        # Prefill prefix
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            prefix_out = self.model(
                input_ids=prefix_input_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        prefix_ms = (time.perf_counter() - start) * 1000

        # Clone and cache prefix KV
        cloned_kv = KVEntry.clone_kv(prefix_out.past_key_values)
        kv_bytes = KVEntry.measure_kv_size(cloned_kv)

        entry = KVEntry(
            prefix_hash=prefix_hash,
            prefix_tokens=prefix_tokens,
            prompt_text=prompt_text,
            num_tokens=len(prefix_tokens),
            past_key_values=cloned_kv,
            size_bytes=kv_bytes,
            worker_id=self.worker_id,
            tier="gpu",
        )
        self.cache.put(entry)

        # Process remaining new tokens using prefix KV
        new_input_ids = torch.tensor([new_tokens], device=self.device)
        start_pos = len(prefix_tokens)
        position_ids = torch.arange(
            start_pos, start_pos + len(new_tokens), device=self.device
        ).unsqueeze(0)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=prefix_out.past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        query_ms = (time.perf_counter() - start) * 1000

        total_prefill_ms = prefix_ms + query_ms

        # Decode
        start = time.perf_counter()
        generated = self._generate(
            outputs.past_key_values, outputs.logits,
            max_new_tokens, start_pos + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        return {
            "prefill_ms": total_prefill_ms,
            "prefix_only_ms": prefix_ms,
            "decode_ms": decode_ms,
            "total_ms": total_prefill_ms + decode_ms,
            "kv_cached_bytes": kv_bytes,
            "kv_cached_mb": kv_bytes / 1024 / 1024,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": len(generated),
        }

    def _process_without_cache(self, tokens: list[int], max_new_tokens: int) -> dict:
        """Fallback: full prefill without any caching."""
        input_ids = torch.tensor([tokens], device=self.device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        generated = self._generate(
            outputs.past_key_values, outputs.logits,
            max_new_tokens, len(tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": len(generated),
        }

    def _generate(self, past_kv, logits, max_tokens: int, current_pos: int) -> list[int]:
        """Autoregressive token generation."""
        generated = []
        for i in range(max_tokens):
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            generated.append(next_token.item())

            pos_ids = torch.tensor([[current_pos + i]], device=self.device)
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_kv,
                    position_ids=pos_ids,
                    use_cache=True,
                )
            past_kv = out.past_key_values
            logits = out.logits

        return generated

    def get_stats(self) -> dict:
        cache_stats = self.cache.get_stats()
        return {
            **cache_stats,
            "requests_processed": self.requests_processed,
            "total_prefill_ms": self.total_prefill_ms,
            "total_decode_ms": self.total_decode_ms,
            "total_cache_saved_ms": self.total_cache_saved_ms,
            "model": self.model_config.model_path,
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        }

    def reset(self) -> None:
        self.cache.reset()
        self.requests_processed = 0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_cache_saved_ms = 0.0
        self._miss_prefill_times.clear()