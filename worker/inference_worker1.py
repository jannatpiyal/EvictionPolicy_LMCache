"""
Real Inference Worker: Runs Llama 3.1-8B with actual KV tensor caching.

Supports:
1) Legacy mode: system_prompt + user_query
2) LMCache-style mode: full prompt (long document + question)

On cache miss: full prefill, extract real KV tensors, store in tiered cache.
On cache hit: retrieve cached KV, only process new tokens via past_key_values.
"""

import time
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import WorkerConfig, ModelConfig
from cache.kv_entry import KVEntry
from cache.tiered_cache import TieredCache
from cache.eviction import EvictionPolicy

logger = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_config: ModelConfig,
        eviction_policy: EvictionPolicy,
        disk_dir: str = "/tmp/kv_cache",
    ):
        self.worker_id = worker_config.worker_id
        self.model_config = model_config

        # Assign GPU
        if model_config.device.startswith("cuda") and ":" not in model_config.device:
            self.device = f"cuda:{worker_config.worker_id}"
        else:
            self.device = model_config.device

        logger.info(f"Worker {self.worker_id}: Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            torch_dtype=getattr(torch, model_config.torch_dtype),
            device_map=self.device,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # KV cache
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
        self._miss_prefill_times: dict[str, float] = {}

    # -----------------------------
    # TOKENIZATION
    # -----------------------------
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    # -----------------------------
    # MAIN ENTRY POINT
    # -----------------------------
    def process_request(
        self,
        system_prompt: Optional[str] = None,
        user_query: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 50,
    ) -> dict:
        """
        Supports:
        - system_prompt + user_query (old)
        - prompt (new LMCache-style)
        """

        self.requests_processed += 1

        # -----------------------------
        # Normalize input
        # -----------------------------
        if prompt is not None:
            full_text = prompt

            # Try structured split
            if "Question:" in prompt:
                parts = prompt.split("Question:")
                system_prompt = parts[0]
                user_query = "Question:" + parts[-1]
            else:
                # fallback split (last 200 chars)
                split_idx = max(len(prompt) - 200, 0)
                system_prompt = prompt[:split_idx]
                user_query = prompt[split_idx:]

        else:
            full_text = system_prompt + "\n" + user_query

        # -----------------------------
        # Tokenization
        # -----------------------------
        prefix_tokens = self.tokenize(system_prompt)
        full_tokens = self.tokenize(full_text)

        prefix_len = min(len(prefix_tokens), len(full_tokens))
        new_tokens = full_tokens[prefix_len:]

        if not new_tokens:
            new_tokens = [self.tokenizer.eos_token_id]

        prefix_hash = KVEntry.compute_prefix_hash(prefix_tokens)

        # -----------------------------
        # CACHE LOOKUP
        # -----------------------------
        cached_entry = self.cache.get(prefix_hash)

        if cached_entry is not None:
            result = self._process_with_cached_kv(
                cached_entry,
                new_tokens,
                prefix_len,
                max_new_tokens,
            )
            result["cache_hit"] = True
            result["tier_hit"] = cached_entry.tier

            # savings calculation
            miss_time = self._miss_prefill_times.get(prefix_hash)
            if miss_time is not None:
                savings = miss_time - result["prefill_ms"]
                result["savings_ms"] = savings
                self.total_cache_saved_ms += savings
            else:
                result["savings_ms"] = 0.0

        else:
            result = self._process_full_and_cache(
                prefix_tokens,
                new_tokens,
                prefix_hash,
                system_prompt,
                max_new_tokens,
            )
            result["cache_hit"] = False
            result["tier_hit"] = None
            result["savings_ms"] = 0.0

            self._miss_prefill_times[prefix_hash] = result["prefill_ms"]

        self.total_prefill_ms += result["prefill_ms"]
        self.total_decode_ms += result["decode_ms"]

        return result

    # -----------------------------
    # CACHE HIT PATH
    # -----------------------------
    def _process_with_cached_kv(
        self,
        entry: KVEntry,
        new_tokens: list[int],
        prefix_len: int,
        max_new_tokens: int,
    ) -> dict:

        kv_on_gpu = entry.get_kv_on_device(self.device)

        new_input_ids = torch.tensor([new_tokens], device=self.device)
        position_ids = torch.arange(
            prefix_len,
            prefix_len + len(new_tokens),
            device=self.device,
        ).unsqueeze(0)

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
            outputs.past_key_values,
            outputs.logits,
            max_new_tokens,
            prefix_len + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
        }

    # -----------------------------
    # CACHE MISS PATH
    # -----------------------------
    def _process_full_and_cache(
        self,
        prefix_tokens: list[int],
        new_tokens: list[int],
        prefix_hash: str,
        prompt_text: str,
        max_new_tokens: int,
    ) -> dict:

        prefix_input_ids = torch.tensor([prefix_tokens], device=self.device)

        # Prefill prefix
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            prefix_out = self.model(input_ids=prefix_input_ids, use_cache=True)

        torch.cuda.synchronize()
        prefix_ms = (time.perf_counter() - start) * 1000

        # Cache KV
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

        # Process query
        new_input_ids = torch.tensor([new_tokens], device=self.device)
        position_ids = torch.arange(
            len(prefix_tokens),
            len(prefix_tokens) + len(new_tokens),
            device=self.device,
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
            outputs.past_key_values,
            outputs.logits,
            max_new_tokens,
            len(prefix_tokens) + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) * 1000

        return {
            "prefill_ms": total_prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": total_prefill_ms + decode_ms,
            "kv_cached_mb": kv_bytes / 1024 / 1024,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
        }

    # -----------------------------
    # GENERATION
    # -----------------------------
    def _generate(self, past_kv, logits, max_tokens: int, current_pos: int):
        generated = []

        for i in range(max_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

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

    # -----------------------------
    # STATS
    # -----------------------------
    def get_stats(self) -> dict:
        cache_stats = self.cache.get_stats()
        return {
            **cache_stats,
            "requests_processed": self.requests_processed,
            "total_prefill_ms": self.total_prefill_ms,
            "total_decode_ms": self.total_decode_ms,
            "total_cache_saved_ms": self.total_cache_saved_ms,
        }

    def reset(self):
        self.cache.reset()
        self.requests_processed = 0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_cache_saved_ms = 0.0
        self._miss_prefill_times.clear()