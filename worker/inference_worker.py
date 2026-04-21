"""
Real Inference Worker: Runs Llama 3.1-8B with actual KV tensor caching.

Supports:
1) Legacy mode: system_prompt + user_query
2) LMCache-style mode: full prompt (long document + question)

On cache miss: full prefill, extract real KV tensors, store in tiered cache.
On cache hit: retrieve cached KV, only process new tokens via past_key_values.

Adds LMCache-style metrics:
- ttft_ms: time to first token
- avg_itl_ms: average inter-token latency after first token
- output_tokens_per_s: decode throughput for generated tokens
"""

import time
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import WorkerConfig, ModelConfig, StorageTier
from cache.kv_entry import KVEntry
from cache.tiered_cache import TieredCache
from cache.eviction import EvictionPolicy
from store.central_kv_store import CentralKVStore
from metadata.registry import MetadataRegistry

logger = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_config: ModelConfig,
        eviction_policy: EvictionPolicy,
        disk_dir: str = "/tmp/kv_cache",
        central_store: Optional[CentralKVStore] = None,
        metadata_registry: Optional[MetadataRegistry] = None,
        metadata_worker_id: Optional[str] = None,
        metadata_worker_addr: Optional[str] = None,
        lease_ttl_s: int = 30,
    ):
        self.worker_id = worker_config.worker_id
        self.model_config = model_config

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

        self.cache = TieredCache(
            worker_config=worker_config,
            eviction_policy=eviction_policy,
            disk_dir=disk_dir,
            device=self.device,
        )
        self.central_store = central_store
        self.metadata_registry = metadata_registry
        self.metadata_worker_id = metadata_worker_id or str(self.worker_id)
        self.metadata_worker_addr = metadata_worker_addr or f"worker-{self.worker_id}"
        self.lease_ttl_s = int(lease_ttl_s)

        self.requests_processed = 0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_cache_saved_ms = 0.0
        self.total_ttft_ms = 0.0
        self.total_itl_ms = 0.0
        self.total_generated_tokens = 0

        self._miss_prefill_times: dict[str, float] = {}

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def process_request(
        self,
        system_prompt: Optional[str] = None,
        user_query: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 50,
    ) -> dict:
        self.requests_processed += 1

        # Best-effort heartbeat/registration (multi-node fault tolerance).
        if self.metadata_registry is not None:
            try:
                self.metadata_registry.register_worker(
                    worker_id=self.metadata_worker_id,
                    address=self.metadata_worker_addr,
                    ttl_s=self.lease_ttl_s,
                )
            except Exception:
                pass

        if prompt is not None:
            full_text = prompt
            if "Question:" in prompt:
                parts = prompt.split("Question:")
                system_prompt = parts[0]
                user_query = "Question:" + parts[-1]
            else:
                split_idx = max(len(prompt) - 200, 0)
                system_prompt = prompt[:split_idx]
                user_query = prompt[split_idx:]
        else:
            system_prompt = system_prompt or ""
            user_query = user_query or ""
            full_text = system_prompt + "\n" + user_query

        prefix_tokens = self.tokenize(system_prompt)
        full_tokens = self.tokenize(full_text)

        prefix_len = min(len(prefix_tokens), len(full_tokens))
        new_tokens = full_tokens[prefix_len:]
        if not new_tokens:
            new_tokens = [self.tokenizer.eos_token_id]

        prefix_hash = KVEntry.compute_prefix_hash(prefix_tokens)
        cached_entry = self.cache.get(prefix_hash)

        central_hit = False
        if cached_entry is None and self.central_store is not None:
            # Central-store hit: fetch KV in CPU memory, insert into local CPU tier,
            # then proceed as a normal cache hit.
            rec = self.central_store.get(prefix_hash)
            if rec is not None:
                central_hit = True
                kv_tuple = rec.kv_tuple
                kv_bytes = KVEntry.measure_kv_size(kv_tuple)
                fetched = KVEntry(
                    prefix_hash=prefix_hash,
                    prefix_tokens=prefix_tokens,
                    prompt_text=system_prompt,
                    num_tokens=len(prefix_tokens),
                    past_key_values=kv_tuple,
                    size_bytes=kv_bytes,
                    worker_id=self.worker_id,
                    tier="cpu",
                )
                self.cache.put_cpu(fetched)
                cached_entry = self.cache.get(prefix_hash)

        if cached_entry is not None:
            result = self._process_with_cached_kv(
                cached_entry,
                new_tokens,
                prefix_len,
                max_new_tokens,
            )
            result["cache_hit"] = True
            result["tier_hit"] = cached_entry.last_hit_tier or cached_entry.tier
            result["central_hit"] = central_hit

            if self.metadata_registry is not None:
                try:
                    self.metadata_registry.claim_replica(
                        prefix_hash=prefix_hash,
                        worker_id=self.metadata_worker_id,
                        ttl_s=self.lease_ttl_s,
                    )
                except Exception:
                    pass

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
            result["central_hit"] = False
            result["savings_ms"] = 0.0
            self._miss_prefill_times[prefix_hash] = result["prefill_ms"]

            if self.metadata_registry is not None:
                try:
                    self.metadata_registry.claim_replica(
                        prefix_hash=prefix_hash,
                        worker_id=self.metadata_worker_id,
                        ttl_s=self.lease_ttl_s,
                    )
                except Exception:
                    pass

            # Write-through to central store (best-effort).
            if self.central_store is not None:
                try:
                    # We have a GPU-resident cloned KV stored in local cache; create a CPU copy for sharing.
                    # This is intentionally explicit to keep CentralKVStore a pure storage layer.
                    gpu_entry = self.cache.tiers[StorageTier.GPU].get(prefix_hash)
                    if gpu_entry is not None and gpu_entry.past_key_values is not None:
                        cpu_kv = tuple(
                            (k.to("cpu"), v.to("cpu")) for k, v in gpu_entry.past_key_values
                        )
                        self.central_store.put(prefix_hash, cpu_kv)
                except Exception as e:
                    logger.debug("Central store put failed for %s: %s", prefix_hash, e)

        self.total_prefill_ms += result["prefill_ms"]
        self.total_decode_ms += result["decode_ms"]
        self.total_ttft_ms += result["ttft_ms"]
        self.total_itl_ms += result["avg_itl_ms"]
        self.total_generated_tokens += result["generated_tokens"]

        result["prefix_hash"] = prefix_hash
        return result

    def _process_with_cached_kv(
        self,
        entry: KVEntry,
        new_tokens: list[int],
        prefix_len: int,
        max_new_tokens: int,
    ) -> dict:
        kv_on_gpu = entry.get_kv_on_device(self.device)
        if kv_on_gpu is None:
            logger.warning(f"Cached KV missing for {entry.prefix_hash}; falling back.")
            return self._process_without_cache(
                self.tokenize(entry.prompt_text) + new_tokens,
                max_new_tokens,
            )

        new_input_ids = torch.tensor([new_tokens], device=self.device)
        position_ids = torch.arange(
            prefix_len,
            prefix_len + len(new_tokens),
            device=self.device,
        ).unsqueeze(0)

        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=kv_on_gpu,
                position_ids=position_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1000

        decode_start = time.perf_counter()
        gen = self._generate_with_timing(
            outputs.past_key_values,
            outputs.logits,
            max_new_tokens,
            prefix_len + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - decode_start) * 1000

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "ttft_ms": prefill_ms + gen["first_token_ms"],
            "avg_itl_ms": gen["avg_itl_ms"],
            "generated_text": self.tokenizer.decode(gen["tokens"], skip_special_tokens=True),
            "generated_tokens": len(gen["tokens"]),
            "output_tokens_per_s": gen["output_tokens_per_s"],
        }

    def _process_full_and_cache(
        self,
        prefix_tokens: list[int],
        new_tokens: list[int],
        prefix_hash: str,
        prompt_text: str,
        max_new_tokens: int,
    ) -> dict:
        prefix_input_ids = torch.tensor([prefix_tokens], device=self.device)

        torch.cuda.synchronize()
        prefix_start = time.perf_counter()
        with torch.no_grad():
            prefix_out = self.model(input_ids=prefix_input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefix_ms = (time.perf_counter() - prefix_start) * 1000

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

        new_input_ids = torch.tensor([new_tokens], device=self.device)
        position_ids = torch.arange(
            len(prefix_tokens),
            len(prefix_tokens) + len(new_tokens),
            device=self.device,
        ).unsqueeze(0)

        torch.cuda.synchronize()
        query_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=prefix_out.past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        query_ms = (time.perf_counter() - query_start) * 1000

        total_prefill_ms = prefix_ms + query_ms

        decode_start = time.perf_counter()
        gen = self._generate_with_timing(
            outputs.past_key_values,
            outputs.logits,
            max_new_tokens,
            len(prefix_tokens) + len(new_tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - decode_start) * 1000

        return {
            "prefill_ms": total_prefill_ms,
            "prefix_only_ms": prefix_ms,
            "decode_ms": decode_ms,
            "total_ms": total_prefill_ms + decode_ms,
            "ttft_ms": total_prefill_ms + gen["first_token_ms"],
            "avg_itl_ms": gen["avg_itl_ms"],
            "kv_cached_mb": kv_bytes / 1024 / 1024,
            "generated_text": self.tokenizer.decode(gen["tokens"], skip_special_tokens=True),
            "generated_tokens": len(gen["tokens"]),
            "output_tokens_per_s": gen["output_tokens_per_s"],
        }

    def _process_without_cache(self, tokens: list[int], max_new_tokens: int) -> dict:
        input_ids = torch.tensor([tokens], device=self.device)

        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1000

        decode_start = time.perf_counter()
        gen = self._generate_with_timing(
            outputs.past_key_values,
            outputs.logits,
            max_new_tokens,
            len(tokens),
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - decode_start) * 1000

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": prefill_ms + decode_ms,
            "ttft_ms": prefill_ms + gen["first_token_ms"],
            "avg_itl_ms": gen["avg_itl_ms"],
            "generated_text": self.tokenizer.decode(gen["tokens"], skip_special_tokens=True),
            "generated_tokens": len(gen["tokens"]),
            "output_tokens_per_s": gen["output_tokens_per_s"],
        }

    def _generate_with_timing(self, past_kv, logits, max_tokens: int, current_pos: int) -> dict:
        generated = []
        first_token_ms = 0.0
        inter_token_ms = []
        token_starts = []

        for i in range(max_tokens):
            step_start = time.perf_counter()

            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                if i == 0:
                    torch.cuda.synchronize()
                    first_token_ms = (time.perf_counter() - step_start) * 1000
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
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1000

            if i == 0:
                first_token_ms = step_ms
            else:
                inter_token_ms.append(step_ms)

            past_kv = out.past_key_values
            logits = out.logits
            token_starts.append(step_ms)

        avg_itl_ms = float(sum(inter_token_ms) / len(inter_token_ms)) if inter_token_ms else 0.0
        total_decode_ms = float(sum(token_starts))
        output_tokens_per_s = (len(generated) / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0

        return {
            "tokens": generated,
            "first_token_ms": first_token_ms,
            "avg_itl_ms": avg_itl_ms,
            "output_tokens_per_s": output_tokens_per_s,
        }

    def get_stats(self) -> dict:
        cache_stats = self.cache.get_stats()
        n = max(self.requests_processed, 1)
        return {
            **cache_stats,
            "requests_processed": self.requests_processed,
            "total_prefill_ms": self.total_prefill_ms,
            "total_decode_ms": self.total_decode_ms,
            "total_cache_saved_ms": self.total_cache_saved_ms,
            "total_ttft_ms": self.total_ttft_ms,
            "avg_ttft_ms": self.total_ttft_ms / n,
            "total_itl_ms": self.total_itl_ms,
            "avg_itl_ms": self.total_itl_ms / n,
            "total_generated_tokens": self.total_generated_tokens,
            "model": self.model_config.model_path,
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        }

    def reset(self):
        self.cache.reset()
        self.requests_processed = 0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_cache_saved_ms = 0.0
        self.total_ttft_ms = 0.0
        self.total_itl_ms = 0.0
        self.total_generated_tokens = 0
        self._miss_prefill_times.clear()
