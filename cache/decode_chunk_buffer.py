from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecodeChunkBuffer:
    """
    Tracks when decode-time KV should be flushed to storage.

    The worker can append tokens one by one during generation, but only flush
    the updated KV snapshot when at least one full chunk of newly generated
    tokens has accumulated. A final partial chunk can be flushed at request end.
    """

    chunk_size_tokens: int
    flushed_tokens: int = 0

    def should_flush(self, total_generated_tokens: int) -> bool:
        if self.chunk_size_tokens <= 0:
            return False
        return (total_generated_tokens - self.flushed_tokens) >= self.chunk_size_tokens

    def mark_flushed(self, total_generated_tokens: int) -> int:
        self.flushed_tokens = total_generated_tokens
        return self.flushed_tokens

    def has_pending(self, total_generated_tokens: int) -> bool:
        return total_generated_tokens > self.flushed_tokens
