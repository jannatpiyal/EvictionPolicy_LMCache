"""
Workload Generator: LMCache-style long-document multi-round QA workload.

Key behavior:
- Round 0 = warmup (all misses)
- Later rounds = mix of hits + misses controlled by hit_ratio
- Same document → same prefix → KV reuse
"""

import random
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import WorkloadConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: int
    prompt: str
    max_new_tokens: int
    prefix_group: int
    timestamp: float
    round_id: int
    document_id: int
    question_id: int


class WorkloadGenerator:

    QUESTION_TEMPLATES = [
        "Summarize the main idea of this document.",
        "What are the key takeaways?",
        "Explain this document for a beginner.",
        "What are the main conclusions?",
        "What are the key techniques discussed?",
    ]

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.py_rng = random.Random(config.seed)

        self.num_documents = getattr(config, "num_documents", 40)
        self.document_length = getattr(config, "document_length_tokens", 10000)
        self.num_rounds = getattr(config, "num_rounds", 2)
        self.max_new_tokens = getattr(config, "max_new_tokens", 100)
        self.hit_ratio = getattr(config, "hit_ratio", 1.0)

    # -----------------------------
    # DOCUMENT GENERATION
    # -----------------------------
    def _make_document(self, doc_id: int) -> str:
        base = f"Document {doc_id}. "
        words = base.split()

        while len(words) < self.document_length:
            words += [
                "This", "document", "contains", "important", "information",
                "about", "systems", "performance", "design", "tradeoffs."
            ]

        return " ".join(words[:self.document_length])

    def _build_prompt(self, doc_text: str, question: str) -> str:
        return (
            "You are a helpful assistant.\n\n"
            "Document:\n"
            f"{doc_text}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )

    # -----------------------------
    # CORE GENERATION
    # -----------------------------
    def generate(self, num_requests: Optional[int] = None):

        base_documents = {
            i: self._make_document(i)
            for i in range(self.num_documents)
        }

        requests = []
        request_id = 0
        timestamp = 0.0

        for round_id in range(self.num_rounds):

            # -----------------------------
            # Decide hits vs misses
            # -----------------------------
            if round_id == 0:
                hit_docs = []
                miss_docs = list(base_documents.keys())
            else:
                num_hits = int(self.hit_ratio * self.num_documents)

                hit_docs = self.py_rng.sample(
                    list(base_documents.keys()),
                    k=num_hits
                )

                num_misses = self.num_documents - num_hits

                # generate new doc ids for misses
                miss_docs = [
                    self.num_documents + round_id * 1000 + i
                    for i in range(num_misses)
                ]

            # -----------------------------
            # Combine
            # -----------------------------
            doc_sequence = hit_docs + miss_docs
            self.py_rng.shuffle(doc_sequence)

            # -----------------------------
            # Generate requests
            # -----------------------------
            for doc_id in doc_sequence:

                if doc_id in base_documents:
                    doc_text = base_documents[doc_id]
                else:
                    doc_text = self._make_document(doc_id)

                question = self.py_rng.choice(self.QUESTION_TEMPLATES)
                prompt = self._build_prompt(doc_text, question)

                req = InferenceRequest(
                    request_id=request_id,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    prefix_group=doc_id,
                    timestamp=timestamp,
                    round_id=round_id,
                    document_id=doc_id,
                    question_id=request_id,
                )

                requests.append(req)
                request_id += 1

                # simple arrival spacing
                timestamp += self.rng.exponential(0.2)

        if num_requests is not None:
            requests = requests[:num_requests]

        logger.info(
            f"Generated workload: "
            f"{len(requests)} requests | "
            f"{self.num_documents} docs | "
            f"{self.num_rounds} rounds | "
            f"hit_ratio={self.hit_ratio}"
        )

        return requests

    # -----------------------------
    # STATS
    # -----------------------------
    def get_reuse_stats(self, requests):

        freq = {}
        round_stats = {}

        for r in requests:
            freq[r.prefix_group] = freq.get(r.prefix_group, 0) + 1
            round_stats[r.round_id] = round_stats.get(r.round_id, 0) + 1

        reused = sum(1 for v in freq.values() if v > 1)

        return {
            "total_requests": len(requests),
            "unique_prefixes": len(freq),
            "reused_prefixes": reused,
            "max_reuse": max(freq.values()),
            "round_distribution": round_stats,
        }