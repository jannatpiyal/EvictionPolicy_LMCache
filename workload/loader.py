"""
Workload Generator: LMCache-style Long Doc QA workload.

Key behavior:
- Round 0 = warmup (all misses)
- Later rounds = a mix of repeated documents and new documents
- Same document text -> same long prefix -> KV reuse
- Repeat ordering can mimic LMCache's random/tile/interleave styles
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional

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
    TOPICS = [
        "distributed systems",
        "database internals",
        "LLM serving infrastructure",
        "retrieval augmented generation",
        "GPU memory management",
        "operating systems",
        "scientific computing",
        "compiler optimization",
        "network performance engineering",
        "fault tolerant storage",
        "machine learning systems",
        "real-time analytics",
    ]

    STYLES = [
        "technical design review",
        "incident postmortem",
        "research summary",
        "implementation guide",
        "benchmark report",
        "architecture memo",
    ]

    DETAIL_SNIPPETS = [
        "The authors compare baseline throughput, tail latency, and memory residency under realistic bursty traffic.",
        "A recurring theme is the tradeoff between transfer overhead and saved prefill computation for long-context prompts.",
        "Several sections focus on how scheduling, batching, and admission control interact under sustained load.",
        "The document highlights operational concerns such as observability, fault isolation, and rollback behavior.",
        "A large portion of the analysis is devoted to cache locality, request skew, and the impact of repeated prefixes.",
        "The discussion includes concrete implementation details, expected bottlenecks, and empirical findings.",
        "One section explains why similar workloads can still lead to very different cache hit rates depending on reuse order.",
        "Another section emphasizes hardware-aware optimization, especially transfer pipelining and memory hierarchy effects.",
        "The report presents several tables and examples that contrast warmup behavior with steady-state execution.",
        "The closing section offers deployment guidance and limitations that future iterations should address.",
    ]

    QUESTION_TEMPLATES = {
        "document_qa": [
            "What problem is this document trying to solve?",
            "Which design decisions matter most in this document?",
            "What performance bottlenecks are highlighted here?",
            "How would you explain the main tradeoffs to an engineer?",
            "What risks or limitations does the document mention?",
        ],
        "summarization": [
            "Summarize the main idea of this document in a few sentences.",
            "Give a concise executive summary of the document.",
            "Summarize the most important findings and conclusions.",
            "Write a short brief for someone who has not read the document.",
        ],
        "mixed": [
            "Summarize the main idea of this document.",
            "What are the key takeaways?",
            "Explain this document for a beginner.",
            "What are the main conclusions?",
            "What are the key techniques discussed?",
            "What tradeoffs are most important here?",
            "Which operational details are most relevant for deployment?",
        ],
    }

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.py_rng = random.Random(config.seed)

        self.num_documents = getattr(config, "num_documents", 40)
        self.document_length = getattr(config, "document_length_tokens", 10000)
        self.num_rounds = getattr(config, "num_rounds", 2)
        self.max_new_tokens = getattr(config, "max_new_tokens", 100)
        self.hit_ratio = getattr(config, "hit_ratio", 1.0)
        self.arrival_mode = getattr(config, "arrival_mode", "bursty")
        self.interarrival_mean_sec = getattr(config, "interarrival_mean_sec", 0.2)
        self.question_style = getattr(config, "question_style", "document_qa")
        self.num_questions_per_document = getattr(config, "num_questions_per_document", 4)
        self.repeat_mode = getattr(config, "repeat_mode", "tile")

    # -----------------------------
    # DOCUMENT GENERATION
    # -----------------------------
    def _document_seed(self, doc_id: int) -> int:
        return self.config.seed * 1_000_003 + doc_id * 97

    def _make_document(self, doc_id: int) -> str:
        doc_rng = random.Random(self._document_seed(doc_id))
        topic = self.TOPICS[doc_id % len(self.TOPICS)]
        style = self.STYLES[doc_id % len(self.STYLES)]

        header = (
            f"Document {doc_id}: A {style} about {topic}.\n"
            f"Section 1 introduces the workload assumptions, architecture, and evaluation goals.\n"
        )
        words = header.split()

        paragraph_idx = 0
        while len(words) < self.document_length:
            para_topic = self.TOPICS[(doc_id + paragraph_idx) % len(self.TOPICS)]
            para_style = self.STYLES[(doc_id + 2 * paragraph_idx) % len(self.STYLES)]
            details = doc_rng.sample(self.DETAIL_SNIPPETS, k=min(4, len(self.DETAIL_SNIPPETS)))
            paragraph = (
                f"Section {paragraph_idx + 2}. This {para_style} discusses {para_topic}. "
                f"{' '.join(details)} "
                f"Example {paragraph_idx}: latency={10 + (doc_id + paragraph_idx) % 37}ms, "
                f"throughput={100 + ((doc_id * 13 + paragraph_idx * 17) % 900)} requests per second. "
                f"The scenario references workers, storage tiers, prefix reuse, and scheduling choices."
            )
            words.extend(paragraph.split())
            paragraph_idx += 1

        return " ".join(words[:self.document_length])

    def _question_pool(self) -> list[str]:
        return list(self.QUESTION_TEMPLATES.get(self.question_style, self.QUESTION_TEMPLATES["mixed"]))

    def _make_questions(self, doc_id: int) -> list[str]:
        pool = self._question_pool()
        doc_rng = random.Random(self._document_seed(doc_id) + 17)
        if len(pool) >= self.num_questions_per_document:
            return doc_rng.sample(pool, k=self.num_questions_per_document)
        return [doc_rng.choice(pool) for _ in range(self.num_questions_per_document)]

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
    # REQUEST ORDERING
    # -----------------------------
    def _order_requests(self, docs: list[int], round_id: int) -> list[int]:
        if self.repeat_mode == "random":
            ordered = list(docs)
            self.py_rng.shuffle(ordered)
            return ordered

        if self.repeat_mode == "interleave":
            evens = docs[::2]
            odds = docs[1::2]
            ordered: list[int] = []
            while evens or odds:
                if evens:
                    ordered.append(evens.pop(0))
                if odds:
                    ordered.append(odds.pop(0))
            return ordered

        if self.repeat_mode == "tile":
            if round_id == 0:
                return list(docs)
            prior_docs = [doc_id for doc_id in docs if doc_id < self.num_documents]
            new_docs = [doc_id for doc_id in docs if doc_id >= self.num_documents]
            return prior_docs + new_docs

        ordered = list(docs)
        self.py_rng.shuffle(ordered)
        return ordered

    def _advance_timestamp(self, timestamp: float, req_idx_in_round: int) -> float:
        if self.arrival_mode == "poisson":
            return timestamp + self.py_rng.expovariate(1.0 / max(self.interarrival_mean_sec, 1e-6))

        # "bursty": launch an initial burst, then add spacing after the first wave.
        initial_window = max(1, int(getattr(self.config, "initial_concurrency", 1)))
        if req_idx_in_round < initial_window:
            return timestamp
        return timestamp + self.interarrival_mean_sec

    # -----------------------------
    # CORE GENERATION
    # -----------------------------
    def generate(self, num_requests: Optional[int] = None):
        base_documents = {i: self._make_document(i) for i in range(self.num_documents)}
        base_questions = {i: self._make_questions(i) for i in base_documents}

        requests = []
        request_id = 0
        timestamp = 0.0

        for round_id in range(self.num_rounds):
            if round_id == 0:
                hit_docs: list[int] = []
                miss_docs = list(base_documents.keys())
            else:
                num_hits = min(self.num_documents, max(0, int(round(self.hit_ratio * self.num_documents))))
                hit_docs = self.py_rng.sample(list(base_documents.keys()), k=num_hits)
                num_misses = self.num_documents - num_hits
                miss_docs = [self.num_documents + round_id * 1000 + i for i in range(num_misses)]

            doc_sequence = self._order_requests(hit_docs + miss_docs, round_id=round_id)

            for req_idx_in_round, doc_id in enumerate(doc_sequence):
                if doc_id in base_documents:
                    doc_text = base_documents[doc_id]
                    questions = base_questions[doc_id]
                else:
                    doc_text = self._make_document(doc_id)
                    questions = self._make_questions(doc_id)

                question = questions[round_id % len(questions)]
                prompt = self._build_prompt(doc_text, question)

                requests.append(
                    InferenceRequest(
                        request_id=request_id,
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        prefix_group=doc_id,
                        timestamp=timestamp,
                        round_id=round_id,
                        document_id=doc_id,
                        question_id=(doc_id * 100) + (round_id % len(questions)),
                    )
                )
                request_id += 1
                timestamp = self._advance_timestamp(timestamp, req_idx_in_round)

        if num_requests is not None:
            requests = requests[:num_requests]

        logger.info(
            "Generated workload: %d requests | %d docs | %d rounds | hit_ratio=%.2f | repeat_mode=%s",
            len(requests),
            self.num_documents,
            self.num_rounds,
            self.hit_ratio,
            self.repeat_mode,
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
            "max_reuse": max(freq.values()) if freq else 0,
            "round_distribution": round_stats,
            "repeat_mode": self.repeat_mode,
            "question_style": self.question_style,
        }
