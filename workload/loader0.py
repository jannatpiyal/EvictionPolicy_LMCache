"""
Workload Generator: Creates inference request streams with shared system prompts.
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
    """A single inference request."""
    request_id: int
    system_prompt: str
    user_query: str
    max_new_tokens: int
    prefix_group: int          # Which system prompt group this belongs to
    timestamp: float


class WorkloadGenerator:
    """
    Generates request streams with shared system prompts (prefixes).
    Prefix popularity follows Zipf distribution.
    """

    SYSTEM_PROMPTS = [
        "You are a helpful AI assistant. Answer the user's questions accurately and concisely.",
        "You are an expert software engineer. Help the user debug code and design systems.",
        "You are a creative writing assistant. Help craft engaging stories and narratives.",
        "You are a math tutor. Explain mathematical concepts step by step with examples.",
        "You are a data science expert. Help with analysis, ML models, and data pipelines.",
        "You are a medical information assistant providing general health guidance.",
        "You are a financial advisor assistant helping with budgets and investments.",
        "You are a travel planning expert. Suggest destinations, itineraries, and tips.",
        "You are a cooking assistant. Help with recipes, techniques, and meal planning.",
        "You are a legal information assistant providing general guidance on common questions.",
        "You are a customer support agent for a technology company. Be patient and thorough.",
        "You are a history professor. Explain historical events with context and nuance.",
        "You are a science communicator. Make complex scientific topics accessible.",
        "You are a career counselor. Help with job searches, resumes, and career planning.",
        "You are a language learning tutor. Help students practice and improve their skills.",
    ]

    USER_QUERIES = [
        "What is a linked list and when should I use one?",
        "How does quicksort work? Walk me through an example.",
        "Explain the difference between TCP and UDP.",
        "What is dynamic programming? Give me a simple example.",
        "How do hash tables handle collisions?",
        "What is the CAP theorem in distributed systems?",
        "Explain Big O notation with common examples.",
        "What is a binary search tree?",
        "How does garbage collection work in Java?",
        "What is the difference between a process and a thread?",
        "Explain how HTTPS encryption works.",
        "What are design patterns? Name three common ones.",
        "How does a load balancer distribute traffic?",
        "What is eventual consistency?",
        "Explain the difference between SQL and NoSQL databases.",
        "What is containerization and why is Docker popular?",
        "How does Git branching work?",
        "What is CI/CD and why is it important?",
        "Explain RESTful API design principles.",
        "What is microservices architecture?",
        "How do recommendation systems work?",
        "What is transfer learning in deep learning?",
        "Explain the attention mechanism in transformers.",
        "What is gradient descent and how does it work?",
        "How does backpropagation work in neural networks?",
    ]

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.py_rng = random.Random(config.seed)

    def _sample_prefix_index(self) -> int:
        n = min(self.config.num_unique_prefixes, len(self.SYSTEM_PROMPTS))
        weights = np.array([1.0 / (i + 1) ** self.config.zipf_alpha for i in range(n)])
        weights /= weights.sum()
        return int(self.rng.choice(n, p=weights))

    def generate(self, num_requests: Optional[int] = None) -> list[InferenceRequest]:
        n = num_requests or self.config.num_requests
        requests = []
        arrival_time = 0.0
        num_prefixes = min(self.config.num_unique_prefixes, len(self.SYSTEM_PROMPTS))

        for i in range(n):
            if self.rng.random() < self.config.prefix_reuse_ratio:
                prefix_idx = self._sample_prefix_index()
            else:
                prefix_idx = int(self.rng.integers(0, num_prefixes))

            query = self.py_rng.choice(self.USER_QUERIES)

            request = InferenceRequest(
                request_id=i,
                system_prompt=self.SYSTEM_PROMPTS[prefix_idx],
                user_query=query,
                max_new_tokens=self.config.max_new_tokens,
                prefix_group=prefix_idx,
                timestamp=arrival_time,
            )
            requests.append(request)
            arrival_time += self.rng.exponential(10.0)

        logger.info(
            f"Generated {len(requests)} requests with "
            f"{num_prefixes} unique prefixes, "
            f"reuse_ratio={self.config.prefix_reuse_ratio}"
        )
        return requests

    def get_reuse_stats(self, requests: list[InferenceRequest]) -> dict:
        freq = {}
        for r in requests:
            freq[r.prefix_group] = freq.get(r.prefix_group, 0) + 1
        return {
            "total_requests": len(requests),
            "unique_prefixes_seen": len(freq),
            "max_reuse": max(freq.values()) if freq else 0,
            "prefix_distribution": dict(sorted(freq.items(), key=lambda x: -x[1])),
        }
