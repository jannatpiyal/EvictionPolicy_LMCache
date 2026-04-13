"""
Eviction policies: base class and all implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class EvictionPolicy(ABC):
    """Base class for all eviction policies."""

    def __init__(self, name: str):
        self.name = name
        self._eviction_count = 0

    @abstractmethod
    def select_victim(self, entries: list) -> Optional[object]:
        """Select a KVEntry to evict."""
        pass

    def on_access(self, entry) -> None:
        pass

    def on_insert(self, entry) -> None:
        pass

    def on_evict(self, entry) -> None:
        self._eviction_count += 1

    def reset(self) -> None:
        self._eviction_count = 0

    @property
    def eviction_count(self) -> int:
        return self._eviction_count


class LRUPolicy(EvictionPolicy):
    """Evict least recently accessed entry."""

    def __init__(self):
        super().__init__(name="LRU")

    def select_victim(self, entries):
        if not entries:
            return None
        return min(entries, key=lambda e: e.last_accessed_at)


class LFUPolicy(EvictionPolicy):
    """Evict least frequently accessed entry (LRU tie-break)."""

    def __init__(self):
        super().__init__(name="LFU")

    def select_victim(self, entries):
        if not entries:
            return None
        return min(entries, key=lambda e: (e.access_count, e.last_accessed_at))


class SemanticEvictionPolicy(EvictionPolicy):
    """
    Evict entries least semantically similar to recent request patterns.
    Uses sentence-transformers for embeddings.
    """

    def __init__(self, window_size=50, similarity_weight=0.7, recency_weight=0.3):
        super().__init__(name="Semantic")
        self.window_size = window_size
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.recent_embeddings = []
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def compute_embedding(self, text):
        return self._get_encoder().encode(text, normalize_embeddings=True)

    def _avg_similarity(self, entry):
        if entry.embedding is None or not self.recent_embeddings:
            return 0.0
        entry_emb = np.array(entry.embedding)
        sims = [float(np.dot(entry_emb, r)) for r in self.recent_embeddings]
        return float(np.mean(sims))

    def _score(self, entry):
        sim = self._avg_similarity(entry)
        recency = 1.0 / (1.0 + entry.time_since_last_access())
        return self.similarity_weight * sim + self.recency_weight * recency

    def select_victim(self, entries):
        if not entries:
            return None
        has_emb = any(e.embedding is not None for e in entries)
        if not has_emb:
            return min(entries, key=lambda e: e.last_accessed_at)
        return min(entries, key=lambda e: self._score(e))

    def on_access(self, entry):
        if entry.embedding is not None:
            self.recent_embeddings.append(np.array(entry.embedding))
            if len(self.recent_embeddings) > self.window_size:
                self.recent_embeddings.pop(0)

    def on_insert(self, entry):
        if entry.prompt_text and entry.embedding is None:
            try:
                entry.embedding = self.compute_embedding(entry.prompt_text).tolist()
            except Exception:
                pass
        if entry.embedding is not None:
            self.recent_embeddings.append(np.array(entry.embedding))
            if len(self.recent_embeddings) > self.window_size:
                self.recent_embeddings.pop(0)

    def reset(self):
        super().reset()
        self.recent_embeddings.clear()


class LearnedEvictionPolicy(EvictionPolicy):
    """ML-based eviction using online logistic regression."""

    def __init__(self, retrain_interval=200, learning_rate=0.01):
        super().__init__(name="Learned")
        self.retrain_interval = retrain_interval
        self.learning_rate = learning_rate
        self.weights = np.zeros(6)
        self.training_buffer = []
        self._decisions = 0

    def _features(self, entry):
        return np.array([
            entry.access_count / 100.0,
            1.0 / (1.0 + entry.time_since_last_access()),
            1.0 / (1.0 + entry.age()),
            entry.size_bytes / (1024 * 1024),
            1.0 / (1.0 + entry.last_reuse_gap),
            1.0,
        ])

    def _predict(self, entry):
        f = self._features(entry)
        logit = np.dot(self.weights, f)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20))))

    def _retrain(self):
        if len(self.training_buffer) < 10:
            return
        for _ in range(5):
            for features, label in self.training_buffer:
                pred = 1.0 / (1.0 + np.exp(-np.clip(np.dot(self.weights, features), -20, 20)))
                self.weights += self.learning_rate * features * (label - pred)
        if len(self.training_buffer) > 2000:
            self.training_buffer = self.training_buffer[-1000:]
        self._decisions = 0

    def select_victim(self, entries):
        if not entries:
            return None
        self._decisions += 1
        if self._decisions >= self.retrain_interval:
            self._retrain()
        return min(entries, key=lambda e: self._predict(e))

    def on_access(self, entry):
        self.training_buffer.append((self._features(entry), 1.0))

    def on_evict(self, entry):
        super().on_evict(entry)
        self.training_buffer.append((self._features(entry), 0.0))

    def reset(self):
        super().reset()
        self.weights = np.zeros(6)
        self.training_buffer.clear()
        self._decisions = 0


# --- Factory ---

POLICY_REGISTRY = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "semantic": SemanticEvictionPolicy,
    "learned": LearnedEvictionPolicy,
}


def create_policy(policy_type, **kwargs):
    """Create an eviction policy. Accepts enum or string."""
    from config import EvictionPolicyType
    if isinstance(policy_type, EvictionPolicyType):
        key = policy_type.value
    else:
        key = str(policy_type)
    cls = POLICY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown policy: {key}. Available: {list(POLICY_REGISTRY.keys())}")
    return cls(**kwargs)
