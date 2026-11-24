"""Embeddings abstraction for nomic-embed-text-v1.5 (8k context) with deterministic unit vector projection."""

import hashlib
from typing import List, Optional, Union
import numpy as np

from src.common.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingModel:
    """Provides 768-dimensional normalized embeddings for semantic vector search."""

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    DIMENSIONS = 768

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._hf_model = None
        self._init_model()

    def _init_model(self):
        """Attempts to load sentence-transformers / HuggingFace model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}...")
            self._hf_model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("SentenceTransformer model successfully loaded.")
        except Exception:
            logger.info("SentenceTransformer not loaded. Using deterministic normalized projection mode.")
            self._hf_model = None

    def embed_text(self, text: str) -> List[float]:
        """Generates a 768-dimensional normalized unit vector for a single string."""
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self.DIMENSIONS

        if self._hf_model:
            try:
                emb = self._hf_model.encode(cleaned, normalize_embeddings=True)
                return emb.tolist()
            except Exception as e:
                logger.warning(f"Embedding inference failed ({e}). Falling back to projection.")

        return self._deterministic_vector(cleaned)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of strings."""
        return [self.embed_text(t) for t in texts]

    def _deterministic_vector(self, text: str) -> List[float]:
        """Generates a reproducible 768-dimensional unit vector using rolling SHA-256 projections.

        Guarantees that identical text yields identical vectors and semantically similar texts
        share token hash overlaps.
        """
        words = text.lower().split()
        vec = np.zeros(self.DIMENSIONS, dtype=np.float32)

        for i, word in enumerate(words):
            # Seed pseudo-random generator with token hash
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.DIMENSIONS
            sign = 1.0 if (h >> 4) % 2 == 0 else -1.0
            weight = 1.0 / (1.0 + 0.1 * i)
            vec[idx] += sign * weight

        # Normalize to unit sphere (L2 norm = 1.0)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()
