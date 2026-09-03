"""Embedding functions for the two vector indexes.

Two options, chosen by CDD_EMBEDDINGS:

* ``default`` - Chroma's bundled ONNX model. Runs locally, so data-room text never
  leaves the machine, which matters given the confidentiality carry-through in design
  spec s VIII.
* ``hash`` - a deterministic hashing embedder. Not semantically meaningful; it exists
  so the retrieval *plumbing* (metadata filters, supersession, similarity floor,
  de-duplication) can be tested without downloading a model or making network calls.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from cdd_agent.config import get_settings

_HASH_DIM = 256


class HashingEmbeddingFunction:
    """Deterministic bag-of-words hashing embedder, L2-normalised.

    Implements both halves of Chroma's embedding-function protocol: the legacy
    callable form and the newer `embed_documents` / `embed_query` pair, so it keeps
    working across Chroma versions rather than failing at query time.
    """

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "cdd-hashing"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def get_config(self) -> dict[str, int]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict[str, int]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(dim=int(config.get("dim", _HASH_DIM)))

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in _as_sequence(input)]

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: Sequence[str] | str) -> list[list[float]]:
        return self(input)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in _normalise(text).split() if t]
        for token in tokens:
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return [0.0] * self.dim
        return [v / norm for v in vec]


def _as_sequence(value: Sequence[str] | str) -> Sequence[str]:
    return [value] if isinstance(value, str) else value


def _normalise(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text)


def get_embedding_function() -> Any:
    settings = get_settings()
    if settings.embeddings == "hash":
        return HashingEmbeddingFunction()
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()
