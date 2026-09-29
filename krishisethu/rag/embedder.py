"""Embedding interface + impls (Tier 3).

Wraps ``OllamaEmbeddings`` (bge-m3:567m) behind a swappable ``Embedder``
interface so a different multilingual embedder can be dropped in without
touching the store / retriever. The live embedder asserts the dimension is
constant (bge-m3 is 1024-d) and rejects any embedding model that changes the
space, because index and query must never mix embedding models.

``StubEmbedder`` is a deterministic, network-free stand-in used by the hermetic
unit tests: it is seeded by ``(seed, text)`` so the same text always maps to the
same unit vector, and a fixed seed makes every run byte-reproducible.
"""
from __future__ import annotations

import math
import random
from typing import Protocol


class Embedder(Protocol):
    """Minimal embedding contract used by the store and retriever."""

    dim: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


#: bge-m3:567m is 1024-d; never index in one dim and query in another.
EXPECTED_DIM = 1024


class OllamaEmbedder:
    """Live embedder backed by a local Ollama server (bge-m3:567m)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, expected_dim: int = EXPECTED_DIM):
        from langchain_ollama import OllamaEmbeddings

        from krishisethu.config.settings import get_settings

        s = get_settings()
        model = model or s.ollama.embedding_model
        base_url = base_url or s.ollama.embed_base()
        self.model = model
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)
        probe = self.embed_query("dimension probe text")
        self.dim = len(probe)
        if self.dim != expected_dim:
            raise ValueError(
                f"embedding model {model!r} returns {self.dim}-d, expected {expected_dim}-d "
                "(index and query must never mix embedding models)"
            )

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)


class StubEmbedder:
    """Deterministic, network-free embedder for hermetic tests.

    Vectors are seeded by ``(seed, text)`` and l2-normalized so cosine similarity
    is meaningful. Same text -> same bytes; different text -> different vector.
    """

    def __init__(self, dim: int = EXPECTED_DIM, seed: int = 0):
        self.dim = dim
        self._seed = seed

    def _vec(self, text: str) -> list[float]:
        rng = random.Random(f"{self._seed}:{text}")
        v = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]
