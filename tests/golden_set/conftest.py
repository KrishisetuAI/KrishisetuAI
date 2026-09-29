"""Golden-set fixtures: a deterministic, CI-safe retriever (stub embedder).

No live Ollama and no ChromaDB persistence: ``s_rag`` is computed in-process over
the corpus with a deterministic stub embedder, so the safety-verdict harness is
100% reproducible on CI and immune to ChromaDB's Windows index-flush race. The
real governed retrieval is covered by ``tests/test_rag.py``.
"""
from __future__ import annotations

import pytest

from harness import InProcessGoldenRetriever


@pytest.fixture(scope="session")
def golden_retriever():
    """Chunk the real RAG corpus once, return an in-process S_RAG retriever."""
    return InProcessGoldenRetriever()
