"""TIER 3 — Domain-Governed Vector RAG.

Long-tail KVK Q&A / ICAR circulars / advisories. Every chunk tagged crop/
district/season/source_type; pre-retrieval metadata filter; top-k <= 5.
ChromaDB (local) / pgvector (hosted) behind a VectorStore protocol.

Public API:
  schema     RagDoc, rag_doc_from_text/file, crop_season_collection, parse_frontmatter
  chunker    Chunk, chunk_markdown, dedupe_docs, TARGET_MIN/MAX_TOKENS, OVERLAP_TOKENS
  embedder   Embedder, OllamaEmbedder, StubEmbedder, EXPECTED_DIM
  store      VectorStore, ChromaLocal, StoredChunk
  ingest     ingest_corpus, IngestResult
  retriever  Retriever, PlotContext, metadata_filter, RetrievalResult, MAX_TOP_K
"""
from __future__ import annotations

from krishisethu.rag.chunker import (
    Chunk,
    OVERLAP_TOKENS,
    TARGET_MAX_TOKENS,
    TARGET_MIN_TOKENS,
    chunk_markdown,
    dedupe_docs,
)
from krishisethu.rag.embedder import EXPECTED_DIM, Embedder, OllamaEmbedder, StubEmbedder
from krishisethu.rag.ingest import IngestResult, ingest_corpus
from krishisethu.rag.retriever import (
    MAX_TOP_K,
    PlotContext,
    RetrievalResult,
    Retriever,
    metadata_filter,
)
from krishisethu.rag.schema import RagDoc, crop_season_collection, rag_doc_from_file, rag_doc_from_text
from krishisethu.rag.store import ChromaLocal, StoredChunk, VectorStore

__all__ = [
    # schema
    "RagDoc",
    "rag_doc_from_file",
    "rag_doc_from_text",
    "crop_season_collection",
    # chunker
    "Chunk",
    "chunk_markdown",
    "dedupe_docs",
    "TARGET_MIN_TOKENS",
    "TARGET_MAX_TOKENS",
    "OVERLAP_TOKENS",
    # embedder
    "Embedder",
    "OllamaEmbedder",
    "StubEmbedder",
    "EXPECTED_DIM",
    # store
    "VectorStore",
    "ChromaLocal",
    "StoredChunk",
    # ingest
    "ingest_corpus",
    "IngestResult",
    # retriever
    "Retriever",
    "PlotContext",
    "metadata_filter",
    "RetrievalResult",
    "MAX_TOP_K",
]
