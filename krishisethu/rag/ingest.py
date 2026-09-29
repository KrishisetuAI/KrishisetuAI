"""Corpus ingestion / idempotent seed command (Tier 3).

Reads normalized Markdown advisories from ``knowledge/rag_corpus/``, validates
each through the closed-enum schema, dedupes near-identical advisories repeated
across districts, chunks, embeds and upserts into ChromaDB.

Ids are deterministic (``<source_id>::<chunk_index>``) and ingestion uses
upsert, so rerunning the command never duplicates — the collection count is
identical after the second pass. A fresh pass on an unchanged corpus is a no-op.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krishisethu.rag.chunker import Chunk, chunk_markdown, dedupe_docs
from krishisethu.rag.embedder import Embedder
from krishisethu.rag.schema import RagDoc, rag_doc_from_file
from krishisethu.rag.store import VectorStore


@dataclass(frozen=True)
class IngestResult:
    """Summary of one ingest pass."""

    docs_scanned: int
    docs_indexed: int
    docs_deduped: int
    chunks_indexed: int
    collections: dict[str, int]

    def summary(self) -> str:
        parts = [f"{c}:{n}" for c, n in sorted(self.collections.items())]
        return (
            f"scanned={self.docs_scanned} indexed={self.docs_indexed} "
            f"deduped={self.docs_deduped} chunks={self.chunks_indexed} "
            f"collections=[{', '.join(parts)}]"
        )


def _doc_meta(doc: RagDoc) -> dict:
    """Flatten the validated frontmatter to Chroma-friendly scalar metadata."""
    m = {
        "source_id": doc.source_id,
        "title": doc.title,
        "crop": doc.crop.value,
        "season": doc.season.value,
        "district": doc.district,
        "state": doc.state.value,
        "source_type": doc.source_type.value,
        "source_authority": doc.source_authority,
        "language": doc.language,
    }
    if doc.source_url:
        m["source_url"] = doc.source_url
    if doc.valid_from:
        m["valid_from"] = doc.valid_from.isoformat()
    return m


def _chunk_id(doc: RagDoc, chunk: Chunk) -> str:
    return f"{doc.source_id}::{chunk.chunk_index}"


def ingest_corpus(
    corpus_dir: str | Path,
    embedder: Embedder,
    store: VectorStore,
    *,
    dedup: bool = True,
) -> IngestResult:
    """Validate, dedupe, chunk, embed and index every corpus document."""
    root = Path(corpus_dir)
    paths = sorted(root.rglob("*.md"))
    scanned = len(paths)
    docs: list[RagDoc] = [rag_doc_from_file(p) for p in paths]

    if dedup:
        docs = dedupe_docs(docs)

    collections: dict[str, int] = {}
    chunks_indexed = 0
    for doc in docs:
        meta = _doc_meta(doc)
        chunks = chunk_markdown(doc.body, base_metadata=meta)
        if not chunks:
            continue
        ids = [_chunk_id(doc, c) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = embedder.embed_documents(texts)
        store.add(doc.collection_name, ids, embeddings, texts, metadatas)
        collections[doc.collection_name] = store.count(doc.collection_name)
        chunks_indexed += len(chunks)

    return IngestResult(
        docs_scanned=scanned,
        docs_indexed=len(docs),
        docs_deduped=scanned - len(docs),
        chunks_indexed=chunks_indexed,
        collections=collections,
    )


def main() -> None:
    """CLI entrypoint: seed knowledge/rag_corpus into data/chroma with bge-m3."""
    from krishisethu.config.settings import get_settings
    from krishisethu.rag.embedder import OllamaEmbedder
    from krishisethu.rag.store import ChromaLocal

    s = get_settings()
    embedder = OllamaEmbedder()
    store = ChromaLocal(s.chroma.persist_dir)
    corpus = Path(__file__).resolve().parents[2] / "knowledge" / "rag_corpus"
    result = ingest_corpus(corpus, embedder, store)
    print(f"ingest: {result.summary()}")


if __name__ == "__main__":
    main()
