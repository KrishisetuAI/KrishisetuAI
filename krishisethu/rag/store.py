"""Vector store protocol + ChromaDB adapter (Tier 3).

``VectorStore`` is the swappable persistence contract (ChromaDB locally,
pgvector hosted later) with exactly three operations: ``add`` (upsert),
``search_with_filter`` (metadata pre-filter, then cosine inside the filtered
set, top-k <= 5), and ``delete``.

``ChromaLocal`` backs it with ``chromadb.PersistentClient`` using the cosine
metric so per-chunk scores are true cosine similarities. The collection name is
``crop_<crop>_<season>`` and every vector is stored alongside its raw text and
full metadata — provenance travels with the chunk, not just the vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredChunk:
    """A retrieved chunk: text + provenance metadata + cosine similarity."""

    id: str
    text: str
    metadata: dict
    distance: float  # cosine distance; identical vectors -> 0.0

    @property
    def score(self) -> float:
        """Cosine similarity for the confidence index (1.0 - cosine distance)."""
        return 1.0 - self.distance

    def provenance(self) -> tuple[str, int, str | None]:
        """(source_id, chunk_index, title) for the source-traceability gate."""
        return (
            str(self.metadata.get("source_id", "")),
            int(self.metadata.get("chunk_index", -1)),
            self.metadata.get("title"),
        )


class VectorStore(Protocol):
    """Persistence contract used by ``ingest`` and ``retriever``."""

    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def search_with_filter(
        self,
        collection: str,
        query_embedding: list[float],
        where: dict | None = None,
        k: int = 5,
    ) -> list[StoredChunk]: ...

    def delete(self, collection: str, ids: list[str] | None = None, where: dict | None = None) -> None: ...

    def count(self, collection: str) -> int: ...


class ChromaLocal:
    """Persistent ChromaDB adapter (cosine metric)."""

    def __init__(self, persist_dir: str | Path):
        import chromadb

        self._persist_dir = str(Path(persist_dir).resolve())
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collections: dict[str, object] = {}

    def _col(self, name: str):
        col = self._collections.get(name)
        if col is None:
            col = self._client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"}
            )
            self._collections[name] = col
        return col

    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._col(collection).upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def search_with_filter(
        self,
        collection: str,
        query_embedding: list[float],
        where: dict | None = None,
        k: int = 5,
    ) -> list[StoredChunk]:
        col = self._col(collection)
        if col.count() == 0:
            return []
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            StoredChunk(id=i, text=t, metadata=m, distance=d)
            for i, t, m, d in zip(ids, docs, metas, dists)
        ]

    def delete(self, collection: str, ids: list[str] | None = None, where: dict | None = None) -> None:
        self._col(collection).delete(ids=ids, where=where)

    def count(self, collection: str) -> int:
        return self._col(collection).count()

    def collections(self) -> list[str]:
        return sorted(c.name for c in self._client.list_collections())
