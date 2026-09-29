"""Plot-context retrieval (Tier 3).

Retrieval is *domain-governed*: a ``PlotContext`` carries crop / season /
district / allowed source types, and the search is metadata-filter-first,
similarity-second. The filter is applied before cosine so a sugarcane query
never sees a rice chunk even if it leaked into the same collection. Top-k is
hard-capped at 5.

``Retriever.s_rag`` returns the max cosine similarity over the top-k — the w2
term of the composite Confidence Index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from krishisethu.domain import Crop, Season, SourceType
from krishisethu.rag.embedder import Embedder
from krishisethu.rag.schema import crop_season_collection
from krishisethu.rag.store import StoredChunk, VectorStore

#: Hard cap from the white paper (plan Sec 5.3 / Sec 8).
MAX_TOP_K = 5


@dataclass(frozen=True)
class PlotContext:
    """The plot's grounded context — the pre-retrieval filter's ground truth."""

    crop: Crop | str
    district: str
    season: Season | str
    state: str | None = None
    source_types: tuple[SourceType | str, ...] | None = None

    @property
    def collection_name(self) -> str:
        return crop_season_collection(self.crop, self.season)

    @property
    def allowed_source_types(self) -> tuple[str, ...]:
        if self.source_types:
            return tuple(s.value if isinstance(s, SourceType) else str(s) for s in self.source_types)
        # Default: no source_type constraint means "any document class is in scope".
        return tuple(s.value for s in SourceType)


def metadata_filter(context: PlotContext) -> dict:
    """Build the ``{crop, season, district, source_type IN allowed}`` filter.

    Returned as a Chroma-compatible ``where`` dict (``$and`` of four legs). The
    district leg is exact so a Sohna advisory is never served to a Palwal plot.
    """
    legs = [
        {"crop": {"$eq": context.crop.value if isinstance(context.crop, Crop) else str(context.crop)}},
        {"season": {"$eq": context.season.value if isinstance(context.season, Season) else str(context.season)}},
        {"district": {"$eq": context.district}},
        {"source_type": {"$in": list(context.allowed_source_types)}},
    ]
    return {"$and": legs}


@dataclass(frozen=True)
class RetrievalResult:
    """A governed retrieval: the top-k chunks plus the S_RAG confidence input."""

    query: str
    context: PlotContext
    chunks: tuple[StoredChunk, ...] = field(default_factory=tuple)

    @property
    def s_rag(self) -> float:
        """Max cosine similarity over the top-k (0.0 when nothing retrieved)."""
        if not self.chunks:
            return 0.0
        return max(c.score for c in self.chunks)

    @property
    def top_k(self) -> int:
        return len(self.chunks)


class Retriever:
    """Metadata-filter-first, cosine-second retrieval over a VectorStore."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = MAX_TOP_K):
        self.embedder = embedder
        self.store = store
        self.top_k = max(1, min(top_k or MAX_TOP_K, MAX_TOP_K))

    def retrieve(
        self,
        query: str,
        context: PlotContext,
        k: int | None = None,
    ) -> RetrievalResult:
        k = self.top_k if k is None else max(1, min(k, self.top_k))
        query_vec = self.embedder.embed_query(query)
        where = metadata_filter(context)
        chunks = self.store.search_with_filter(
            context.collection_name, query_vec, where=where, k=k
        )
        return RetrievalResult(query=query, context=context, chunks=tuple(chunks))

    def s_rag(self, query: str, context: PlotContext, k: int | None = None) -> float:
        """S_RAG = max cosine over top-k; 0.0 if retrieval returned nothing."""
        return self.retrieve(query, context, k=k).s_rag
