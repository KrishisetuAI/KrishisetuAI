"""Tier-3 RAG tests (Prompt 4).

Hermetic by default: a deterministic StubEmbedder + a temp ChromaDB so the suite
runs offline anywhere. The one live test (real bge-m3 ingestion + latency) is
gated on an Ollama server being reachable.

Coverage of the acceptance criteria:
  - pre-retrieval filter excludes off-crop / off-district / off-source chunks
  - top-k never exceeds 5
  - embeddings byte-reproducible under a fixed seed
  - retrieve returns provenance + per-chunk scores; S_RAG = max cosine
  - ingestion idempotent (run twice, same collection counts)
  - chunker: never splits a table row; token budget; overlap; dedup
"""
from __future__ import annotations

import os
import time
from datetime import date

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path

import pytest

from krishisethu.domain import Crop, Season, SourceType
from krishisethu.rag import (
    ChromaLocal,
    PlotContext,
    Retriever,
    StubEmbedder,
    chunk_markdown,
    dedupe_docs,
    ingest_corpus,
)
from krishisethu.rag.chunker import OVERLAP_TOKENS, TARGET_MAX_TOKENS, TARGET_MIN_TOKENS
from krishisethu.rag.embedder import EXPECTED_DIM
from krishisethu.rag.schema import RagDoc, crop_season_collection

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "knowledge" / "rag_corpus"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _meta(district="sohna", crop="sugarcane", season="kharif", source_type="kvk_advisory", **extra):
    m = {
        "source_id": f"{district}_{crop}_{extra.get('n', 0)}",
        "title": f"{crop} advisory at {district}",
        "crop": crop,
        "season": season,
        "district": district,
        "state": "haryana",
        "source_type": source_type,
        "source_authority": "KVK test",
        "language": "en",
    }
    m.update(extra)
    return m


def _seed_chunk(store, emb, collection, text, **meta):
    """Add one chunk to a collection directly, stamping the required metadata."""
    m = dict(meta)
    store.add(
        collection,
        ids=[m["source_id"]],
        embeddings=[emb.embed_query(text)],
        documents=[text],
        metadatas=[m],
    )


@pytest.fixture
def store(tmp_path):
    return ChromaLocal(tmp_path / "chroma")


@pytest.fixture
def emb():
    return StubEmbedder(seed=7)


# --------------------------------------------------------------------------
# chunker: table safety, budget, overlap, metadata, coverage
# --------------------------------------------------------------------------
def _table(n_rows: int) -> str:
    rows = ["| Col A | Col B | Col C |", "|---|---|---|"]
    rows += [f"| row {i} | value {i} | note {i} |" for i in range(n_rows)]
    return "\n".join(rows)


def test_table_block_never_split(emb):
    text = f"# Alpha\n\nIntro paragraph.\n\n{_table(40)}\n\n## Beta\n\nTrailing paragraph."
    chunks = chunk_markdown(text)
    # The 40-row table is a single atomic unit -> must live entirely in one chunk.
    table_rows = [i for i, c in enumerate(chunks) if "Col A" in c.text]
    assert len(table_rows) == 1, "table must appear in exactly one chunk"
    tbl_chunk = chunks[table_rows[0]]
    assert "row 39" in tbl_chunk.text
    for j, c in enumerate(chunks):
        # no chunk may contain a partial table (first row without the header etc.)
        if "| row" in c.text:
            assert "Col A" in c.text and "row 39" in c.text


def test_chunk_token_budget_and_coverage():
    text = "\n\n".join(
        f"## Section {i}\n\n" + "\n".join(f"point {i}.{j} — some descriptive prose here." for j in range(6))
        for i in range(8)
    )
    chunks = chunk_markdown(text)
    assert len(chunks) >= 3, "long document must split into several chunks"
    assert all(c.ntokens <= TARGET_MAX_TOKENS for c in chunks), "no chunk may exceed the hard cap"
    # at least one chunk must be a normal-size chunk within the target band
    assert any(TARGET_MIN_TOKENS <= c.ntokens <= TARGET_MAX_TOKENS for c in chunks)
    # coverage: every chunk's span is within the source and ordered (overlap allowed)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts), "chunk start offsets must be non-decreasing"
    for c in chunks:
        assert 0 <= c.start < c.end <= len(text)
    # coverage of the whole body
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)


def test_chunk_overlap_and_metadata():
    text = "\n\n".join(f"### Part {i}\n\n" + "\n".join(f"line {i}.{j} token text." for j in range(5)) for i in range(6))
    chunks = chunk_markdown(text)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["ntokens"] == c.ntokens
        assert c.metadata["char_start"] == c.start
        assert c.metadata["char_end"] == c.end
    # consecutive chunks share an overlapping char range
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end, "consecutive chunks must overlap"
        assert b.start > a.start, "overlap must not swallow the previous chunk"


def test_chunk_inherits_base_metadata():
    chunks = chunk_markdown("## A\n\nsome body.\n\n## B\n\nmore body.", base_metadata={"source_id": "x", "crop": "sugarcane"})
    assert all(c.metadata["source_id"] == "x" and c.metadata["crop"] == "sugarcane" for c in chunks)


def test_dedupe_collapses_near_identical_across_districts():
    body = "# T\n\nSome common advisory body text.\n\n## Note\n\nExtra detail here."
    a = RagDoc(source_id="a", title="t", crop="sugarcane", state="haryana", district="sohna",
               season="kharif", source_type=SourceType.KVK_ADVISORY, body=body)
    b = RagDoc(source_id="b", title="t", crop="sugarcane", state="haryana", district="nuh",
               season="kharif", source_type=SourceType.KVK_ADVISORY, body=body)
    c = RagDoc(source_id="c", title="t2", crop="rice", state="haryana", district="karnal",
               season="kharif", source_type=SourceType.KVK_ADVISORY,
               body="# R\n\nA completely different rice advisory.")
    kept = dedupe_docs([a, b, c])
    assert len(kept) == 2
    assert {d.source_id for d in kept} == {"a", "c"}


# --------------------------------------------------------------------------
# embedder: reproducibility under a fixed seed
# --------------------------------------------------------------------------
def test_stub_embedder_byte_reproducible():
    emb = StubEmbedder(seed=0)
    v1 = emb.embed_query("ratoon gap fill advisory")
    v2 = emb.embed_query("ratoon gap fill advisory")
    assert v1 == v2, "same text must embed to identical bytes under a fixed seed"
    assert len(v1) == EXPECTED_DIM
    d1 = emb.embed_documents(["a", "b"])
    d2 = emb.embed_documents(["a", "b"])
    assert d1 == d2
    q = emb.embed_query("a")
    assert d1[0] == q, "embed_query and embed_documents must agree for the same text"


def test_stub_embedder_distinguishes():
    emb = StubEmbedder(seed=0)
    assert emb.embed_query("sugarcane") != emb.embed_query("rice")


def test_stub_embedder_normalized():
    emb = StubEmbedder(seed=3)
    v = emb.embed_query("x")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-9, "stub vectors are l2-normalized for cosine semantics"


# --------------------------------------------------------------------------
# store: add / search_with_filter / delete on real (temp) Chroma
# --------------------------------------------------------------------------
def test_chroma_add_search_delete(store, emb):
    col = "crop_sugarcane_kharif"
    _seed_chunk(store, emb, col, "suspend irrigation", **_meta(district="sohna", n=1))
    _seed_chunk(store, emb, col, "clear furrow outlets", **_meta(district="sohna", n=2))
    assert store.count(col) == 2
    qv = emb.embed_query("suspend irrigation on heavy rain")
    hits = store.search_with_filter(col, qv, where={"crop": {"$eq": "sugarcane"}}, k=5)
    assert len(hits) == 2
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    store.delete(col, where={"district": {"$eq": "sohna"}})
    assert store.count(col) == 0


# --------------------------------------------------------------------------
# governed retrieval: filter, top-k cap, provenance + scores, S_RAG
# --------------------------------------------------------------------------
def test_prefilter_excludes_off_crop_contamination_canary(store, emb):
    col = "crop_sugarcane_kharif"
    for i in range(6):
        _seed_chunk(store, emb, col, f"valid sugarcane advisory {i}", **_meta(district="sohna", crop="sugarcane", n=i + 1))
    # adversarial: a rice-tagged chunk that leaked into the sugarcane collection
    _seed_chunk(store, emb, col, "rice blast advisory", **_meta(district="karnal", crop="rice", source_type="kvk_advisory", n="rice"))
    ret = Retriever(emb, store)
    ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif")
    out = ret.retrieve("sugarcane advisory", ctx, k=5)
    assert out.top_k == 5, "exactly top-k=5 plot-matching chunks must be returned"
    assert all(c.metadata["crop"] == "sugarcane" for c in out.chunks), "zero off-crop chunks"
    assert all(c.metadata["district"] == "sohna" for c in out.chunks), "zero off-district chunks"
    assert all(c.metadata["crop"] != "rice" for c in out.chunks), "contamination canary: rice must never surface"


def test_prefilter_excludes_off_source_type(store, emb):
    col = "crop_sugarcane_kharif"
    _seed_chunk(store, emb, col, "a kvk advisory", **_meta(crop="sugarcane", district="sohna", source_type="kvk_advisory", n=1))
    _seed_chunk(store, emb, col, "a research note", **_meta(crop="sugarcane", district="sohna", source_type="research_paper", n=2))
    ret = Retriever(emb, store)
    ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif",
                      source_types=(SourceType.KVK_ADVISORY,))
    out = ret.retrieve("advisory", ctx, k=5)
    assert out.chunks and all(c.metadata["source_type"] == "kvk_advisory" for c in out.chunks)


def test_top_k_never_exceeds_five(store, emb):
    col = "crop_sugarcane_kharif"
    for i in range(9):
        _seed_chunk(store, emb, col, f"advisory {i}", **_meta(district="sohna", n=i + 1))
    ret = Retriever(emb, store)
    ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif")
    assert ret.retrieve("x", ctx, k=5).top_k == 5
    assert ret.retrieve("x", ctx, k=100).top_k <= 5, "a large requested k must be clamped to 5"
    assert ret.retrieve("x", ctx, k=3).top_k == 3


def test_retrieve_returns_provenance_and_scores(store, emb):
    col = "crop_sugarcane_kharif"
    _seed_chunk(store, emb, col, "ratoon gap fill guide", **_meta(district="sohna", n=1))
    _seed_chunk(store, emb, col, "heavy rain actions", **_meta(district="sohna", n=2))
    ret = Retriever(emb, store)
    ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif")
    out = ret.retrieve("ratoon", ctx, k=2)
    assert out.top_k == 2
    for c in out.chunks:
        sid, idx, title = c.provenance()
        assert sid and isinstance(idx, int) and title
        assert -1.0 <= c.score <= 1.0
        assert c.metadata["source_id"] == sid
    assert 0.0 <= out.s_rag <= 1.0, "S_RAG is the max cosine in [0,1]"


def test_s_rag_zero_when_empty(store, emb):
    ret = Retriever(emb, store)
    ctx = PlotContext(crop="rice", district="karnal", season="kharif")
    assert ret.s_rag("anything", ctx) == 0.0


# --------------------------------------------------------------------------
# ingestion: idempotent + dedup on the seed corpus
# --------------------------------------------------------------------------
def test_ingest_idempotent(emb, store):
    r1 = ingest_corpus(CORPUS, emb, store)
    counts1 = dict(r1.collections)
    r2 = ingest_corpus(CORPUS, emb, store)
    assert r2.collections == counts1, "rerunning must not duplicate"
    assert r2.chunks_indexed == r1.chunks_indexed


def test_ingest_dedupes_cross_district(emb, store):
    res = ingest_corpus(CORPUS, emb, store)
    assert res.docs_scanned == 9
    assert res.docs_deduped == 1, "the nuh red-rot circular is near-identical to the sohna one"
    assert res.docs_indexed == 8
    # collections are crop_<crop>_<season>
    assert set(res.collections) == {"crop_sugarcane_kharif", "crop_rice_kharif", "crop_wheat_rabi"}
    assert res.collections["crop_sugarcane_kharif"] >= 8


def test_chunk_list_items():
    text = "# Checklist\n\n1. Suspend irrigation\n2. Clear furrow outlets\n3. Delay foliar spray"
    chunks = chunk_markdown(text)
    assert chunks
    assert "1. Suspend irrigation" in chunks[0].text


def test_chunk_trailing_blank_flush():
    text = "## A\n\nbody.\n\n\n"  # ends with a blank line -> final flush sees an empty buffer
    chunks = chunk_markdown(text)
    assert chunks and "body." in chunks[-1].text


def test_doc_meta_optional_fields():
    from krishisethu.rag.ingest import _doc_meta
    doc = RagDoc(source_id="x", title="t", crop="sugarcane", state="haryana", district="sohna",
                 season="kharif", source_type=SourceType.KVK_ADVISORY,
                 source_url="https://example.org", valid_from=date(2026, 7, 1))
    m = _doc_meta(doc)
    assert m["source_url"] == "https://example.org"
    assert m["valid_from"] == "2026-07-01"
    bare = _doc_meta(RagDoc(source_id="y", title="t", crop="rice", state="haryana", district="karnal",
                            season="kharif", source_type=SourceType.KVK_ADVISORY))
    assert "source_url" not in bare and "valid_from" not in bare


def test_ingest_summary_string(emb, store):
    res = ingest_corpus(CORPUS, emb, store)
    s = res.summary()
    assert "scanned=9" in s and "collections=[" in s


def test_ingest_empty_body_doc_skipped(tmp_path, emb, store):
    """A doc with an empty body must produce no chunk (and not be indexed)."""
    (tmp_path / "empty.md").write_text(
        "---\nsource_id: z\ntitle: t\ncrop: sugarcane\nstate: haryana\ndistrict: sohna\n"
        "season: kharif\nsource_type: kvk_advisory\n---\n",
        encoding="utf-8",
    )
    res = ingest_corpus(tmp_path, emb, store)
    assert res.docs_indexed == 1
    assert res.chunks_indexed == 0
    assert res.collections == {}


def test_ingest_main_seeds(tmp_path, monkeypatch, capsys):
    from krishisethu.config.settings import ChromaSettings, Settings
    import krishisethu.rag.ingest as ingest_mod

    embedder = StubEmbedder(seed=2)
    store_dir = tmp_path / "chroma"
    s = Settings(chroma=ChromaSettings(persist_dir=str(store_dir)))
    monkeypatch.setattr("krishisethu.config.settings.get_settings", lambda: s)
    monkeypatch.setattr("krishisethu.rag.embedder.OllamaEmbedder", lambda *a, **k: embedder)

    ingest_mod.main()
    out = capsys.readouterr().out
    assert "ingest:" in out
    probe = ChromaLocal(str(store_dir))
    assert probe.count("crop_sugarcane_kharif") >= 8


def test_ingest_rejects_bad_metadata(tmp_path, emb, store):
    bad = tmp_path / "bad.md"
    bad.write_text(
        "---\n"
        "source_id: kvk_x\n"
        "title: t\n"
        "crop: corn\n"  # not a closed-enum crop
        "state: haryana\n"
        "district: sohna\n"
        "season: kharif\n"
        "source_type: kvk_advisory\n"
        "---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ingest_corpus(tmp_path, emb, store)


def test_collection_name():
    assert crop_season_collection(Crop.SUGARCANE, Season.KHARIF) == "crop_sugarcane_kharif"


# --------------------------------------------------------------------------
# live (gated): real bge-m3 provenance + median latency
# --------------------------------------------------------------------------
def _ollama_up() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200 and "models" in r.text
    except Exception:
        return False


OLLAMA_UP = _ollama_up()


@pytest.mark.skipif(not OLLAMA_UP, reason="Ollama server not reachable")
def test_live_ollama_ingest_retrieve_latency(tmp_path):
    from krishisethu.rag.embedder import OllamaEmbedder

    emb = OllamaEmbedder()
    assert emb.dim == EXPECTED_DIM, "bge-m3 must be 1024-d"
    store = ChromaLocal(tmp_path / "live")
    res = ingest_corpus(CORPUS, emb, store)
    assert res.chunks_indexed >= 10

    ret = Retriever(emb, store)
    ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif")
    sample = ret.retrieve("how should I manage a 45% ratoon gap in my third ratoon", ctx, k=5)
    # real semantics: the ratoon-gap advisory should surface in the top-5
    assert any("ratoon" in c.metadata["source_id"] for c in sample.chunks), "ratoon advisory should rank"

    # contamination canary on the real store
    assert all(c.metadata["crop"] == "sugarcane" for c in sample.chunks)
    assert sample.top_k <= 5

    # median retrieval latency over 5 repeated same-context queries
    lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        ret.retrieve("sugarcane irrigation scheduling", ctx, k=5)
        lat.append(time.perf_counter() - t0)
    median = sorted(lat)[len(lat) // 2]
    assert median > 0.0 and median < 5.0, f"median retrieval latency should be sane, got {median:.3f}s"
