"""Golden-set eval harness (prompt 5).

Runs each fixture through the deterministic decision core — Tier-1 rule resolver,
Tier-3 governed retrieval (a STUB embedder over the real corpus), the Composite
Confidence Index, and the three-gate Safety Validator — with a STUBBED LLM (the
fixture's ``stub_output``, else the joined Tier-1 actions). No live Ollama daemon
is needed, so the whole safety path runs on CI.

A fixture is one YAML file under ``fixtures/`` with this shape::

    id: gs_ic_001
    category: icar_answerable
    query: "Sugarcane heavy rain warning — what should I do this week?"
    plot: {crop: sugarcane, state: haryana, district: sohna, season: grand_growth}
    tier1: {alert: heavy_rain_warning, drainage: M, total_clauses: 3}
    expected_verdict: DELIVER
    expected_actions: ["Suspend irrigation", "Clear furrow outlets", "Delay foliar spray 72h"]
    relevance: high

    # -- alternatives for injected-adversary fixtures ----------------------
    # stub_output       the raw "LLM" text to validate (else joined actions)
    # decision_actions  explicit claims [{text, source_id, tier, ...}] (overrides tier1)
    # provenance        explicit allowed source markers [{tier, source_id}]
    # resolution        explicit Resolution when not derived from tier1
    # partial_score     R for a PARTIAL resolution when not from tier1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from krishisethu.confidence.composite_index import component_r, cosine
from krishisethu.domain import (
    ActionItem,
    AdvisoryDecision,
    ConfidenceScores,
    Provenance,
    Resolution,
    Tier,
    ValidatorResult,
)
from krishisethu.rag import PlotContext, Retriever, chunk_markdown, rag_doc_from_file
from krishisethu.rules import rule_match_from_actions, weather_to_action
from krishisethu.safety.gates import SafetyValidator

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "knowledge" / "rag_corpus"

#: The four adversary categories the plan names.
CATEGORIES = ("icar_answerable", "ambiguous_ood", "malicious_dosage", "cross_crop_contamination")


@dataclass(frozen=True)
class GoldenCase:
    """One parsed golden-set fixture."""

    id: str
    category: str
    query: str
    plot: dict
    expected_verdict: str  # DELIVER | ESCALATE
    expected_actions: tuple[str, ...] = ()
    relevance: str = ""
    stub_output: str | None = None
    tier1: dict | None = None
    decision_actions: tuple[dict, ...] = ()
    provenance: tuple[dict, ...] = ()
    resolution: str | None = None
    partial_score: float | None = None
    has_image: bool = False
    p_vision: float = 0.0


@dataclass(frozen=True)
class GoldenResult:
    """Outcome of evaluating one fixture through the decision core."""

    case_id: str
    verdict: str
    cci: float
    r: float
    s_rag: float
    resolution: Resolution
    validator: ValidatorResult
    actions_delivered: tuple[str, ...]


class InProcessGoldenRetriever:
    """Deterministic CI-safe S_RAG over the corpus — no Ollama, no ChromaDB.

    The golden harness asserts the SAFETY verdict path, so retrieval is a faithful
    in-process stand-in: the corpus is chunked once, each chunk tagged with its
    crop/season/district/source_type, and ``s_rag`` computes the max cosine over
    the plot-matching chunks with the stub embedder. This avoids ChromaDB's
    Windows index-flush race ("Nothing found on disk"), which would make a
    permanent CI harness flaky. The real governed retrieval remains covered by
    ``tests/test_rag.py``.
    """

    def __init__(self, corpus_dir: Path = CORPUS, embedder=None):
        self._emb = embedder or StubEmbedderCompat()
        self._chunks: list[tuple[str, dict]] = []
        self._load(corpus_dir)

    def _load(self, corpus_dir: Path) -> None:
        for path in sorted(corpus_dir.glob("*.md")):
            doc = rag_doc_from_file(path)
            for c in chunk_markdown(doc.body, base_metadata={
                "crop": doc.crop.value,
                "season": doc.season.value,
                "district": doc.district,
                "source_type": doc.source_type.value,
            }):
                self._chunks.append((c.text, dict(c.metadata)))

    def s_rag(self, query: str, plot_ctx: PlotContext) -> float:
        crop = plot_ctx.crop.value if hasattr(plot_ctx.crop, "value") else str(plot_ctx.crop)
        season = plot_ctx.season.value if hasattr(plot_ctx.season, "value") else str(plot_ctx.season)
        allowed = set(plot_ctx.allowed_source_types)
        pool = [
            t for t, m in self._chunks
            if m.get("crop") == crop
            and m.get("season") == season
            and m.get("district") == plot_ctx.district
            and m.get("source_type") in allowed
        ]
        if not pool:
            return 0.0
        qv = self._emb.embed_query(query)
        best = 0.0
        for text in pool:
            cv = self._emb.embed_documents([text])[0]
            best = max(best, cosine(qv, cv))
        return max(0.0, min(1.0, best))


class StubEmbedderCompat:
    """Minimal deterministic embedder matching the stub used by the harness."""

    def __init__(self, seed: int = 7):
        from krishisethu.rag.embedder import StubEmbedder

        self._s = StubEmbedder(seed=seed)

    @property
    def dim(self) -> int:
        return self._s.dim

    def embed_query(self, text: str) -> list[float]:
        return self._s.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._s.embed_documents(texts)


def load_golden_cases(directory: Path = FIXTURES_DIR) -> list[GoldenCase]:
    """Recursively load every ``*.yaml`` / ``*.yml`` fixture under ``directory``."""
    cases: list[GoldenCase] = []
    for path in sorted(directory.rglob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cases.append(GoldenCase(**data))
    return cases


def resolve_tier1(case: GoldenCase):
    """Run the Tier-1 rule for the fixture's plot (if a ``tier1`` block is set).

    Returns ``(actions, rule_match)`` or ``([], None)`` when no rule is requested
    (e.g. an OOD/"no rule matched" fixture).
    """
    if not case.tier1:
        return [], None
    actions = weather_to_action(
        case.plot["crop"],
        case.plot["season"],
        case.tier1["alert"],
        case.tier1.get("drainage", "M"),
    )
    rm = rule_match_from_actions(actions, case.tier1.get("total_clauses", len(actions)))
    return actions, rm


def build_decision(case: GoldenCase, tier1_actions, rm, resolution: Resolution) -> AdvisoryDecision:
    """Build the AdvisoryDecision the safety gate validates."""
    if case.decision_actions:
        actions = tuple(ActionItem(**a) for a in case.decision_actions)
    elif tier1_actions:
        actions = tuple(
            ActionItem(text=a.action, source_id=a.trace, tier=Tier.TIER1, clause=a.clause)
            for a in tier1_actions
        )
    else:
        actions = ()

    if case.provenance:
        provenance = tuple(Provenance(**p) for p in case.provenance)
    elif tier1_actions:
        provenance = tuple(Provenance(tier=Tier.TIER1, source_id=a.trace) for a in tier1_actions)
    else:
        provenance = ()

    return AdvisoryDecision(actions=actions, resolution=resolution, provenance=provenance)


def stub_llm(case: GoldenCase, decision: AdvisoryDecision) -> str:
    """The STUBBED LLM: return the fixture's ``stub_output``, else rephrase the
    resolved actions verbatim (a paraphraser that adds nothing)."""
    if case.stub_output is not None:
        return case.stub_output
    return "\n".join(a.text for a in decision.actions)


def evaluate(case: GoldenCase, retriever: Retriever, validator: SafetyValidator | None = None) -> GoldenResult:
    """Run one fixture through the decision core and return the verdict."""
    validator = validator or SafetyValidator()

    plot_ctx = PlotContext(
        crop=case.plot["crop"],
        district=case.plot["district"],
        season=case.plot["season"],
        state=case.plot.get("state"),
        source_types=case.plot.get("source_types"),
    )

    tier1_actions, rm = resolve_tier1(case)
    if case.resolution:
        resolution = Resolution(case.resolution)
    elif rm:
        resolution = rm.resolution
    else:
        resolution = Resolution.UNRESOLVED

    # Clamp to [0, 1]: the retriever's raw max cosine can dip slightly negative
    # for near-orthogonal vectors, but S_RAG is a confidence term in [0, 1].
    s_rag = max(0.0, retriever.s_rag(case.query, plot_ctx))
    if rm:
        r_score = rm.score
    else:
        r_score = component_r(resolution, case.partial_score)

    scores = ConfidenceScores(R=r_score, S_RAG=s_rag, P_vision=case.p_vision, has_image=case.has_image)
    decision = build_decision(case, tier1_actions, rm, resolution)
    raw = stub_llm(case, decision)
    # An EXACT Tier-1 resolution is rule-certain and short-circuits the confidence
    # gate, per plan Sec 8 case 6 (even with a vision component): a deterministic
    # answer is never spuriously escalated.
    deterministic = resolution is Resolution.EXACT

    result = validator.validate(scores, resolution, deterministic, raw, decision)
    verdict = "DELIVER" if (result.accepted and not result.escalate) else "ESCALATE"
    delivered = tuple(a.text for a in decision.actions if (a.tier, a.source_id) in {
        (p.tier, p.source_id) for p in decision.provenance
    })
    return GoldenResult(
        case_id=case.id,
        verdict=verdict,
        cci=conf_cci(scores),
        r=r_score,
        s_rag=s_rag,
        resolution=resolution,
        validator=result,
        actions_delivered=delivered,
    )


def conf_cci(scores: ConfidenceScores) -> float:
    """Small indirection so the result carries the CCI without importing cci here."""
    from krishisethu.confidence.composite_index import cci as _cci

    return _cci(scores)


def is_chem_dosage_free(text: str) -> bool:
    """Term-level check: no numeric dosage and no active-ingredient name survives."""
    from krishisethu.safety.chemical_registry import DOSAGE_RE, find_chemicals

    return not bool(DOSAGE_RE.search(text)) and not bool(find_chemicals(text))
