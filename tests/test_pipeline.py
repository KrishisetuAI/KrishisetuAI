"""E2E pipeline tests (Prompt 7).

Runs the full LangGraph pipeline with stub retriever and mocked renderer
to verify:
  - Sohna demo plot returns a valid advisory with evidence.
  - Malicious dosage query escalates and produces zero dosage.
  - Below‑gate OOD query creates a KVK ticket.
  - Median end‑to‑end latency stays within budget (~5s).
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from krishisethu.config.settings import get_settings
from krishisethu.escalation import EscalationService
from krishisethu.knowledge.loader import CanonicalLoader
from krishisethu.pipeline import Pipeline, PipelineDependencies
from krishisethu.rag import ChromaLocal, OllamaEmbedder, Retriever, StubEmbedder
from krishisethu.rag.schema import RagDoc
from krishisethu.rag.store import StoredChunk
from krishisethu.renderer import OllamaRenderer, RenderResult
from krishisethu.renderer.structured_output import ActionItem, Advisory, ConfidenceBreakdown, EvidenceRef
from krishisethu.safety import SafetyValidator

REPO = Path(__file__).resolve().parent
TEST_DATA = REPO / "data"
TEST_DATA.mkdir(exist_ok=True)
ESCALATION_STORE = TEST_DATA / "test_escalations.json"
CANONICAL_ROOT = REPO.parent / "knowledge" / "canonical"
RAG_CORPUS = REPO.parent / "knowledge" / "rag_corpus"


@pytest.fixture
def mock_renderer():
    """Return a mocked OllamaRenderer that always returns a valid advisory."""
    mock = MagicMock(spec=OllamaRenderer)
    mock.render.return_value = RenderResult(
        advisory=Advisory(
            actions=(
                ActionItem(
                    text="Suspend irrigation due to heavy rain.",
                    input_class=None,
                    consult_expert=False,
                    evidence=(EvidenceRef(source_id="WEATHER:sugarcane:grand_growth:heavy_rain_warning:SUSPEND_IRRIGATION", tier="tier1"),),
                ),
                ActionItem(
                    text="Clear furrow outlets to prevent waterlogging.",
                    input_class=None,
                    consult_expert=False,
                    evidence=(EvidenceRef(source_id="WEATHER:sugarcane:grand_growth:heavy_rain_warning:CLEAR_FURROW", tier="tier1"),),
                ),
                ActionItem(
                    text="Delay foliar spray 72h.",
                    input_class=None,
                    consult_expert=False,
                    evidence=(EvidenceRef(source_id="WEATHER:sugarcane:grand_growth:heavy_rain_warning:DELAY_FOLIAR_SPRAY", tier="tier1"),),
                ),
            ),
            confidence_breakdown=ConfidenceBreakdown(R=1.0, S_RAG=0.0, P_vision=0.0, has_image=False, cci=0.625),
        ),
        model_used="qwen3.5:4b",
        attempts=1,
        error=None,
    )
    return mock


@pytest.fixture
def pipeline_deps(mock_renderer):
    """Build real dependencies with a stub retriever and mocked renderer."""
    settings = get_settings()
    loader = CanonicalLoader(CANONICAL_ROOT)
    # Use stub embedder for deterministic S_RAG
    embedder = StubEmbedder(seed=0)
    store = ChromaLocal(settings.chroma.persist_dir)
    retriever = Retriever(embedder, store, top_k=settings.rag.top_k)
    validator = SafetyValidator(threshold=settings.gate.confidence_gate)
    escalation = EscalationService(ESCALATION_STORE)
    return PipelineDependencies(
        loader=loader,
        retriever=retriever,
        renderer=mock_renderer,
        validator=validator,
        escalation_service=escalation,
    )


def test_sohna_demo_returns_advisory(pipeline_deps):
    """Sohna demo plot should return a valid advisory with evidence."""
    pipeline = Pipeline(pipeline_deps)
    result = pipeline.run(
        "What should I do for my sugarcane crop this week?",
        plot_id="demo",
        demo=True,
    )
    assert result.final_output, "Advisory text should be non‑empty"
    assert not result.escalate, "Should deliver, not escalate"
    assert result.validator_result is not None
    assert result.validator_result.accepted is True
    # Evidence traceability
    assert result.final_decision is not None
    actions = result.final_decision.actions
    assert len(actions) > 0, "At least one action should be delivered"
    for action in actions:
        assert action.source_id, "Every action must have a traceable source_id"
    # Check that the gate outcome is pass_exact (deterministic)
    assert result.gate_outcome == "pass_exact"


def test_malicious_dosage_query_escalates(pipeline_deps):
    """A query with a dosage request must be escalated and stripped."""
    # Override the mock to return an advisory with a dosage.
    # Use a valid Tier-3 marker that exists in the RAG corpus.
    pipeline_deps.renderer.render.return_value = RenderResult(
        advisory=Advisory(
            actions=(
                ActionItem(
                    text="Apply chlorpyrifos 2.5 ml/L to control shoot borer.",
                    input_class="organophosphate insecticide",
                    consult_expert=False,
                    evidence=(EvidenceRef(source_id="kvk_sohna_sugarcane_heavy_rain", tier="tier3"),),
                ),
            ),
            confidence_breakdown=ConfidenceBreakdown(R=0.0, S_RAG=0.8, P_vision=0.0, has_image=False, cci=0.24),
        ),
        model_used="qwen3.5:4b",
        attempts=1,
        error=None,
    )
    pipeline = Pipeline(pipeline_deps)
    result = pipeline.run(
        "I want to apply chlorpyrifos 2.5 ml/L to control shoot borer.",
        plot_id="demo",
        demo=True,
    )
    assert result.escalate is True, "Should escalate"
    assert "escalated" in result.final_output.lower() or "forwarded" in result.final_output.lower()
    assert result.validator_result is not None
    assert result.validator_result.escalate is True
    # The final output should be the neutral escalation message, not the dosage.
    assert "2.5" not in result.final_output
    assert "chlorpyrifos" not in result.final_output.lower()


def test_below_gate_ood_query_creates_ticket(pipeline_deps):
    """An OOD query with CCI below threshold should create an escalation ticket."""
    # We'll patch the retriever.retrieve to return a RetrievalResult with low S_RAG.
    from krishisethu.rag.retriever import RetrievalResult

    class FakeRetrievalResult(RetrievalResult):
        @property
        def s_rag(self):
            return 0.05
        @property
        def chunks(self):
            return ()
        def __init__(self, *args, **kwargs):
            pass

    with patch.object(pipeline_deps.retriever, "retrieve") as mock_retrieve:
        mock_retrieve.return_value = FakeRetrievalResult()
        pipeline = Pipeline(pipeline_deps)
        result = pipeline.run(
            "What is the best fertilizer for wheat in August?",
            plot_id="demo",
            demo=True,
        )
        assert result.escalate is True, "Should escalate due to low CCI"
        assert result.escalation_ticket is not None, "A ticket should have been created"
        assert result.escalation_ticket.dedup_key, "Ticket should have a dedup key"
        # Check that the gate outcome is fail_cci
        assert result.gate_outcome == "fail_cci"


def test_median_latency_within_budget(pipeline_deps):
    """Median end‑to‑end latency should be under 5 seconds (5000 ms)."""
    queries = [
        "What should I do for my sugarcane crop this week?",
        "Heavy rain forecast, suspend irrigation?",
        "Should I apply zinc on yellowing leaves?",
    ]
    latencies = []
    pipeline = Pipeline(pipeline_deps)
    for q in queries:
        t0 = time.perf_counter()
        pipeline.run(q, plot_id="demo", demo=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median = sorted(latencies)[len(latencies) // 2]
    # The budget target is ~5s; we allow a generous margin for CI.
    assert median < 5000, f"Median latency {median:.0f} ms exceeds 5s limit"