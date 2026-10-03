"""Default dependency wiring for the decision pipeline.

Shared by the CLI and the FastAPI/IVR service so every channel — console,
webhook, voice — runs the exact same safety-gated pipeline: canonical loader,
Chroma retriever, HybridRenderer (cloud pool + Ollama fallback), SafetyValidator,
and the KVK escalation service. No channel is ever allowed to build its own
shortcut path around the gates.
"""
from __future__ import annotations

import logging
from pathlib import Path

from krishisethu.config.settings import get_settings
from krishisethu.escalation import EscalationService
from krishisethu.knowledge.loader import CanonicalLoader
from krishisethu.pipeline.graph import Pipeline, PipelineDependencies
from krishisethu.rag import ChromaLocal, OllamaEmbedder, Retriever, StubEmbedder
from krishisethu.renderer import OllamaRenderer
from krishisethu.renderer.router import HybridRenderer
from krishisethu.safety import SafetyValidator

logger = logging.getLogger("krishisethu.pipeline.factory")

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_default_deps() -> PipelineDependencies:
    """Live dependencies (Ollama when reachable, hermetic stubs otherwise).

    Embeddings prefer the live Ollama ``bge-m3`` endpoint; when Ollama is
    unreachable the pipeline degrades to the deterministic StubEmbedder so the
    Tier-1 EXACT short-circuit still delivers (never an ungated answer).
    """
    settings = get_settings()

    loader = CanonicalLoader(REPO_ROOT / "knowledge" / "canonical")

    embedder: object
    try:
        embedder = OllamaEmbedder(base_url=settings.ollama.embed_base())
        logger.debug("using live Ollama embedder (bge-m3)")
    except Exception:  # Ollama down -> hermetic stub
        embedder = StubEmbedder(seed=0)
        logger.debug("Ollama unreachable; using deterministic StubEmbedder")

    store = ChromaLocal(settings.chroma.persist_dir)
    retriever = Retriever(embedder, store, top_k=settings.rag.top_k)

    fallback_renderer = OllamaRenderer(
        base_url=settings.ollama.embed_base(),
        model=settings.ollama.llm_model,
        fallback_model=settings.ollama.llm_fallback,
        timeout_seconds=60.0,
    )
    renderer = HybridRenderer(fallback_renderer=fallback_renderer)
    validator = SafetyValidator(threshold=settings.gate.confidence_gate)
    escalation = EscalationService(REPO_ROOT / "data" / "escalations.json")

    return PipelineDependencies(
        loader=loader,
        retriever=retriever,
        renderer=renderer,
        validator=validator,
        escalation_service=escalation,
    )


def build_default_pipeline() -> Pipeline:
    """One fully wired decision pipeline (the only construction path)."""
    return Pipeline(build_default_deps())
