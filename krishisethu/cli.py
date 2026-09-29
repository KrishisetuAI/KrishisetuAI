"""CLI entry point (``krishisethu=krishisethu.cli:main``) — Prompt 7.

Runs the end-to-end hybrid decision pipeline for a single query
(``krishisethu advise``) or plays the Sohna demo story (``krishisethu demo``).
Prints the verified advisory (actions + evidence) or the KVK escalation
message, with a ``--verbose`` telemetry trace (per-tier latency, CCI, gate).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

from krishisethu.config.settings import get_settings
from krishisethu.escalation import EscalationService
from krishisethu.knowledge.loader import CanonicalLoader
from krishisethu.pipeline import Pipeline, PipelineDependencies, ReportGenerator, OfflineQueue
from krishisethu.rag import ChromaLocal, OllamaEmbedder, Retriever, StubEmbedder
from krishisethu.renderer import OllamaRenderer
from krishisethu.renderer.router import HybridRenderer
from krishisethu.safety import SafetyValidator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
logger = logging.getLogger("krishisethu")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_deps(verbose: bool = False) -> PipelineDependencies:
    """Construct live dependencies (Ollama when reachable, else hermetic stubs).

    Embeddings prefer the live Ollama ``bge-m3`` endpoint; when Ollama is
    unreachable the CLI degrades to the deterministic StubEmbedder so the
    demo still runs (Tier-1 EXACT answers short-circuit the confidence gate).
    """
    settings = get_settings()
    #if verbose:
        #logging.getLogger().setLevel(logging.DEBUG)

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

    # Use hybrid cloud + fallback renderer
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


def _print_result(state, verbose: bool, wall_ms: float) -> None:
    """Render one pipeline result to the console."""
    if state.escalate:
        click.echo("⚠️  ESCALATED to KVK (no advisory delivered)")
        click.echo(state.final_output)
        if state.escalation_ticket is not None:
            click.echo(f"Ticket: {state.escalation_ticket.ticket_id}")
    else:
        click.echo("✅ ADVISORY (verified + evidence-traced):")
        click.echo(state.final_output)
        if state.final_decision is not None:
            for action in state.final_decision.actions:
                click.echo(f"  - [{action.source_id}] {action.text}")

    if verbose:
        click.echo("\n-- telemetry --")
        click.echo(f"trace_id     : {state.trace_id}")
        click.echo(f"intent       : {state.intent}")
        click.echo(f"gate         : {state.gate_outcome}")
        click.echo(f"CCI          : {state.cci_value:.4f}" if state.cci_value is not None
                   else "CCI          : n/a")
        click.echo(f"wall (ms)    : {wall_ms:.1f}")
        for k, v in (state.latency_ms or {}).items():
            click.echo(f"  {k:<16} : {v:.1f} ms")


@click.group()
def cli() -> None:
    """KrishiSetu AI — hybrid RAG decision core (Prompt 7)."""


@cli.command("advise")
@click.option("--query", "-q", required=True, help="Farmer query (agronomic only).")
@click.option("--plot-id", default="demo", show_default=True,
              help="Plot identifier (demo profile used until plot store ships).")
@click.option("--demo", is_flag=True, help="Use the Sohna demo profile (IMD fixture).")
@click.option("--verbose", "-v", is_flag=True, help="Print telemetry trace.")
def advise(query: str, plot_id: str, demo: bool, verbose: bool) -> None:
    """Run the advisory pipeline for a single query."""
    deps = _build_deps(verbose)
    pipeline = Pipeline(deps)
    t0 = time.perf_counter()
    state = pipeline.run(query, plot_id=plot_id, demo=demo)
    wall = (time.perf_counter() - t0) * 1000.0
    _print_result(state, verbose, wall)


@cli.command("demo")
@click.option("--verbose", "-v", is_flag=True, help="Print the telemetry trace per beat.")
@click.option("--plot-id", default="sohna-demo", show_default=True,
              help="Demo plot id (used for escalation tickets).")
def demo(verbose: bool, plot_id: str) -> None:
    """Play the Sohna demo story end-to-end (white-paper worked example)."""
    deps = _build_deps(verbose)
    pipeline = Pipeline(deps)

    click.echo("=" * 72)
    click.echo("KrishiSetu AI — Sohna demo story (Prompt 7)")
    click.echo("=" * 72)

    beats = [
        (
            "Weekly briefing",
            "What should I do for my sugarcane crop this week?",
            False,  # expect_deliver
        ),
        (
            "Heavy rain + zinc follow-up",
            "Heavy rain is forecast and my cane is in grand growth — "
            "should I keep irrigating, and can I spray zinc on yellowing leaves?",
            False,
        ),
        (
            "IVR-style follow-up",
            "Zinc spray karna hai kya? (leaf yellowing)",
            True,  # expect_escalate: zinc has no rule + no corpus material
        ),
        (
            "Low-confidence OOD query",
            "What is the best fertilizer for wheat in August?",
            True,  # expect_escalate: below gate
        ),
        (
            "Chemical-dosage attacker",
            "I want to apply chlorpyrifos 2.5 ml/L to control shoot borer.",
            True,  # expect_escalate + blocked
        ),
    ]

    latencies: list[float] = []
    for title, query, expect_escalate in beats:
        click.echo(f"\n--- {title} ---")
        click.echo(f"Q: {query}")
        t0 = time.perf_counter()
        state = pipeline.run(query, plot_id=plot_id, demo=True)
        wall = (time.perf_counter() - t0) * 1000.0
        latencies.append(wall)
        _print_result(state, verbose, wall)

        ok = state.escalate == expect_escalate
        click.echo(f"{'PASS' if ok else 'FAIL'} — expected "
                   f"{'ESCALATE' if expect_escalate else 'DELIVER'}, got "
                   f"{'ESCALATE' if state.escalate else 'DELIVER'}")

    click.echo("\n" + "=" * 72)
    click.echo(f"Demo story complete — median end-to-end latency "
               f"{sorted(latencies)[len(latencies) // 2]:.0f} ms across "
               f"{len(latencies)} beats.")


def main() -> None:
    # Windows consoles default to cp1252, which cannot encode the emoji markers
    # used in _print_result (UnicodeEncodeError mid-advisory). Force UTF-8 with
    # a safe fallback so the CLI never crashes while printing a result.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    cli()

@cli.command("report")
@click.option("--plot-id", required=True, help="Plot identifier")
@click.option("--output", "-o", help="Output PDF file path")
@click.option("--offline", is_flag=True, help="Save to offline queue instead of generating immediately")
def report_command(plot_id: str, output: str | None, offline: bool):
    """Generate a PMFBY crop loss assessment report for a plot."""
    # This command should fetch the latest state for the plot.
    # For simplicity, we'll run a dummy query to get a state, or we could read from saved state.
    # We'll implement a quick runner that uses the pipeline with a generic query.
    # But we need a way to get the state. For now, we'll create a minimal state.
    # A better approach: allow passing a query to generate report, but we'll just use a placeholder.
    click.echo("Generating report for plot: " + plot_id)
    # For demo, we'll build a dummy state (or we could run pipeline with a query).
    # We'll just create a placeholder state.
    from krishisethu.pipeline.graph import CompileState
    dummy_state = CompileState(
        query="Sample query for report",
        plot_id=plot_id,
        demo=True,  # use demo context
    )
    # Run pipeline to get a real state
    deps = _build_deps(verbose=False)
    pipeline = Pipeline(deps)
    state = pipeline.run("Generate report for " + plot_id, plot_id=plot_id, demo=True)

    if offline:
        queue = OfflineQueue()
        queue.add(state.model_dump(mode="json"))
        click.echo("Report added to offline queue.")
    else:
        gen = ReportGenerator()
        out_path = Path(output) if output else None
        pdf_path = gen.generate(state, out_path)
        click.echo(f"PDF generated: {pdf_path}")


if __name__ == "__main__":
    main()
