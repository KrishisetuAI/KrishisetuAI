"""Pipeline contracts (Prompt 7).

``TierOrchestrator`` is the interface the LangGraph pipeline is capped behind:
a caller runs a query through the whole hybrid decision core (Tiers 1-4 plus
the Safety Validator) and gets back one typed result state. The graph
(:mod:`krishisethu.pipeline.graph`) is the only LangGraph implementation;
Tiers 1 and 2 stay framework-free and are never imported here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krishisethu.pipeline.graph import CompileState


@runtime_checkable
class TierOrchestrator(Protocol):
    """Cap over the LangGraph pipeline: run query -> typed final state."""

    def run(
        self,
        query: str,
        plot_id: str = "demo",
        demo: bool = False,
    ) -> "CompileState":
        """Run one query end-to-end and return the final pipeline state.

        ``demo`` switches the context assembler to the Sohna demo profile
        (IMD weather fixture + crop stage + soil) instead of a plot-store
        lookup. The returned state carries the advisory, gate outcome,
        per-tier latencies, and the escalation ticket when one was raised.
        """
        ...
