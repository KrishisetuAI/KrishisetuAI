"""Shared contracts for the Tier-1 rule engine.

Every Tier-1 decision function returns a list (or single instance) of
:class:`RuleAction`. Each action is immutable and carries:

* ``trace``  — a stable, machine-readable code (rule-id + clause) that the
  source-traceability gate reads, e.g. ``WEATHER:SUGARCANE:GRAND_GROWTH:
  HEAVY_RAIN_WARNING:SUSPEND_IRRIGATION``.
* ``clause`` — the simple clause label that fired, used by
  :mod:`krishisethu.rules.rule_match` to count matched clauses for R(C).
* ``priority`` — orders ranked action lists (lower is more urgent).

Nothing in this module may import an LLM, a vector store, or open a socket.
Tier 1 is PURE PYTHON: identical inputs must always yield identical actions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleAction:
    """One emitted Tier-1 action with its source-trace code.

    ``trace`` is the opaque code the safety gate maps back to a rule+clause;
    ``clause`` is the same label in a short form for clause-match counting;
    ``action`` is the human-readable instruction that reaches the farmer.
    """

    trace: str
    action: str
    clause: str
    priority: int = 0

    def __lt__(self, other: "RuleAction") -> bool:
        """Order by ``priority`` (then trace) so ``sorted()`` is the recommended rank."""
        return (self.priority, self.trace) < (other.priority, other.trace)
