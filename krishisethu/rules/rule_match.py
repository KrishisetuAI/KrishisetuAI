"""Rule-match indicator R(C) (Tier 1).

R(C) is the w1 term of the composite Confidence Index. In [0, 1]:

* ``1.0`` on exact resolution — every clause of the matched rule fired,
* a normalized partial-match score in ``(0, 1)`` when only some clauses fired,
* ``0.0`` when unresolved — no candidate rule matched.

PURE PYTHON, deterministic. ``rule_match_from_actions`` counts matched clauses
from a list of emitted :class:`~krishisethu.rules.contracts.RuleAction` (the
``clause`` field is the label; ``trace`` is the code the safety gate reads).
"""
from __future__ import annotations

from dataclasses import dataclass

from krishisethu.domain import Resolution
from krishisethu.rules.contracts import RuleAction


@dataclass(frozen=True)
class RuleMatch:
    """R(C) result: resolution, score and the trace codes behind it."""

    resolution: Resolution
    score: float
    matched_clauses: int
    total_clauses: int
    trace_codes: tuple[str, ...]


def classify(matched_clauses: int, total_clauses: int) -> Resolution:
    """Derive resolution from the match counts (0 -> UNRESOLVED, all -> EXACT)."""

    if matched_clauses <= 0:
        return Resolution.UNRESOLVED
    if matched_clauses >= total_clauses:
        return Resolution.EXACT
    return Resolution.PARTIAL


def rule_match(
    resolution: Resolution,
    matched_clauses: int = 0,
    total_clauses: int = 0,
    trace_codes: tuple[str, ...] = (),
) -> RuleMatch:
    """Compute R(C) from an explicit resolution and clause counts.

    Use this when the caller already knows the resolution (e.g. the Tier-1
    short-circuit reports EXACT). ``trace_codes`` are carried through so the
    source-traceability gate can cite which rule + clause fired.
    """
    if resolution is Resolution.EXACT:
        score = 1.0
    elif resolution is Resolution.UNRESOLVED:
        score = 0.0
    else:
        score = 0.0 if total_clauses <= 0 else min(1.0, matched_clauses / total_clauses)

    return RuleMatch(
        resolution=resolution,
        score=round(score, 4),
        matched_clauses=matched_clauses,
        total_clauses=total_clauses,
        trace_codes=tuple(trace_codes),
    )


def rule_match_from_actions(actions: list[RuleAction], total_clauses: int) -> RuleMatch:
    """Compute R(C) from the emitted actions for a rule.

    ``matched_clauses`` = number of distinct clause labels among ``actions``.
    Resolution is auto-derived: 0 matches -> UNRESOLVED (0.0), all clauses ->
    EXACT (1.0), otherwise PARTIAL (matched / total).
    """
    matched = len({a.clause for a in actions})
    resolution = classify(matched, total_clauses)
    return rule_match(
        resolution,
        matched_clauses=matched,
        total_clauses=total_clauses,
        trace_codes=tuple(a.trace for a in actions),
    )
