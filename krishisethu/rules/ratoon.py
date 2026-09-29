"""Sugarcane ratoon / gap planner (Tier 1).

Stage-based ratoon checklist, a gap-fill decision (preserve vs selective gap-
fill vs replant vs rotate, keyed on gap percentage per row), and a rotation
recommendation when ratoon viability falls below the economic threshold.
Reproduces the Sohna case: a 3rd-ratoon field with a 45% gap and 0.50
viability is below the economic threshold, so the recommendation is rotation
(a legume green-manure / maize break) rather than re-planting cane again.

PURE PYTHON, deterministic. Every emitted action carries a ``RATOON:...``
trace code. All thresholds are named constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from krishisethu.rules.contracts import RuleAction


class RatoonStage(str, Enum):
    """Ratoon development stages the checklist keys on."""

    ESTABLISHMENT = "establishment"
    TILLERING = "tillering"
    GRAND_GROWTH = "grand_growth"


#: Gap-fill thresholds (fraction of missing stands per row).
GAP_PRESERVE_REPLANT_MAX = 0.20  # <= 20% gap -> preserve the ratoon as-is
GAP_SELECTIVE_REPLANT_MIN = 0.20  # 20% .. 40% -> selective gap-fill (replant gaps)
GAP_REPLANT_MIN = 0.40  # >= 40% gap -> whole-field replant (or rotate if uneconomic)
#: Below this viability the ratoon is not worth keeping -> recommend rotation.
ECONOMIC_VIABILITY_THRESHOLD = 0.60
#: Beyond this ratoon number multi-harvest is usually uneconomic if gappy.
MAX_RATOON_NUMBER = 3

#: Standard rotation break for a declining cane ratoon (NF-fixing legume then maize).
ROTATION_RECOMMENDED_CROP = "legume green-manure (dhaincha) then maize"


@dataclass(frozen=True)
class GapDecision:
    """Gap-fill recommendation for the measured stand loss."""

    decision: str  # PRESERVE | SELECTIVE_GAP_FILL | REPLANT | ROTATE
    gap_fraction: float
    trace: str


@dataclass(frozen=True)
class RotationRecommendation:
    """Rotation decision based on ratoon viability / age / gap."""

    rotate: bool
    recommended_crop: str | None
    ratoon_number: int
    viability: float
    reason: str
    trace: str


@dataclass(frozen=True)
class RatoonPlan:
    """Complete ratoon plan: stage checklist + gap decision + rotation call."""

    stage: RatoonStage
    checklist: tuple[RuleAction, ...]
    gap: GapDecision
    rotation: RotationRecommendation


#: Stage-based checklist: (clause, action, priority).
_STAGE_CHECKLIST: dict[RatoonStage, list[tuple[str, str, int]]] = {
    RatoonStage.ESTABLISHMENT: [
        ("TRASH_MANAGEMENT", "Manage trash (retain as mulch / remove heavy residue)", 0),
        ("STUBBLE_SHAVING", "Shave stubble to encourage uniform sprouting", 1),
        ("GAP_FILL", "Gap-fill where sprouting is poor", 2),
        ("FERTILIZER_BASAL", "Apply basal dose as per PoP", 3),
    ],
    RatoonStage.TILLERING: [
        ("TILLER_CHECK", "Monitor tiller count; maintain soil moisture", 0),
        ("WEEDING", "Weed control in inter-rows", 1),
    ],
    RatoonStage.GRAND_GROWTH: [
        ("IRRIGATION", "Irrigate on schedule during grand growth", 0),
        ("TOPSOIL_HAULM", "Keep trash between rows for moisture", 1),
    ],
}


def ratoon_stage_checklist(stage: RatoonStage | str) -> tuple[RuleAction, ...]:
    """Return the stage-specific ratoon management checklist (trace-coded)."""

    stage_e = stage if isinstance(stage, RatoonStage) else RatoonStage(stage)
    entries = _STAGE_CHECKLIST[stage_e]
    return tuple(
        RuleAction(
            trace=f"RATOON:{stage_e.value}:{clause}",
            action=human,
            clause=clause,
            priority=prio,
        )
        for clause, human, prio in entries
    )


def gap_fill_decision(gap_fraction: float) -> GapDecision:
    """Preserve, selectively gap-fill, replant or rotate based on gap %.

    Overrides the pure gap rule when the ratoon is uneconomic (see
    :func:`rotation_recommendation`) — the plan-level function decides which
    wins; this function reports the gap-only outcome.
    """
    if gap_fraction < 0.0:
        raise ValueError("gap_fraction must be >= 0")
    if gap_fraction <= GAP_PRESERVE_REPLANT_MAX:
        decision, clause = "PRESERVE", "GAP_PRESERVE"
    elif gap_fraction < GAP_REPLANT_MIN:
        decision, clause = "SELECTIVE_GAP_FILL", "GAP_SELECTIVE"
    else:
        decision, clause = "REPLANT", "GAP_REPLANT"
    return GapDecision(
        decision=decision,
        gap_fraction=round(gap_fraction, 4),
        trace=f"RATOON:GAP_DECISION:{clause}",
    )


def rotation_recommendation(
    ratoon_number: int,
    viability: float,
    gap_fraction: float,
) -> RotationRecommendation:
    """Recommend a rotation break when the ratoon is below economic viability.

    A ratoon is below the economic threshold when its viability index is under
    ``ECONOMIC_VIABILITY_THRESHOLD``; a high gap compounds it. Multi-ratoon
    fields (ratoon_number > MAX_RATOON_NUMBER) are flagged as the main driver.
    """
    if viability < 0.0 or viability > 1.0:
        raise ValueError("viability must be in [0, 1]")
    if ratoon_number < 1:
        raise ValueError("ratoon_number must be >= 1")

    uneconomic = viability < ECONOMIC_VIABILITY_THRESHOLD
    if uneconomic:
        driver = (
            "ratoon age (ratoon_number > 3)"
            if ratoon_number > MAX_RATOON_NUMBER
            else "low ratoon viability"
        )
        reason = (
            f"{driver} plus {gap_fraction:.0%} gap — below the "
            f"{ECONOMIC_VIABILITY_THRESHOLD:.2f} economic threshold; recommend rotation"
        )
        return RotationRecommendation(
            rotate=True,
            recommended_crop=ROTATION_RECOMMENDED_CROP,
            ratoon_number=ratoon_number,
            viability=round(viability, 4),
            reason=reason,
            trace="RATOON:ROTATION:RECOMMEND",
        )
    return RotationRecommendation(
        rotate=False,
        recommended_crop=None,
        ratoon_number=ratoon_number,
        viability=round(viability, 4),
        reason=f"ratoon above the {ECONOMIC_VIABILITY_THRESHOLD:.2f} economic threshold — keep the ratoon",
        trace="RATOON:ROTATION:KEEP",
    )


def ratoon_plan(
    stage: RatoonStage | str,
    gap_fraction: float,
    ratoon_number: int,
    viability: float,
) -> RatoonPlan:
    """Full ratoon plan.

    Combines the stage checklist, the gap decision and the rotation call. When
    the ratoon is uneconomic the rotation recommendation takes precedence over
    a bare gap-only REPLANT, so the farmer is told to rotate rather than spend
    on replanting a failing ratoon (the Sohna case).
    """
    stage_e = stage if isinstance(stage, RatoonStage) else RatoonStage(stage)
    gap = gap_fill_decision(gap_fraction)
    rot = rotation_recommendation(ratoon_number, viability, gap_fraction)

    if rot.rotate:
        # Uneconomic ratoon: recommend a rotation break rather than replant.
        gap = GapDecision(
            decision="ROTATE",
            gap_fraction=gap.gap_fraction,
            trace=f"RATOON:GAP_DECISION:ROTATE",
        )

    return RatoonPlan(
        stage=stage_e,
        checklist=ratoon_stage_checklist(stage_e),
        gap=gap,
        rotation=rot,
    )
