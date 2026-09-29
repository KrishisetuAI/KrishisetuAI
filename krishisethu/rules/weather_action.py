"""Weather-to-action matrix (Tier 1).

Deterministic. Indexed by ``(crop, crop_stage, IMD alert type, soil drainage
class)`` and returning a ranked, trace-coded action list. Reproduces the
white-paper Sohna example: (SUGARCANE, GRAND_GROWTH, HEAVY_RAIN_WARNING,
drainage M) -> [Suspend irrigation, Clear furrow outlets, Delay foliar spray
72h]. PURE PYTHON: no model, no network, no I/O. A new rule is a row in the
matrix tables below, never a model call.
"""
from __future__ import annotations

from enum import Enum

from krishisethu.domain import Crop, Season
from krishisethu.rules.contracts import RuleAction


class IMDAlertType(str, Enum):
    """IMD alert kinds the matrix keys on (subset used by the rules)."""

    HEAVY_RAIN_WARNING = "heavy_rain_warning"
    HEAT_WAVE = "heat_wave"
    FROST_WARNING = "frost_warning"
    STRONG_WIND = "strong_wind"
    NO_ALERT = "no_alert"


class SoilDrainageClass(str, Enum):
    """Soil-drainage classes, matching the canonical docs' shorthand letters.

    ``S`` sandy-loam (fast draining, low water-holding); ``M`` loam/clayey
    (moderate); ``C`` heavy clay (very slow, waterlogging-prone).
    """

    SANDY = "S"
    MODERATE = "M"
    CLAYEY = "C"

    @classmethod
    def coerce(cls, value: "SoilDrainageClass | str") -> "SoilDrainageClass":
        """Accept an enum, a single letter (S/M/C) or a full name (case-insensitive)."""
        if isinstance(value, cls):
            return value
        s = str(value).strip()
        for member in cls:
            if member.value.lower() == s.lower():
                return member
        by_name = {m.name.lower(): m for m in cls}
        if s.lower() in by_name:
            return by_name[s.lower()]
        raise ValueError(f"unknown soil drainage class: {value!r}")


#: Base actions per (stage, alert): list of (clause, human action, priority).
#: Explicit priorities let the drainage modifier insert between fixed ranks.
_BASE_ACTIONS: dict[tuple[Season, IMDAlertType], list[tuple[str, str, int]]] = {
    (Season.GRAND_GROWTH, IMDAlertType.HEAVY_RAIN_WARNING): [
        ("SUSPEND_IRRIGATION", "Suspend irrigation", 0),
        ("DELAY_FOLIAR_SPRAY", "Delay foliar spray 72h", 2),
    ],
    (Season.MATURITY, IMDAlertType.HEAVY_RAIN_WARNING): [
        ("SUSPEND_IRRIGATION", "Suspend irrigation", 0),
        ("DELAY_FOLIAR_SPRAY", "Delay foliar spray 72h", 2),
        ("HARVEST_WINDOW_CHECK", "Check harvest-window schedule", 3),
    ],
    (Season.RATOON, IMDAlertType.HEAVY_RAIN_WARNING): [
        ("SUSPEND_IRRIGATION", "Suspend irrigation", 0),
        ("CLEAR_FURROW", "Clear furrow outlets", 1),
    ],
    (Season.GRAND_GROWTH, IMDAlertType.HEAT_WAVE): [
        ("OPTIMAL_IRRIGATION", "Irrigate to maintain soil moisture", 0),
        ("AVOID_MIDDAY_FOLIAR", "Avoid midday foliar spray", 1),
        ("DAWN_IRRIGATION", "Prefer dawn irrigation", 2),
    ],
    (Season.MATURITY, IMDAlertType.HEAT_WAVE): [
        ("IRRIGATION_MAINTENANCE", "Maintain irrigation to avoid moisture stress", 0),
    ],
    (Season.GRAND_GROWTH, IMDAlertType.FROST_WARNING): [
        ("DELAY_IRRIGATION", "Delay irrigation to avoid ice risk", 0),
    ],
    (Season.MATURITY, IMDAlertType.FROST_WARNING): [
        ("DELAY_IRRIGATION", "Delay irrigation to avoid ice risk", 0),
    ],
    (Season.MATURITY, IMDAlertType.STRONG_WIND): [
        ("SUSPEND_FOLIAR_SPRAY", "Suspend foliar spray", 0),
    ],
    (Season.GRAND_GROWTH, IMDAlertType.STRONG_WIND): [
        ("SUSPEND_FOLIAR_SPRAY", "Suspend foliar spray", 0),
    ],
}

#: Drainage modifiers appended to the base list for the same (stage, alert).
#: Empty list means "no change from base" for that drainage class.
_DRAINAGE_EXTRA: dict[tuple[Season, IMDAlertType, SoilDrainageClass], list[tuple[str, str, int]]] = {
    (Season.GRAND_GROWTH, IMDAlertType.HEAVY_RAIN_WARNING, SoilDrainageClass.MODERATE): [
        ("CLEAR_FURROW", "Clear furrow outlets", 1),
    ],
    (Season.GRAND_GROWTH, IMDAlertType.HEAVY_RAIN_WARNING, SoilDrainageClass.CLAYEY): [
        ("CLEAR_FURROW", "Clear furrow outlets", 1),
        ("AVOID_FIELD_TRAFFIC", "Avoid field traffic", 3),
    ],
}

#: Universal safe action when (stage, alert) is not in the matrix — the engine
#: must be total (never raise, never return empty), so an unheard-of combo
#: degrades to the routine advisory rather than inventing a recommendation.
_FALLBACK_ACTIONS: list[tuple[str, str, int]] = [
    ("GENERIC_CAUTION", "Follow routine advisory — no defensive action in matrix", 0),
]

_ROUTINE_ALERT: list[tuple[str, str, int]] = [
    ("ROUTINE_SCHEDULE", "Proceed with routine schedule", 0),
]


def weather_to_action(
    crop: Crop | str,
    crop_stage: Season | str,
    alert: IMDAlertType | str,
    drainage: SoilDrainageClass | str,
) -> list[RuleAction]:
    """Ranked defensive actions for a ``(crop, stage, alert, drainage)`` cell.

    Deterministic. Returns actions in recommended order (lowest priority
    first). Every action carries a ``WEATHER:<crop>:<stage>:<alert>:<clause>``
    trace code for the source-traceability gate.
    """
    crop_e = crop if isinstance(crop, Crop) else Crop(crop)
    stage_e = crop_stage if isinstance(crop_stage, Season) else Season(crop_stage)
    alert_e = alert if isinstance(alert, IMDAlertType) else IMDAlertType(alert)
    drainage_e = SoilDrainageClass.coerce(drainage)

    if alert_e is IMDAlertType.NO_ALERT:
        base: list[tuple[str, str, int]] = _ROUTINE_ALERT
    else:
        base = _BASE_ACTIONS.get((stage_e, alert_e), _FALLBACK_ACTIONS)
    extra = _DRAINAGE_EXTRA.get((stage_e, alert_e, drainage_e), [])

    actions = [
        RuleAction(
            trace=f"WEATHER:{crop_e.value}:{stage_e.value}:{alert_e.value}:{clause}",
            action=human,
            clause=clause,
            priority=prio,
        )
        for clause, human, prio in base + extra
    ]
    # RuleAction.__lt__ ranks by (priority, trace) — the recommended order.
    actions.sort()
    return actions
