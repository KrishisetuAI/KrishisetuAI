"""TIER 1 — Deterministic Rule Engine.

PURE PYTHON, immutable against LLM influence. No model calls, no network, no
non-determinism. Weather-to-action matrices, spray windows, irrigation, ratoon.
NEVER import LangChain or an LLM here; a new rule = a pure function + a matrix
row (+ a unit test).
"""
from __future__ import annotations

from krishisethu.rules.contracts import RuleAction
from krishisethu.rules.irrigation import (
    DEFAULT_MAX_SINGLE_APPLICATION_MM,
    DEFICIT_THRESHOLD_FRACTION,
    IrrigationRecommendation,
    irrigation_due,
    irrigation_from_et0_whc,
    soil_water_holding_capacity_mm,
)
from krishisethu.rules.ratoon import (
    ECONOMIC_VIABILITY_THRESHOLD,
    GapDecision,
    RatoonPlan,
    RatoonStage,
    RotationRecommendation,
    gap_fill_decision,
    ratoon_plan,
    ratoon_stage_checklist,
    rotation_recommendation,
)
from krishisethu.rules.rule_match import (
    RuleMatch,
    classify,
    rule_match,
    rule_match_from_actions,
)
from krishisethu.rules.spray_windows import (
    RH_MAX_PCT,
    TEMP_MAX_C,
    TEMP_MIN_C,
    WIND_MAX_KMH,
    SprayConditions,
    SprayWindowResult,
    sprayOK,
)
from krishisethu.rules.weather_action import (
    IMDAlertType,
    SoilDrainageClass,
    weather_to_action,
)

__all__ = [
    # contracts
    "RuleAction",
    # weather-to-action
    "IMDAlertType",
    "SoilDrainageClass",
    "weather_to_action",
    # spray windows
    "SprayConditions",
    "SprayWindowResult",
    "sprayOK",
    "WIND_MAX_KMH",
    "RH_MAX_PCT",
    "TEMP_MIN_C",
    "TEMP_MAX_C",
    # irrigation
    "IrrigationRecommendation",
    "irrigation_due",
    "irrigation_from_et0_whc",
    "soil_water_holding_capacity_mm",
    "DEFICIT_THRESHOLD_FRACTION",
    "DEFAULT_MAX_SINGLE_APPLICATION_MM",
    # ratoon
    "RatoonStage",
    "GapDecision",
    "RotationRecommendation",
    "RatoonPlan",
    "ratoon_stage_checklist",
    "gap_fill_decision",
    "rotation_recommendation",
    "ratoon_plan",
    "ECONOMIC_VIABILITY_THRESHOLD",
    # rule match
    "RuleMatch",
    "rule_match",
    "rule_match_from_actions",
    "classify",
]
