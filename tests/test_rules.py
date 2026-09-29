"""Tests for the Tier-1 deterministic rule engine (Prompt 3).

Covers: exhaustive sprayOK boolean cases, the white-paper Sohna weather-to-
action example, irrigation trigger arithmetic, ratoon gap-fill + rotation
thresholds, R(C) exact/partial/unresolved, a determinism property test, and
an architecture assertion that this tier has no LLM/network dependency.
"""
from __future__ import annotations

import ast
from itertools import product
from pathlib import Path

import pytest

import krishisethu.rules as rules_pkg
from krishisethu.domain import Crop, Resolution, Season
from krishisethu.rules import (
    ECONOMIC_VIABILITY_THRESHOLD,
    IMDAlertType,
    RuleAction,
    RatoonStage,
    SoilDrainageClass,
    SprayConditions,
    classify,
    gap_fill_decision,
    irrigation_due,
    irrigation_from_et0_whc,
    ratoon_plan,
    ratoon_stage_checklist,
    rotation_recommendation,
    rule_match,
    rule_match_from_actions,
    soil_water_holding_capacity_mm,
    sprayOK,
    weather_to_action,
)

RULES_DIR = Path(rules_pkg.__file__).parent


# --------------------------------------------------------------------------
# weather-to-action: Sohna example + matrix totality
# --------------------------------------------------------------------------
def test_weather_action_sohna_example_order_and_trace():
    acts = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, IMDAlertType.HEAVY_RAIN_WARNING, "M")
    assert [a.action for a in acts] == [
        "Suspend irrigation",
        "Clear furrow outlets",
        "Delay foliar spray 72h",
    ]
    assert all(a.trace.startswith("WEATHER:sugarcane:grand_growth:heavy_rain_warning:") for a in acts)
    # priorities are monotonic (recommended order)
    assert [a.priority for a in acts] == sorted(a.priority for a in acts)


def test_weather_action_drainage_varies():
    # Sandy loam (S): base actions, no furrow-extra inserted.
    sands = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "S")
    assert [a.action for a in sands] == ["Suspend irrigation", "Delay foliar spray 72h"]
    # Heavy clay (C): furrow clearance + avoid traffic added.
    clays = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "C")
    assert "Avoid field traffic" in [a.action for a in clays]
    assert "Clear furrow outlets" in [a.action for a in clays]


def test_weather_action_no_alert_routine():
    acts = weather_to_action(Crop.WHEAT, Season.RABI, "no_alert", "M")
    assert [a.clause for a in acts] == ["ROUTINE_SCHEDULE"]
    assert acts[0].trace.startswith("WEATHER:wheat:rabi:no_alert:")


def test_weather_action_totality_unknown_combo_falls_back():
    # (KHARIF, HEAT_WAVE) is not in the matrix -> must degrade, never raise/empty.
    acts = weather_to_action(Crop.WHEAT, "kharif", "heat_wave", "S")
    assert len(acts) >= 1
    assert acts[0].clause == "GENERIC_CAUTION"


def test_weather_action_coerce_drainage_by_name():
    acts = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "moderate")
    assert "Clear furrow outlets" in [a.action for a in acts]


def test_weather_action_coerce_drainage_enum_directly():
    # Passing a SoilDrainageClass enum (not a string) must be accepted.
    acts = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning",
                             SoilDrainageClass.MODERATE)
    assert "Clear furrow outlets" in [a.action for a in acts]


def test_weather_action_unknown_drainage_raises():
    with pytest.raises(ValueError):
        weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "Z")


# --------------------------------------------------------------------------
# spray windows: exhaustive 16-combo boolean cases + boundaries
# --------------------------------------------------------------------------
def _conditions(temp_ok, rh_ok, wind_ok, no_precip_ok):
    """Build SprayConditions from 4 'condition is safe' flags."""
    return SprayConditions(
        temperature_c=25.0 if temp_ok else 40.0,  # 40 is outside [18, 35]
        relative_humidity_pct=60.0 if rh_ok else 90.0,  # 90 >= 85
        wind_kmh=5.0 if wind_ok else 12.0,  # 12 >= 10
        precipitation_within_24h=not no_precip_ok,  # no_precip_ok True -> dry
    )


@pytest.mark.parametrize(
    ("temp_ok", "rh_ok", "wind_ok", "no_precip_ok"),
    list(product([True, False], repeat=4)),
)
def test_sprayok_exhaustive_all_16_combos(temp_ok, rh_ok, wind_ok, no_precip_ok):
    res = sprayOK(_conditions(temp_ok, rh_ok, wind_ok, no_precip_ok))
    all_ok = temp_ok and rh_ok and wind_ok and no_precip_ok
    assert res.ok is all_ok
    if all_ok:
        assert res.violations == ()
    else:
        expected = set()
        if not wind_ok:
            expected.add("wind_gt_eq_10")
        if not no_precip_ok:
            expected.add("precip_within_24h")
        if not rh_ok:
            expected.add("humidity_ge_85")
        if not temp_ok:
            expected.add("temperature_outside_18_35")
        assert set(res.violations) == expected


def test_sprayok_boundaries():
    # wind exactly 10 -> violated (must be < 10)
    assert "wind_gt_eq_10" in sprayOK(SprayConditions(25, 60, 10.0, False)).violations
    # RH exactly 85 -> violated (must be < 85)
    assert "humidity_ge_85" in sprayOK(SprayConditions(25, 85.0, 5, False)).violations
    # temperature 18 and 35 inclusive -> OK; 17.9 / 35.1 -> violated
    assert sprayOK(SprayConditions(18.0, 60, 5, False)).ok
    assert sprayOK(SprayConditions(35.0, 60, 5, False)).ok
    assert "temperature_outside_18_35" in sprayOK(SprayConditions(17.9, 60, 5, False)).violations
    assert "temperature_outside_18_35" in sprayOK(SprayConditions(35.1, 60, 5, False)).violations


def test_sprayok_precip_override_arg_w():
    # Two-arg signature: w overrides the flag on t.
    ok = sprayOK(SprayConditions(25, 60, 5, False), w=True)
    assert not ok.ok
    assert ok.violations == ("precip_within_24h",)


# --------------------------------------------------------------------------
# irrigation: trigger arithmetic + AWC helper
# --------------------------------------------------------------------------
def test_irrigation_due_threshold_hit():
    rec = irrigation_due(et0_mm=60.0, effective_rainfall_mm=0.0, awc_mm=100.0)
    assert rec.due is True
    assert rec.recommended_volume_mm == 60.0
    assert rec.deficit_mm == 60.0
    assert rec.clamped_to_single_application is False
    assert "SUGAR-IRR-014:DUE" in rec.trace


def test_irrigation_not_due_below_threshold():
    rec = irrigation_due(et0_mm=30.0, effective_rainfall_mm=10.0, awc_mm=100.0)
    assert rec.due is False
    assert rec.available_water_mm == 80.0
    assert rec.recommended_volume_mm == 0.0


def test_irrigation_clamped_to_single_application():
    rec = irrigation_due(et0_mm=120.0, effective_rainfall_mm=0.0, awc_mm=150.0)
    assert rec.due is True
    assert rec.deficit_mm == 120.0
    assert rec.recommended_volume_mm == 75.0  # capped at max single application
    assert rec.clamped_to_single_application is True


def test_irrigation_rainfall_offsets_demand():
    rec = irrigation_due(et0_mm=70.0, effective_rainfall_mm=40.0, awc_mm=100.0)
    assert rec.due is False
    assert rec.deficit_mm == 0.0


def test_irrigation_net_use_never_negative():
    rec = irrigation_due(et0_mm=5.0, effective_rainfall_mm=50.0, awc_mm=100.0)
    assert rec.available_water_mm == 100.0
    assert rec.due is False
    assert rec.recommended_volume_mm == 0.0


def test_irrigation_whc_helper_and_convenience():
    assert soil_water_holding_capacity_mm(150.0, 0.6) == 90.0
    rec = irrigation_from_et0_whc(et0_mm=60.0, effective_rainfall_mm=0.0, whc_mm_per_m=150.0, root_depth_m=0.6)
    assert rec.awc_mm == 90.0
    assert rec.due is True
    assert rec.recommended_volume_mm == 60.0


def test_irrigation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        irrigation_due(et0_mm=10.0, effective_rainfall_mm=0.0, awc_mm=0.0)
    with pytest.raises(ValueError):
        irrigation_due(et0_mm=10.0, effective_rainfall_mm=0.0, awc_mm=100.0, deficit_threshold_fraction=0.0)
    with pytest.raises(ValueError):
        irrigation_due(et0_mm=10.0, effective_rainfall_mm=0.0, awc_mm=100.0,
                       deficit_threshold_fraction=1.5)
    with pytest.raises(ValueError):
        irrigation_due(et0_mm=10.0, effective_rainfall_mm=0.0, awc_mm=100.0,
                       max_single_application_mm=0.0)
    with pytest.raises(ValueError):
        soil_water_holding_capacity_mm(0.0, 0.6)
    with pytest.raises(ValueError):
        soil_water_holding_capacity_mm(150.0, 0.0)


# --------------------------------------------------------------------------
# ratoon: gap-fill thresholds, rotation thresholds, Sohna case
# --------------------------------------------------------------------------
def test_gap_fill_thresholds():
    assert gap_fill_decision(0.15).decision == "PRESERVE"
    assert gap_fill_decision(0.20).decision == "PRESERVE"
    assert gap_fill_decision(0.21).decision == "SELECTIVE_GAP_FILL"
    assert gap_fill_decision(0.30).decision == "SELECTIVE_GAP_FILL"
    assert gap_fill_decision(0.40).decision == "REPLANT"
    assert gap_fill_decision(0.45).decision == "REPLANT"


def test_rotation_below_economic_threshold():
    rot = rotation_recommendation(ratoon_number=3, viability=0.50, gap_fraction=0.45)
    assert rot.rotate is True
    assert rot.recommended_crop is not None
    assert f"{ECONOMIC_VIABILITY_THRESHOLD:.2f}" in rot.reason
    assert "low ratoon viability" in rot.reason


def test_rotation_above_threshold_keeps():
    rot = rotation_recommendation(ratoon_number=1, viability=0.90, gap_fraction=0.15)
    assert rot.rotate is False
    assert rot.recommended_crop is None


def test_rotation_bounded_at_threshold():
    assert rotation_recommendation(1, 0.60, 0.10).rotate is False
    assert rotation_recommendation(1, 0.59, 0.10).rotate is True


def test_rotation_high_ratoon_number_driver():
    rot = rotation_recommendation(ratoon_number=4, viability=0.55, gap_fraction=0.50)
    assert rot.rotate is True
    assert "ratoon age" in rot.reason


def test_gap_fill_rejects_negative():
    with pytest.raises(ValueError):
        gap_fill_decision(-0.1)


def test_rotation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        rotation_recommendation(1, 1.5, 0.10)
    with pytest.raises(ValueError):
        rotation_recommendation(0, 0.5, 0.10)


def test_ratoon_plan_sohna_case():
    plan = ratoon_plan(stage=RatoonStage.GRAND_GROWTH, gap_fraction=0.45, ratoon_number=3, viability=0.50)
    # Sohna: uneconomic 3rd ratoon -> recommend rotation, not replant.
    assert plan.rotation.rotate is True
    assert plan.rotation.recommended_crop is not None
    assert plan.gap.decision == "ROTATE"
    assert plan.checklist  # stage checklist populated
    assert all(c.trace.startswith("RATOON:grand_growth:") for c in plan.checklist)


def test_ratoon_plan_healthy_ratoon():
    plan = ratoon_plan(stage="establishment", gap_fraction=0.10, ratoon_number=1, viability=0.90)
    assert plan.rotation.rotate is False
    assert plan.gap.decision == "PRESERVE"
    assert all(c.trace.startswith("RATOON:establishment:") for c in plan.checklist)


def test_ratoon_stage_checklists_all_stages():
    for stage in RatoonStage:
        checklist = ratoon_stage_checklist(stage)
        assert checklist
        assert all(c.trace.startswith(f"RATOON:{stage.value}:") for c in checklist)


# --------------------------------------------------------------------------
# R(C): exact / partial / unresolved
# --------------------------------------------------------------------------
def _three_weather_actions():
    return [
        RuleAction(trace="WEATHER:SUGARCANE:GRAND_GROWTH:HEAVY_RAIN_WARNING:SUSPEND_IRRIGATION",
                   action="Suspend irrigation", clause="SUSPEND_IRRIGATION", priority=0),
        RuleAction(trace="WEATHER:SUGARCANE:GRAND_GROWTH:HEAVY_RAIN_WARNING:CLEAR_FURROW",
                   action="Clear furrow outlets", clause="CLEAR_FURROW", priority=1),
        RuleAction(trace="WEATHER:SUGARCANE:GRAND_GROWTH:HEAVY_RAIN_WARNING:DELAY_FOLIAR_SPRAY",
                   action="Delay foliar spray 72h", clause="DELAY_FOLIAR_SPRAY", priority=2),
    ]


def test_rule_match_exact():
    m = rule_match(Resolution.EXACT, matched_clauses=4, total_clauses=4)
    assert m.resolution is Resolution.EXACT
    assert m.score == 1.0


def test_rule_match_unresolved():
    m = rule_match(Resolution.UNRESOLVED, matched_clauses=0, total_clauses=4)
    assert m.resolution is Resolution.UNRESOLVED
    assert m.score == 0.0


def test_rule_match_partial_normalized():
    m = rule_match(Resolution.PARTIAL, matched_clauses=2, total_clauses=5)
    assert m.resolution is Resolution.PARTIAL
    assert m.score == 0.4


def test_rule_match_partial_zero_total_is_zero():
    m = rule_match(Resolution.PARTIAL, matched_clauses=0, total_clauses=0)
    assert m.score == 0.0


def test_rule_match_from_actions_exact_and_trace():
    acts = _three_weather_actions()
    m = rule_match_from_actions(acts, total_clauses=3)
    assert m.resolution is Resolution.EXACT
    assert m.score == 1.0
    assert m.matched_clauses == 3
    assert m.trace_codes[0].startswith("WEATHER:")


def test_rule_match_from_actions_partial():
    m = rule_match_from_actions(_three_weather_actions(), total_clauses=5)
    assert m.resolution is Resolution.PARTIAL
    assert m.score == 0.6


def test_rule_match_from_actions_unresolved():
    m = rule_match_from_actions([], total_clauses=4)
    assert m.resolution is Resolution.UNRESOLVED
    assert m.score == 0.0


def test_rule_match_distinct_clause_not_double_counted():
    acts = [
        RuleAction(trace="R:1:C1", action="a1", clause="C1"),
        RuleAction(trace="R:1:C1", action="a1-again", clause="C1"),
    ]
    m = rule_match_from_actions(acts, total_clauses=2)
    assert m.matched_clauses == 1
    assert m.score == 0.5


def test_classify():
    assert classify(0, 5) is Resolution.UNRESOLVED
    assert classify(3, 5) is Resolution.PARTIAL
    assert classify(5, 5) is Resolution.EXACT
    assert classify(6, 5) is Resolution.EXACT  # all matched (>= total)


# --------------------------------------------------------------------------
# determinism property: identical inputs -> identical outputs
# --------------------------------------------------------------------------
def test_determinism_property():
    a = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "M")
    b = weather_to_action(Crop.SUGARCANE, Season.GRAND_GROWTH, "heavy_rain_warning", "M")
    assert [(x.trace, x.action, x.clause, x.priority) for x in a] == \
        [(x.trace, x.action, x.clause, x.priority) for x in b]

    sc = SprayConditions(25.0, 60.0, 5.0, False)
    assert sprayOK(sc) == sprayOK(sc)

    assert irrigation_due(60.0, 0.0, 100.0) == irrigation_due(60.0, 0.0, 100.0)

    p1 = ratoon_plan("grand_growth", 0.45, 3, 0.50)
    p2 = ratoon_plan("grand_growth", 0.45, 3, 0.50)
    assert p1 == p2

    assert rule_match(Resolution.PARTIAL, 2, 5) == rule_match(Resolution.PARTIAL, 2, 5)


# --------------------------------------------------------------------------
# architecture assertion: no LLM / network dependency in this tier
# --------------------------------------------------------------------------
def test_rules_tier_has_no_llm_or_network_imports():
    forbidden_prefixes = (
        "langchain", "chromadb", "qdrant", "faiss",
        "sentence_transformers", "transformers", "torch",
        "ollama", "requests", "httpx", "aiohttp", "urllib", "socket", "fastapi",
    )
    for py in sorted(RULES_DIR.rglob("*.py")):
        src = py.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"{py.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith(forbidden_prefixes), f"{py.name} imports from {mod}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name.lower()
                assert "embed" not in name and "chunk" not in name, (
                    f"{py.name} defines function {node.name} (embedding/chunking not allowed in Tier 1)"
                )
