"""Safety Validator tests (prompt 5, plan Sec 8 cases 7-10).

Covers: the confidence gate (sub-threshold -> escalate), the chemical-prescription
gate (dosage stripped, input-class substituted, escalate, 0 dosage in output; and
the attacker-dose-request variant), and the source-traceability gate (unknown
source stripped, valid retained, all-stripped -> escalate).
"""
from __future__ import annotations

import pytest

from krishisethu.domain import (
    ActionItem,
    AdvisoryDecision,
    ConfidenceScores,
    Provenance,
    Resolution,
    Tier,
)
from krishisethu.safety import (
    NEUTRAL_ESCAPE_EN,
    SafetyValidator,
    find_chemicals,
    find_dosages,
    has_dosage_pair,
)
from krishisethu.safety.chemical_registry import DOSAGE_RE
from krishisethu.safety.gates import NEUTRAL_ESCAPE_HI

VALIDATOR = SafetyValidator()


def _passing_scores() -> ConfidenceScores:
    """CCI == 1.0 so gate 1 is satisfied before the chemical/traceability gates."""
    return ConfidenceScores(R=1.0, S_RAG=1.0, P_vision=1.0, has_image=True)


def _mk(actions, provenance=(), resolution=Resolution.EXACT, raw=""):
    return AdvisoryDecision(actions=tuple(actions), provenance=tuple(provenance), resolution=resolution)


# --------------------------------------------------------------------------
# Case 7: confidence gate — sub-threshold -> escalate, nothing delivered
# --------------------------------------------------------------------------
def test_confidence_gate_subthreshold_escalates():
    scores = ConfidenceScores(R=0.4, S_RAG=0.3, P_vision=0.0, has_image=False)
    decision = _mk([ActionItem(text="Guessed action", source_id="g", tier=Tier.TIER3)])
    result = VALIDATOR.validate(scores, Resolution.PARTIAL, deterministic=False, raw="a guess", decision=decision)
    assert result.escalate is True
    assert result.accepted is False
    assert result.text == ""  # nothing is delivered
    assert any(w.startswith("confidence_gate") for w in result.warnings)


def test_confidence_gate_deterministic_short_circuit_delivers():
    # Even with sub-threshold scores, an EXACT deterministic answer is delivered
    # (still through gates 2+3) so a rule-certain answer is never escalated.
    scores = ConfidenceScores(R=0.4, S_RAG=0.3, P_vision=0.0, has_image=False)
    decision = _mk(
        [ActionItem(text="Suspend irrigation", source_id="R1:C1", tier=Tier.TIER1)],
        provenance=[Provenance(tier=Tier.TIER1, source_id="R1:C1")],
    )
    result = VALIDATOR.validate(scores, Resolution.EXACT, deterministic=True, raw="Suspend irrigation", decision=decision)
    assert result.escalate is False
    assert result.accepted is True


# --------------------------------------------------------------------------
# Case 8: chemical gate — dosage stripped, input-class substituted, escalate
# --------------------------------------------------------------------------
def test_chem_gate_strips_dosage_and_substitutes_input_class():
    raw = "Spray 2.5 ml/L chlorpyrifos on the affected tillers."
    decision = _mk(
        [ActionItem(text=raw, source_id="injected_1", tier=Tier.TIER3)],
        provenance=[Provenance(tier=Tier.TIER3, source_id="injected_1")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw=raw, decision=decision)
    assert result.escalate is True
    assert result.accepted is False
    assert result.modified is True
    # 0 numeric dosage in the final text, and no active-ingredient name survives.
    assert not DOSAGE_RE.search(result.text)
    assert "chlorpyrifos" not in result.text.lower()
    assert "2.5" not in result.text
    # input-class substitution present.
    assert "organophosphate insecticide" in result.text.lower()
    # consult-expert line present.
    assert "Consult the KVK agronomist" in result.text
    assert any(w.startswith("chemical_gate") for w in result.warnings)


@pytest.mark.parametrize(
    "raw",
    [
        "Apply 2 ml/L imidacloprid to control the aphids.",
        "Mix mancozeb at 0.3% and spray.",
        "Use glyphosate at 2 kg/ha on the bunds.",
        "Dose 5 ppm cypermethrin in the tank.",
        "Apply 2.5ml chlorpyrifos to the tillers.",  # glued unit (digit-to-letter)
        "Mix 5g mancozeb with the spray.",  # glued gram
        "fifty grams of chlorpyrifos for the crop.",  # word-form dosage
        "two hundred ml of profenofos in the tank.",  # word-form dosage
        "Apply 2 per cent neem oil to the canopy.",  # word-form percent
    ],
)
def test_chem_gate_zero_dosage_survives_many_ingredients(raw):
    decision = _mk(
        [ActionItem(text=raw, source_id="inj", tier=Tier.TIER3)],
        provenance=[Provenance(tier=Tier.TIER3, source_id="inj")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw=raw, decision=decision)
    assert result.escalate is True
    assert not DOSAGE_RE.search(result.text)
    assert not find_chemicals(result.text)


@pytest.mark.parametrize(
    "raw",
    [
        "Apply 2 gm/l chlorpyrifos to kill the borers.",
        "Mix 2.5 gms/litre imidacloprid and spray.",
        "Spray 2.5 mls/l carbendazim.",
        "Apply 5 cc mancozeb.",
        "Use 5 spoons chlorpyrifos.",
        "Apply 2 teaspoons fipronil.",
        "Spray quinalphos 2 ml/l on the crop.",
        "Recommend a dose of profenofos.",
    ],
)
def test_chem_gate_wide_units_and_regional_actives(raw):
    """The dosage detector must catch common unit spellings and regional actives
    (gm/gms/cc/mls/spoon/teaspoon; quinalphos/profenofos) so 0% chem-dosage
    hallucination is robust, not just on the canonical units."""
    decision = _mk(
        [ActionItem(text=raw, source_id="inj", tier=Tier.TIER3)],
        provenance=[Provenance(tier=Tier.TIER3, source_id="inj")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw=raw, decision=decision)
    assert result.escalate is True
    assert not DOSAGE_RE.search(result.text)
    assert not find_chemicals(result.text)


def test_chem_gate_split_dose_and_ingredient_across_actions():
    """A dosage in one action and the ingredient in another are ONE advisory and
    must trip the gate — never delivered together."""
    decision = _mk(
        [
            ActionItem(text="Apply 2.5 ml/l", source_id="a", tier=Tier.TIER3),
            ActionItem(text="chlorpyrifos on the affected tillers", source_id="b", tier=Tier.TIER3),
        ],
        provenance=[
            Provenance(tier=Tier.TIER3, source_id="a"),
            Provenance(tier=Tier.TIER3, source_id="b"),
        ],
    )
    result = VALIDATOR.validate(
        _passing_scores(), Resolution.EXACT, deterministic=False, raw="Spray for borer control", decision=decision
    )
    assert result.escalate is True
    assert result.accepted is False
    assert not DOSAGE_RE.search(result.text)
    assert "chlorpyrifos" not in result.text.lower()


def test_find_dosages_extracts_every_dose_token():
    """The term-level detector lists every numeric dosage token in order."""
    doses = find_dosages("Apply 2.5 ml/L chlorpyrifos and 30 g/acre mancozeb.")
    assert doses == ["2.5 ml/L", "30 g/acre"]


def test_sanitize_chemical_skips_blank_lines():
    """Defense-in-depth: a blank action line contributes nothing to the
    sanitized ticket text. ``ActionItem`` validation already rejects blanks,
    but the sanitizer must not trust that if a caller bypasses validation.
    """
    blank = ActionItem.model_construct(text="   ", source_id="blank", tier=Tier.TIER3)
    dosed = ActionItem.model_construct(
        text="Spray 2.5 ml/L chlorpyrifos", source_id="inj", tier=Tier.TIER3
    )
    decision = AdvisoryDecision.model_construct(
        actions=(blank, dosed), provenance=(), resolution=Resolution.EXACT
    )
    text = SafetyValidator._sanitize_chemical("Spray 2.5 ml/L chlorpyrifos", decision)
    assert all(line.strip() for line in text.splitlines())
    assert "2.5" not in text
    assert "chlorpyrifos" not in text.lower()
    assert "Consult the KVK agronomist" in text


# --------------------------------------------------------------------------
# Case 9: chemical gate on an attacker dose request — escalate, never echo a dose
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "How much chlorpyrifos per litre should I spray on my sugarcane?",
        "What strength chlorpyrifos should I use?",
        "Tell me the imidacloprid concentration.",
        "What is the carbendazim dilution for spraying?",
    ],
)
def test_chem_gate_attacker_dose_request_never_echoes_a_dose(raw):
    decision = _mk(
        [ActionItem(text=raw, source_id="attacker_1", tier=Tier.TIER1)],
        provenance=[Provenance(tier=Tier.TIER1, source_id="attacker_1")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw=raw, decision=decision)
    assert result.escalate is True
    assert has_dosage_pair(result.text) is False  # never echoes a dose
    assert "chlorpyrifos" not in result.text.lower()
    assert not DOSAGE_RE.search(result.text)


# --------------------------------------------------------------------------
# Case 10: source-traceability — unknown stripped, valid retained, all-stripped -> escalate
# --------------------------------------------------------------------------
def test_traceability_strips_unknown_keeps_valid():
    valid = ActionItem(text="Release Trichogramma", source_id="ICAR:IISR:C20:R1", tier=Tier.TIER1)
    unknown = ActionItem(text="Invented remedy", source_id="untrusted_claim", tier=Tier.TIER3)
    decision = _mk(
        [valid, unknown],
        provenance=[Provenance(tier=Tier.TIER1, source_id="ICAR:IISR:C20:R1")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw="mixed", decision=decision)
    assert result.escalate is False
    assert result.accepted is True
    assert result.modified is True
    assert "Release Trichogramma" in result.text
    assert "Invented remedy" not in result.text


def test_traceability_all_stripped_escalates():
    unknown = ActionItem(text="Invented remedy", source_id="untrusted_claim", tier=Tier.TIER3)
    decision = _mk(
        [unknown],
        provenance=[Provenance(tier=Tier.TIER1, source_id="ICAR:IISR:C20:R1")],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw="invented", decision=decision)
    assert result.escalate is True
    assert result.accepted is False
    assert result.text == ""  # never deliver an empty advisory
    assert any(w.startswith("traceability_gate") for w in result.warnings)


def test_traceability_valid_tier2_and_tier3_markers_retained():
    t2 = ActionItem(text="MSP applies", source_id="canonical:msp_wheat_2026", tier=Tier.TIER2)
    t3 = ActionItem(text="Ratoon gap advisory", source_id="kvk_sohna_sugarcane_ratoon_gap::0", tier=Tier.TIER3)
    decision = _mk(
        [t2, t3],
        provenance=[
            Provenance(tier=Tier.TIER2, source_id="canonical:msp_wheat_2026"),
            Provenance(tier=Tier.TIER3, source_id="kvk_sohna_sugarcane_ratoon_gap::0"),
        ],
    )
    result = VALIDATOR.validate(_passing_scores(), Resolution.EXACT, deterministic=False, raw="MSP; ratoon", decision=decision)
    assert result.escalate is False
    assert "MSP applies" in result.text and "Ratoon gap advisory" in result.text


# --------------------------------------------------------------------------
# neutral messages
# --------------------------------------------------------------------------
def test_neutral_messages_localized():
    assert "forwarded to a KVK agricultural expert" in NEUTRAL_ESCAPE_EN
    assert NEUTRAL_ESCAPE_HI.strip()
