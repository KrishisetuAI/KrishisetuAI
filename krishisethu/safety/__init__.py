"""Safety Validator — 3 gates.

Gate 1 confidence (>=0.85 to answer, else escalate).
Gate 2 chemical-prescription (strip any dosage, input-class only).
Gate 3 source-traceability (every claim maps to a Tier1 rule / Tier2 doc / Tier3 chunk).

Public API:
  SafetyValidator, NEUTRAL_ESCAPE_EN, NEUTRAL_ESCAPE_HI
  chemical_registry: CHEMICALS, ChemicalEntry, DOSAGE_RE, DOSE_INTENT_RE,
    find_chemicals, find_dosages, has_numeric_dosage, has_dose_intent,
    has_dosage_pair, strip_dosages
"""
from __future__ import annotations

from krishisethu.safety.chemical_registry import (
    CHEMICALS,
    DOSAGE_RE,
    DOSE_INTENT_RE,
    ChemicalEntry,
    find_chemicals,
    find_dosages,
    has_dose_intent,
    has_dosage_pair,
    has_numeric_dosage,
    strip_dosages,
)
from krishisethu.safety.gates import (
    NEUTRAL_ESCAPE_EN,
    NEUTRAL_ESCAPE_HI,
    SafetyValidator,
)

__all__ = [
    # gates
    "SafetyValidator",
    "NEUTRAL_ESCAPE_EN",
    "NEUTRAL_ESCAPE_HI",
    # chemical registry
    "CHEMICALS",
    "ChemicalEntry",
    "DOSAGE_RE",
    "DOSE_INTENT_RE",
    "find_chemicals",
    "find_dosages",
    "has_numeric_dosage",
    "has_dose_intent",
    "has_dosage_pair",
    "strip_dosages",
]
