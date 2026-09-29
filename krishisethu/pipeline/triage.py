"""Query triage (Algorithm 1 entry).

Deterministic, framework-free intent classification. Tier 1 must only fire on
weather/irrigation-answerable queries: an EXACT weather resolution is rule
certainty, so an OOD or malicious query must NEVER get the weather short-circuit
just because the demo plot happens to have a rain warning. The classifier keeps
that separation:

* ``WEATHER``      — rain/water/irrigation/weekly-briefing queries -> run Tier 1.
* ``CHEMICAL``     — dosage intent or an active-ingredient name -> flagged for the
                     safety backbone; Tier 1 is skipped so the query escalates.
* ``OOD``          — anything else (zinc deficiency, fertilizer, pests, other
                     crops/seasons) -> Tier 1 skipped, retrieval + confidence
                     decide, and a below-gate CCI escalates.

Pure Python, no LLM, no network — identical inputs always yield the same intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from krishisethu.safety.chemical_registry import (
    DOSE_INTENT_RE,
    has_dosage_pair,
)


class Intent(str, Enum):
    WEATHER = "weather"     # run the Tier-1 weather/irrigation rules
    CHEMICAL = "chemical"   # dosage/active-ingredient request -> escalate
    OOD = "ood"             # out-of-distribution -> retrieval + confidence gate


#: Substrings that mark a query as weather/irrigation-answerable.
_WEATHER_MARKERS = (
    "weekly", "briefing", "rain", "water", "irrigat", "spray", "monsoon",
    "furrow", "what should i do", "advisory", "weather",
)

#: Substrings that mark a query as clearly out-of-distribution for the rules.
_OOD_MARKERS = (
    "zinc", "yellow", "fertilizer", "fertiliser", "wheat", "pest", "disease",
    "fungicide", "herbicide", "insecticide", "borer", "rust", "blast",
)


@dataclass(frozen=True)
class TriageResult:
    """Intent + safety flags for one query."""

    intent: Intent
    flagged_dosage: bool = False  # dosage intent detected -> escalate path
    chemical_hint: str | None = None  # first registry ingredient matched


def classify(query: str) -> TriageResult:
    """Classify a farmer query into :class:`Intent`.

    Order matters: a dosage pair is a chemical request first and foremost; a
    chemical intent also overrides the weather markers (an attacker phrasing a
    dosage request inside a weather sentence must still escalate).
    """
    text = " ".join(str(query or "").lower().split())

    if has_dosage_pair(text) or DOSE_INTENT_RE.search(text):
        return TriageResult(intent=Intent.CHEMICAL, flagged_dosage=True)

    if any(marker in text for marker in _CHEMICAL_NAME_MARKERS):
        return TriageResult(intent=Intent.CHEMICAL)

    if any(marker in text for marker in _OOD_MARKERS):
        return TriageResult(intent=Intent.OOD)

    if any(marker in text for marker in _WEATHER_MARKERS):
        return TriageResult(intent=Intent.WEATHER)

    # No signal at all: default to weather so the weekly briefing still resolves.
    return TriageResult(intent=Intent.WEATHER)


#: Registry active-ingredient names (lowercase) used as chemical markers.
_CHEMICAL_NAME_MARKERS: tuple[str, ...] = (
    "chlorpyrifos", "chlorpyriphos", "imidacloprid", "glyphosate",
    "carbendazim", "cypermethrin", "mancozeb", "deltamethrin", "neem oil",
    "quinalphos", "profenofos", "acephate", "thiamethoxam", "lambda",
    "fipronil", "indoxacarb", "atrazine", "ml/l", "ml per litre",
)
