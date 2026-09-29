"""Spray-window check (Tier 1).

``sprayOK(t, w)`` returns True only when ALL four conditions hold:

* wind < 10 km/h
* no precipitation forecast within 24h
* relative humidity < 85%
* temperature in [18, 35] degrees C

It returns the list of violated clause labels so the UI can explain WHY
spraying is not OK (the gate is only a boolean; the explanation is the point).
PURE PYTHON, deterministic — the four bounds are named constants so a change
to any threshold is a one-line edit with an obvious test.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Clause labels (stable machine strings; used in tests + UI copy).
CLAUSE_WIND = "wind_gt_eq_10"
CLAUSE_PRECIP = "precip_within_24h"
CLAUSE_HUMIDITY = "humidity_ge_85"
CLAUSE_TEMPERATURE = "temperature_outside_18_35"

#: Hard bounds from the white paper / PoP (spray window).
WIND_MAX_KMH = 10.0  # strictly less than
RH_MAX_PCT = 85.0  # strictly less than
TEMP_MIN_C = 18.0  # inclusive
TEMP_MAX_C = 35.0  # inclusive


@dataclass(frozen=True)
class SprayConditions:
    """Live conditions at the spray decision point."""

    temperature_c: float
    relative_humidity_pct: float
    wind_kmh: float
    precipitation_within_24h: bool


@dataclass(frozen=True)
class SprayWindowResult:
    """Verdict + the clause labels that failed (empty means spraying is OK)."""

    ok: bool
    violations: tuple[str, ...]


def sprayOK(
    t: SprayConditions,
    w: bool | None = None,
) -> SprayWindowResult:
    """Return whether spraying is safe, plus the violated clause labels.

    ``t`` is the conditions. ``w`` is an optional override for the
    precipitation-within-24h flag (kept for the two-argument ``sprayOK(t, w)``
    signature in the white paper); when ``None`` the flag on ``t`` is used.
    """
    precipitation = t.precipitation_within_24h if w is None else bool(w)

    violations: list[str] = []
    if not (t.wind_kmh < WIND_MAX_KMH):
        violations.append(CLAUSE_WIND)
    if precipitation:
        violations.append(CLAUSE_PRECIP)
    if not (t.relative_humidity_pct < RH_MAX_PCT):
        violations.append(CLAUSE_HUMIDITY)
    if not (TEMP_MIN_C <= t.temperature_c <= TEMP_MAX_C):
        violations.append(CLAUSE_TEMPERATURE)

    return SprayWindowResult(ok=not violations, violations=tuple(violations))
