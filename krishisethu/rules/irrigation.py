"""Deficit-irrigation decision (Tier 1).

From IMD district ET0 (mm, accumulated since the last irrigation) and the soil
water-holding capacity of the root zone, decide whether irrigation is due and,
if so, how much water to apply. Uses a funded deficit threshold: irrigate when
the root-zone depletion reaches ``DEFICIT_THRESHOLD_FRACTION`` of available
water capacity (canonical norm: irrigate as soil moisture drops toward 40% of
AWC, i.e. at 60% depletion).

PURE PYTHON, deterministic. The defaults mirror the Tier-2 canonical document
``SUGAR-IRR-014`` (75 mm per irrigation during grand growth); the number is a
parameter, so changing it is a one-line edit with an obvious test. No model,
no network, no I/O. 1 mm of water over 1 ha = 10 cubic metres (10 m^3/ha).
"""
from __future__ import annotations

from dataclasses import dataclass

#: Irrigate once root-zone depletion reaches this fraction of AWC (0.60 ->
#:  40% of AWC remains, matching the canonical SUGAR-IRR-014 wording).
DEFICIT_THRESHOLD_FRACTION = 0.60
#: Cap on a single application (canonical grand-growth norm, SUGAR-IRR-014).
DEFAULT_MAX_SINGLE_APPLICATION_MM = 75.0
#: Default root-zone depth used by :func:`soil_water_holding_capacity_mm`.
DEFAULT_ROOT_DEPTH_M = 0.60


@dataclass(frozen=True)
class IrrigationRecommendation:
    """Decision + numbers for the irrigation action (mm depth)."""

    due: bool
    recommended_volume_mm: float
    deficit_mm: float
    available_water_mm: float
    awc_mm: float
    clamped_to_single_application: bool
    trace: str


def soil_water_holding_capacity_mm(
    whc_mm_per_m: float,
    root_depth_m: float = DEFAULT_ROOT_DEPTH_M,
) -> float:
    """AWC of the root zone = WHC (mm/m) * root depth (m)."""

    if whc_mm_per_m <= 0:
        raise ValueError("whc_mm_per_m must be > 0")
    if root_depth_m <= 0:
        raise ValueError("root_depth_m must be > 0")
    return round(whc_mm_per_m * root_depth_m, 2)


def irrigation_due(
    et0_mm: float,
    effective_rainfall_mm: float,
    awc_mm: float,
    deficit_threshold_fraction: float = DEFICIT_THRESHOLD_FRACTION,
    max_single_application_mm: float = DEFAULT_MAX_SINGLE_APPLICATION_MM,
    rule_id: str = "SUGAR-IRR-014",
) -> IrrigationRecommendation:
    """Return the irrigation decision for an accumulated ET0 period.

    Depletion = max(0, ET0 - effective rainfall). Irrigation is due when that
    depletion reaches ``deficit_threshold_fraction`` of AWC. The recommended
    volume refills the root zone to field capacity, capped at a single
    application (split if the deficit exceeds the cap).
    """
    if awc_mm <= 0:
        raise ValueError("awc_mm must be > 0")
    if not (0.0 < deficit_threshold_fraction <= 1.0):
        raise ValueError("deficit_threshold_fraction must be in (0, 1]")
    if max_single_application_mm <= 0:
        raise ValueError("max_single_application_mm must be > 0")

    net_use = max(0.0, et0_mm - effective_rainfall_mm)
    available = awc_mm - net_use
    threshold_use = deficit_threshold_fraction * awc_mm

    due = net_use >= threshold_use
    deficit = net_use if due else 0.0
    volume = min(deficit, max_single_application_mm) if due else 0.0
    clamped = bool(due and volume < deficit - 1e-9)

    trace = f"IRR:{rule_id}:{'DUE' if due else 'NOT_DUE'}"

    return IrrigationRecommendation(
        due=due,
        recommended_volume_mm=round(volume, 1),
        deficit_mm=round(deficit, 1),
        available_water_mm=round(max(0.0, available), 1),
        awc_mm=round(awc_mm, 2),
        clamped_to_single_application=clamped,
        trace=trace,
    )


def irrigation_from_et0_whc(
    et0_mm: float,
    effective_rainfall_mm: float,
    whc_mm_per_m: float,
    root_depth_m: float = DEFAULT_ROOT_DEPTH_M,
    **kwargs,
) -> IrrigationRecommendation:
    """Convenience: compute AWC from soil water-holding capacity then decide."""

    awc = soil_water_holding_capacity_mm(whc_mm_per_m, root_depth_m)
    return irrigation_due(et0_mm, effective_rainfall_mm, awc, **kwargs)
