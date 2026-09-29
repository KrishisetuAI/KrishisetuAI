"""Context assembler (Prompt 7 step 2).

Builds the plot context from a demo profile (Demo Farmer Sohna) plus the IMD
weather fixture, crop stage, soil drainage, and an optional image placeholder
(P_vision = 0). No live API yet: the plot profile store is a stub that returns
the Sohna profile for any ``plot_id`` until the real DPI data connectors land.

Season/crop-stage note: the Tier-1 rules key on *crop stage* (``grand_growth``)
while the Tier-3 collection key is the *growing season* (``kharif`` for the
monsoon sugarcane corpus). The assembled context therefore carries BOTH: the
``PlotContext`` used by retrieval is ``kharif`` (the corpus is seeded with
``crop_sugarcane_kharif`` chunks), and the stage fed to the weather matrix is
``grand_growth``. This mirrors the production mapping stage -> season.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from krishisethu.domain import Crop, DecisionType, Season, State
from krishisethu.pipeline.triage import Intent, TriageResult
from krishisethu.rag import PlotContext
from krishisethu.rules.weather_action import IMDAlertType, SoilDrainageClass


@dataclass(frozen=True)
class AssembledContext:
    """Everything downstream tiers need, derived from a plot + triage."""

    plot_context: PlotContext
    crop_stage: Season
    weather_alert: IMDAlertType
    soil_drainage: SoilDrainageClass
    has_image: bool
    on_date: date
    run_tier1: bool
    decision_type: DecisionType = DecisionType.IRRIGATION_RULE


@dataclass(frozen=True)
class DemoProfile:
    """Demo Farmer Sohna — the scripted plot used by ``--demo``.

    Mirrors the white-paper worked example: sugarcane in grand growth, IMD
    heavy-rain warning, moderate (loam) drainage. No image in this build, so
    P_vision = 0.0 and the vision weight is redistributed to (0.625, 0.375).
    """

    plot_id: str = "sohna-demo"
    crop: Crop = Crop.SUGARCANE
    state: State = State.HARYANA
    district: str = "sohna"
    season: Season = Season.KHARIF      # Tier-3 collection key
    crop_stage: Season = Season.GRAND_GROWTH  # Tier-1 weather-matrix key
    weather_alert: IMDAlertType = IMDAlertType.HEAVY_RAIN_WARNING
    soil_drainage: SoilDrainageClass = SoilDrainageClass.MODERATE
    has_image: bool = False

    def plot_context(self) -> PlotContext:
        return PlotContext(
            crop=self.crop,
            district=self.district,
            season=self.season,
            state=self.state.value,
        )


#: Default IMD fixture for non-demo (plot lookup not yet implemented): no alert,
#: so the demo story's escalation beats are driven by intent + confidence, not
#: by a phantom storm on every plot.
_NON_DEMO_ALERT = IMDAlertType.NO_ALERT


def assemble(
    query: str,
    plot_id: str = "demo",
    demo: bool = False,
    on_date: date | None = None,
    triage: TriageResult | None = None,
) -> AssembledContext:
    """Assemble the context for one query.

    ``demo`` selects the Sohna profile (weekly-heavy-rain fixture). Otherwise a
    stub profile lookup returns the same plot shape with ``no_alert`` weather —
    a real plot store replaces this in a later phase. ``triage`` decides whether
    Tier 1 may run (weather intent only).
    """
    profile = DemoProfile()
    triage = triage or TriageResult(intent=Intent.WEATHER)
    today = on_date or date.today()

    if demo:
        alert = profile.weather_alert
        stage = profile.crop_stage
        plot_ctx = profile.plot_context()
    else:
        alert = _NON_DEMO_ALERT
        stage = profile.crop_stage
        plot_ctx = profile.plot_context()

    # A CHEMICAL / OOD query never gets the Tier-1 short-circuit.
    run_tier1 = triage.intent is Intent.WEATHER

    return AssembledContext(
        plot_context=plot_ctx,
        crop_stage=stage,
        weather_alert=alert,
        soil_drainage=profile.soil_drainage,
        has_image=profile.has_image,
        on_date=today,
        run_tier1=run_tier1,
    )
