"""Domain models + enumerations shared across tiers.

Enums anchor the Tier-2 canonical schema and, later, Tier-3 chunk metadata.
"""
from __future__ import annotations

from enum import Enum


class Crop(str, Enum):
    SUGARCANE = "sugarcane"
    WHEAT = "wheat"
    RICE = "rice"


class State(str, Enum):
    HARYANA = "haryana"
    UTTAR_PRADESH = "uttar_pradesh"


class Season(str, Enum):
    """Growing season OR crop-stage token (the paper mixes both in one axis)."""
    KHARIF = "kharif"
    RABI = "rabi"
    ZAID = "zaid"
    GRAND_GROWTH = "grand_growth"
    RATOON = "ratoon"
    MATURITY = "maturity"


class DecisionType(str, Enum):
    AGRONOMIC_PRACTICE = "agronomic_practice"
    CHEMICAL_PRESCRIPTION_CLASS = "chemical_prescription_class"
    IRRIGATION_RULE = "irrigation_rule"
    SPACING_NORM = "spacing_norm"
    MSP_RATE = "msp_rate"
    SCHEME_ELIGIBILITY = "scheme_eligibility"
    CROP_CALENDAR = "crop_calendar"


class Tier(str, Enum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


class Resolution(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class SourceType(str, Enum):
    """Document class for a Tier-3 long-tail source (plan Sec 5.3).

    Tier 2 holds stable compliance facts; Tier 3 indexes long-tail / dynamic /
    region-specific content. The pre-retrieval metadata filter constrains
    ``source_type IN allowed`` so an advisory can never surface a research-paper
    methodology note (and vice versa).
    """

    ICAR_POP_EXCERPT = "icar_pop_excerpt"
    ICAR_CIRCULAR = "icar_circular"
    KVK_ADVISORY = "kvk_advisory"
    KVK_QA = "kvk_qa"
    IMD_AGROMET = "imd_agromet"
    STATE_ADVISORY = "state_advisory"
    RESEARCH_PAPER = "research_paper"


#: Re-export the shared safety contracts from the submodule so
#: ``from krishisethu.domain import ConfidenceScores`` works as the single root.
from krishisethu.domain.models import (  # noqa: E402
    ActionItem,
    AdvisoryDecision,
    ConfidenceScores,
    EscalationTicket,
    Provenance,
    TicketStatus,
    ValidatorResult,
)
