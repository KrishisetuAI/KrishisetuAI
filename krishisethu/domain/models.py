"""Domain contracts shared by the confidence index, safety validator and escalation.

Each is a typed Pydantic model (frozen where the shape is immutable) so the
safety backbone is validated and serializable end to end — the CCI is computed by
pure functions over :class:`ConfidenceScores`, the gates return a
:class:`ValidatorResult`, and escalation persists :class:`EscalationTicket`.

The enums (:class:`Tier`, :class:`Resolution`) live in the package root; the
models here import them, so ``krishisethu.domain`` stays the single import root.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from krishisethu.domain import Resolution, Tier


class TicketStatus(str, Enum):
    """Lifecycle of a KVK escalation ticket."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


#: Closure of the unit interval used by every confidence sub-score.
def _clamp01(v: float) -> float:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"confidence component must be in [0, 1], got {v}")
    return float(v)


class ConfidenceScores(BaseModel):
    """The three CCI sub-scores (Eq 5-7) plus the image flag that selects weights.

    ``R`` (Tier-1 rule-match), ``S_RAG`` (max cosine over top-k), ``P_vision``
    (vision softmax of the top class; 0.0 when there is no image). ``has_image``
    decides whether the 0.20 vision weight is counted or redistributed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    R: float = 0.0
    S_RAG: float = 0.0
    P_vision: float = 0.0
    has_image: bool = False

    @field_validator("R", "S_RAG", "P_vision")
    @classmethod
    def _subscores_in_unit(cls, v: float) -> float:
        return _clamp01(v)

    def summary(self) -> dict:
        """Plain-dict view for tickets / logs."""
        return {
            "R": self.R,
            "S_RAG": self.S_RAG,
            "P_vision": self.P_vision,
            "has_image": self.has_image,
        }


class Provenance(BaseModel):
    """One allowed source marker: which tier plus its source id.

    Gate 3 builds the traceable set from these; a claim whose ``(tier,
    source_id)`` is not in the set is stripped. ``claim_span`` optionally records
    the char span in the rendered advisory the marker covers (audit aid).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Tier
    source_id: str
    claim_span: tuple[int, int] | None = None

    @field_validator("source_id")
    @classmethod
    def _source_id_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("source_id must be a non-empty marker")
        return str(v).strip()


class ActionItem(BaseModel):
    """One recommendation line with its traceable source marker.

    ``source_id`` is the marker the traceability gate matches against
    :class:`Provenance` — a Tier-1 rule-id/trace code, a Tier-2 doc id, or a
    Tier-3 chunk id. ``input_class`` and ``consult_expert`` flag the
    chemical-prescription gate's substitution. There is deliberately NO numeric
    dosage field here — dosage is impossible structurally.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    source_id: str
    tier: Tier
    clause: str | None = None
    input_class: str | None = None
    consult_expert: bool = False

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("action text must be non-empty")
        return str(v).strip()

    @field_validator("source_id")
    @classmethod
    def _source_id_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("source_id must be a non-empty marker")
        return str(v).strip()


class AdvisoryDecision(BaseModel):
    """The resolved decision the safety gate validates before delivery.

    ``actions`` are the claims to surface (each already source-marked);
    ``provenance`` is the allowlist of source markers a claim may cite.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionItem, ...] = ()
    resolution: Resolution = Resolution.UNRESOLVED
    provenance: tuple[Provenance, ...] = ()


class ValidatorResult(BaseModel):
    """Outcome of the three-gate Safety Validator.

    ``text`` is the advisory content after all gates ran (empty when nothing is
    deliverable). ``escalate`` means the system must raise a KVK ticket and send
    only the neutral hand-off message to the farmer channel, never ``text``.
    ``accepted`` is True only when the advisory may be delivered verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool = False
    modified: bool = False
    text: str = ""
    escalate: bool = False
    warnings: tuple[str, ...] = ()


class EscalationTicket(BaseModel):
    """A KVK escalation ticket (plan Sec 8).

    Carries the full decision context for the human-in-the-loop officer; the
    farmer channel receives only the neutral message. ``dedup_key`` is the
    sha256(query + plot_id + date_window) used to make replay idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    query: str
    context: dict = {}
    cci: float | None = None
    retrieved_chunks: tuple[dict, ...] = ()
    validator_warnings: tuple[str, ...] = ()
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime
    resolved_at: datetime | None = None
    dedup_key: str = ""
