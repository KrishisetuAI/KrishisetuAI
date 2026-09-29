"""Structured output models for the Tier-4 renderer (prompt 6).

The renderer is a PARAPHRASER ONLY. These Pydantic models are the strict JSON
contract the Ollama model must satisfy; anything that fails validation is
rejected (never returned as raw prose). ``extra="forbid"`` everywhere means a
field the schema does not define — including any numeric-dosage field — fails
validation structurally: dosage is impossible in the schema.

This is the *LLM-facing* contract. The safety-core contract lives in
:mod:`krishisethu.domain` (:class:`~krishisethu.domain.ActionItem` carries one
flat ``source_id``/``tier`` per claim). Here the model may cite MULTIPLE
:class:`EvidenceRef` per action; :func:`to_domain_decision` maps the validated
advisory onto the domain :class:`~krishisethu.domain.AdvisoryDecision` so the
Prompt-5 Safety Validator remains the final gate.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, field_validator

from krishisethu.domain import (
    ActionItem as DomainActionItem,
)
from krishisethu.domain import (
    AdvisoryDecision,
    Provenance,
    Resolution,
    Tier,
)

#: No dosage or active-ingredient name may survive in rendered text. The registry
#: is the single source of truth for what counts as one (Tier-2 backed).
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class RendererValidationError(ValueError):
    """Raised when LLM output is not schema-valid or fails a post-parse check."""


class EvidenceRef(BaseModel):
    """One source marker the rendered claim cites (Tier-1 rule id | Tier-2 doc id | Tier-3 chunk id)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    tier: Tier

    @field_validator("source_id")
    @classmethod
    def _source_id_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("source_id must be a non-empty marker")
        return str(v).strip()


class ActionItem(BaseModel):
    """One rendered recommendation line with its evidence list.

    There is deliberately NO numeric dosage field and no ``input_class``-bypass:
    ``input_class`` is the *input class* (e.g. "organophosphate insecticide"),
    never a product name or a dose. ``consult_expert`` flags a consult-KVK line.
    ``evidence`` is REQUIRED to be non-empty — a claim without a source marker is
    not a rendered claim at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ``evidence`` is REQUIRED (no default) so a claim that omits it — or sends
    # an empty list — fails the schema: an untraceable claim is not a claim.
    text: str
    input_class: str | None = None
    consult_expert: bool = False
    evidence: tuple[EvidenceRef, ...]

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("action text must be non-empty")
        return str(v).strip()

    @field_validator("input_class")
    @classmethod
    def _input_class_is_a_class(cls, v: str | None) -> str | None:
        """Constrain ``input_class`` to a registry class label — never a product
        name or a dose. The registry is the single source of truth for what an
        input class is; a free-text value (e.g. "chlorpyrifos 2.5 ml/L") fails
        the schema, closing the free-text bypass."""
        if v is None:
            return None
        from krishisethu.safety.chemical_registry import CHEMICALS

        allowed = {c.input_class for c in CHEMICALS}
        if v not in allowed:
            raise ValueError(
                f"input_class must be a recognized input class from the registry, got {v!r}"
            )
        return v

    @field_validator("evidence")
    @classmethod
    def _evidence_required(cls, v: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        if not v:
            raise ValueError("every rendered action must carry at least one evidence ref")
        return v


class ConfidenceBreakdown(BaseModel):
    """The CCI components the advisory reports.

    The LLM never computes these — the system injects the true values into the
    prompt as FACTS and the model echoes them; :func:`validate_advisory`
    verifies the echo matches the ground truth within tolerance, so the LLM
    cannot override the confidence index.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    R: float = 0.0
    S_RAG: float = 0.0
    P_vision: float = 0.0
    has_image: bool = False
    cci: float = 0.0

    @field_validator("R", "S_RAG", "P_vision", "cci")
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"confidence component must be in [0, 1], got {v}")
        return float(v)


class Advisory(BaseModel):
    """The full renderer output: an action list + the echoed confidence breakdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionItem, ...] = ()
    confidence_breakdown: ConfidenceBreakdown = ConfidenceBreakdown()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_advisory(raw: str) -> Advisory:
    """Parse LLM JSON text into an :class:`Advisory`.

    Tolerates a ```json`` fence around the payload. Raises
    :class:`RendererValidationError` on any failure — the caller never receives
    raw prose.
    """
    text = _FENCE_RE.sub("", str(raw or "")).strip()
    if not text:
        raise RendererValidationError("empty model output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RendererValidationError(f"output is not valid JSON: {e.msg}") from e
    if not isinstance(data, dict):
        raise RendererValidationError("output JSON must be an object")
    try:
        return Advisory.model_validate(data)
    except Exception as e:  # pydantic.ValidationError — surface as renderer error
        raise RendererValidationError(f"output fails the Advisory schema: {e}") from e


# ---------------------------------------------------------------------------
# post-parse verification (traceability, chemistry, confidence echo)
# ---------------------------------------------------------------------------
def _scan_forbidden_chemistry(advisory: Advisory) -> list[str]:
    """Term-level scan: an active-ingredient name or numeric dosage in any text.

    A paraphraser of Tier 1-3 material has no legitimate reason to emit either —
    Tier-1 rules and the seed corpus are qualitative. The registry is the single
    source of truth for what counts.
    """
    from krishisethu.safety.chemical_registry import DOSAGE_RE, find_chemicals

    problems: list[str] = []
    for i, action in enumerate(advisory.actions):
        if find_chemicals(action.text):
            problems.append(f"action[{i}] names an active ingredient")
        if DOSAGE_RE.search(action.text):
            problems.append(f"action[{i}] carries a numeric dosage")
    return problems


def validate_advisory(
    advisory: Advisory,
    *,
    allowed_markers: set[tuple[Tier, str]] | None = None,
    expected_breakdown: ConfidenceBreakdown | None = None,
    tolerance: float = 1e-3,
) -> list[str]:
    """Post-parse checks. Returns an empty list when the advisory is acceptable.

    Checks, in order:
      1. every action's evidence refs are within the allowed source markers;
      2. no action text names an active ingredient or carries a numeric dosage;
      3. the echoed confidence breakdown matches the injected ground truth.
    """
    problems: list[str] = []

    if allowed_markers is not None:
        for i, action in enumerate(advisory.actions):
            for ref in action.evidence:
                if (ref.tier, ref.source_id) not in allowed_markers:
                    problems.append(
                        f"action[{i}] cites out-of-set source {ref.tier.value}:{ref.source_id}"
                    )

    problems += _scan_forbidden_chemistry(advisory)

    if expected_breakdown is not None:
        b = advisory.confidence_breakdown
        for field in ("R", "S_RAG", "P_vision", "cci"):
            if abs(getattr(b, field) - getattr(expected_breakdown, field)) > tolerance:
                problems.append(
                    f"confidence_breakdown.{field} echo {getattr(b, field)} != "
                    f"ground truth {getattr(expected_breakdown, field)}"
                )
        if b.has_image != expected_breakdown.has_image:
            problems.append("confidence_breakdown.has_image echo differs from ground truth")

    return problems


# ---------------------------------------------------------------------------
# mapping onto the safety-core domain contract
# ---------------------------------------------------------------------------
def to_domain_decision(
    advisory: Advisory,
    *,
    resolution: Resolution = Resolution.UNRESOLVED,
    allowed_markers: set[tuple[Tier, str]] | None = None,
) -> AdvisoryDecision:
    """Map a *validated* renderer advisory onto the domain decision for the gates.

    Every rendered action becomes a domain :class:`ActionItem` whose marker is
    its first evidence ref (post-parse verification has already ensured every
    evidence ref is allowed, so the flattened marker is traceable). The domain
    decision then runs through the Prompt-5 SafetyValidator unchanged.
    """
    actions: list[DomainActionItem] = []
    provenance: list[Provenance] = []
    for action in advisory.actions:
        if not action.evidence:
            raise RendererValidationError("rendered action carries no evidence ref")
        primary = action.evidence[0]
        if allowed_markers is not None and (primary.tier, primary.source_id) not in allowed_markers:
            raise RendererValidationError(
                f"action {primary.source_id!r} is not in the allowed marker set"
            )
        actions.append(
            DomainActionItem(
                text=action.text,
                source_id=primary.source_id,
                tier=primary.tier,
                input_class=action.input_class,
                consult_expert=action.consult_expert,
            )
        )
        provenance.append(Provenance(tier=primary.tier, source_id=primary.source_id))
    return AdvisoryDecision(actions=tuple(actions), resolution=resolution, provenance=tuple(provenance))
