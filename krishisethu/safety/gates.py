"""Safety Validator — the three gates, applied in order (plan Sec 8).

Every candidate response passes through this validator before it can leave the
system:

1. **Confidence gate** — if not deterministic and CCI < 0.85, withhold and raise
   a KVK escalation ticket; the farmer channel gets only the neutral message.
2. **Chemical-prescription gate** — if output pairs an active ingredient with a
   numeric dose (or a dose request), strip the dosage, substitute a
   region-verified input-class recommendation, add a consult-expert line, and
   escalate regardless of CCI. Never a specific dosage.
3. **Source-traceability gate** — every claim must cite a source in the
   traceable set (Tier-1 rule id | Tier-2 doc id | Tier-3 chunk id). Strip any
   untraceable claim; if all are stripped, escalate rather than deliver an empty
   advisory.

An EXACT Tier-1 resolution short-circuits gate 1 (the answer is rule-certain and
must never be spuriously escalated) but still runs through gates 2 and 3.
"""
from __future__ import annotations

from krishisethu.confidence.composite_index import CONFIDENCE_GATE, cci
from krishisethu.domain import (
    ActionItem,
    AdvisoryDecision,
    ConfidenceScores,
    Resolution,
    ValidatorResult,
)
from krishisethu.safety.chemical_registry import (
    find_chemicals,
    has_dosage_pair,
    strip_dosages,
)

#: The neutral, localized hand-off message the farmer channel receives on
#: escalation. The officer gets the full ticket; the farmer gets only this.
NEUTRAL_ESCAPE_EN = (
    "This query has been forwarded to a KVK agricultural expert for review. "
    "You will receive guidance shortly."
)
NEUTRAL_ESCAPE_HI = (
    "आपका प्रश्न कृषि विज्ञान केंद्र (KVK) विशेषज्ञ को भेज दिया गया है। "
    "शीघ्र ही आपको मार्गदर्शन प्राप्त होगा।"
)


class SafetyValidator:
    """Runs the three gates in order and returns a :class:`ValidatorResult`."""

    def __init__(self, threshold: float = CONFIDENCE_GATE):
        self.threshold = threshold

    def validate(
        self,
        scores: ConfidenceScores,
        resolution: Resolution,
        deterministic: bool,
        raw: str,
        decision: AdvisoryDecision,
    ) -> ValidatorResult:
        """Validate a candidate advisory.

        ``raw`` is the candidate rendered text (the stubbed/LLM output); the
        advisory's claims live in ``decision.actions`` and its allowed source
        markers in ``decision.provenance``.
        """
        warnings: list[str] = []

        # ---- Gate 1: confidence ------------------------------------------
        if not deterministic:
            value = cci(scores)
            if value < self.threshold:
                warnings.append(f"confidence_gate: cci={value:.4f} < {self.threshold:.2f}")
                return ValidatorResult(
                    accepted=False, modified=False, text="", escalate=True, warnings=tuple(warnings)
                )

        # ---- Gate 2: chemical-prescription -------------------------------
        # Evaluate the pair over the JOINED rendered output (raw + every action),
        # not per item — a dosage and an ingredient split across two actions are
        # still one combined advisory and must trip the gate.
        if has_dosage_pair("\n".join(self._candidate_texts(raw, decision))):
            sanitized = self._sanitize_chemical(raw, decision)
            warnings.append(
                "chemical_gate: active ingredient paired with a dosage; dosage stripped, "
                "input-class substituted, consult-expert line added"
            )
            return ValidatorResult(
                accepted=False, modified=True, text=sanitized, escalate=True, warnings=tuple(warnings)
            )

        # ---- Gate 3: source-traceability ---------------------------------
        kept = [a for a in decision.actions if self._is_traceable(a, decision)]
        removed = len(decision.actions) - len(kept)
        if not kept:
            warnings.append("traceability_gate: every claim's source marker is absent; escalated")
            return ValidatorResult(
                accepted=False, modified=True, text="", escalate=True, warnings=tuple(warnings)
            )
        text = "\n".join(a.text for a in kept).strip() or (raw or "").strip()
        return ValidatorResult(
            accepted=True, modified=(removed > 0), text=text, escalate=False, warnings=tuple(warnings)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_texts(raw: str, decision: AdvisoryDecision) -> list[str]:
        """The raw output plus every action's text AND input-class, all scanned.

        ``input_class`` is included because a domain :class:`ActionItem` may be
        constructed outside the renderer (whose schema already constrains the
        field to registry class labels); the gate must not trust the field's
        provenance.
        """
        texts = [raw or ""]
        texts += [a.text for a in decision.actions]
        texts += [a.input_class for a in decision.actions if a.input_class]
        return texts

    @staticmethod
    def _is_traceable(action: ActionItem, decision: AdvisoryDecision) -> bool:
        """True when the action's ``(tier, source_id)`` is in the allowed set."""
        return (action.tier, action.source_id) in {(p.tier, p.source_id) for p in decision.provenance}

    @staticmethod
    def _sanitize_chemical(raw: str, decision: AdvisoryDecision) -> str:
        """Strip every dosage and replace each ingredient with its input class.

        The result is the sanitized advisory recorded on the escalation ticket:
        no numeric dosage and no specific active ingredient survives, only
        input-class phrasing plus a consult-expert line.
        """
        lines = [a.text for a in decision.actions] if decision.actions else ([raw] if raw else [])
        out: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            cleaned = strip_dosages(line)
            chemicals = find_chemicals(cleaned)
            if chemicals:
                out.append(f"Use a region-verified {chemicals[0].input_class} class input.")
            else:
                out.append(cleaned)
        out.append("Consult the KVK agronomist before any chemical application.")
        return "\n".join(out)
