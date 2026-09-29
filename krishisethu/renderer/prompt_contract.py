"""Structured prompt contract for the Tier-4 renderer (prompt 6).

``build_prompt`` assembles the ONLY content the LLM ever sees: (a) the resolved
Tier-1 action list, (b) the Tier-2 canonical excerpt, (c) the top-k Tier-3
fragments with their ``(source_id, chunk_index)`` markers, (d) the confidence
breakdown as FACTS the model must echo, and a system instruction that says:
rephrase only; do not add agronomic content; reference retrieved fragments by
their source marker; if a fact is not in the provided material, do not state it.
The returned :class:`PromptPackage` also carries the allowed source-marker set
and the expected confidence breakdown, so the caller can verify the model's
output against what it was actually given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from krishisethu.domain import Tier
from krishisethu.renderer.structured_output import ConfidenceBreakdown

#: The behavioural law of the paraphraser, phrased imperatively.
SYSTEM_RULES = """You are the explanation layer of KrishiSetu AI, an agricultural advisory system.
You are a PARAPHRASER ONLY. You reformulate guidance that has already been decided by deterministic,
evidence-backed tiers. You never originate a recommendation.

Rules (non-negotiable):
1. Rephrase only. Do not add any agronomic content, crop fact, chemical, rate, dose or practice that
   is not present in the MATERIAL below.
2. If a fact is not in the provided material, do not state it.
3. Reference retrieved fragments by their source marker (evidence[]). You may only cite a marker that
   appears in the MATERIAL. Never invent a source_id.
4. Never give a numeric dosage, application rate or specific chemical name. If the material mentions a
   chemical class, say the class only (input_class), never a product.
5. Output STRICT JSON matching the schema: {"actions":[{"text","input_class","consult_expert",
   "evidence":[{"source_id","tier"}]}],"confidence_breakdown":{"R","S_RAG","P_vision","has_image","cci"}}.
   No prose outside the JSON, no markdown fence.
6. Echo confidence_breakdown EXACTLY as given in the MATERIAL FACTS — you do not compute it."""


def _marker(obj) -> tuple[str, str]:
    """(tier-value, source marker) of a rule action or a domain ActionItem."""
    tier = getattr(obj, "tier", None)
    tier_value = tier.value if hasattr(tier, "value") else str(tier or "tier1")
    source = getattr(obj, "trace", None) or getattr(obj, "source_id", None) or ""
    return tier_value, str(source)


@dataclass(frozen=True)
class ExcerptSource:
    """A Tier-2 canonical excerpt with its doc id (never chunked/embedded)."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Fragment:
    """A Tier-3 top-k chunk with its provenance markers."""

    source_id: str  # the full chunk id, e.g. "kvk_sohna_sugarcane_ratoon_gap::0"
    chunk_index: int
    text: str


@dataclass(frozen=True)
class PromptPackage:
    """The assembled structured prompt + the ground truth the output is verified against."""

    system: str
    user: str
    allowed_markers: frozenset[tuple[Tier, str]] = frozenset()
    expected_breakdown: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    def to_messages(self) -> list[dict]:
        """Chat-Ollama message shape: [system, human]."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_prompt(
    actions: Sequence,
    canonical_excerpt: ExcerptSource | str | None = None,
    topk_fragments: Sequence[Fragment] = (),
    *,
    query: str = "",
    breakdown: ConfidenceBreakdown | None = None,
) -> PromptPackage:
    """Assemble the structured prompt from the resolved decision material.

    ``actions`` — the resolved Tier-1 action list (objects exposing ``text`` and
    a source marker: ``trace`` for :class:`~krishisethu.rules.RuleAction`,
    ``source_id`` for :class:`~krishisethu.domain.ActionItem`).
    ``canonical_excerpt`` — the verbatim Tier-2 excerpt (an :class:`ExcerptSource`
    or a plain string; plain strings contribute no marker).
    ``topk_fragments`` — the top-k Tier-3 fragments with their provenance.

    Keyword-only extras: ``query`` (the farmer's question, agronomic only) and
    ``breakdown`` (the true CCI components to echo).
    """
    material: list[str] = []

    allowed: set[tuple[Tier, str]] = set()

    # (a) resolved Tier-1 actions
    for i, action in enumerate(actions or ()):
        tier_value, source = _marker(action)
        text = getattr(action, "action", None) or getattr(action, "text", "")
        material.append(f"  - [{tier_value}:{source}] {text}")
        allowed.add((Tier(tier_value), source))

    # (b) Tier-2 canonical excerpt
    excerpt_text = ""
    if isinstance(canonical_excerpt, ExcerptSource):
        excerpt_text = canonical_excerpt.text
        # bracket form matches the other tiers so the model sees one uniform
        # "[tier:marker]" citation vocabulary.
        material.append(
            "\nTIER-2 CANONICAL EXCERPT:\n"
            + f"  [{Tier.TIER2.value}:{canonical_excerpt.doc_id}] {excerpt_text}"
        )
        allowed.add((Tier.TIER2, canonical_excerpt.doc_id))
    elif isinstance(canonical_excerpt, str) and canonical_excerpt.strip():
        excerpt_text = canonical_excerpt.strip()
        material.append("\nTIER-2 CANONICAL EXCERPT:\n" + excerpt_text)

    # (c) top-k Tier-3 fragments with provenance
    if topk_fragments:
        material.append("\nTIER-3 RETRIEVED FRAGMENTS (cite by source marker):")
        for frag in topk_fragments:
            material.append(f"  [{Tier.TIER3.value}:{frag.source_id}] (chunk {frag.chunk_index}) {frag.text}")
            allowed.add((Tier.TIER3, frag.source_id))

    expected = breakdown or ConfidenceBreakdown()
    facts: list[str] = [
        f"  - R (rule confidence) = {expected.R}",
        f"  - S_RAG (retrieval confidence) = {expected.S_RAG}",
        f"  - P_vision = {expected.P_vision}",
        f"  - has_image = {str(expected.has_image).lower()}",
        f"  - CCI = {expected.cci}  (echo these exact numbers)",
    ]

    user_parts = [
        "You are paraphrasing the decision for this advisory.",
    ]
    if query:
        user_parts.append(f"\nQUERY (context only): {query}")
    user_parts.append("\nMATERIAL — the only facts you may use:")
    user_parts.append("\n".join(material) if material else "  (no material provided)")
    user_parts.append("\n\nCONFIDENCE FACTS (echo exactly):\n" + "\n".join(facts))
    user_parts.append(
        "\n\nProduce the advisory JSON now: actions paraphrased from the MATERIAL above, "
        "each with evidence[] citing its source markers, and the echoed confidence_breakdown."
    )

    return PromptPackage(
        system=SYSTEM_RULES,
        user="\n".join(user_parts),
        allowed_markers=frozenset(allowed),
        expected_breakdown=expected,
    )
