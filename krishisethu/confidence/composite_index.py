"""Composite Confidence Index (white paper Sec IV-B, Eqs 5-7).

PURE, structural, and framework-free. Nothing in the LLM layer computes or
overrides the CCI — it is derived here from the three typed sub-scores and the
resolution, so it is always reproducible from the same inputs.

Eq 5: CCI = w1*R + w2*S_RAG + w3*P_vision
Eq 6: weights (image) = (0.50, 0.30, 0.20); (no image) w3 dropped and
      w_rule/w_rag renormalized to fill 1.0 -> (0.625, 0.375, 0.00), ratio 5/3 kept.
Eq 7: deliver iff CCI >= 0.85 (with the exact-resolution short-circuit).

The no-image redistribution keeps CCI on the same 0-1 scale as the with-image
case (sum of weights always 1.0), so the 0.85 gate threshold stays comparable.

Weights are read from Settings (CONFIDENCE_W_RULE, CONFIDENCE_W_RAG, CONFIDENCE_W_VISION)
so environment overrides take effect. The sum-to-one and 5/3 ratio invariants are
enforced by GateSettings validation.
"""
from __future__ import annotations

import math
from typing import Sequence

from krishisethu.config.settings import get_settings
from krishisethu.domain import ConfidenceScores, Resolution


#: Module-level constants initialized from Settings (cached at import time).
#: These reflect the validated weights and gate from Settings.
_settings = get_settings()
W_RULE = _settings.gate.w_rule
W_RAG = _settings.gate.w_rag
W_VISION = _settings.gate.w_vision
CONFIDENCE_GATE = _settings.gate.confidence_gate


def _default_weights() -> tuple[float, float, float]:
    """Read the canonical weights from Settings (validated sum=1.0, ratio 5/3)."""
    s = get_settings()
    return (s.gate.w_rule, s.gate.w_rag, s.gate.w_vision)


def _gate_threshold() -> float:
    """Read the confidence gate threshold from Settings."""
    return get_settings().gate.confidence_gate


def weights(has_image: bool) -> tuple[float, float, float]:
    """Weight triple for the CCI.

    With an image: ``(w_rule, w_rag, w_vision)`` from settings. Without one the
    vision weight is dropped and the rule/rag weights are renormalized to fill
    1.0 preserving the 5/3 ratio. Either way the triple sums to 1.0.
    """
    w_rule, w_rag, w_vision = _default_weights()
    if has_image:
        return (w_rule, w_rag, w_vision)
    total = w_rule + w_rag
    # Round so the no-image triple is exact (e.g. 0.625, 0.375, 0.0) — no float
    # noise that would trip an equality-asserted test.
    return (round(w_rule / total, 4), round(w_rag / total, 4), 0.0)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, clamped to [-1, 1].

    A zero vector is treated as orthogonal (0.0) so an empty/whitespace
    embedding never produces a spurious similarity.
    """
    if len(a) != len(b) or not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def rag_score(chunks: Sequence) -> float:
    """S_RAG = max cosine similarity over the top-k retrieved chunks.

    Accepts a sequence of objects exposing ``.score`` (e.g.
    :class:`~krishisethu.rag.store.StoredChunk`) or a sequence of raw floats.
    Empty top-k -> 0.0, per the spec.
    """
    scores = [float(getattr(c, "score", c)) for c in chunks]
    # Clamp to [0, 1]: a cosine can be slightly negative for near-orthogonal
    # vectors, but a confidence term must never drag the CCI below 0.
    return max(0.0, min(1.0, max(scores, default=0.0)))


def component_r(resolution: Resolution, partial_score: float | None = None) -> float:
    """Map a :class:`Resolution` to the R sub-score in [0, 1].

    EXACT -> 1.0; UNRESOLVED -> 0.0; PARTIAL -> ``partial_score`` (the
    normalized clause-match fraction, e.g. 2/5 = 0.4, or an explicit 0.6), which
    is clamped to [0, 1].
    """
    if resolution is Resolution.EXACT:
        r = 1.0
    elif resolution is Resolution.UNRESOLVED:
        r = 0.0
    else:
        r = 0.0 if partial_score is None else float(partial_score)
    return round(min(1.0, max(0.0, r)), 4)


def cci(scores: ConfidenceScores) -> float:
    """Eq 5: w1*R + w2*S_RAG + w3*P_vision using the image-appropriate weights."""
    wr, wrag, wv = weights(scores.has_image)
    return round(wr * scores.R + wrag * scores.S_RAG + wv * scores.P_vision, 4)


def pass_gate(cci_value: float, deterministic: bool = False, threshold: float | None = None) -> bool:
    """Eq 7: True iff the CCI clears the gate (or the exact short-circuit holds).

    ``deterministic`` is the caller's pre-computed flag: True when the answer is
    fully determined by an EXACT Tier-1 resolution with no probabilistic
    component. A deterministic answer is delivered without consulting the gate
    (but still passes the chemical + traceability gates), so a rule-certain
    answer is never spuriously escalated.

    ``threshold`` if provided overrides the settings value; otherwise the
    CONFIDENCE_GATE from Settings is used.
    """
    if deterministic:
        return True
    gate = threshold if threshold is not None else _gate_threshold()
    return cci_value >= gate
