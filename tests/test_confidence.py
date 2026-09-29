"""Composite Confidence Index tests (prompt 5, plan Sec 8 cases 1-6).

Covers: weights (with/without image, renormalization), R mapping EXACT/PARTIAL/
UNRESOLVED, S_RAG = max cosine (and 0.0 on empty), CCI arithmetic, the 0.85 gate
boundary, and the exact-resolution short-circuit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from krishisethu.confidence import (
    CONFIDENCE_GATE,
    W_RAG,
    W_RULE,
    W_VISION,
    cci,
    component_r,
    cosine,
    pass_gate,
    rag_score,
    weights,
)
from krishisethu.domain import ConfidenceScores, Resolution


@dataclass(frozen=True)
class _Chunk:
    """Minimal stand-in for :class:`~krishisethu.rag.store.StoredChunk` (has .score)."""

    score: float


# --------------------------------------------------------------------------
# Case 1: weights
# --------------------------------------------------------------------------
def test_weights_with_image():
    w = weights(True)
    assert w == (0.50, 0.30, 0.20)
    assert sum(w) == pytest.approx(1.0)


def test_weights_no_image_renormalized():
    # w3 dropped; w_rule/w_rag renormalized to fill 1.0 keeping the 5/3 ratio.
    w = weights(False)
    assert w == (0.625, 0.375, 0.0)
    assert sum(w) == pytest.approx(1.0)
    # ratio preserved: 0.625 / 0.375 == 5 / 3
    assert w[0] / w[1] == pytest.approx(5 / 3)


def test_weight_ptrs_respect_spec():
    assert (W_RULE, W_RAG, W_VISION) == (0.50, 0.30, 0.20)


# --------------------------------------------------------------------------
# Case 2: R maps Resolution -> [0, 1]
# --------------------------------------------------------------------------
def test_component_r_exact_partial_unresolved():
    assert component_r(Resolution.EXACT) == 1.0
    assert component_r(Resolution.PARTIAL, partial_score=0.6) == 0.6
    assert component_r(Resolution.UNRESOLVED) == 0.0


def test_component_r_clamped_to_unit_interval():
    assert component_r(Resolution.PARTIAL, partial_score=1.3) == 1.0
    assert component_r(Resolution.PARTIAL, partial_score=-0.2) == 0.0
    assert component_r(Resolution.PARTIAL, partial_score=0.4) == 0.4


def test_component_r_partial_without_score_is_zero():
    assert component_r(Resolution.PARTIAL) == 0.0


# --------------------------------------------------------------------------
# Case 3: S_RAG = max cosine over top-k; empty -> 0.0
# --------------------------------------------------------------------------
def test_rag_score_max_over_chunks():
    assert rag_score([_Chunk(0.1), _Chunk(0.72), _Chunk(0.5)]) == pytest.approx(0.72)


def test_rag_score_empty_is_zero():
    assert rag_score([]) == 0.0


def test_rag_score_accepts_raw_floats():
    assert rag_score([0.3, 0.9, 0.4]) == pytest.approx(0.9)


def test_cosine_basic_and_bounds():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert cosine(v1, v1) == pytest.approx(1.0)
    assert cosine(v1, v2) == pytest.approx(0.0)
    v3 = [2.0, 0.0]
    assert cosine(v1, v3) == pytest.approx(1.0)  # parallel, normalized away
    assert cosine([1.0, 0.0], [0.0]) == 0.0  # dim mismatch
    assert cosine([], []) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector


# --------------------------------------------------------------------------
# Case 4: CCI arithmetic spot checks
# --------------------------------------------------------------------------
def test_cci_no_image_arithmetic():
    # no-image: 0.625*R + 0.375*S_RAG
    s = ConfidenceScores(R=0.5, S_RAG=0.5, P_vision=0.0, has_image=False)
    assert cci(s) == pytest.approx(0.625 * 0.5 + 0.375 * 0.5)


def test_cci_with_image_arithmetic():
    s = ConfidenceScores(R=0.8, S_RAG=0.6, P_vision=0.4, has_image=True)
    assert cci(s) == pytest.approx(0.50 * 0.8 + 0.30 * 0.6 + 0.20 * 0.4)


def test_cci_defaults_to_zero():
    assert cci(ConfidenceScores()) == 0.0


def test_cci_rejects_out_of_range_component():
    with pytest.raises(ValueError):
        ConfidenceScores(R=1.2)
    with pytest.raises(ValueError):
        ConfidenceScores(S_RAG=-0.1)


# --------------------------------------------------------------------------
# Case 5: gate boundary 0.85 / 0.845
# --------------------------------------------------------------------------
def test_gate_boundary_delivers_at_0_85():
    s = ConfidenceScores(R=0.7, S_RAG=1.0, P_vision=1.0, has_image=True)
    assert cci(s) == pytest.approx(0.85)
    assert pass_gate(cci(s)) is True


def test_gate_boundary_escalates_below_0_85():
    s = ConfidenceScores(R=0.69, S_RAG=1.0, P_vision=1.0, has_image=True)
    assert cci(s) == pytest.approx(0.845)
    assert pass_gate(cci(s)) is False


def test_gate_threshold_constant():
    assert CONFIDENCE_GATE == 0.85


# --------------------------------------------------------------------------
# Case 6: exact + vision strict-formula edge -> deterministic short-circuit delivers
# --------------------------------------------------------------------------
def test_exact_vision_strict_formula_would_fail_but_short_circuit_delivers():
    s = ConfidenceScores(R=1.0, S_RAG=0.0, P_vision=1.0, has_image=True)
    assert cci(s) == pytest.approx(0.70)  # strict formula < 0.85
    assert pass_gate(cci(s)) is False  # formula alone would escalate
    assert pass_gate(cci(s), deterministic=True) is True  # but rule-certain -> deliver


def test_deterministic_short_circuit_ignores_low_cci():
    assert pass_gate(0.1, deterministic=True) is True
    assert pass_gate(0.1, deterministic=False) is False
