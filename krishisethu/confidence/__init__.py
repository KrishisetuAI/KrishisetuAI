"""Composite Confidence Index.

CCI = w1*R + w2*S_RAG + w3*P_vision (0.50/0.30/0.20 with image; 0.625/0.375/0.00
without), gate 0.85. Pure functions; nothing in the LLM layer computes or
overrides CCI.

Public API:
  cci, weights, cosine, rag_score, component_r, pass_gate, W_RULE, W_RAG,
  W_VISION, CONFIDENCE_GATE
"""
from __future__ import annotations

from krishisethu.confidence.composite_index import (
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

__all__ = [
    "cci",
    "weights",
    "cosine",
    "rag_score",
    "component_r",
    "pass_gate",
    "W_RULE",
    "W_RAG",
    "W_VISION",
    "CONFIDENCE_GATE",
]
