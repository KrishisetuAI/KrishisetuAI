"""TIER 4 — LLM Explanation Layer (prompt 6).

PARAPHRASER ONLY. Quantized qwen3.5:4b (Ollama) reformulates output already
produced by Tiers 1-3; qwen3.5:2b fallback. The structured prompt contract
contains only Tier1-3 outputs; the output must be schema-valid JSON (no free
prose); an advisory that fails schema / traceability / chemistry / confidence
checks is rejected, never delivered.

Public API:
  prompt_contract  ExcerptSource, Fragment, PromptPackage, build_prompt, SYSTEM_RULES
  structured_output EvidenceRef, ActionItem, ConfidenceBreakdown, Advisory,
      parse_advisory, validate_advisory, to_domain_decision, RendererValidationError
  ollama_client    OllamaRenderer, RenderResult
  translation      translate_text, localize, LocalizedAdvisory
"""
from __future__ import annotations

from krishisethu.renderer.ollama_client import OllamaRenderer, RenderResult
from krishisethu.renderer.prompt_contract import (
    SYSTEM_RULES,
    ExcerptSource,
    Fragment,
    PromptPackage,
    build_prompt,
)
from krishisethu.renderer.structured_output import (
    ActionItem,
    Advisory,
    ConfidenceBreakdown,
    EvidenceRef,
    RendererValidationError,
    parse_advisory,
    to_domain_decision,
    validate_advisory,
)
from krishisethu.renderer.translation import (
    LANG_EN,
    LANG_HI,
    LocalizedAdvisory,
    localize,
    translate_text,
)

__all__ = [
    # prompt_contract
    "build_prompt",
    "PromptPackage",
    "ExcerptSource",
    "Fragment",
    "SYSTEM_RULES",
    # structured_output
    "EvidenceRef",
    "ActionItem",
    "ConfidenceBreakdown",
    "Advisory",
    "parse_advisory",
    "validate_advisory",
    "to_domain_decision",
    "RendererValidationError",
    # ollama_client
    "OllamaRenderer",
    "RenderResult",
    # translation
    "translate_text",
    "localize",
    "LocalizedAdvisory",
    "LANG_EN",
    "LANG_HI",
]
