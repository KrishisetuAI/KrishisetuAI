"""Hybrid renderer that routes to a pool of cloud (LiteLLM) endpoints with
fallback to local Ollama.

Multi-provider cloud key-rolling: the renderer dynamically inspects the
environment for up to 10 cloud keys (4x NVIDIA NIM, 4x Groq, 2x OpenRouter) and
builds an ordered pool of active endpoints. Each request is attempted against
every configured key in order; on 429/401/timeout/schema failure the next key is
tried, and if the pool is empty or fully exhausted the request is served by the
local OllamaRenderer (Qwen 4B/2B). Farmer PII is never forwarded to any cloud
endpoint (Safety Review Finding B3).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm
litellm.suppress_debug_info = True
from krishisethu.renderer.ollama_client import OllamaRenderer, RenderResult
from krishisethu.renderer.prompt_contract import PromptPackage
from krishisethu.renderer.structured_output import Advisory

logger = logging.getLogger(__name__)

# Provider routing configuration
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "openai/meta/llama-3.3-70b-instruct"
NVIDIA_PROVIDER = "openai"
NVIDIA_SLOTS = 4

GROQ_MODEL = "groq/llama-3.3-70b-versatile"
GROQ_SLOTS = 4

OPENROUTER_MODEL = "openrouter/qwen/qwen-2.5-coder-32b-instruct"
OPENROUTER_SLOTS = 2

DEFAULT_TEMPERATURE = 0.1
MAX_TOKENS = 768


@dataclass(frozen=True)
class CloudEndpoint:
    """A single configured cloud LLM endpoint with credentials."""
    provider: str
    model: str
    api_key: str
    slot: int
    api_base: Optional[str] = None
    custom_llm_provider: Optional[str] = None


def _build_cloud_pool() -> List[CloudEndpoint]:
    """Read environment and return ordered list of configured cloud endpoints.

    Order: NVIDIA (1..4) -> Groq (1..4) -> OpenRouter (1..2).
    Empty/missing keys are skipped.
    """
    pool: List[CloudEndpoint] = []

    # NVIDIA NIM
    for i in range(1, NVIDIA_SLOTS + 1):
        key = os.getenv(f"NVIDIA_API_KEY_{i}")
        if key:
            pool.append(CloudEndpoint(
                provider="nvidia",
                model=NVIDIA_MODEL,
                api_key=key,
                slot=i,
                api_base=NVIDIA_API_BASE,
                custom_llm_provider=NVIDIA_PROVIDER,
            ))

    # Groq Cloud
    for i in range(1, GROQ_SLOTS + 1):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            pool.append(CloudEndpoint(
                provider="groq",
                model=GROQ_MODEL,
                api_key=key,
                slot=i,
            ))

    # OpenRouter
    for i in range(1, OPENROUTER_SLOTS + 1):
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if key:
            pool.append(CloudEndpoint(
                provider="openrouter",
                model=OPENROUTER_MODEL,
                api_key=key,
                slot=i,
            ))

    return pool


# Legacy compatibility: GROQ-only key getter (used when cloud_models is explicit)
def _get_api_keys() -> List[str]:
    """Return flattened list of all configured cloud API keys (legacy helper).

    Order matches _build_cloud_pool: NVIDIA 1..4, Groq 1..4, OpenRouter 1..2.
    """
    keys: List[str] = []
    for i in range(1, NVIDIA_SLOTS + 1):
        key = os.getenv(f"NVIDIA_API_KEY_{i}")
        if key:
            keys.append(key)
    for i in range(1, GROQ_SLOTS + 1):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
    for i in range(1, OPENROUTER_SLOTS + 1):
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if key:
            keys.append(key)
    return keys


# Backwards-compatible constant for any external reference
CLOUD_MODELS = [NVIDIA_MODEL, GROQ_MODEL, OPENROUTER_MODEL]


class HybridRenderer:
    """Renderer that tries cloud models first, falls back to Ollama."""

    def __init__(
        self,
        cloud_models: Optional[List[str]] = None,
        fallback_renderer: Optional[OllamaRenderer] = None,
        timeout_seconds: float = 10.0,
    ):
        self.fallback = fallback_renderer or OllamaRenderer()
        self.timeout = timeout_seconds

        if cloud_models is not None:
            # Legacy explicit model list: pair each model with all available keys
            keys = _get_api_keys()
            self._endpoints: List[CloudEndpoint] = [
                CloudEndpoint(provider="legacy", model=m, api_key=k, slot=0)
                for m in cloud_models
                for k in keys
            ]
            if not keys:
                logger.warning("No cloud API keys set; cloud routing will fail, using fallback only.")
        else:
            # New multi-provider pool built from environment
            self._endpoints = _build_cloud_pool()
            if not self._endpoints:
                logger.warning("No cloud API keys configured for HybridRenderer; using local Ollama fallback.")

    def render(self, package: PromptPackage) -> RenderResult:
        """Attempt cloud endpoints in order, fallback to Ollama on any failure."""
        # Safety Review Finding B3: PII guard before ANY cloud call
        self.fallback._assert_no_pii(package)

        if not self._endpoints:
            logger.debug("No cloud endpoints configured, using fallback directly.")
            return self.fallback.render(package)

        # Try each cloud endpoint in order
        for endpoint in self._endpoints:
            result = self._try_cloud_endpoint(endpoint, package)
            if result is not None:
                return result
            # else fall through to next endpoint

        # All cloud attempts failed – fallback to Ollama
        logger.warning("All configured cloud endpoints failed or exhausted; falling back to local Ollama.")
        return self.fallback.render(package)

    def _try_cloud_endpoint(self, endpoint: CloudEndpoint, package: PromptPackage) -> Optional[RenderResult]:
        """Try a single cloud endpoint; return RenderResult on success, None on failure."""
        # Defensive PII guard before any cloud call (also guarded in render())
        self.fallback._assert_no_pii(package)

        messages = [
            {"role": "system", "content": package.system},
            {"role": "user", "content": package.user},
        ]

        kwargs = {
            "model": endpoint.model,
            "messages": messages,
            "api_key": endpoint.api_key,
            "timeout": self.timeout,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        if endpoint.api_base:
            kwargs["api_base"] = endpoint.api_base
        if endpoint.custom_llm_provider:
            kwargs["custom_llm_provider"] = endpoint.custom_llm_provider

        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if not content:
                logger.debug("Empty response from %s slot %s; trying next key.", endpoint.provider, endpoint.slot)
                return None
            advisory = self._parse_response(content, package)
            if advisory:
                return RenderResult(
                    advisory=advisory,
                    model_used=endpoint.model,
                    attempts=1,
                    error=None,
                )
            logger.debug("Schema/validation failure from %s slot %s; trying next key.", endpoint.provider, endpoint.slot)
            return None
        except Exception as e:
            logger.debug("Cloud call failed on %s slot %s (%s): %s; trying next key.", endpoint.provider, endpoint.slot, type(e).__name__, e)
            return None

    def _parse_response(self, content: str, package: PromptPackage) -> Optional[Advisory]:
        """Parse JSON content into Advisory, validating against expected markers."""
        try:
            data = json.loads(content)
            from krishisethu.renderer.structured_output import parse_advisory, validate_advisory
            advisory = parse_advisory(content)
            problems = validate_advisory(
                advisory,
                allowed_markers=package.allowed_markers,
                expected_breakdown=package.expected_breakdown,
            )
            if not problems:
                return advisory
            logger.debug(f"Validation problems: {problems}")
        except Exception as e:
            logger.debug(f"Failed to parse cloud response: {e}")
        return None