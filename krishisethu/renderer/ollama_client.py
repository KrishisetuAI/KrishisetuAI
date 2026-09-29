"""Thin Ollama wrapper for the Tier-4 paraphraser (prompt 6).

A bounded parse-validate-retry loop over ``ChatOllama(format="json")``:

  * the primary model (qwen3.5:4b) is tried first; on a transport error
    (model missing / slow / unreachable) OR a validation failure we fall back to
    qwen3.5:2b, and each model gets at most ``max_retries`` retries before the
    response is REJECTED — the caller never receives raw prose.
  * Never passes farmer PII: the prompt is built only from typed agronomic
    context (:class:`~krishisethu.renderer.prompt_contract.PromptPackage`), and a
    defensive PII scan rejects obviously personal payloads before the call.

``ChatOllama`` is imported at module scope so tests can monkeypatch
``krishisethu.renderer.ollama_client.ChatOllama`` with a fake (hermetic CI).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_ollama import ChatOllama  # noqa: F401  (monkeypatch seam for tests)

from krishisethu.renderer.prompt_contract import PromptPackage
from krishisethu.renderer.structured_output import (
    Advisory,
    RendererValidationError,
    parse_advisory,
    validate_advisory,
)

#: Default paraphraser hyper-parameters. Near-deterministic by design: a
#: paraphraser must not drift between runs of the same decision.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_NUM_PREDICT = 768
#: Model tags from settings (qwen3.5 family).
PRIMARY_MODEL_ENV = "qwen3.5:4b"
FALLBACK_MODEL_ENV = "qwen3.5:2b"

#: Phone / Aadhaar-ish tokens the renderer must never see (defensive).
#: Catches 10-digit runs and spaced/grouped forms (987 654 3210, 9876-5432-10,
#: 3-4-4 Aadhaar-style grouping, 5-5 grouping) so a personal identifier is not
#: forwarded to the model endpoint regardless of formatting.
_PII_RE = re.compile(
    r"(?<!\d)(?:\d[\s-]?){9}\d(?!\d)|"      # 10 digits, optional single sep between
    r"(?<!\d)(?:\d[\s-]?){11}\d(?!\d)",     # 12 digits (Aadhaar), optional sep
    re.IGNORECASE,
)

#: Transport errors that mean "this model is unavailable / too slow" -> fallback.
_TRANSPORT_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_transport_error(exc: BaseException) -> bool:
    """True when the exception means the model backend is unreachable/missing."""
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in ("connection", "timeout", "responserror", "notfound", "unavailable"))


@dataclass(frozen=True)
class RenderResult:
    """Outcome of one render call. Either ``advisory`` or an ``error`` (never both)."""

    advisory: Advisory | None = None
    model_used: str | None = None
    attempts: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.advisory is not None


class OllamaRenderer:
    """The Tier-4 paraphraser: structured prompt in -> schema-valid Advisory out."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
        max_retries: int = 2,
        timeout_seconds: float = 60.0,
    ):
        from krishisethu.config.settings import get_settings

        s = get_settings()
        self.base_url = base_url or s.ollama.embed_base()
        self.model = model or s.ollama.llm_model
        self.fallback_model = fallback_model or s.ollama.llm_fallback
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_retries = max_retries  # retries AFTER the first attempt per model
        self.timeout_seconds = timeout_seconds
        self._llms: dict[str, object] = {}

    # -- model handle ---------------------------------------------------
    def _llm_for(self, model: str):
        """Lazily build (and cache) a ChatOllama for ``model``."""
        llm = self._llms.get(model)
        if llm is None:
            llm = ChatOllama(
                model=model,
                base_url=self.base_url,
                format="json",
                temperature=self.temperature,
                num_predict=self.num_predict,
                # wall-clock bound on each model call so a hung server triggers
                # the transport fallback instead of hanging render() forever.
                client_kwargs={"timeout": self.timeout_seconds},
            )
            self._llms[model] = llm
        return llm

    def _invoke(self, model: str, package: PromptPackage) -> str:
        """One raw model call; returns the content string or raises."""
        llm = self._llm_for(model)
        message = llm.invoke(package.to_messages())
        return str(getattr(message, "content", message) or "")

    # -- PII guard ------------------------------------------------------
    @staticmethod
    def _assert_no_pii(package: PromptPackage) -> None:
        """The renderer is passed only agronomic context — never farmer PII.

        Defensive scan over the assembled user prompt for phone / Aadhaar-like
        tokens; refuse loudly rather than forward a personal identifier to the
        model endpoint.
        """
        if _PII_RE.search(package.user):
            raise RendererValidationError(
                "refusing to render: the prompt contains a phone/Aadhaar-like token (PII)"
            )

    # -- public API -----------------------------------------------------
    def render(self, package: PromptPackage) -> RenderResult:
        """Render a structured prompt into a schema-valid :class:`Advisory`.

        Bounded: the primary model is tried, then the fallback, each with at most
        ``max_retries`` retries (total attempts <= 2*(max_retries+1)). A response
        is accepted only when it parses to the schema AND passes every post-parse
        check (evidence within the allowed markers, no chemical actives/doses,
        confidence echo matches ground truth). On failure the result carries
        ``error`` — never raw prose.
        """
        self._assert_no_pii(package)

        order = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            order.append(self.fallback_model)

        attempts = 0
        errors: list[str] = []
        for model in order:
            model_error = "retries exhausted without a valid response"
            for _attempt in range(self.max_retries + 1):  # first try + retries
                attempts += 1
                try:
                    raw = self._invoke(model, package)
                except Exception as e:  # transport / model backend
                    if _is_transport_error(e):
                        # Model unavailable/slow -> abandon it, use the fallback.
                        model_error = f"{model}: transport error ({type(e).__name__})"
                        break
                    model_error = f"{model}: call error ({type(e).__name__}: {e})"
                    continue  # transient: retry the same model

                if not str(raw or "").strip():
                    # Empty output (e.g. the model's whole budget went to a
                    # thinking block) is a model-behaviour failure, not a retryable
                    # parse miss — abandon this model for the fallback immediately.
                    model_error = f"{model}: empty model output"
                    break

                try:
                    advisory = parse_advisory(raw)
                except RendererValidationError as e:
                    model_error = f"{model}: {e}"
                    continue  # garbage: retry (bounded by max_retries)

                problems = validate_advisory(
                    advisory,
                    allowed_markers=package.allowed_markers,
                    expected_breakdown=package.expected_breakdown,
                )
                if not problems:
                    return RenderResult(advisory=advisory, model_used=model, attempts=attempts)
                model_error = f"{model}: validation failed ({'; '.join(problems)})"
                # invalid output: retry the same model first
            errors.append(model_error)

        return RenderResult(advisory=None, attempts=attempts, error="; ".join(errors))
