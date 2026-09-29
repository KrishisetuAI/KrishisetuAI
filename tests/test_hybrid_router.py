"""Tests for the hybrid cloud/Ollama router (Bonus Phase).

Covers multi-provider key-rolling, PII guard, fallback behavior, and schema validation.
"""

from unittest.mock import MagicMock, patch

import litellm
import pytest

from krishisethu.renderer.router import HybridRenderer, CloudEndpoint, _build_cloud_pool
from krishisethu.renderer.ollama_client import OllamaRenderer, RenderResult
from krishisethu.renderer.prompt_contract import PromptPackage
from krishisethu.renderer.structured_output import Advisory, ConfidenceBreakdown


# List of all cloud env var names for test isolation
CLOUD_ENV_VARS = (
    [f"NVIDIA_API_KEY_{i}" for i in range(1, 5)]
    + [f"GROQ_API_KEY_{i}" for i in range(1, 5)]
    + [f"OPENROUTER_API_KEY_{i}" for i in range(1, 3)]
)


@pytest.fixture(autouse=True)
def _clear_cloud_env(monkeypatch):
    """Ensure no cloud env vars leak between tests."""
    for name in CLOUD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def package():
    """A minimal PromptPackage for router tests."""
    return PromptPackage(
        system="You are a paraphraser.",
        user="Paraphrase: Suspend irrigation.",
        allowed_markers=frozenset(),
        expected_breakdown=None,
    )


@pytest.fixture
def pii_package():
    """A PromptPackage containing a phone number (PII)."""
    return PromptPackage(
        system="You are a paraphraser.",
        user="Call me at 9876543210 please.",
        allowed_markers=frozenset(),
        expected_breakdown=None,
    )


@pytest.fixture
def valid_advisory():
    """A valid Advisory for mocking parse_advisory."""
    return Advisory(
        actions=(),
        confidence_breakdown=ConfidenceBreakdown(R=0.0, S_RAG=0.0, P_vision=0.0, has_image=False, cci=0.0),
    )


def _mock_litellm_success(content: str):
    """Create a fake litellm completion response with given content."""
    fake = MagicMock()
    fake.choices = [MagicMock()]
    fake.choices[0].message.content = content
    return fake


# --------------------------------------------------------------------------
# Pool building tests
# --------------------------------------------------------------------------
def test_build_cloud_pool_empty(monkeypatch):
    """No env vars -> empty pool."""
    pool = _build_cloud_pool()
    assert pool == []


def test_build_cloud_pool_nvidia(monkeypatch):
    """NVIDIA keys create endpoints with correct api_base and provider."""
    monkeypatch.setenv("NVIDIA_API_KEY_1", "nvidia-key-1")
    monkeypatch.setenv("NVIDIA_API_KEY_2", "nvidia-key-2")
    pool = _build_cloud_pool()
    assert len(pool) == 2
    for ep in pool:
        assert ep.provider == "nvidia"
        assert ep.model == "openai/meta/llama-3.3-70b-instruct"
        assert ep.api_base == "https://integrate.api.nvidia.com/v1"
        assert ep.custom_llm_provider == "openai"
    assert pool[0].slot == 1
    assert pool[1].slot == 2


def test_build_cloud_pool_groq(monkeypatch):
    """Groq keys create endpoints with correct model."""
    monkeypatch.setenv("GROQ_API_KEY_1", "groq-key-1")
    pool = _build_cloud_pool()
    assert len(pool) == 1
    ep = pool[0]
    assert ep.provider == "groq"
    assert ep.model == "groq/llama-3.3-70b-versatile"
    assert ep.api_base is None
    assert ep.custom_llm_provider is None


def test_build_cloud_pool_openrouter(monkeypatch):
    """OpenRouter keys create endpoints with correct model."""
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "or-key-1")
    pool = _build_cloud_pool()
    assert len(pool) == 1
    ep = pool[0]
    assert ep.provider == "openrouter"
    assert ep.model == "openrouter/qwen/qwen-2.5-coder-32b-instruct"


def test_build_cloud_pool_order(monkeypatch):
    """Pool order is NVIDIA -> Groq -> OpenRouter."""
    monkeypatch.setenv("GROQ_API_KEY_1", "groq-1")
    monkeypatch.setenv("NVIDIA_API_KEY_1", "nvidia-1")
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "or-1")
    pool = _build_cloud_pool()
    assert [ep.provider for ep in pool] == ["nvidia", "groq", "openrouter"]


def test_build_cloud_pool_skips_empty(monkeypatch):
    """Empty string keys are skipped."""
    monkeypatch.setenv("NVIDIA_API_KEY_1", "")
    monkeypatch.setenv("NVIDIA_API_KEY_2", "real-key")
    pool = _build_cloud_pool()
    assert len(pool) == 1
    assert pool[0].slot == 2


# --------------------------------------------------------------------------
# Router fallback tests (no keys configured)
# --------------------------------------------------------------------------
def test_router_no_keys_uses_fallback(package):
    """With no env keys, router calls fallback directly without any cloud call."""
    fallback = MagicMock(spec=OllamaRenderer)
    fallback.render.return_value = RenderResult(advisory=None, model_used="fallback", attempts=1, error=None)

    router = HybridRenderer(fallback_renderer=fallback)

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        result = router.render(package)
        mock_completion.assert_not_called()
        fallback.render.assert_called_once_with(package)
        assert result.model_used == "fallback"


def test_router_pii_guard_blocks_before_cloud(pii_package):
    """PII in prompt raises RendererValidationError before any cloud call."""
    fallback = OllamaRenderer(base_url="http://fake:11434")
    router = HybridRenderer(fallback_renderer=fallback)
    # Set a key so cloud would be attempted
    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        with pytest.raises(Exception) as exc_info:
            router.render(pii_package)
        # Should be the PII error from OllamaRenderer._assert_no_pii
        assert "PII" in str(exc_info.value) or "phone" in str(exc_info.value).lower()
        mock_completion.assert_not_called()


# --------------------------------------------------------------------------
# Router success and key-rolling tests
# --------------------------------------------------------------------------
def test_router_uses_nvidia_when_available(package, valid_advisory, monkeypatch):
    """NVIDIA endpoint used with custom provider and api_base when key set."""
    monkeypatch.setenv("NVIDIA_API_KEY_1", "nvidia-test-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_success(json_content)
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert result.model_used == "openai/meta/llama-3.3-70b-instruct"
    # Verify NVIDIA-specific kwargs were passed
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert call_kwargs["custom_llm_provider"] == "openai"


def test_router_uses_groq_when_available(package, valid_advisory, monkeypatch):
    """Groq endpoint used with correct model when key set."""
    monkeypatch.setenv("GROQ_API_KEY_1", "groq-test-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_success(json_content)
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert result.model_used == "groq/llama-3.3-70b-versatile"
    call_kwargs = mock_completion.call_args.kwargs
    assert "api_base" not in call_kwargs
    assert "custom_llm_provider" not in call_kwargs


def test_router_uses_openrouter_when_available(package, valid_advisory, monkeypatch):
    """OpenRouter endpoint used with correct model when key set."""
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "or-test-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_success(json_content)
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert result.model_used == "openrouter/qwen/qwen-2.5-coder-32b-instruct"


def test_router_rolls_over_keys_on_429(package, valid_advisory, monkeypatch):
    """On 429 rate limit, router tries next key in pool."""
    monkeypatch.setenv("GROQ_API_KEY_1", "groq-key-1")
    monkeypatch.setenv("GROQ_API_KEY_2", "groq-key-2")
    fallback = MagicMock(spec=OllamaRenderer)

    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="groq",
                model="llama-3.3-70b-versatile",
                response=MagicMock(),
            )
        return _mock_litellm_success(json_content)

    with patch("krishisethu.renderer.router.litellm.completion", side_effect=side_effect) as mock_completion:
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert result.model_used == "groq/llama-3.3-70b-versatile"
    assert call_count["n"] == 2
    # Second call should use the second key
    assert mock_completion.call_args_list[1].kwargs["api_key"] == "groq-key-2"


def test_router_rolls_over_keys_on_401(package, valid_advisory, monkeypatch):
    """On 401 auth error, router tries next key."""
    monkeypatch.setenv("GROQ_API_KEY_1", "bad-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "good-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise litellm.AuthenticationError(
                message="Invalid API key",
                llm_provider="groq",
                model="llama-3.3-70b-versatile",
            )
        return _mock_litellm_success(json_content)

    with patch("krishisethu.renderer.router.litellm.completion", side_effect=side_effect) as mock_completion:
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert call_count["n"] == 2


def test_router_rolls_over_keys_on_timeout(package, valid_advisory, monkeypatch):
    """On timeout, router tries next key."""
    monkeypatch.setenv("GROQ_API_KEY_1", "slow-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "fast-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("Request timed out")
        return _mock_litellm_success(json_content)

    with patch("krishisethu.renderer.router.litellm.completion", side_effect=side_effect) as mock_completion:
        with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert call_count["n"] == 2


def test_router_rolls_over_keys_on_schema_error(package, valid_advisory, monkeypatch):
    """On schema validation failure, router tries next key."""
    from krishisethu.renderer.structured_output import RendererValidationError
    monkeypatch.setenv("GROQ_API_KEY_1", "bad-response-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "good-response-key")
    fallback = MagicMock(spec=OllamaRenderer)
    router = HybridRenderer(fallback_renderer=fallback)

    # First response is valid JSON but fails schema validation
    # Second response passes
    good_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        return _mock_litellm_success(good_content)

    def parse_side_effect(content):
        if call_count["n"] == 1:
            # First call: simulate schema validation failure
            raise RendererValidationError("missing confidence_breakdown")
        return valid_advisory

    with patch("krishisethu.renderer.router.litellm.completion", side_effect=side_effect) as mock_completion:
        with patch("krishisethu.renderer.structured_output.parse_advisory", side_effect=parse_side_effect):
            result = router.render(package)

    fallback.render.assert_not_called()
    assert call_count["n"] == 2


def test_router_falls_back_when_all_keys_fail(package, monkeypatch):
    """When all keys fail, router falls back to Ollama."""
    monkeypatch.setenv("GROQ_API_KEY_1", "fail-1")
    monkeypatch.setenv("GROQ_API_KEY_2", "fail-2")
    fallback = MagicMock(spec=OllamaRenderer)
    fallback.render.return_value = RenderResult(advisory=None, model_used="fallback", attempts=1, error=None)
    router = HybridRenderer(fallback_renderer=fallback)

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.side_effect = litellm.RateLimitError(
            message="Rate limit", llm_provider="groq", model="x", response=MagicMock()
        )
        result = router.render(package)

    fallback.render.assert_called_once_with(package)
    assert result.model_used == "fallback"
    assert mock_completion.call_count == 2


# --------------------------------------------------------------------------
# Legacy cloud_models parameter tests (backwards compat)
# --------------------------------------------------------------------------
def test_router_legacy_cloud_models_uses_fallback_on_429(package):
    """Legacy: explicit cloud_models with 429 -> fallback."""
    fallback = MagicMock(spec=OllamaRenderer)
    fallback.render.return_value = RenderResult(advisory=None, model_used="fallback", attempts=1, error=None)
    router = HybridRenderer(cloud_models=["groq/llama3-8b-8192"], fallback_renderer=fallback)
    # No keys set -> empty pool -> fallback directly
    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.side_effect = litellm.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="groq",
            model="llama3-8b-8192",
            response=MagicMock(),
        )
        result = router.render(package)
        fallback.render.assert_called_once_with(package)
        assert result.model_used == "fallback"


def test_router_legacy_cloud_models_uses_fallback_on_network_error(package):
    """Legacy: explicit cloud_models with network error -> fallback."""
    fallback = MagicMock(spec=OllamaRenderer)
    fallback.render.return_value = RenderResult(advisory=None, model_used="fallback", attempts=1, error=None)
    router = HybridRenderer(cloud_models=["groq/llama3-8b-8192"], fallback_renderer=fallback)

    with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
        mock_completion.side_effect = ConnectionError("Network down")
        result = router.render(package)
        fallback.render.assert_called_once()
        assert result.model_used == "fallback"


def test_router_legacy_cloud_models_success(package, valid_advisory):
    """Legacy: explicit cloud_models with patched _get_api_keys -> cloud success."""
    # Patch the legacy key getter to return a dummy key
    with patch("krishisethu.renderer.router._get_api_keys", return_value=["legacy-dummy-key"]):
        fallback = MagicMock(spec=OllamaRenderer)
        router = HybridRenderer(cloud_models=["groq/llama3-8b-8192"], fallback_renderer=fallback)

        json_content = '{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}'

        with patch("krishisethu.renderer.router.litellm.completion") as mock_completion:
            mock_completion.return_value = _mock_litellm_success(json_content)
            with patch("krishisethu.renderer.structured_output.parse_advisory", return_value=valid_advisory):
                result = router.render(package)

        fallback.render.assert_not_called()
        # Legacy model name preserved
        assert result.model_used == "groq/llama3-8b-8192"