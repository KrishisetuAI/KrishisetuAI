"""Tests for krishisethu.config.settings (Prompt 1)."""
from __future__ import annotations

from krishisethu.config.settings import (
    GateSettings,
    OllamaSettings,
    RagSettings,
    Settings,
    get_settings,
)


def test_settings_load_env():
    """Each nested group resolves from .env; base_url has no trailing slash."""
    s = Settings()
    assert s.ollama.host == "http://localhost:11434"
    assert s.ollama.llm_model == "qwen3.5:4b"
    assert s.ollama.llm_fallback == "qwen3.5:2b"
    assert s.ollama.embedding_model == "bge-m3:567m"
    assert s.chroma.collection_name == "krishisethu_kvk"
    assert s.rag.top_k == 5
    assert s.gate.confidence_gate == 0.85


def test_base_url_no_trailing_slash():
    s = Settings()
    assert s.ollama.base_url == "http://localhost:11434"
    assert not s.ollama.base_url.endswith("/")
    assert s.ollama.embed_base().endswith(":11434")


def test_base_url_derived_from_host(monkeypatch):
    """When OLLAMA_BASE_URL is absent, base_url derives from host (slashes stripped)."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    o = OllamaSettings(_env_file=None, host="http://example:11434/")
    assert o.base_url == "http://example:11434"
    assert o.embed_base() == "http://example:11434"
    assert o.host == "http://example:11434"


def test_absolute_chroma_path():
    s = Settings()
    # persisted dir resolves to an absolute Windows path
    import os
    assert os.path.isabs(s.chroma.persist_dir)


def test_weights_sum_to_one():
    g = GateSettings(w_rule=0.50, w_rag=0.30, w_vision=0.20)
    assert abs(g.w_rule + g.w_rag + g.w_vision - 1.0) < 1e-6


def test_validate_weights_rejects_bad():
    import pytest
    with pytest.raises(ValueError):
        GateSettings(w_rule=0.60, w_rag=0.30, w_vision=0.20)  # sums 1.10


def test_redistribute_without_vision():
    g = GateSettings()
    wr, wrg = g.redistribute_without_vision()
    assert abs(wr + wrg - 1.0) < 1e-6  # w1'+w2' == 1.0
    assert abs(wr / wrg - 5.0 / 3.0) < 1e-6  # ratio 5/3 preserved
    assert abs(wr - 0.625) < 1e-6
    assert abs(wrg - 0.375) < 1e-6


def test_topk_positivity():
    assert RagSettings(top_k=5).top_k == 5


def test_topk_rejects_zero(monkeypatch):
    import pytest
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    with pytest.raises(ValueError):
        RagSettings(_env_file=None, top_k=0)


def test_settings_are_cached():
    assert get_settings() is get_settings()
