"""Tests for krishisethu.api.main (Prompt 1)."""
from __future__ import annotations

import httpx
import pytest

from krishisethu.api.main import create_app
from krishisethu.config.settings import OllamaSettings, Settings


@pytest.mark.asyncio
async def test_health():
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_status_reachable():
    """Ollama is live here -> status 200, ollama_reachable bool, version + models populated."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert "settings" in body
        assert isinstance(body.get("ollama_reachable"), bool)
        if body.get("ollama_reachable"):
            assert body.get("ollama_version") is not None
            assert "bge-m3:567m" in body.get("models_present", [])
            assert "qwen3.5:4b" in body.get("models_present", [])


@pytest.mark.asyncio
async def test_status_no_exception_when_ollama_down():
    """Never 500 and never leak an exception when Ollama is unreachable."""
    dead = Settings(
        ollama=OllamaSettings(host="http://127.0.0.1:1", base_url="http://127.0.0.1:1")
    )
    app = create_app(settings=dead)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/api/v1/status")
        assert r.status_code == 200
        body = r.json()
        assert body["ollama_reachable"] is False
        assert body["ollama_version"] is None
