"""Tests for krishisethu.api.main (Prompt 1)."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from krishisethu.api.main import create_app
from krishisethu.config.settings import OllamaSettings, Settings


class StubAdvisePipeline:
    """Hermetic pipeline stand-in (no Ollama, Chroma, or network)."""

    def __init__(self, *, escalate: bool, actions: tuple = ()) -> None:
        self.escalate = escalate
        self.actions = actions
        self.calls: list[tuple] = []

    def run(self, query: str, plot_id: str = "demo", demo: bool = False):
        self.calls.append((query, plot_id, demo))
        decision = SimpleNamespace(
            actions=[SimpleNamespace(text=t, source_id=s) for t, s in self.actions]
        )
        return SimpleNamespace(
            escalate=self.escalate,
            final_output=(
                "Your query has been forwarded to a KVK agricultural expert."
                if self.escalate
                else "Suspend irrigation. Clear furrow outlets."
            ),
            final_decision=decision,
            cci_value=0.23 if self.escalate else 0.95,
            gate_outcome="fail_cci" if self.escalate else "pass_exact",
            trace_id="trace-api-test",
            escalation_ticket=(
                SimpleNamespace(ticket_id="KVK-TEST") if self.escalate else None
            ),
        )


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


@pytest.mark.asyncio
async def test_advise_returns_validated_actions():
    """The REST channel serializes the pipeline's validated, traceable actions."""
    pipeline = StubAdvisePipeline(
        escalate=False,
        actions=(
            ("Suspend irrigation for the next 72 hours.", "SUGAR-IRR-014"),
            ("Clear furrow outlets to drain excess water.", "SUGAR-M-HEAVYRAIN"),
        ),
    )
    app = create_app(pipeline=pipeline)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.post(
            "/api/v1/advise",
            json={
                "query": "Heavy rain is forecast for my sugarcane field.",
                "plot_id": "sohna-demo",
                "demo": True,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["escalate"] is False
    assert body["ticket_id"] is None
    assert body["gate_outcome"] == "pass_exact"
    assert [a["source_id"] for a in body["actions"]] == [
        "SUGAR-IRR-014",
        "SUGAR-M-HEAVYRAIN",
    ]
    assert "Suspend irrigation" in body["output"]
    assert pipeline.calls == [("Heavy rain is forecast for my sugarcane field.", "sohna-demo", True)]


@pytest.mark.asyncio
async def test_advise_escalation_never_returns_the_advisory():
    """A withheld (poisoned) advisory must never be serialized to the client."""
    pipeline = StubAdvisePipeline(
        escalate=True,
        actions=(("POISON chlorpyrifos 2.5 ml/L never return this", "bad"),),
    )
    app = create_app(pipeline=pipeline)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.post(
            "/api/v1/advise", json={"query": "Zinc spray karna hai kya?"}
        )
    assert r.status_code == 200
    body = r.json()
    assert body["escalate"] is True
    assert body["actions"] == []  # withheld advisory never serialized
    assert body["ticket_id"] == "KVK-TEST"
    assert "forwarded to a KVK" in body["output"]
    assert "POISON" not in body["output"]
    assert "2.5" not in body["output"]
    assert pipeline.calls == [("Zinc spray karna hai kya?", "demo", False)]


@pytest.mark.asyncio
async def test_advise_rejects_empty_or_missing_query():
    app = create_app(pipeline=StubAdvisePipeline(escalate=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r_empty = await c.post("/api/v1/advise", json={"query": "   "})
        r_missing = await c.post("/api/v1/advise", json={})
    assert r_empty.status_code == 422
    assert r_missing.status_code == 422
