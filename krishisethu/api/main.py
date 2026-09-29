"""FastAPI REST skeleton for KrishiSetu AI.

Exposes:
    GET /health          -> {"status": "ok"}
    GET /api/v1/status   -> settings summary + live Ollama reachability

The /status endpoint never 500s when Ollama is down: it reports
``ollama_reachable: false`` and moves on.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from krishisethu.config.settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "settings"):
        app.state.settings = get_settings()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="KrishiSetu AI", version="0.1.0", lifespan=lifespan)
    # Set at construction so tests that inject a custom settings object (e.g. a
    # dead Ollama host) work even when httpx.ASGITransport skips lifespan events.
    app.state.settings = settings or get_settings()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/status")
    async def status() -> dict:
        s: Settings = app.state.settings
        result = {
            "settings": s.summary(),
            "ollama_reachable": False,
            "ollama_version": None,
            "models_present": [],
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{s.ollama.embed_base()}/api/version")
                if r.status_code == 200:
                    result["ollama_reachable"] = True
                    result["ollama_version"] = r.json().get("version")
                    tr = await client.get(f"{s.ollama.embed_base()}/api/tags")
                    if tr.status_code == 200:
                        result["models_present"] = [
                            m.get("name") for m in tr.json().get("models", [])
                        ]
        except Exception:
            # Never 500 when Ollama is down; report unreachable and return.
            pass
        return result

    return app


app = create_app()
