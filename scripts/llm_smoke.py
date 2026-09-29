"""KrishiSetu AI — LLM smoke test (Prompt 0, step 8).

Confirms the Tier-4 paraphraser (qwen3.5:4b) and its fallback (qwen3.5:2b)
complete a trivial prompt through BOTH the langchain_ollama ChatOllama wrapper
and the raw `ollama` Python client. Prints first ~20 chars and wall latency.

Usage (venv active or absolute path):
    .venv/Scripts/python scripts/llm_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def via_chatollama(model: str) -> tuple[str, float]:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, base_url=BASE_URL)
    t0 = time.time()
    resp = llm.invoke("Reply with the single word: READY")
    dt = time.time() - t0
    text = str(resp.content).strip()
    return text, dt


def via_raw_client(model: str) -> tuple[str, float]:
    import ollama

    t0 = time.time()
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": "Reply with the single word: READY"}])
    dt = time.time() - t0
    text = resp["message"]["content"].strip()
    return text, dt


def main() -> int:
    models = [
        os.environ.get("OLLAMA_LLM_MODEL", "qwen3.5:4b"),
        os.environ.get("OLLAMA_LLM_FALLBACK", "qwen3.5:2b"),
    ]
    fail = False
    for m in models:
        for label, fn in (("ChatOllama", via_chatollama), ("raw-ollama", via_raw_client)):
            try:
                text, dt = fn(m)
                ok = "READY" in text.upper() or text
                print(f"  {m:14s} [{label:11s}] latency={dt:.2f}s  first20={text[:20]!r}")
                if not text:
                    fail = True
            except Exception as e:  # noqa: BLE001
                print(f"  {m:14s} [{label:11s}] FAILED: {e}")
                fail = True
    print("PASS" if not fail else "FAIL")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
