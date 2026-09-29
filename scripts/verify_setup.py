"""
KrishiSetu AI — environment verification script.
Run AFTER: python installed, venv created, packages installed, .env created, models pulled.

    .venv\Scripts\python scripts\verify_setup.py

Checks: (1) Ollama server reachable, (2) models present, (3) embeddings work end-to-end.
Exits non-zero with actionable message if a check fails.
"""
from __future__ import annotations

import os
import sys

def banner(t: str) -> None:
    print("\n" + "=" * 62)
    print("  " + t)
    print("=" * 62)

def main() -> int:
    fail = False
    banner("STEP 1/3 - Ollama server reachable (curl-equivalent via urllib)")
    import urllib.request
    import json as _json

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(host + "/api/version", timeout=5) as r:
            data = _json.loads(r.read().decode())
        print(f"  OK   Ollama server at {host}  version={data.get('version')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL Cannot reach Ollama at {host}: {e}")
        print("  TIP  Ensure the Ollama desktop app is running (system tray icon).")
        return 1

    banner("STEP 2/3 - Required models present")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=5) as r:
            tags = _json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL Cannot list models: {e}")
        return 1

    have = {m.get("name") for m in tags.get("models", [])}
    wanted = {
        os.environ.get("OLLAMA_LLM_MODEL", "qwen3.5:4b"),
        os.environ.get("OLLAMA_LLM_FALLBACK", "qwen3.5:2b"),
        os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m"),
    }
    for w in sorted(wanted):
        if w in have:
            print(f"  OK   model present: {w}")
        else:
            print(f"  MISS model NOT pulled: {w}")
            fail = True
    if fail:
        print("  TIP  Pull models from PowerShell (Ollama is NOT on Git Bash PATH):")
        print('       & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:4b')
        print('       & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:2b')
        print('       & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull bge-m3:567m')
        return 1

    banner("STEP 3/3 - End-to-end embedding via Ollama (bge-m3, multilingual 1024-d)")
    try:
        from langchain_ollama import OllamaEmbeddings
        embed_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
        emb = OllamaEmbeddings(model=embed_model, base_url=host)
        vecs = emb.embed_documents(
            ["Late blight on potato: spray window", "Wheat MSP for Rabi 2026"]
        )
        dim = len(vecs[0])
        assert dim > 0, "embedding returned 0 dimension"
        import numpy as np
        sim = float(np.dot(vecs[0], vecs[1])
                    / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
        print(f"  OK   embedded 2 docs -> dim {dim}; cosine sim between them = {sim:.3f}")
        print("  NOTE dimension varies by model (bge-m3 ~1024 (multilingual); nomic ~768).")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL embedding step: {e}")
        print("  TIP  confirm langchain-ollama is installed; re-run pip install -r requirements.txt")
        return 1

    banner("ALL CHECKS PASSED")
    print("Environment is ready for the hybrid RAG core.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
