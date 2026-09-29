"""KrishiSetu AI — fast daily health check (Prompt 0, step 10).

Verifies the import chain + Python version + .env completeness without touching
Ollama or the network. Exit 0 only if every import works and .env is complete.

Usage (venv active or absolute path):
    .venv/Scripts/python scripts/verify_all.py
"""
from __future__ import annotations

import os
import sys

REQUIRED_ENV_KEYS = [
    "OLLAMA_HOST", "OLLAMA_BASE_URL", "OLLAMA_LLM_MODEL", "OLLAMA_LLM_FALLBACK",
    "OLLAMA_EMBEDDING_MODEL", "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION_NAME",
    "RAG_TOP_K", "CONFIDENCE_GATE", "CONFIDENCE_W_RULE", "CONFIDENCE_W_RAG",
    "CONFIDENCE_W_VISION", "APP_HOST", "APP_PORT",
]

IMPORTS = [
    ("langchain", "langchain"),
    ("langchain_ollama", "langchain_ollama"),
    ("chromadb", "chromadb"),
    ("ollama", "ollama"),
    ("pydantic", "pydantic"),
    ("yaml", "yaml"),
    ("dotenv", "dotenv"),
]


def main() -> int:
    fail = False

    print(f"  python      : {sys.version.split()[0]}")
    print(f"  interpreter : {sys.executable}")

    print("  imports:")
    for module, label in IMPORTS:
        try:
            __import__(module)
            print(f"    OK   {label}")
        except Exception as e:  # noqa: BLE001
            print(f"    FAIL {label}: {e}")
            fail = True

    print("  .env completeness:")
    env_path = os.path.join(os.getcwd(), ".env")
    missing = []
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            loaded = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, _ = line.partition("=")
                    loaded[k.strip()] = 1
        for k in REQUIRED_ENV_KEYS:
            if k not in loaded:
                missing.append(k)
        if missing:
            print(f"    FAIL missing keys: {missing}")
            fail = True
        else:
            print(f"    OK   all {len(REQUIRED_ENV_KEYS)} required keys present")
    else:
        print("    FAIL .env not found (cp .env.example .env)")
        fail = True

    print("RESULT:", "PASS" if not fail else "FAIL")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
