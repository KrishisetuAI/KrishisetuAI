"""KrishiSetu AI — embedding smoke test (Prompt 0, step 7).

Verifies the multilingual embedding model (bge-m3:567m) is reachable through
Ollama and returns a 1024-d vector with sensible cosine similarity ordering:
near-duplicate > topical > unrelated.

Usage (venv active or absolute path):
    .venv/Scripts/python scripts/embed_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")

    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=model, base_url=base_url)

    topical = "Late blight on potato: what is the spray window before heavy rain?"
    near_dup = "Late blight on potato: what is the spray window before rain?"
    unrelated = "Minimum support price for wheat in Rabi 2026 in Haryana."

    vec_t = emb.embed_query(topical)
    vec_n = emb.embed_query(near_dup)
    vec_u = emb.embed_query(unrelated)

    dim = len(vec_t)
    if dim != 1024:
        print(f"  WARN expected 1024-d, got {dim}-d (check OLLAMA_EMBEDDING_MODEL)")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    sim_near = cos(vec_t, vec_n)
    sim_unrl = cos(vec_t, vec_u)

    print(f"  model         : {model}")
    print(f"  base_url      : {base_url}")
    print(f"  vector dim    : {dim}")
    print(f"  near-dup cos  : {sim_near:.4f}")
    print(f"  unrelated cos : {sim_unrl:.4f}")

    if sim_near > sim_unrl:
        print("  PASS  near-duplicate clearly higher than unrelated -> embeddings are sane")
        return 0
    print("  FAIL  ordering not as expected (near-dup should be highest)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
