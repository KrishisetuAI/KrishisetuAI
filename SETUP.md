# KrishiSetu AI — Local Environment Setup (Windows 11 + Ollama + Python 3.14)

This README stands up the hybrid RAG decision core (Tiers 1-4) on a local
Windows 11 machine. Everything runs on-device: the deterministic rule engine
(pure Python), the canonical YAML/Markdown knowledge layer, ChromaDB for vector
RAG, and the local Ollama LLM + embedding stack — no external API keys.

Time: ~20-40 min (mostly pip install and three ollama pulls).

---

## 1. Prerequisites (install in this order)

| # | Item | Why / Note |
|---|------|-----------|
| 1 | Windows 11 (or 10) | Host OS. Anything >= 10 works. |
| 2 | Python 3.14 (3.14.x, 64-bit) | Interpreter the venv + LangChain 1.x stack run on. **3.14.7 is already installed on this machine.** |
| 3 | Ollama desktop app (v0.33.3+) | Runs the local LLM + embeddings; owns the models. |
| 4 | Git Bash (via Git for Windows) | Where dev commands run. PowerShell also fine. |

> Model sizing assumes this machine: NVIDIA RTX 4050 Laptop (6 GB VRAM), 24 GB RAM.

### 1a. Confirm Python 3.14
A real Python 3.14 is already on this machine. Verify in a fresh terminal:

    python --version        # -> Python 3.14.x
    python -m venv .venv

NOTE: If python says "from the Microsoft Store", the Store alias is ahead of
the real interpreter on PATH. Disable the alias (Settings -> Apps -> Advanced
app settings -> App execution aliases) or use the py launcher:

    py -3.14 -m pip --version

### 1b. Install + run Ollama
1. Download from https://ollama.com/download (Windows installer .exe) if not installed.
2. After install Ollama auto-starts (system-tray llama icon). Server API on http://127.0.0.1:11434.
3. IMPORTANT: Ollama is a Windows desktop app — its CLI (ollama.exe) is NOT on
   the Git Bash PATH. To confirm the server is up, curl the API from Git Bash:

    curl -s http://localhost:11434/api/version     # -> {"version":"0.33.3"}

---

## 2. Install steps

### 2.1 Create & activate the virtual environment
    cd "C:/Users/lenovo/OneDrive/Documents/KrishiSetu_AI"
    python -m venv .venv
    source .venv/Scripts/activate        # Git Bash
    # PowerShell alternative:
    #   .venv\Scripts\Activate.ps1
    #   Set-ExecutionPolicy -Scope Process Bypass   # if blocked

### 2.2 Install Python packages (pinned in requirements.txt)
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Why these packages / versions (pinned to the CURRENT lines — LangChain 1.x and
ChromaDB 1.x are the maintained, Python 3.14-compatible releases):
- langchain==1.4.0 — orchestration (1.x folded community integrations in; Ollama
  now ships as its own langchain-ollama package).
- langchain-ollama==1.1.0 — provides ChatOllama, OllamaEmbeddings, OllamaLLM;
  this is how we reach the local models.
- chromadb==1.5.9 — persistent vector store (PersistentClient).
- ollama==0.6.2 — official Python client (direct API + fallback).
- fastapi==0.141.1 + uvicorn[standard]==0.52.4 — API layer.
- pydantic==2.13.5, pydantic-settings==2.15.0 — config / models.
- pyyaml==6.0.3 — load canonical YAML-frontmatter knowledge docs.
- python-dotenv==1.2.3 — load .env.
- pandas==3.0.5, numpy==2.5.2, tiktoken==0.14.0 — data + token counting.

### 2.3 Configure environment (.env)
    cp .env.example .env
Open .env and confirm:
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_LLM_MODEL=qwen3.5:4b
    OLLAMA_LLM_FALLBACK=qwen3.5:2b
    OLLAMA_EMBEDDING_MODEL=bge-m3:567m
    CHROMA_PERSIST_DIR=./data/chroma
    CHROMA_COLLECTION_NAME=krishisethu_kvk
    RAG_TOP_K=5
    CONFIDENCE_GATE=0.85
    APP_PORT=8000

OLLAMA_HOST is the source of truth; OLLAMA_BASE_URL is what the LangChain
classes consume via base_url=. They are identical here.

---

## 3. Which Ollama models to pull, and why

Run in PowerShell or CMD (NOT Git Bash — Ollama is not on that PATH), with the
Ollama app running. The three are sized to the 4-tier design AND this machine's
RTX 4050 (6 GB VRAM):

    powershell -ExecutionPolicy Bypass -File .\scripts\pull_models.ps1

or manually:
    & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:4b
    & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:2b
    & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull bge-m3:567m

| Model | Tag | Role in KrishiSetu | Why this one |
|------|-----|--------------------|--------------|
| Qwen3.5 4B | qwen3.5:4b | Tier-4 PARAPHRASER only | Current-gen Qwen; ~3.4GB (Q4) fits the 6GB GPU; excellent Hindi + English, so briefings render well in both languages. |
| Qwen3.5 2B | qwen3.5:2b | Fallback LLM | Tiny and fast (~2.7GB); used when the 4B is slow/unavailable so briefings still render. |
| BGE-M3 | bge-m3:567m | Embeddings (1024-d) | Multilingual (100+ languages); far better Hindi/English cross-lingual retrieval than the old nomic-embed-text. Runs INSIDE Ollama — no separate API key. |

Optional higher-quality paraphraser: `qwen3.5:9b` (~6.6GB) gives the best local
quality but spills off the 6GB VRAM and is slower, so it is the non-default
option. All embeddings run through Ollama, keeping the whole stack local.

If a pull is slow or errors: the server is down. Launch the Ollama app from
the tray, then retry.

---

## 4. Verification steps

### 4.1 curl the Ollama API (server up?)
    curl -s http://localhost:11434/api/version       # -> {"version":...}
    curl -s http://localhost:11434/api/tags          # -> {"models":[...]}

### 4.2 model list in PowerShell
    & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list

### 4.3 end-to-end verification script (needs venv + packages)
    source .venv/Scripts/activate
    python scripts/verify_setup.py
This checks (1) Ollama reachable, (2) all three models present, (3) a real
embedding via OllamaEmbeddings returns a non-empty vector and prints cosine
similarity between two docs. Exit 0 = ready.

### 4.4 quick manual embed (from venv python)
    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(model="bge-m3:567m", base_url="http://localhost:11434")
    v = emb.embed_query("potato late blight spray window")
    print(len(v), v[:3])
    # ~1024  [-0.0014, 0.0121, ...]

---

## 5. Next steps (first build target)
1. pip install, then start the FastAPI app skeleton.
2. Canonical YAML layer in knowledge/ (MSP, spacing, crop calendar).
3. Deterministic rule engine in rules/ (weather-to-action matrices).
4. Seed data/chroma with KVK long-tail Q&As chunked with crop/district/season
   tags, pre-retrieval metadata filter, top-k=5.
5. Re-run python scripts/verify_setup.py to confirm RAG end-to-end.

---

## 6. Troubleshooting quick reference

| Symptom | Fix |
|---------|-----|
| python opens Microsoft Store | Disable the Store alias, or fix PATH so the real 3.14 comes first. |
| curl /api/version returns empty | Ollama app not running — start it from tray; check tasklist for "ollama". |
| ollama: command not found (Git Bash) | Expected — Ollama is not on PATH. Use PowerShell / full path to ollama.exe. |
| ModuleNotFoundError: langchain_ollama | pip install -r requirements.txt inside the ACTIVATED venv. |
| Chroma cannot open ./data/chroma | Use the absolute CHROMA_PERSIST_DIR from .env; create the folder. |
| Embedding dim mismatch across models | Keep ONE embedding model for index + query; do not mix bge-m3 (1024) and nomic (768). |
| Connection refused on :11434 in Python | OLLAMA_HOST wrong or app down; Ollama binds 127.0.0.1 by default. |
