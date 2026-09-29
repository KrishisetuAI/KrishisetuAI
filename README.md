<div align="center">

# 🌾 KrishiSetu AI

### Plot-specific, evidence-backed farm advisories — safe by construction

**Smart India Hackathon 2026 Submission**

Official DPI data in. Verified, sourced, farmer-ready advice out.
The LLM never originates a recommendation. It paraphrases.

[Architecture](docs/01-ARCHITECTURE.md) · [Safety Contract](docs/06-SAFETY-AND-CONFIDENCE.md) · [Benchmarks](docs/11-BENCHMARKS.md) · [Demo Script](commands.py) · [Setup](SETUP.md)

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.11-orange?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Qwen3.5--4B%20%7C%20BGE--M3-black?logo=ollama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local%20vector%20store-green?logo=chromadb&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-teal?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-238%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## The Problem

India's 100M+ smallholder farmers face a flood of agronomic questions every week: *When do I irrigate? Heavy rain is coming — do I still spray? Is this leaf yellowing zinc deficiency?* The answers exist — in IMD weather alerts, Soil Health Cards, ICAR Package of Practices, KVK advisories — but they are scattered across portals, PDFs, and helplines a farmer on a 2G feature phone cannot reach.

Generic LLM chatbots are not the answer. They hallucinate pesticide dosages, cite nothing, and cannot be held accountable. A wrong spray recommendation destroys a crop, a season, and a family's income.

**KrishiSetu AI closes this gap safely**: it fuses official DPI data with a hybrid decision engine whose advice is always traceable to a rule, a canonical document, or a verified corpus chunk — and escalates to a human KVK expert whenever it is not certain enough to speak.

## Why We Are Different: Safe by Construction

Most "agri-AI" projects put an LLM at the center and bolt safety on afterwards. We inverted it:

| Guarantee | How it is enforced |
|---|---|
| **The LLM never originates advice** | Tier 4 is a *paraphraser only* — its prompt contract contains exclusively Tier 1–3 output; anything untraceable is rejected at parse time |
| **Never below-threshold guessing** | Composite Confidence Index `CCI = 0.50·R + 0.30·S_RAG + 0.20·P_vision`; below `0.85` we escalate to KVK instead of answering |
| **Zero chemical dosage hallucination** | Safety Gate 2 strips any active-ingredient + dosage pair structurally and at term level; the query always escalates to a human |
| **Every claim carries a source** | Safety Gate 3 requires a Tier-1 rule id / Tier-2 doc id / Tier-3 chunk id marker on every action; untraceable claims are stripped, and if all are stripped we escalate |
| **Generate, never submit** | PMFBY crop-loss PDFs are produced for the farmer's review only; there is no code path to any insurance portal |

These are not prompts asking a model to behave — they are hard gates in the pipeline (`krishisethu/safety/gates.py`) that run on every single response.

## The 4-Tier Architecture

```
        Farmer query (EN / Hindi / Hinglish, web or IVR)
                          │
                          ▼
   ┌──────────── LangGraph pipeline (typed Pydantic state) ────────────┐
   │  triage → context → Tier 1 → Tier 2 → Tier 3 → CCI gate           │
   │                       │ (pass_exact short-circuit)                 │
   │                       ▼                                            │
   │                  Tier 4 render (paraphrase only)                   │
   │                       ▼                                            │
   │             3 safety gates ──→ ✅ DELIVER  or  ⚠️ KVK ESCALATE     │
   └────────────────────────────────────────────────────────────────────┘
```

| Tier | Name | What it does | Why it cannot hallucinate |
|---|---|---|---|
| **1** | Deterministic Rule Engine | Weather-to-action matrices, spray windows, deficit irrigation, ratoon/gap planner — pure Python | Identical inputs → identical outputs; immune by construction |
| **2** | Canonical Knowledge Layer | ICAR PoP / MSP / spacing / scheme eligibility as Markdown + YAML, **exact metadata lookup** (never chunked, never embedded) | Supersession chains + authority-rank conflict resolution; expired facts are never served |
| **3** | Domain-Governed Vector RAG | KVK Q&A, ICAR circulars, state advisories in ChromaDB — **metadata-filter-first, similarity-second, top-k=5** | A sugarcane query can never retrieve a rice chunk; every chunk is sourced |
| **4** | LLM Explanation Layer | Qwen3.5-4B (Ollama) or cloud pool via HybridRenderer | Receives *only* Tier 1–3 output; must emit schema-valid JSON with evidence refs, else rejected |

**Confidence & escalation spine:** CCI in `krishisethu/confidence/composite_index.py`, three-gate `SafetyValidator` in `krishisethu/safety/gates.py`, chemical registry in `krishisethu/safety/chemical_registry.py`, KVK `EscalationService` in `krishisethu/escalation/kvk_ticket.py` (tickets deduplicated by `sha256(query + plot_id + date_window)`).

**Evidence & reporting:** Sentinel-2 NDVI engine (`krishisethu/evidence/ndvi.py`) with SCL cloud masking and anomaly severity classification, feeding the PMFBY PDF report generator (`krishisethu/pipeline/report.py`) with an offline queue for connectivity-poor regions.

**Hybrid cloud-local rendering:** `krishisethu/renderer/router.py` pools up to 10 cloud keys (4× NVIDIA NIM, 4× Groq, 2× OpenRouter) with automatic key-rolling on 429/401/timeout, a PII guard before any cloud call (farmer data never leaves the device), and clean fallback to local Ollama when all cloud keys fail — the system works fully offline.

## The Demo: 4 Scenarios, One Command Each

The exact commands for the video demo live in [`commands.py`](commands.py) — a fully commented reference script, in order:

1. **Standard weekly briefing** — sugarcane, grand growth, heavy-rain warning → Tier-1 EXACT advisory (suspend irrigation, clear furrow outlets, delay foliar spray 72h), each action with its rule trace.
2. **English weather-alert query** — "Heavy rain is forecast for my sugarcane field…" → verified advisory, CCI ≥ 0.85, evidence-traced actions.
3. **Hindi/Hinglish query triggering escalation** — "Zinc spray karna hai kya? (leaf yellowing)" → CCI fails the gate → neutral farmer hand-off + KVK escalation ticket. We do not guess; we route to a human expert.
4. **Offline PMFBY PDF report** — plot history + NDVI flood anomaly (baseline 0.79 → observed 0.44, **severe**) rendered into an insurance-ready assessment PDF, queued offline.

## Live-Verified Numbers

Measured on our own golden set and test corpus (not vendor claims) — see [docs/11-BENCHMARKS.md](docs/11-BENCHMARKS.md):

| Metric | Value |
|---|---|
| **Full test suite** | **238 passed**, 1 skipped — rules, knowledge, RAG, confidence, safety, escalation, NDVI, pipeline, router, report |
| **Chem-dosage hallucination** | **0%** (blocked by Safety Gate 2, malicious suite) |
| **Traceability failures** | **0%** (blocked by Safety Gate 3) |
| **OOD / ambiguous abstention** | **100%** — escalate, never low-confidence delivery |
| **Escalation idempotency** | 1 ticket on replay (sha256 dedup) |
| **Tier-1 rule latency** | ~0.2 ms (sub-millisecond CPU math) |
| **Sohna flood NDVI anomaly** | Δ = −0.35, severity **severe** (44.3% drop > 30% threshold) |

## Quickstart

See [SETUP.md](SETUP.md) for full detail. The short path (Windows PowerShell):

```powershell
cd "C:/Users/lenovo/OneDrive/Documents/KrishiSetu_AI"
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env

# Pull the 3 Ollama models (PowerShell, not Git Bash):
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:4b
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.5:2b
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull bge-m3:567m

python scripts/verify_setup.py    # exit 0 = ready
```

Run the demo story and a single advisory:

```powershell
python demo.py
python -m krishisethu.cli advise -q "Heavy rain is forecast for my sugarcane field in grand growth — what should I do this week?" --plot-id sohna-demo --demo
python -m krishisethu.cli report --plot-id sohna-demo -o report.pdf
```

Optional: add one cloud key (`NVIDIA_API_KEY_1`, `GROQ_API_KEY_1`, or `OPENROUTER_API_KEY_1`) in `.env` to demo the hybrid cloud router; with none set, everything runs locally on Ollama.

## Repo Map

```
krishisethu/
├── rules/        Tier 1 — deterministic rule engine
├── knowledge/    Tier 2 — canonical knowledge (exact lookup)
├── rag/          Tier 3 — domain-governed vector RAG (ChromaDB)
├── renderer/     Tier 4 — LLM paraphraser + HybridRenderer + translation
├── pipeline/     LangGraph orchestration, PMFBY report, offline queue
├── confidence/   CCI composite confidence index
├── safety/       three-gate validator + chemical registry
├── escalation/   KVK ticket service (deduplicated)
├── evidence/     Sentinel-2 NDVI engine
├── domain/       typed enums & models
├── api/          FastAPI service skeleton
└── cli.py        `krishisethu advise | demo | report` CLI
commands.py       ← video-demo command script (this submission)
docs/             architecture, safety, benchmarks, roadmap, prompt log
tests/            238-test suite + golden set
```

## Tech Stack

**Core:** Python 3.11+ · LangGraph 1.2.11 · Pydantic v2 · Click CLI
**AI:** Ollama (Qwen3.5-4B / 2B paraphraser, BGE-M3 1024-d multilingual embeddings) · LiteLLM (NVIDIA NIM / Groq / OpenRouter cloud pool)
**Data:** ChromaDB vector store · Jinja2 + xhtml2pdf report rendering · NumPy NDVI band math
**Service:** FastAPI · uvicorn

## Roadmap Status

**Phase 1 (Prompts 0–10) — COMPLETE and verified.** Hybrid RAG decision core, NDVI evidence engine, PMFBY report, hybrid cloud-local routing, and full documentation are shipped; all tests green. Phase 2 targets: vision-based disease diagnosis, live IMD/SHC/Sentinel-2 ingestion, WhatsApp/SMS/Bhashini IVR delivery hardening. See [docs/12-ROADMAP.md](docs/12-ROADMAP.md) and [docs/PROMPT-LOG.md](docs/PROMPT-LOG.md).

## License & PMFBY Boundary

> **PMFBY Boundary Disclaimer:** Advisory output only — not a substitute for professional agronomic or insurance advice. PMFBY reports are generated for the farmer's review and are **never submitted automatically**; submission is always a documented human action. KrishiSetu AI does not interface with PMFBY portals, banks, or insurers.

Licensed under the **MIT License**.
