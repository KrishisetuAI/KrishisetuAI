# KrishiSetu AI — Architecture (white paper)

KrishiSetu AI is a **safe-by-construction, 4-tier hybrid decision engine** that turns official DPI
data into a weekly, plot-specific action briefing. The defining property: the LLM never originates a
recommendation — it only paraphrases. Every answer is gated by a composite confidence score, and
below the gate we escalate to a KVK expert, never guess.

The four tiers:

## Tier 1 — Deterministic Rule Engine (`krishisethu/rules/`)
Pure Python, no model calls, no network. Weather-to-action matrices, spray windows, deficit
irrigation, and a sugarcane ratoon/gap planner. Returns `R(C)` in [0,1] (1.0 on exact resolution)
with a full rule trace. **Immune to hallucination by construction.** Adds a rule = a pure function +
a matrix row. Deterministic: identical inputs, identical outputs.

## Tier 2 — Canonical Knowledge Layer (`krishisethu/knowledge/`)
ICAR Package of Practices / MSP / spacing / scheme eligibility as Markdown + YAML frontmatter, loaded
by **exact metadata lookup** keyed on crop, state, season, valid_from/valid_to, source_authority.
This layer is NEVER chunked, embedded, or sent through a vector store. Includes validity windows plus
a supersession chain and authority-rank conflict resolution so expired or superseded facts are never
served. The source-traceability gate can cite every fact to its source_doc + valid_from.

## Tier 3 — Domain-Governed Vector RAG (`krishisethu/rag/`)
Long-tail, dynamic, region-specific content only: KVK Q&A, ICAR circulars, state advisories,
research papers. ChromaDB (local) or pgvector (hosted). Every chunk carries crop / district / season
/ source_type; retrieval is **metadata-filter-first, similarity-second**, capped at **top-k=5**. A
sugarcane query can never see a rice chunk. Returns `S_RAG` = max cosine to the confidence index.

## Tier 4 — LLM Explanation Layer (`krishisethu/renderer/`)
Quantized Qwen3.5-4B (Ollama) as a **paraphraser only**, Qwen3.5-2B fallback. Receives a strictly
structured prompt containing only Tier 1-3 output and a hard prohibition: any recommendation not
traceable to Tiers 1-3 is rejected. Emits schema-valid JSON (action items + evidence[]), not free
prose. If a fact is not in the provided material, it must not be stated.

## Orchestration (`krishisethu/pipeline/`)
A thin LangGraph state machine over typed Pydantic state: triage -> context assembler ->
Tiers 1/2/3 -> confidence -> Tier 4 render -> safety validate -> DELIVER or ESCALATE. Tiers 1 and 2
stay framework-free.

## Confidence + Safety (`krishisethu/confidence/`, `krishisethu/safety/`, `krishisethu/escalation/`)
Composite Confidence Index `C = 0.50*R + 0.30*S_RAG + 0.20*P_vision`, gate `0.85`. Three validator
gates (confidence, chemical-prescription strip, source-traceability). Below-gate and malicious
queries become a KVK escalation ticket. See [06-SAFETY-AND-CONFIDENCE.md](06-SAFETY-AND-CONFIDENCE.md).

## Deliveries
Mobile web (rich), WhatsApp/SMS (alerts + briefing summary), Bhashini IVR (feature phones), and the
PMFBY claim report — generated, never submitted.

## Architecture decisions (see docs/ADR/)
- Confidence gate 0.85 with w=(0.50, 0.30, 0.20).
- Tier 2 is never chunked/embedded (exact metadata lookup only).
- Tier 4 is paraphraser-only behind a source-traceability contract.
- ChromaDB first (local), pgvector adapter behind a VectorStore protocol when shared.
- PMFBY generate-never-submit enforced as code.

See [plan.md](../plan.md) for the build plan and God-tier prompts that flesh this out tier by tier.
