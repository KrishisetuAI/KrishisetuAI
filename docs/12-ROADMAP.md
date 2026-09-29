# KrishiSetu AI — Roadmap

The first deliverable is the **hybrid RAG decision core** (Phases 0-6). Each phase
is executed as a god-tier prompt (see [plan.md](../plan.md) section 7) and logged in
[PROMPT-LOG.md](PROMPT-LOG.md). Vision, NDVI, claim report, and multi-channel
delivery come after the core is verified.

## First deliverable — hybrid RAG decision core

| Phase | Prompt | Tier / scope | Status |
|-------|--------|--------------|--------|
| 0 | Prompt 0 | Environment, package scaffold, config | DONE |
| 1 | Prompt 1 | Domain enums, config, FastAPI skeleton | DONE |
| 2 | Prompt 2 | Tier-2 canonical knowledge (exact lookup) | DONE |
| 3 | Prompt 3 | Tier-1 deterministic rule engine + R(C) | DONE |
| 4 | Prompt 4 | Tier-3 domain-governed vector RAG + S_RAG | DONE |
| 5 | Prompt 5 | CCI + three-gate safety validator + escalation + golden set | DONE |
| 6 | Prompt 6 | Tier-4 LLM paraphraser (renderer) | DONE |
| 7 | Prompt 7 | LangGraph integration pipeline + CLI demo | DONE |
| 8 | Prompt 8 | NDVI Evidence Engine & PMFBY Report Integration | DONE |
| 9 | Prompt 9 | Expert-Grade Root README + Documentation Sync | DONE |

## After the core

The later models, each its own prompted phase: vision diagnosis, NDVI evidence,
PMFBY claim report, and multi-channel delivery (mobile web / WhatsApp-SMS /
Bhashini IVR).

---

**Phase 1 Build — COMPLETE.** All 223 tests passing. Ready for Phase 2.