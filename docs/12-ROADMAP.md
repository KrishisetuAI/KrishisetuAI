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
| 10 | Prompt 10 | SIH 2026 Final Documentation & Demo Command Pack | DONE |

**Phase 1 (Core 4-Tier Deterministic/RAG Engine) — COMPLETED.**

## After the core

The later models, each its own prompted phase: vision diagnosis, NDVI evidence,
PMFBY claim report, and multi-channel delivery (mobile web / WhatsApp-SMS /
Bhashini IVR). NDVI evidence (PROMPT-08) and the PMFBY report generator
already shipped with the core build.

| Post-core phase | Scope | Status |
|---|---|---|
| 2 | Multi-Channel Voice Telephony — Twilio + Bhashini IVR (PROMPT-11) + REST advise endpoint (PROMPT-12) | **COMPLETED** — voice and REST answers flow only through the safety-gated 4-tier pipeline; WhatsApp/SMS + admin dashboard pending |
| 3 | Computer Vision Triage (edge vision feeding P_vision) | **ACTIVE / NEXT** |
| — | Field pilot, DPI integration & hardening | PENDING |

---

**Phase 1 Build — COMPLETE.**

**Phase 2 (Multi-Channel Voice Telephony / Twilio + Bhashini Integration) — COMPLETED (2026-10-03, PROMPT-11; REST endpoint + documentation sync in PROMPT-12):**

- TwiML webhook routing: `POST /ivr/incoming` (Hindi speech gather, hi-IN) and
  `POST /ivr/gather` (recognized speech -> full 4-tier pipeline via the shared
  `krishisethu/pipeline/factory.py`; no channel can build a shortcut around the gates).
- Bhashini async STT/TTS service (`krishisethu/api/services/bhashini.py`): injectable,
  `httpx.MockTransport`-testable, `BhashiniError` instead of bare exceptions.
- Per-call WAV isolation (`data/ivr_audio/{CallSid}.wav`) with CallSid-regex
  path-traversal guarding and `Cache-Control: no-store` serving.
- Strict escalation speech lock: escalations speak only the hardcoded neutral Hindi KVK
  hand-off; a hermetic regression test proves a poisoned advisory never reaches TTS.
- `POST /api/v1/advise` (PROMPT-12): the REST channel on the identical gated pipeline;
  escalations never serialize the withheld advisory to the client.
- Verification: full suite 249 passed + 1 skipped at IVR integration; 255 passed + 1
  skipped after PROMPT-12; deterministic-core coverage 100% (Tier-1 rules, CCI, safety
  gates, escalation).

**Next up: Phase 3 — Computer Vision Triage (MobileNetV3-Small TFLite INT8 feeding the P_vision term of the CCI).**
