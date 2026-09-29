# KrishiSetu AI — Benchmarks

Measured on our own test set (controlled lab numbers), not vendor claims. Updated
after each god-tier prompt that produces a measurable number.

## NDVI Evidence Engine (Prompt 8)

| Metric | Value | Notes |
|--------|-------|-------|
| NDVI bounds | [-1.0, 1.0] | mathematically guaranteed by (B08-B04)/(B08+B04) |
| Division-by-zero safety | 0% crashes | `np.errstate` + `np.where` guard |
| SCL cloud classes masked | 4/12 | 3=shadow, 8=medium, 9=high, 10=cirrus |
| Sohna flood anomaly (baseline 0.79 → obs 0.44) | delta -0.35, **severe** | drop = 44.3% > 30% threshold |
| Severity boundaries | 15% / 30% | normal < 15%, moderate 15-30%, severe > 30% |
| Test coverage | 7/7 pass | bounds, div0, masking, anomaly, boundaries, zero-baseline, dataclass |

## Safety & confidence (Prompt 5)

| Metric | Baseline | Now | Delta / target |
|--------|----------|-----|----------------|
| CCI weights (no image) | — | (0.625, 0.375, 0.00) | sums to 1.0, 5/3 ratio kept |
| CCI weights (image) | — | (0.50, 0.30, 0.20) | sums to 1.0 |
| Gate threshold | — | 0.85 | deliver iff `CCI >= 0.85` |
| Exact-resolution short-circuit | — | delivers for EXACT | deterministic answer never spuriously escalated |
| Chem-dosage hallucination (malicious suite) | n/a | **0%** | target 0% (structural + term-level) |
| OOD / ambiguous abstention | n/a | **100%** | abstain (escalate), never low-confidence delivery |
| Golden-set verdicts | n/a | DELIVER 3 / ESCALATE 5 | 8 fixtures, 4 adversary categories |
| Escalation idempotency | n/a | 1 ticket on replay | sha256(query + plot + week) |
| Safety-suite coverage | 0 | 97% | `confidence/safety/escalation/domain` |

## Tier-3 retrieval (Prompt 4)

| Metric | Baseline | Now | Delta |
|--------|----------|-----|-------|
| Median retrieval latency | n/a | **199 ms** | first call ~5.5 s (model load) |
| S_RAG (max cosine) | none | 0.0 on empty; 0.7264 on the ratoon gap-fill query | w2 of the CCI |
| Seed corpus docs | 0 | 9 (8 indexed after dedup) | +9 |
| Seed chunks indexed | 0 | 10 | +10 |
| Collections | 0 | `crop_sugarcane_kharif` (8), `crop_rice_kharif` (1), `crop_wheat_rabi` (1) | +3 |
| Tier-3 coverage | 0 | 96% | chunker/retriever/store/embedder/ingest |

## Regression totals

- **Full suite: 223 tests pass** across `rules` (Tier 1), `knowledge` (Tier 2),
  `rag` (Tier 3), `confidence`/`safety`/`escalation`, `ndvi` (Prompt 8), golden set, pipeline, hybrid router, and report.
- New safety modules coverage: **97%**; composite index **100%**; NDVI engine **100%**.

Notes: numbers are measured on our own seed corpus and golden set; the real live
corpus and the Tier-4 renderer (Prompt 6) will land in later prompts. The 3.8 s
end-to-end latency target is measured once the LangGraph pipeline (Prompt 7) exists.
