# KrishiSetu AI — Safety & Confidence

The trust backbone. Everything below is enforced in code and tested, not a policy statement.

## Composite Confidence Index

```
CCI(C,Q) = w1*R(C) + w2*S_RAG(Q,D) + w3*P_vision
  R        = Tier-1 rule-match indicator: 1.0 exact, normalized partial (0,1), 0.0 unresolved
  S_RAG    = max cosine similarity between query embedding and top-k retrieved chunks (0.0 if skipped)
  P_vision = vision softmax of top class (0.0 when no image)
  weights (image)   = 0.50, 0.30, 0.20
  weights (no image)= 0.625, 0.375, 0.00  (w3 redistributed proportionally, preserving the 5/3 ratio; sums to 1.0)
Gate: deliver iff CCI >= 0.85. If CCI < 0.85 -> withhold + KVK escalation ticket, never answer.
Short-circuit: an EXACT Tier-1 resolution with zero probabilistic components delivers directly
(still through the chemical + traceability gates) — a deterministic answer is never spuriously escalated.
```

## Safety Validator — three gates (ordered)
1. **Confidence gate** — if not deterministic and CCI < 0.85, withhold and escalate.
2. **Chemical-prescription gate** — if output pairs an active ingredient with a numeric dose, strip
   the dosage, substitute a region-verified input-class recommendation, add a consult-expert line,
   and escalate regardless of CCI. Never emit a specific dosage.
3. **Source-traceability gate** — every claim must cite a source in the traceable set (Tier-1 rule
   id | Tier-2 doc id | Tier-3 chunk id). Strip untraceable claims; if all are stripped, escalate.

## Escalation
A below-gate or malicious result becomes an `EscalationTicket` (query, context, component scores,
retrieved chunks, warnings, status). Deduplicated by sha256(query + plot + week). The farmer channel
receives only the neutral localized hand-off message ("forwarded to a KVK expert"); full context goes
to the officer. Persist locally now; Supabase (RLS) later.

## Operating numbers (targets, reproduced on our own test set)
- 0% chemical-dosage hallucination on the malicious suite.
- Abstention (escalate, never guess) on ambiguous / out-of-distribution input.
- Every delivered advisory passes source-traceability.

## Implementation notes
- CCI is pure and structural: nothing in the LLM layer computes or overrides it.
- The structured advisory schema has NO numeric-dosage field, so dosage is impossible structurally.
- See the mandatory test cases in [plan.md](../plan.md) section 8. The eval harness lives in
  `tests/golden_set/` and runs on CI with a stubbed LLM (four adversary categories: ICAR-answerable,
  ambiguous/OOD, malicious-dosage, cross-crop contamination).
