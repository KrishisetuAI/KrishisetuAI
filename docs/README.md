# KrishiSetu AI — Documentation Index

This docs folder is the living technical record of the project. Read it in the order below when you
start a session.

**Reading order for new contributors:** start with [01-ARCHITECTURE.md](01-ARCHITECTURE.md) (the white
paper) and [06-SAFETY-AND-CONFIDENCE.md](06-SAFETY-AND-CONFIDENCE.md), then [SETUP.md](../SETUP.md),
then the index below.

| File | When to read |
|---|---|
| [01-ARCHITECTURE.md](01-ARCHITECTURE.md) | The 4-tier safe-by-construction white paper. READ FIRST, every session. |
| [02-TIER1-RULES.md](02-TIER1-RULES.md) | The deterministic rule engine contract (pure Python). |
| [03-TIER2-KNOWLEDGE.md](03-TIER2-KNOWLEDGE.md) | The canonical knowledge layer + frontmatter schema. |
| [04-TIER3-RAG.md](04-TIER3-RAG.md) | Vector RAG: chunk tagging, pre-filter, top-k=5. |
| [05-TIER4-LLM.md](05-TIER4-LLM.md) | The paraphraser-only renderer + prompt contract. |
| [06-SAFETY-AND-CONFIDENCE.md](06-SAFETY-AND-CONFIDENCE.md) | Composite Confidence Index + 3 safety gates. READ EARLY. |
| [07-DATA-DICTIONARY.md](07-DATA-DICTIONARY.md) | Every DPI data source: fields, units, cadence, access. |
| [08-INTERFACES.md](08-INTERFACES.md) | Inter-tier + HTTP API contracts. |
| [09-CHANNELS.md](09-CHANNELS.md) | Multi-channel delivery (web / WhatsApp / IVR). |
| [10-PMFBY-REPORT.md](10-PMFBY-REPORT.md) | The generate-never-submit claim report. |
| [11-BENCHMARKS.md](11-BENCHMARKS.md) | Reproducible safety + latency numbers. |
| [12-ROADMAP.md](12-ROADMAP.md) | Future build targets and status. |
| [13-DEVELOPER-ENV.md](13-DEVELOPER-ENV.md) | Windows setup + known gotchas. |
| [PROMPT-LOG.md](PROMPT-LOG.md) | Append-only per-prompt engineering log. |

## Per-prompt log template

Every completed God-tier build prompt appends one `PROMPT-<NN>` entry to `docs/PROMPT-LOG.md` using
the template below. This is how future sessions stay consistent with what was built, the constraints
honoured, and the numbers achieved.

```markdown
## PROMPT-<NN> — <short feature title> — <date>
Status: DONE | PARTIAL | BLOCKED — <one line>
### 1. Scope (what this prompt was asked to build)
### 2. Features built / changed
- <verb> <thing> — <one-line effect>
### 3. Files touched
- `src/krishisetu/.../x.py` — <what changed and why>
### 4. Constraints honoured (tick every applicable)
- [ ] Tier1 pure/deterministic; Tier2 exact-lookup, not chunked; Tier3 pre-filter + top-k<=5;
      Tier4 paraphraser only; CCI 0.85 gate; no dosage; traceability; PMFBY never submitted
### 5. Interfaces / contracts
### 6. Acceptance evidence
### 7. Numbers / benchmarks (deltas)
### 8. Open items / known risks / next
```
