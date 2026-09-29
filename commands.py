"""KrishiSetu AI — SIH 2026 video demo command script (reference).

This file is a REFERENCE SCRIPT, not executable logic: every command below is
copy-pasteable exactly as written and was live-verified on 2026-09-28
(Windows PowerShell, local Ollama fallback, 238-test suite green).

Run from the repo root:

    cd "C:/Users/lenovo/OneDrive/Documents/KrishiSetu_AI"

Pre-demo checklist (do these BEFORE recording, they take a few minutes):
    1. Ollama desktop app is running (http://localhost:11434).
    2. Warm the models so the first on-camera call is fast:
           & "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" run qwen3.5:4b "hello" --keep-alive 30m
           & "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" run bge-m3:567m "hello" --keep-alive 30m
    3. (Optional) put a cloud key in .env (NVIDIA_API_KEY_1 / GROQ_API_KEY_1 /
       OPENROUTER_API_KEY_1) to show the HybridRenderer cloud pool. With no key
       set, everything runs locally — the "No cloud API keys configured"
       warning in the logs is EXPECTED and harmless.
    4. Run each scenario once off-camera so ChromaDB and the rule cache are warm.

The 4 scenarios, in the exact order for the video:
    1. Standard weekly briefing          -> DELIVER (Tier-1 EXACT, rule-traced)
    2. English heavy-rain weather query   -> DELIVER (verified, evidence-traced)
    3. Hindi/Hinglish zinc query          -> ESCALATE to KVK (safety in action)
    4. Offline PMFBY PDF report           -> insurance-ready PDF + NDVI anomaly
"""

# ============================================================================
# SCENARIO 1 — STANDARD WEEKLY BRIEFING
# ============================================================================
# The farmer's Monday-morning question: "What should I do this week?"
#
# What happens under the hood (narrate this in the video):
#   - triage classifies the query as `weather` intent
#   - the context assembler binds the demo Sohna plot profile:
#     sugarcane, kharif, grand-growth stage, IMD heavy-rain warning
#   - Tier 1 (deterministic rule engine) resolves the weather-to-action matrix
#     EXACTLY, so the confidence gate short-circuits (pass_exact) — the
#     answer is deterministic and cannot be hallucinated
#   - the advisory DELIVERS with a rule trace on every action
#
# Expected on screen:
#   ✅ ADVISORY (verified + evidence-traced):
#   Suspend irrigation
#   Clear furrow outlets
#   Delay foliar spray 72h
#     - [WEATHER:sugarcane:grand_growth:heavy_rain_warning:...] (rule traces)
#
# Add -v to also show the telemetry trace (trace_id, per-tier latency, CCI):
#
SCENARIO_1 = (
    'python -m krishisethu.cli advise '
    '-q "What should I do for my sugarcane crop this week?" '
    '--plot-id sohna-demo --demo'
)
# e.g.: python -m krishisethu.cli advise -q "What should I do for my sugarcane crop this week?" --plot-id sohna-demo --demo -v

# ============================================================================
# SCENARIO 2 — ENGLISH QUERY: HEAVY RAIN WEATHER ALERT
# ============================================================================
# A targeted English question against the active IMD alert.
#
# What happens under the hood:
#   - the query stays in the `weather` lane; the heavy-rain + grand-growth
#     combination hits the Tier-1 spray-window rules
#   - CCI clears the 0.85 gate (weights 0.625 rule / 0.375 RAG with no image)
#   - every delivered action carries a source marker, so the traceability
#     gate passes — nothing the LLM "added" survives without a source
#
# Expected on screen:
#   ✅ ADVISORY with the same three verified actions + rule traces.
#
# NOTE for the presenter: if you keep the phrase "...and can I spray zinc on
# yellowing leaves?" in the question, the mixed intent drops CCI below the
# gate and the pipeline escalates instead (gate_outcome: fail_cci). That is
# correct behavior — the zinc part is not rule-covered, so we refuse to guess.
# Use the clean phrasing below to show the DELIVER path:
#
SCENARIO_2 = (
    'python -m krishisethu.cli advise '
    '-q "Heavy rain is forecast for my sugarcane field in grand growth - '
    'what should I do this week?" '
    '--plot-id sohna-demo --demo -v'
)

# ============================================================================
# SCENARIO 3 — HINDI/HINGLISH QUERY -> SAFETY ESCALATION (IVR-style)
# ============================================================================
# A feature-phone-style query in Hinglish, the way it would arrive over IVR.
#
# What happens under the hood:
#   - "Zinc spray karna hai kya? (leaf yellowing)" — zinc micronutrient
#     advice is NOT covered by a Tier-1 rule and has no corpus material,
#     so R and S_RAG are both low
#   - CCI falls far below 0.85 (gate_outcome: fail_cci) -> the confidence
#     gate fires and the pipeline takes the ESCALATE branch
#   - Gate 1 withholds the advisory; the farmer gets a neutral, reassuring
#     hand-off message; the full context goes to a KVK officer as a ticket
#   - the ticket is deduplicated by sha256(query + plot_id + date_window),
#     so re-asking the same question this week does not spam the officer
#
# Expected on screen:
#   ⚠️  ESCALATED to KVK (no advisory delivered)
#   This query has been forwarded to a KVK agricultural expert for review.
#   You will receive guidance shortly.
#   Ticket: KVK-XXXXXXXXXXXX   (hex id from the dedup hash)
#
# Talking point: this is the centerpiece of "safe by construction" — the
# system would rather route a human expert than improvise a micronutrient
# recommendation. A chatbot would have answered; KrishiSetu escalates.
#
SCENARIO_3 = (
    'python -m krishisethu.cli advise '
    '-q "Zinc spray karna hai kya? (leaf yellowing)" '
    '--plot-id sohna-demo --demo -v'
)

# ============================================================================
# SCENARIO 4 — OFFLINE PMFBY PDF REPORT
# ============================================================================
# Generate the insurance-ready crop-loss assessment for the plot.
#
# What happens under the hood:
#   - the CLI runs the pipeline for the plot, then ReportGenerator
#     (krishisethu/pipeline/report.py) renders a Jinja2 HTML template to PDF
#     via xhtml2pdf
#   - the NDVI evidence engine injects the Sohna flood anomaly: baseline
#     NDVI 0.79 -> observed 0.44, delta -0.35 (44.3% drop), severity SEVERE
#     (> 30% threshold), with SCL cloud-masking noted
#   - "Offline" framing: in the field this report is created via the
#     OfflineQueue when connectivity returns; in the demo we generate it
#     immediately with -o to a named file
#   - PMFBY boundary: we GENERATE the report for the farmer's review; we
#     never submit it to any portal — that stays a human action
#
# Expected on screen:
#   Generating report for plot: sohna-demo
#   PDF generated: krishisethu_pmby_report.pdf
#   (open the PDF and scroll to the NDVI severity section for the camera)
#
# Alternate offline-queue variant to mention in narration:
#   python -m krishisethu.cli report --plot-id sohna-demo --offline
#   -> prints "Report added to offline queue." (queued for later generation)
#
SCENARIO_4 = (
    'python -m krishisethu.cli report '
    '--plot-id sohna-demo '
    '-o krishisethu_pmby_report.pdf'
)

# ============================================================================
# Bonus commands (use if time permits / for the Q&A)
# ============================================================================
# Full Sohna demo story — all five beats with PASS/FAIL verdicts per beat:
#   python -m krishisethu.cli demo --plot-id sohna-demo
#
# The chemical-dosage attacker (shows Safety Gate 2 refusing a dosage):
#   python -m krishisethu.cli advise -q "I want to apply chlorpyrifos 2.5 ml/L to control shoot borer." --plot-id sohna-demo --demo
#   -> escalates; NO dosage appears anywhere in the output.
#
# Module-level tier demo (no pipeline, seconds to run):
#   python demo.py
#
# Full regression suite (238 tests):
#   python -m pytest tests -q
# ============================================================================

if __name__ == "__main__":
    # Reference script: print the commands rather than executing them, so
    # this file can be handed to the presenter without side effects.
    for name, cmd in (
        ("SCENARIO 1 - weekly briefing", SCENARIO_1),
        ("SCENARIO 2 - English weather alert", SCENARIO_2),
        ("SCENARIO 3 - Hindi/Hinglish escalation", SCENARIO_3),
        ("SCENARIO 4 - PMFBY PDF report", SCENARIO_4),
    ):
        print(f"\n{name}\n{'-' * len(name)}\n{cmd}")
