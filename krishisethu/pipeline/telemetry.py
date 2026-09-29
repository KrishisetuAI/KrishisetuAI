"""Structured telemetry for the decision pipeline (Prompt 7 step 5).

Every node records its wall latency into ``CompileState.latency_ms``; the
pipeline emits one structured log line per node with the shared ``trace_id``,
and a final ``pipeline.complete`` line carrying the gate outcome, CCI and
end-to-end latency. ``emit_*`` helpers keep the log schema in one place so a
collector (stdout JSON / ELK / Supabase outbox later) has a stable shape.
"""
from __future__ import annotations

import json
import logging
import time
from statistics import median
from typing import Any

logger = logging.getLogger("krishisethu.pipeline")

#: Node names in execution order — used for the benchmark report.
TIER_NODE_ORDER = (
    "triage",
    "context_assembler",
    "tier1",
    "tier2",
    "tier3",
    "confidence",
    "render",
    "validate",
    "deliver",
    "escalate",
)


def emit(trace_id: str, event: str, **fields: Any) -> None:
    """One structured log line: trace_id + event + free-form fields."""
    payload = {"trace_id": trace_id, "event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, default=str))


def latency_report(latency_ms: dict[str, float], e2e_ms: float) -> dict[str, float]:
    """Total latency (s) by node over the recorded node times."""
    return {name: latency_ms.get(name, 0.0) for name in TIER_NODE_ORDER}


def e2e_ms(latency_ms: dict[str, float]) -> float:
    """End-to-end wall time: sum of the per-node latencies actually recorded."""
    return round(float(sum(v for v in latency_ms.values() if v >= 0)), 3)


def median_ms(samples: list[float]) -> float:
    """Median of a list of per-run end-to-end millisecond samples."""
    return round(float(median(samples or [0.0])), 3)


def now_ms() -> float:
    """Monotonic wall clock in ms (node-local, not injectable by design)."""
    return time.perf_counter() * 1000.0
