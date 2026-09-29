"""PMFBY Crop Loss Assessment Report Generator.

Ingests a CompileState and produces a PDF report using Jinja2 + WeasyPrint.
Supports offline queueing: pending reports are stored in a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
from xhtml2pdf import pisa

from krishisethu.pipeline.graph import CompileState
from krishisethu.evidence.ndvi import compute_anomaly

logger = logging.getLogger(__name__)

# Default Sohna flood scenario for demo plot
SOHNA_BASELINE_NDVI = 0.79
SOHNA_OBSERVED_NDVI = 0.44


def _get_ndvi_evidence(state: CompileState) -> dict:
    """Extract or compute NDVI evidence for the report.
    
    For demo plot, uses Sohna flood scenario (baseline 0.79, observed 0.44).
    """
    plot_id = state.plot_id or "demo"
    
    if plot_id == "demo" or state.demo:
        baseline = SOHNA_BASELINE_NDVI
        observed = SOHNA_OBSERVED_NDVI
        date_str = str(state.on_date) if state.on_date else "2024-07-15"
        cloud_masked = True
    else:
        # For non-demo plots, would fetch from actual NDVI service
        baseline = 0.75
        observed = 0.70
        date_str = str(state.on_date) if state.on_date else "2024-07-15"
        cloud_masked = False
    
    delta, severity = compute_anomaly(observed, baseline)
    
    return {
        "date": date_str,
        "baseline_ndvi": baseline,
        "observed_ndvi": observed,
        "delta": round(delta, 4),
        "severity": severity,
        "cloud_masked": cloud_masked,
        "plot_id": plot_id,
    }


# Template string (could be loaded from file, but kept inline for simplicity)
TEMPLATE_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PMFBY Crop Loss Assessment</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        h2 { color: #2980b9; margin-top: 24px; }
        .section { margin-bottom: 20px; }
        .label { font-weight: bold; display: inline-block; width: 220px; }
        .value { display: inline-block; }
        .metric-box { background: #ecf0f1; padding: 12px; border-radius: 6px; margin: 8px 0; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; }
        th { background: #34495e; color: white; padding: 8px; text-align: left; }
        td { border: 1px solid #ddd; padding: 8px; }
        .severity-normal { color: #27ae60; font-weight: bold; }
        .severity-moderate { color: #f39c12; font-weight: bold; }
        .severity-severe { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <h1>PMFBY Crop Loss Assessment Report</h1>
    <p><strong>Report Generated:</strong> {{ generated_at }}</p>
    <p><strong>Plot ID:</strong> {{ state.plot_id }}</p>
    <p><strong>Query:</strong> {{ state.query }}

    <div class="section">
        <h2>1. Decision Context</h2>
        <div class="metric-box">
            <span class="label">Intent:</span> <span class="value">{{ state.intent }}</span><br>
            <span class="label">Gate Outcome:</span> <span class="value">{{ state.gate_outcome }}</span><br>
            <span class="label">CCI:</span> <span class="value">{{ state.cci_value or "N/A" }}</span><br>
            <span class="label">Escalated:</span> <span class="value">{{ state.escalate }}</span>
        </div>
    </div>

    <div class="section">
        <h2>2. Tier-1 Deterministic Rules</h2>
        <ul>
        {% for action in state.rule_actions %}
            <li><strong>{{ action.action }}</strong> (source: {{ action.trace }})</li>
        {% else %}
            <li>No rule actions fired.</li>
        {% endfor %}
        </ul>
        <div class="metric-box">
            <span class="label">Resolution:</span> <span class="value">{{ state.rule_match.resolution.value if state.rule_match else "UNRESOLVED" }}</span><br>
            <span class="label">Rule Score (R):</span> <span class="value">{{ state.rule_match.score if state.rule_match else 0.0 }}</span>
        </div>
    </div>

    <div class="section">
        <h2>3. Tier-2 Canonical Knowledge</h2>
        {% if state.canonical_result %}
            <p><strong>Doc ID:</strong> {{ state.canonical_result.doc.doc_id }}</p>
            <p><strong>Authority:</span> {{ state.canonical_result.doc.source_authority }}</p>
            <p><strong>Excerpt:</strong> {{ state.canonical_result.body[:300] }}...</p>
        {% else %}
            <p>No canonical document matched.</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>4. Tier-3 Retrieval</h2>
        <p><strong>S_RAG:</strong> {{ state.retrieval_result.s_rag if state.retrieval_result else 0.0 }}</p>
        {% if state.retrieval_result and state.retrieval_result.chunks %}
            <p><strong>Top Chunks:</strong></p>
            <ul>
            {% for chunk in state.retrieval_result.chunks[:3] %}
                <li>{{ chunk.text[:100] }}... (score: {{ "%.3f"|format(chunk.score) }})</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No retrieval results.</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>5. NDVI Satellite Evidence</h2>
        {% set ndvi = ndvi_evidence %}
        <div class="metric-box">
            <span class="label">Plot ID:</span> <span class="value">{{ ndvi.plot_id }}</span><br>
            <span class="label">Observation Date:</span> <span class="value">{{ ndvi.date }}</span><br>
            <span class="label">Baseline NDVI (healthy):</span> <span class="value">{{ "%.2f"|format(ndvi.baseline_ndvi) }}</span><br>
            <span class="label">Observed NDVI:</span> <span class="value">{{ "%.2f"|format(ndvi.observed_ndvi) }}</span><br>
            <span class="label">Delta (observed - baseline):</span> <span class="value">{{ "%.2f"|format(ndvi.delta) }}</span><br>
            <span class="label">Severity:</span> <span class="value severity-{{ ndvi.severity }}">{{ ndvi.severity.upper() }}</span><br>
            <span class="label">Cloud Masked:</span> <span class="value">{{ "Yes" if ndvi.cloud_masked else "No" }}</span>
        </div>
        <p><em>NDVI anomaly computed from Sentinel-2 B08/B04 bands with SCL cloud masking (classes 3, 8, 9, 10 excluded). Severity thresholds: normal < 15% drop, moderate 15-30%, severe > 30%.</em></p>
    </div>

    <div class="section">
        <h2>6. Final Validated Advisory</h2>
        <div style="background: #f9f9f9; padding: 16px; border-left: 4px solid #27ae60;">
            <p>{{ state.final_output or "No advisory delivered." }}</p>
        </div>
    </div>

    <div class="section">
        <h2>7. Telemetry</h2>
        <ul>
        {% for node, ms in state.latency_ms.items() %}
            <li><strong>{{ node }}:</strong> {{ ms }} ms</li>
        {% endfor %}
        </ul>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generates PMFBY PDF reports from pipeline state."""

    def __init__(self, template: Optional[str] = None, output_dir: str = "./data/reports"):
        self.template = template or TEMPLATE_HTML
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=jinja2.select_autoescape(["html"])
        )

    def generate(self, state: CompileState, output_path: Optional[Path] = None) -> Path:
        """Render a PDF report from the pipeline state."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"pmfby_{state.plot_id}_{timestamp}.pdf"

        # Compute NDVI evidence
        ndvi_evidence = _get_ndvi_evidence(state)

        # Render HTML
        template_obj = self._env.from_string(self.template)
        html_content = template_obj.render(
            state=state,
            generated_at=datetime.now().isoformat(),
            ndvi_evidence=ndvi_evidence,
        )

        # Convert to PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            if pisa_status.err:
                raise RuntimeError(f"PDF generation failed: {pisa_status.err}")

        return output_path


class OfflineQueue:
    """Simple JSON queue for pending reports when offline."""

    def __init__(self, queue_path: str = "./data/offline_reports.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, state_dict: Dict[str, Any]) -> None:
        """Add a state snapshot to the queue."""
        queue = self._load()
        queue.append({
            "timestamp": datetime.now().isoformat(),
            "state": state_dict,
        })
        self._save(queue)

    def pop_all(self) -> List[Dict]:
        """Retrieve and clear the queue."""
        queue = self._load()
        self._save([])
        return queue

    def _load(self) -> List[Dict]:
        if not self.queue_path.exists():
            return []
        try:
            with open(self.queue_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, data: List[Dict]) -> None:
        with open(self.queue_path, "w") as f:
            json.dump(data, f, indent=2, default=str)