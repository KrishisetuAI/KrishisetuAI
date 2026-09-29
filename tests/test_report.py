"""Tests for PMFBY report generation (Bonus Phase)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from krishisethu.pipeline import CompileState, ReportGenerator, OfflineQueue
from krishisethu.domain import Resolution
from krishisethu.rules import RuleAction
from krishisethu.rules.rule_match import RuleMatch


def test_report_generator_creates_pdf(tmp_path):
    """Generate a PDF from a minimal state and check it exists."""
    from krishisethu.rules.rule_match import RuleMatch
    from krishisethu.domain import Resolution
    
    state = CompileState(
        query="Test query",
        plot_id="test-plot",
        demo=False,
        intent="weather",
        rule_actions=[
            RuleAction(trace="TEST:RULE", action="Test action", clause="TEST", priority=0)
        ],
        rule_match=RuleMatch(resolution=Resolution.EXACT, score=1.0, matched_clauses=1, total_clauses=1, trace_codes=("TEST:RULE",)),
        final_output="Test advisory",
        latency_ms={"triage": 1.2, "render": 3.4},
    )
    gen = ReportGenerator(output_dir=str(tmp_path))
    out_path = tmp_path / "report.pdf"
    pdf_path = gen.generate(state, out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_offline_queue(tmp_path):
    """Test queue add/pop."""
    queue_path = tmp_path / "queue.json"
    queue = OfflineQueue(str(queue_path))
    state_dict = {"query": "test", "plot_id": "p1"}
    queue.add(state_dict)
    queue.add({"query": "test2", "plot_id": "p2"})

    items = queue.pop_all()
    assert len(items) == 2
    assert items[0]["state"]["query"] == "test"
    assert items[1]["state"]["query"] == "test2"

    # Queue should be empty after pop
    assert queue.pop_all() == []