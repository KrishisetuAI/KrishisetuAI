"""Escalation-service tests (prompt 5, plan Sec 8 case 11).

Covers ticket creation, sha256(query + plot_id + date_window) idempotency (no
double-ticket on replay), the neutral localized hand-off message, read_tickets,
and resolve_ticket. Uses a temp JSON store and an injectable clock.
"""
from __future__ import annotations

from datetime import datetime, timezone

from krishisethu.domain import TicketStatus
from krishisethu.escalation import EscalationService


class FakeClock:
    """Deterministic monotonically-advancing clock for ticket timestamps."""

    def __init__(self, start: datetime | None = None):
        self.t = start or datetime(2026, 9, 8, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        current = self.t
        self.t = self.t.replace(second=self.t.second + 1)  # advance 1s each tick
        return current


def test_dedup_key_sha256_deterministic():
    k1 = EscalationService.dedup_key("irrigation?", "sohna-plot-1", "2026-W36")
    k2 = EscalationService.dedup_key("irrigation?", "sohna-plot-1", "2026-W36")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex
    assert k1 != EscalationService.dedup_key("irrigation?", "sohna-plot-1", "2026-W37")
    assert k1 != EscalationService.dedup_key("irrigation?", "sohna-plot-2", "2026-W36")


def test_raise_ticket_creates_and_persists(tmp_path):
    svc = EscalationService(tmp_path / "tickets.json", now=FakeClock())
    ticket = svc.raise_ticket(
        query="sugarcane ratoon gap",
        plot_id="sohna-plot-1",
        date_window="2026-W37",
        context={"crop": "sugarcane"},
        cci=0.62,
        retrieved_chunks=({"source_id": "a::0", "score": 0.5},),
        validator_warnings=("confidence_gate",),
    )
    assert ticket.ticket_id.startswith("KVK-")
    assert ticket.status is TicketStatus.OPEN
    assert ticket.created_at is not None
    assert ticket.dedup_key == EscalationService.dedup_key("sugarcane ratoon gap", "sohna-plot-1", "2026-W37")
    # persisted one ticket
    assert len(svc.read_tickets()) == 1


def test_idempotency_no_double_ticket_on_replay(tmp_path):
    """Case 11: same (query, plot, week) escalated twice -> one ticket."""
    svc = EscalationService(tmp_path / "tickets.json", now=FakeClock())
    t1 = svc.raise_ticket(query="leaf spot?", plot_id="p9", date_window="2026-W36", cci=0.5)
    t2 = svc.raise_ticket(query="leaf spot?", plot_id="p9", date_window="2026-W36", cci=0.5)
    assert t1.ticket_id == t2.ticket_id, "replay must return the same ticket, not a duplicate"
    assert len(svc.read_tickets()) == 1


def test_distinct_plot_or_week_creates_distinct_tickets(tmp_path):
    svc = EscalationService(tmp_path / "tickets.json", now=FakeClock())
    svc.raise_ticket(query="q", plot_id="p1", date_window="W36")
    svc.raise_ticket(query="q", plot_id="p2", date_window="W36")  # different plot
    svc.raise_ticket(query="q", plot_id="p1", date_window="W37")  # different week
    assert len(svc.read_tickets()) == 3


def test_neutral_message_is_localized_handoff(tmp_path):
    svc = EscalationService(tmp_path / "tickets.json")
    assert "forwarded to a KVK agricultural expert" in svc.neutral_message
    assert svc.neutral_message_hindi.strip()
    assert svc.neutral_message != svc.neutral_message_hindi


def test_farmer_channel_does_not_get_decision_context(tmp_path):
    # The farmer channel gets only the neutral message; no advisory content leaks.
    svc = EscalationService(tmp_path / "tickets.json")
    ticket = svc.raise_ticket(
        query="how much chlorpyrifos per litre?", plot_id="p1", date_window="W36",
        context={"query": "how much chlorpyrifos per litre?"},
        validator_warnings=("chemical_gate",),
    )
    farmer_text = svc.neutral_message
    assert "chlorpyrifos" not in farmer_text.lower()
    assert ticket.query == "how much chlorpyrifos per litre?"  # context only on the ticket
    assert any("chemical_gate" in w for w in ticket.validator_warnings)


def test_resolve_ticket(tmp_path):
    svc = EscalationService(tmp_path / "tickets.json", now=FakeClock())
    ticket = svc.raise_ticket(query="q", plot_id="p1", date_window="W36")
    svc.resolve_ticket(ticket.ticket_id)
    [t] = svc.read_tickets()
    assert t.status is TicketStatus.RESOLVED
    assert t.resolved_at is not None


def test_read_tickets_admin_view(tmp_path):
    svc = EscalationService(tmp_path / "tickets.json")
    svc.raise_ticket(query="q", plot_id="p1", date_window="W36")
    assert len(svc.read_tickets()) == 1
    all_ = svc.raise_ticket(query="q", plot_id="p3", date_window="W36")
    assert len(svc.read_tickets()) == 2
    assert any(t.ticket_id == all_.ticket_id for t in svc.read_tickets())


def test_confidence_fail_produces_ticket_and_neutral_message(tmp_path):
    """Case 7 end-to-end: below-gate -> validator escalates -> ticket + neutral msg.

    The farmer channel receives only the neutral hand-off message; the full
    decision context (including the sub-threshold CCI) lives on the ticket.
    """
    from krishisethu.domain import AdvisoryDecision, ConfidenceScores, Resolution
    from krishisethu.safety.gates import SafetyValidator

    scores = ConfidenceScores(R=0.2, S_RAG=0.0, P_vision=0.0, has_image=False)
    decision = AdvisoryDecision(actions=(), resolution=Resolution.UNRESOLVED, provenance=())
    result = SafetyValidator().validate(scores, Resolution.UNRESOLVED, deterministic=False, raw="", decision=decision)
    assert result.escalate is True

    svc = EscalationService(tmp_path / "tickets.json", now=FakeClock())
    ticket = svc.raise_ticket(
        query="unusual wilting in zaid rice",
        plot_id="karnal-plot-7",
        date_window="2026-W36",
        context={"crop": "rice", "district": "karnal"},
        cci=0.14,
        validator_warnings=tuple(result.warnings),
    )
    # one ticket, populated with the warning + cci
    assert len(svc.read_tickets()) == 1
    assert ticket.cci is not None and ticket.cci < 0.85
    assert ticket.validator_warnings and ticket.validator_warnings[0].startswith("confidence_gate")
    # Farmer sees only the neutral hand-off message, never an advisory.
    assert "forwarded to a KVK agricultural expert" in svc.neutral_message
    assert "unusual wilting" not in svc.neutral_message
