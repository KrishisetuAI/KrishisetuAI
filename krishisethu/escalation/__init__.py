"""KVK escalation (prompt 5).

CRATE_ESCALATION_TICKET + NOTIFY_KVK_OFFICER against a local JSON store (SQLite /
Supabase outbox later). The farmer channel gets the neutral hand-off message; the
officer gets the full ticket.

Public API:
  EscalationService
"""
from __future__ import annotations

from krishisethu.escalation.kvk_ticket import EscalationService

__all__ = ["EscalationService"]
