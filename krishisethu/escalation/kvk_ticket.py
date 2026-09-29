"""KVK escalation-ticket stub (prompt 5).

``CREATE_ESCALATION_TICKET(Q, C, CCI, {D_j})`` + ``NOTIFY_KVK_OFFICER(T)``.
An :class:`EscalationService` creates and raises a ticket, deduplicates it
idempotently by sha256(query + plot_id + date_window) so a replayed briefing does
not double-ticket, persists it to a JSON store (SQLite / Supabase-outbox later),
returns only the localized neutral hand-off message to the farmer channel, and
exposes ``read_tickets()`` for the admin dashboard.

The farmer channel receives only the neutral message (English + Hindi); the full
decision context travels on the ticket to the officer.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from krishisethu.domain import EscalationTicket, TicketStatus
from krishisethu.safety.gates import NEUTRAL_ESCAPE_EN, NEUTRAL_ESCAPE_HI


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


class EscalationService:
    """Create / dedupe / persist KVK escalation tickets against a JSON store."""

    def __init__(self, store_path: str | Path = "./data/escalations.json", now=None):
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        #: Injectable clock so tests are deterministic.
        self._now = now or _default_now

    @staticmethod
    def dedup_key(query: str, plot_id: str, date_window: str) -> str:
        """sha256(query + plot_id + date_window) — the idempotency key.

        A replayed briefing for the same plot in the same week resolves to the
        same key, no matter how many times the gate fires.
        """
        raw = f"{query}|{plot_id}|{date_window}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -- persistence -----------------------------------------------------
    def _load(self) -> list[EscalationTicket]:
        if not self._store_path.exists():
            return []
        try:
            records = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [EscalationTicket(**r) for r in records]

    def _save(self, tickets: list[EscalationTicket]) -> None:
        payload = [t.model_dump(mode="json") for t in tickets]
        self._store_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- operations ------------------------------------------------------
    def raise_ticket(
        self,
        *,
        query: str,
        plot_id: str,
        date_window: str,
        context: dict | None = None,
        cci: float | None = None,
        retrieved_chunks: tuple[dict, ...] = (),
        validator_warnings: tuple[str, ...] = (),
    ) -> EscalationTicket:
        """Create (or return the existing) ticket for this query/plot/week.

        Idempotent: if a ticket with the same dedup key already exists it is
        returned unchanged — a replay never creates a duplicate.
        """
        key = self.dedup_key(query, plot_id, date_window)
        existing = [t for t in self._load() if t.dedup_key == key]
        if existing:
            return existing[0]

        ticket = EscalationTicket(
            ticket_id=f"KVK-{key[:12].upper()}",
            query=query,
            context=dict(context or {}),
            cci=cci,
            retrieved_chunks=tuple(retrieved_chunks),
            validator_warnings=tuple(validator_warnings),
            status=TicketStatus.OPEN,
            created_at=self._now(),
            dedup_key=key,
        )
        self._save(self._load() + [ticket])
        return ticket

    def read_tickets(self) -> list[EscalationTicket]:
        """Admin-dashboard view of every persisted ticket (open + resolved)."""
        return self._load()

    def resolve_ticket(self, ticket_id: str) -> None:
        """Mark a ticket RESOLVED (sets ``resolved_at``). No-op if unknown/already done."""
        tickets = self._load()
        changed = False
        for t in tickets:
            if t.ticket_id == ticket_id and t.status is not TicketStatus.RESOLVED:
                t.status = TicketStatus.RESOLVED
                t.resolved_at = self._now()
                changed = True
        if changed:
            self._save(tickets)

    # -- farmer-channel contract ----------------------------------------
    @property
    def neutral_message(self) -> str:
        """The only text the farmer channel sees on escalation (English)."""
        return NEUTRAL_ESCAPE_EN

    @property
    def neutral_message_hindi(self) -> str:
        """The neutral hand-off message in Hindi (Bhashini/IVR-ready)."""
        return NEUTRAL_ESCAPE_HI
