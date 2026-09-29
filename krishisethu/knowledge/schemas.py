"""Pydantic schema for a Tier-2 canonical knowledge document.

A canonical doc is a Markdown body + typed YAML frontmatter that encodes a single,
stable, compliance-critical fact (ICAR PoP norm, MSP rate, spacing, scheme
eligibility). It is NEVER chunked or embedded. Validation here is the gate that
keeps the corpus schema-clean before it can be served by the exact-metadata
lookup (krishisethu.knowledge.loader).

Supersession note: the white paper only records valid_from; we add ``valid_to``
(valid_until), a ``supersedes`` doc id, and authority-rank conflict resolution so
an expired or superseded fact is never served.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator

from krishisethu.domain import Crop, DecisionType, Season, State

#: Optional backward-compatible alias for the supersession field in frontmatter.
VALID_UNTIL_ALIASES = ("valid_to", "valid_until")


class CanonicalDoc(BaseModel):
    """A parsed canonical document: validated frontmatter + verbatim Markdown body."""

    model_config = ConfigDict(extra="forbid")

    # Stable identifier (derived from the file stem at load time).
    doc_id: str = ""
    crop: Crop
    state: State
    district: str | None = None
    season: Season
    valid_from: date
    valid_until: date | None = None
    source_authority: str
    source_doc: str
    source_url: str | None = None
    schema_version: str = "2.0"
    version: int = 1
    decision_type: DecisionType
    rule_id: str | None = None
    language: str = "en"
    # Extra correctness field the white paper omits: this doc replaces another id.
    supersedes: str | None = None
    # The Markdown body after the YAML frontmatter.
    body: str = ""

    @field_validator("version")
    @classmethod
    def _version_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("version must be >= 1")
        return v

    @field_validator("supersedes")
    @classmethod
    def _not_self(cls, v: str | None, info) -> str | None:
        # A doc must not supersede itself.
        if v is not None and v == info.data.get("doc_id"):
            raise ValueError("doc cannot supersede itself")
        return v

    def is_valid_on(self, on_date: date) -> bool:
        """True when ``valid_from <= on_date < valid_until`` (open-ended if no valid_until)."""
        if on_date < self.valid_from:
            return False
        if self.valid_until is not None and on_date >= self.valid_until:
            return False
        return True

    @property
    def provenance(self) -> tuple[str, str, date, int]:
        """(source_doc, source_authority, valid_from, version) for citations."""
        return (self.source_doc, self.source_authority, self.valid_from, self.version)
