"""Pydantic schema for a Tier-3 RAG corpus document.

A RAG doc is Markdown + typed YAML frontmatter that encodes one long-tail,
region-specific advisory (KVK Q&A, ICAR circular, state advisory, research
paper excerpt). It is chunked + embedded — unlike a Tier-2 canonical doc, which
is never chunked.

Validation here is the admission gate: crop / season / source_type must resolve
through their closed enums (Crop, Season, SourceType), district is a required
short string, and the whole record is rejected (never auto-fixed) on violation.
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator

from krishisethu.domain import Crop, Season, SourceType, State

#: Allowed district shapes: lowercase / mixed alnum, spaces, hyphens, dots.
_DISTRICT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .\-]{0,63}$")
#: Stable micro-id: lowercase alnum, underscores, hyphens, dots.
_SOURCE_ID_RE = re.compile(r"^[a-z0-9_][a-z0-9_.\-]{0,127}$")


class RagDoc(BaseModel):
    """A validated corpus document: frontmatter facts + verbatim Markdown body."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    crop: Crop
    state: State
    district: str
    season: Season
    source_type: SourceType
    source_authority: str = ""
    source_url: str | None = None
    valid_from: date | None = None
    language: str = "en"
    # The Markdown body after the YAML frontmatter.
    body: str = ""

    @field_validator("district")
    @classmethod
    def _district_clean(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("district is required for every Tier-3 chunk")
        if not _DISTRICT_RE.match(s):
            raise ValueError(f"district {v!r} is not a plausible district name")
        return s

    @field_validator("source_id")
    @classmethod
    def _source_id_clean(cls, v: str) -> str:
        if not _SOURCE_ID_RE.match(v):
            raise ValueError(f"source_id {v!r} must be a lowercase slug")
        return v

    @field_validator("title")
    @classmethod
    def _title_clean(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("title is required")
        return str(v).strip()

    @property
    def collection_name(self) -> str:
        """ChromaDB collection that this doc's chunks land in."""
        return crop_season_collection(self.crop, self.season)

    def provenance(self) -> str:
        return f"{self.source_type.value}:{self.source_id}"


def crop_season_collection(crop: Crop | str, season: Season | str) -> str:
    """``crop_<crop>_<season>`` — the collection a query and a doc both resolve to."""
    crop_e = crop if isinstance(crop, Crop) else Crop(crop)
    season_e = season if isinstance(season, Season) else Season(season)
    return f"crop_{crop_e.value}_{season_e.value}"


def parse_frontmatter(text: str) -> tuple[str, str]:
    """Split a markdown document into (yaml_block, body). Raises ValueError if malformed."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening '---' frontmatter delimiter")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("missing closing '---' frontmatter delimiter")
    yaml_block = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    return yaml_block, body


def rag_doc_from_text(text: str) -> RagDoc:
    """Parse + validate one corpus document from its raw text."""
    import yaml

    yaml_block, body = parse_frontmatter(text)
    data = yaml.safe_load(yaml_block) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return RagDoc(body=body, **data)


def rag_doc_from_file(path: str | "Path") -> RagDoc:
    """Parse + validate one corpus document from disk."""
    from pathlib import Path

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return rag_doc_from_text(text)
