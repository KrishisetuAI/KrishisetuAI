"""Deterministic exact-metadata lookup for Tier-2 canonical docs.

No LangChain, no vector store, no embeddings. Docs are indexed once by
(crop, state, season, decision_type); the loader resolves a query by:

  1. validity window  (valid_from <= on_date < valid_until)
  2. supersession     (a doc whose id is another doc's ``supersedes`` is dropped)
  3. authority rank   (highest wins; ties broken by newest valid_from, then version)

Returns the winner's body verbatim plus a provenance tuple for the
source-traceability gate. Expired or superseded docs are never returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from krishisethu.domain import Crop, DecisionType, Season, State
from krishisethu.knowledge.schemas import CanonicalDoc

# Known authorities -> rank (higher = more authoritative for conflict resolution).
AUTHORITY_RANK: dict[str, int] = {
    "ICAR - Indian Institute of Sugarcane Research (ICAR-IISR), Lucknow": 100,
    "ICAR - Indian Institute of Sugarcane Research, Lucknow": 100,
    "ICAR - Indian Institute of Wheat Research (ICAR-IIWBR), Karnal": 90,
    "ICAR - National Rice Research Institute (ICAR-NRRI), Cuttack": 90,
    "ICAR": 80,
    "Government of India, Ministry of Agriculture and Farmers Welfare": 70,
    "Commission for Agricultural Costs and Prices (CACP)": 60,
}
DEFAULT_AUTHORITY_RANK = 10


def authority_rank(authority: str) -> int:
    return AUTHORITY_RANK.get(authority, DEFAULT_AUTHORITY_RANK)


@dataclass
class CanonicalResult:
    """Winner doc + verbatim body + provenance for the traceability gate."""

    doc: CanonicalDoc
    body: str
    provenance: tuple[str, str, date, int]


def parse_frontmatter(text: str) -> tuple[str, str]:
    """Split a markdown doc into (yaml_block, body). Raises ValueError if malformed."""
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


def doc_from_file(path: Path) -> CanonicalDoc:
    """Parse a canonical markdown file into a validated CanonicalDoc."""
    text = path.read_text(encoding="utf-8")
    yaml_block, body = parse_frontmatter(text)
    data = yaml.safe_load(yaml_block) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")

    # Accept valid_to as an alias for valid_until.
    if "valid_to" in data and "valid_until" not in data:
        data["valid_until"] = data.pop("valid_to")

    doc_id = str(data.pop("doc_id", path.stem))
    doc = CanonicalDoc(doc_id=doc_id, body=body, **data)
    # doc_id may have been provided in frontmatter with a comma etc.; ensure non-empty.
    if not doc.doc_id:
        doc.doc_id = path.stem
    return doc


class CanonicalLoader:
    """Index + deterministic lookup over a canonical corpus root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.docs: list[CanonicalDoc] = []
        self._index: dict[tuple[object, object, object, object], list[CanonicalDoc]] = {}
        self._by_id: dict[str, CanonicalDoc] = {}
        self.load()

    def load(self) -> None:
        self.docs = []
        self._index = {}
        self._by_id = {}
        for p in sorted(self.root.rglob("*.md")):
            doc = doc_from_file(p)
            self.docs.append(doc)
            self._by_id[doc.doc_id] = doc
            key = (doc.crop, doc.state, doc.season, doc.decision_type)
            self._index.setdefault(key, []).append(doc)

    def _candidates(
        self, crop: Crop, state: State, season: Season, decision_type: DecisionType
    ) -> list[CanonicalDoc]:
        return self._index.get((crop, state, season, decision_type), [])
    # -- lookup ------------------------------------------------------------
    def read_canonical(
        self,
        crop: Crop | str,
        state: State | str,
        season: Season | str,
        decision_type: DecisionType | str,
        on_date: date,
    ) -> CanonicalResult | None:
        crop = crop.value if isinstance(crop, Crop) else Crop(crop)
        state = state.value if isinstance(state, State) else State(state)
        season = season.value if isinstance(season, Season) else Season(season)
        decision_type = (
            decision_type.value if isinstance(decision_type, DecisionType) else DecisionType(decision_type)
        )

        valid = [
            d
            for d in self._candidates(crop, state, season, decision_type)
            if d.is_valid_on(on_date)
        ]
        if not valid:
            return None

        # Any doc superseded by a doc that is itself VALID on the query date is dropped.
        # (A future-but-not-yet-effective superseder must not erase historical reads.)
        superseded = {
            d.supersedes
            for d in self._candidates(crop, state, season, decision_type)
            if d.supersedes and d.is_valid_on(on_date)
        }
        surviving = [d for d in valid if d.doc_id not in superseded]
        if not surviving:
            # Every valid doc is superseded; never serve a superseded document.
            return None

        winner = max(
            surviving,
            key=lambda d: (authority_rank(d.source_authority), d.valid_from, d.version),
        )
        return CanonicalResult(doc=winner, body=winner.body, provenance=winner.provenance)
