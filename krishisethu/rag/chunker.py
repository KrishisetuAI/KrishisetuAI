"""Markdown-heading-aware semantic chunker + advisory dedup (Tier 3).

Splits a normalized Markdown advisory into self-contained chunks of ~150-250
tokens with 30-50 token overlap, keying separators on headings and numbered
points first. Hard guarantee: a table block is a single atomic unit and is
never split across chunks. Every chunk carries the full inherited metadata plus
``chunk_index``, ``ntokens`` and the ``char_start``/``char_end`` offsets into
the source body.

``dedupe_docs`` collapses near-identical advisories repeated across districts
BEFORE chunking, so the same KVK circular pasted for Sohna / Nuh / Palwal is
indexed once rather than three times.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import tiktoken

from krishisethu.rag.schema import RagDoc

#: Chunk-budget pointers (plan Sec 5.3): target 150-250 tokens, 30-50 overlap.
TARGET_MIN_TOKENS = 150
TARGET_MAX_TOKENS = 250
OVERLAP_TOKENS = 40

#: Near-dup threshold for cross-district advisory collapse.
DEDUP_SIMILARITY = 0.95

#: Base packer fills to max_target - overlap so the actual chunk (overlap added
#: back) stays within the 150-250 pointer.
_FILL_TOKENS = TARGET_MAX_TOKENS - OVERLAP_TOKENS

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[0-9]+[.)]|[-*+]+)\s+\S")

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@dataclass(frozen=True)
class _Unit:
    """One atomic splitting unit — a heading, a whole table block, a list run
    or a paragraph. Never split; a table row can never be cut in two."""

    kind: str  # heading | table | list | para
    text: str
    start: int
    end: int
    tokens: int


def _classify(line: str) -> str:
    stripped = line.lstrip()
    if _HEADING_RE.match(line):
        return "heading"
    if stripped.startswith("|"):
        return "table"
    if _LIST_RE.match(stripped):
        return "list"
    if not stripped:
        return "blank"
    return "para"


def _atomic_units(text: str) -> list[_Unit]:
    """Split ``text`` into atomic units, preserving char offsets."""
    units: list[_Unit] = []
    kind: str | None = None
    buf: list[tuple[str, int]] = []
    offset = 0

    def flush() -> None:
        nonlocal buf, kind
        if not buf:
            return
        start = buf[0][1]
        end = buf[-1][1] + len(buf[-1][0])
        body = "".join(l for l, _ in buf)
        units.append(_Unit(kind, body, start, end, _token_count(body)))
        buf = []
        kind = None

    for line in text.splitlines(keepends=True):
        c = _classify(line)
        if c == "blank":
            flush()
            offset += len(line)
            continue
        if kind is None:
            kind = c
            buf = [(line, offset)]
        elif c == kind:
            buf.append((line, offset))
        else:
            flush()
            kind = c
            buf = [(line, offset)]
        offset += len(line)
    flush()
    return units


def _pack(units: list[_Unit], fill: int, max_target: int, min_tokens: int) -> list[list[_Unit]]:
    """Greedy window pack that never splits a unit; prefer heading boundaries.

    An atomic unit larger than the hard cap (e.g. one huge table) stands alone —
    it cannot be split, so it is allowed to exceed the target. All other chunks
    stay within ``max_target``.
    """
    base: list[list[_Unit]] = []
    i, n = 0, len(units)
    while i < n:
        u = units[i]
        if u.tokens > max_target:
            base.append([u])
            i += 1
            continue
        end, tokens = i, 0
        while end < n:
            un = units[end]
            if un.tokens > max_target:
                break  # oversized unit can't be packed; leave for next iteration
            if un.kind == "heading" and tokens >= min_tokens:
                break
            if tokens + un.tokens > fill and tokens >= min_tokens:
                break
            if tokens + un.tokens > max_target:
                break  # hard cap on accumulated content
            tokens += un.tokens
            end += 1
        if end == i:
            end = i + 1
        base.append(units[i:end])
        i = end
    return base


def chunk_markdown(
    text: str,
    base_metadata: dict | None = None,
    min_target: int = TARGET_MIN_TOKENS,
    max_target: int = TARGET_MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list["Chunk"]:
    """Split a markdown body into self-contained, overlap-bearing chunks.

    ``base_metadata`` is inherited verbatim into every chunk and then augmented
    with ``chunk_index``, ``ntokens``, ``char_start`` and ``char_end``.
    """
    fill = max(1, max_target - overlap)
    units = _atomic_units(text)
    if not units:
        return []
    base = _pack(units, fill, max_target, min_target)

    chunks: list[Chunk] = []
    for idx, base_units in enumerate(base):
        base_tokens = sum(u.tokens for u in base_units)
        pre = _overlap_prefix(base[idx - 1], base_tokens, max_target, overlap) if idx > 0 and base_tokens <= max_target else []
        sel = pre + base_units
        body = "".join(u.text for u in sel)
        meta = dict(base_metadata or {})
        meta.update(
            chunk_index=idx,
            ntokens=_token_count(body),
            char_start=sel[0].start,
            char_end=sel[-1].end,
        )
        chunks.append(Chunk(text=body, start=sel[0].start, end=sel[-1].end, ntokens=meta["ntokens"], chunk_index=idx, metadata=meta))
    return chunks


def _overlap_prefix(prev: list[_Unit], base_tokens: int, max_target: int, overlap: int) -> list[_Unit]:
    """Tail of the previous chunk to carry as overlap, never breaking the cap.

    Only whole units that fit within ``overlap`` tokens are recycled; a unit
    larger than the overlap window (e.g. a whole table) is never duplicated into
    the next chunk. The prefix is trimmed from the front so the final chunk never
    exceeds ``max_target``.
    """
    collected: list[_Unit] = []
    ov = 0
    for u in reversed(prev):
        if u.tokens > overlap:
            break
        collected.append(u)
        ov += u.tokens
        if ov >= overlap:
            break
    collected.reverse()
    pre_tokens = sum(u.tokens for u in collected)
    while collected and pre_tokens + base_tokens > max_target:
        dropped = collected.pop(0)
        pre_tokens -= dropped.tokens
    return collected


@dataclass
class Chunk:
    """A self-contained chunk with its provenance metadata."""

    text: str
    start: int
    end: int
    ntokens: int = 0
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip markdown/table punctuation."""
    t = re.sub(r"[|#*`>_\-{}\[\]()]", " ", text)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def dedupe_docs(docs: list[RagDoc], threshold: float = DEDUP_SIMILARITY) -> list[RagDoc]:
    """Collapse near-identical advisories repeated across districts.

    The first occurrence is the representative; any later doc whose normalized
    body is >= ``threshold`` similar to a kept representative is dropped. Order
    is preserved.
    """
    kept: list[RagDoc] = []
    kept_norms: list[str] = []
    for doc in docs:
        norm = _normalize(doc.body)
        if any(_similarity(norm, prev) >= threshold for prev in kept_norms):
            continue
        kept.append(doc)
        kept_norms.append(norm)
    return kept
