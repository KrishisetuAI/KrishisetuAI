"""CLI linter for the Tier-2 canonical corpus.

Usage:
    python -m krishisethu.knowledge.validator [corpus_root]

Exits non-zero on any violation: malformed/missing frontmatter, schema errors,
duplicate (crop, state, season, decision_type, valid_from) keys, expired docs
(valid_until < today), and unknown/stale source_authority.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import yaml

from krishisethu.knowledge.loader import (
    DEFAULT_AUTHORITY_RANK,
    authority_rank,
    doc_from_file,
    parse_frontmatter,
)

REQUIRED_KEYS = [
    "crop",
    "state",
    "season",
    "valid_from",
    "source_authority",
    "source_doc",
    "decision_type",
]


def validate_corpus(root: Path) -> list[str]:
    """Return a list of violation strings; empty list means clean."""
    issues: list[str] = []
    dup_groups: dict[tuple[str, str, str, str, str], list[Path]] = {}

    for p in sorted(root.rglob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
            yaml_block, _ = parse_frontmatter(text)
            data = yaml.safe_load(yaml_block) or {}
        except Exception as e:  # noqa: BLE001
            issues.append(f"{p}: cannot parse frontmatter: {e}")
            continue

        if not isinstance(data, dict):
            issues.append(f"{p}: frontmatter is not a YAML mapping")
            continue

        for key in REQUIRED_KEYS:
            if key not in data:
                issues.append(f"{p}: missing required key '{key}'")

        try:
            doc = doc_from_file(p)
        except Exception as e:  # noqa: BLE001
            issues.append(f"{p}: schema validation failed: {e}")
            continue

        dup = (
            doc.crop.value,
            doc.state.value,
            doc.season.value,
            doc.decision_type.value,
            doc.valid_from.isoformat(),
        )
        dup_groups.setdefault(dup, []).append(p)

        if doc.valid_until is not None and doc.valid_until < date.today():
            issues.append(f"{p}: expired (valid_until {doc.valid_until.isoformat()})")

        if authority_rank(doc.source_authority) <= DEFAULT_AUTHORITY_RANK:
            issues.append(f"{p}: unknown/stale source_authority '{doc.source_authority}'")

    for dup, paths in dup_groups.items():
        if len(paths) > 1:
            issues.append(f"duplicate key {dup}: {', '.join(str(x) for x in paths)}")

    return issues


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else "knowledge/canonical"
    issues = validate_corpus(Path(root))
    for issue in issues:
        print(f"  [VIOLATION] {issue}", file=sys.stderr)
    if issues:
        print(f"FAIL: {len(issues)} violation(s) in {root}", file=sys.stderr)
        return 1
    print(f"OK: canonical corpus at {root} is clean ({len(list(Path(root).rglob('*.md')))} doc(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
