"""Corpus manifest registrar for the canonical knowledge layer.

A manifest row records (id, path, source_type, crop, district, season, valid_from,
valid_until, sha256) so every doc is traceable and reprocessable. ``register``
upserts a single doc; ``rebuild_manifest`` reseans a corpus and rewrites the CSV.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from krishisethu.knowledge.schemas import CanonicalDoc

FIELDS = [
    "id",
    "path",
    "source_type",
    "crop",
    "district",
    "season",
    "valid_from",
    "valid_until",
    "sha256",
]


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_for(doc: CanonicalDoc, path: Path) -> dict[str, str]:
    return {
        "id": doc.doc_id,
        "path": str(path),
        "source_type": doc.decision_type.value,
        "crop": doc.crop.value,
        "district": doc.district or "",
        "season": doc.season.value,
        "valid_from": doc.valid_from.isoformat(),
        "valid_until": doc.valid_until.isoformat() if doc.valid_until else "",
        "sha256": sha256_of_file(path),
    }


def _read_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def _write_manifest(manifest_path: Path, rows: dict[str, dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for _id in sorted(rows):
            writer.writerow(rows[_id])


def register(doc: CanonicalDoc, path: str | Path, manifest_path: str | Path = "corpus/manifest.csv") -> dict[str, str]:
    """Upsert a single doc into the manifest and return its row."""
    mp = Path(manifest_path)
    rows = _read_manifest(mp)
    rows[doc.doc_id] = _row_for(doc, Path(path))
    _write_manifest(mp, rows)
    return rows[doc.doc_id]


def rebuild_manifest(
    corpus_root: str | Path, manifest_path: str | Path = "corpus/manifest.csv"
) -> Path:
    """Rescan a canonical corpus root and rewrite the manifest CSV. Returns the manifest path."""
    from krishisethu.knowledge.loader import doc_from_file  # local import to avoid cycle

    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, str]] = {}
    for p in sorted(Path(corpus_root).rglob("*.md")):
        doc = doc_from_file(p)
        rows[doc.doc_id] = _row_for(doc, p)
    _write_manifest(mp, rows)
    return mp
