"""Tests for the Tier-2 canonical knowledge layer (Prompt 2)."""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

import krishisethu.knowledge as knowledge_pkg
from krishisethu.knowledge.loader import CanonicalLoader, doc_from_file
from krishisethu.knowledge.manifest import register, rebuild_manifest
from krishisethu.knowledge.schemas import CanonicalDoc
from krishisethu.knowledge.validator import validate_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "knowledge" / "canonical"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "canonical"


# --- loader: exact lookup + validity + supersession -----------------------
def test_exact_lookup_returns_doc_with_provenance():
    loader = CanonicalLoader(CANONICAL)
    res = loader.read_canonical("sugarcane", "haryana", "rabi", "irrigation_rule", date(2026, 9, 8))
    assert res is not None
    assert res.doc.rule_id == "SUGAR-IRR-014"
    assert res.doc.source_authority.startswith("ICAR")
    assert "75 mm" in res.body  # body returned verbatim
    sd, auth, vf, ver = res.provenance
    assert sd and auth and vf and ver >= 1


def test_never_serve_expired_doc():
    loader = CanonicalLoader(FIXTURES)
    # expired doc valid_until 2023-12-31 -> on 2026 it must not be served
    res = loader.read_canonical("sugarcane", "haryana", "rabi", "irrigation_rule", date(2026, 9, 8))
    assert res is None


def test_superseded_resolves_to_newer_doc():
    loader = CanonicalLoader(CANONICAL)
    # After v2.valid_from (2024), v1 (2022 edition) is superseded -> serve v2.
    res_2024 = loader.read_canonical("sugarcane", "haryana", "rabi", "spacing_norm", date(2024, 6, 1))
    assert res_2024 is not None
    assert res_2024.doc.doc_id == "haryana_rabi_spacing_v2"
    assert "revised" in res_2024.body
    # Before v2.valid_from, only v1 is valid -> serve v1.
    res_2023 = loader.read_canonical("sugarcane", "haryana", "rabi", "spacing_norm", date(2023, 6, 1))
    assert res_2023 is not None
    assert res_2023.doc.doc_id == "haryana_rabi_spacing_v1"


def test_determinism():
    loader = CanonicalLoader(CANONICAL)
    a = loader.read_canonical("sugarcane", "haryana", "rabi", "irrigation_rule", date(2026, 9, 8))
    b = loader.read_canonical("sugarcane", "haryana", "rabi", "irrigation_rule", date(2026, 9, 8))
    assert a.doc.doc_id == b.doc.doc_id
    assert a.body == b.body
    assert a.provenance == b.provenance


# --- schema validation ----------------------------------------------------
def test_schema_rejects_missing_required_key(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text(
        "---\nstate: haryana\nseason: rabi\nvalid_from: '2024-01-01'\n"
        "source_authority: X\nsource_doc: Y\ndecision_type: spacing_norm\n---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises((ValidationError, ValueError)):
        doc_from_file(p)


def test_schema_rejects_bad_enum(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text(
        "---\ncrop: banana\nstate: haryana\nseason: rabi\nvalid_from: '2024-01-01'\n"
        "source_authority: X\nsource_doc: Y\ndecision_type: spacing_norm\n---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        doc_from_file(p)


def test_schema_accepts_valid_doc(tmp_path):
    p = tmp_path / "ok.md"
    p.write_text(
        "---\ncrop: sugarcane\nstate: haryana\nseason: rabi\nvalid_from: '2024-01-01'\n"
        "source_authority: 'ICAR - Indian Institute of Sugarcane Research (ICAR-IISR), Lucknow'\n"
        "source_doc: D\ndecision_type: spacing_norm\n---\nbody\n",
        encoding="utf-8",
    )
    doc = doc_from_file(p)
    assert doc.crop.value == "sugarcane"
    assert doc.decision_type.value == "spacing_norm"


# --- validator / linter ---------------------------------------------------
def test_validator_clean_on_seed_corpus():
    assert validate_corpus(CANONICAL) == []


def test_validator_flags_expired():
    issues = validate_corpus(FIXTURES)
    assert any("expired" in i for i in issues)


def test_validator_flags_duplicate_key(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ncrop: sugarcane\nstate: haryana\nseason: rabi\nvalid_from: '2024-01-01'\n"
        "source_authority: 'ICAR - Indian Institute of Sugarcane Research (ICAR-IISR), Lucknow'\n"
        "source_doc: A\ndecision_type: spacing_norm\n---\nbody\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "---\ncrop: sugarcane\nstate: haryana\nseason: rabi\nvalid_from: '2024-01-01'\n"
        "source_authority: 'ICAR - Indian Institute of Sugarcane Research (ICAR-IISR), Lucknow'\n"
        "source_doc: B\ndecision_type: spacing_norm\n---\nbody\n",
        encoding="utf-8",
    )
    issues = validate_corpus(tmp_path)
    assert any("duplicate key" in i for i in issues)


# --- manifest -------------------------------------------------------------
def test_register_and_rebuild_manifest(tmp_path):
    doc_path = CANONICAL / "haryana_rabi_irrigation_grand_growth.md"
    doc = doc_from_file(doc_path)
    manifest = tmp_path / "manifest.csv"
    row = register(doc, doc_path, manifest)
    assert row["id"] == "haryana_rabi_irrigation_grand_growth"
    assert row["source_type"] == "irrigation_rule"
    assert len(row["sha256"]) == 64

    rebuilt = rebuild_manifest(CANONICAL, manifest)
    assert rebuilt.exists()
    text = rebuilt.read_text(encoding="utf-8")
    assert "haryana_rabi_spacing_v2" in text
    assert text.splitlines()[0] == "id,path,source_type,crop,district,season,valid_from,valid_until,sha256"


# --- architecture assertion: no vector/LLM/embedding in this tier ---------
def test_knowledge_tier_has_no_vector_or_llm_imports():
    knowledge_dir = Path(knowledge_pkg.__file__).parent
    forbidden_prefixes = (
        "langchain", "chromadb", "qdrant", "faiss",
        "sentence_transformers", "transformers", "torch",
    )
    for py in sorted(knowledge_dir.rglob("*.py")):
        src = py.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"{py.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith(forbidden_prefixes), f"{py.name} imports from {mod}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name.lower()
                assert "embed" not in name and "chunk" not in name, (
                    f"{py.name} defines function {node.name} (embedding/chunking not allowed in Tier 2)"
                )
