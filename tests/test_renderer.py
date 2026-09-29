"""Tier-4 renderer tests (prompt 6).

The renderer is a PARAPHRASER ONLY with a strict JSON output contract. Tests are
hermetic: ``ChatOllama`` is monkeypatched with a scriptable fake, so the whole
suite runs on CI with no Ollama daemon (one live gated test asserts real-model
latency + schema-validity when Ollama and qwen are reachable).

Coverage of the acceptance criteria:
  - adversarial/garbage context -> only traceable claims, always schema-valid
  - a prompt that tries to introduce a dosage fails schema OR is caught by matching
  - 100% schema-valid JSON on the golden set with bounded retries
  - fallback to the 2b model when the 4b model is mocked unavailable
  - no prose leakage on the reject path
  - the no-dosage-field invariant (structural)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from krishisethu.domain import ActionItem as DomainActionItem
from krishisethu.domain import Resolution, Tier
from krishisethu.renderer import (
    LANG_EN,
    LANG_HI,
    Advisory,
    ConfidenceBreakdown,
    ExcerptSource,
    Fragment,
    OllamaRenderer,
    PromptPackage,
    RendererValidationError,
    build_prompt,
    localize,
    parse_advisory,
    to_domain_decision,
    translate_text,
    validate_advisory,
)
from krishisethu.safety.gates import SafetyValidator

REPO = Path(__file__).resolve().parents[1]
sys_path_golden = str(REPO / "tests" / "golden_set")

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
T1_SUSPEND = ("WEATHER:sugarcane:grand_growth:heavy_rain_warning:SUSPEND_IRRIGATION", "Suspend irrigation")
T1_CLEAR = ("WEATHER:sugarcane:grand_growth:heavy_rain_warning:CLEAR_FURROW", "Clear furrow outlets")


def _actions(*pairs) -> list[DomainActionItem]:
    return [
        DomainActionItem(text=text, source_id=marker, tier=Tier.TIER1, clause=marker.split(":")[-1])
        for marker, text in pairs
    ]


def _breakdown(r=1.0, s=0.72, p=0.0, image=False, cci=None) -> ConfidenceBreakdown:
    if cci is None:
        cci = round(0.625 * r + 0.375 * s, 4) if not image else round(0.5 * r + 0.3 * s + 0.2 * p, 4)
    return ConfidenceBreakdown(R=r, S_RAG=s, P_vision=p, has_image=image, cci=cci)


def _valid_json(pkg: PromptPackage, texts=None, extra_field: str | None = None) -> str:
    """A schema-conformant model echo for ``pkg`` (all allowed markers cited)."""
    markers = sorted(pkg.allowed_markers, key=lambda m: (m[0].value, m[1]))
    actions = []
    for i, (tier, source) in enumerate(markers):
        action = {
            "text": (texts[i] if texts else f"Paraphrased action {i}"),
            "input_class": None,
            "consult_expert": False,
            "evidence": [{"source_id": source, "tier": tier.value}],
        }
        if extra_field:
            action[extra_field] = "2.5 ml/L chlorpyrifos"  # attempt an injection
        actions.append(action)
    bd = pkg.expected_breakdown
    return json.dumps(
        {
            "actions": actions,
            "confidence_breakdown": {
                "R": bd.R, "S_RAG": bd.S_RAG, "P_vision": bd.P_vision,
                "has_image": bd.has_image, "cci": bd.cci,
            },
        }
    )


def _renderer(**kw) -> OllamaRenderer:
    kw.setdefault("model", "qwen3.5:4b")
    kw.setdefault("fallback_model", "qwen3.5:2b")
    return OllamaRenderer(base_url="http://fake:11434", **kw)


@pytest.fixture
def fake_ollama(monkeypatch):
    """Scriptable ChatOllama stand-in: behavior[model](attempt) -> content/raise."""
    state: dict = {"script": {}, "instances": []}

    class FakeChat:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.kwargs = kwargs
            self.calls = 0
            state["instances"].append(self)

        def invoke(self, messages):
            self.calls += 1
            behavior = state["script"].get(self.model)
            if behavior is None:
                raise ConnectionError(f"model {self.model} unavailable")
            out = behavior(self.calls)
            if isinstance(out, BaseException):
                raise out
            return type("AIMessage", (), {"content": out})()

    monkeypatch.setattr("krishisethu.renderer.ollama_client.ChatOllama", FakeChat)
    return state


@pytest.fixture
def package() -> PromptPackage:
    return build_prompt(
        _actions(T1_SUSPEND, T1_CLEAR),
        canonical_excerpt=ExcerptSource("canonical:haryana_sugar_irrigation", "Trigger irrigation ... 75 mm."),
        topk_fragments=[Fragment(source_id="kvk_sohna_ratoon::0", chunk_index=0, text="Gap-fill when below 20%.")],
        query="Sugarcane heavy rain warning — what should I do?",
        breakdown=_breakdown(),
    )


# --------------------------------------------------------------------------
# prompt contract
# --------------------------------------------------------------------------
def test_build_prompt_carries_only_allowed_material(package):
    assert "PARAPHRASER ONLY" in package.system
    assert "do not add any agronomic content" in package.system.lower()
    assert "source marker" in package.system.lower()
    assert "if a fact is not in the provided material, do not state it" in package.system.lower()
    assert (Tier.TIER1, T1_SUSPEND[0]) in package.allowed_markers
    assert (Tier.TIER2, "canonical:haryana_sugar_irrigation") in package.allowed_markers
    assert (Tier.TIER3, "kvk_sohna_ratoon::0") in package.allowed_markers
    assert len(package.allowed_markers) == 4
    # every allowed marker appears in the user material as [tier:marker]
    for tier, marker in package.allowed_markers:
        assert f"[{tier.value}:{marker}]" in package.user
    # the confidence facts are stated as echoes, never as computed values
    assert "echo these exact numbers" in package.user
    assert f"CCI = {package.expected_breakdown.cci}" in package.user


def test_build_prompt_no_material_still_schema_capable():
    pkg = build_prompt([], breakdown=_breakdown(r=0.0, s=0.0, cci=0.0))
    assert pkg.allowed_markers == frozenset()
    assert "no material provided" in pkg.user


def test_build_prompt_rule_action_inputs():
    from krishisethu.rules import RuleAction

    acts = [RuleAction(trace="WEATHER:SUGAR:HEAVY_RAIN:SUSPEND", action="Suspend irrigation", clause="SUSPEND")]
    pkg = build_prompt(acts, breakdown=_breakdown())
    assert (Tier.TIER1, "WEATHER:SUGAR:HEAVY_RAIN:SUSPEND") in pkg.allowed_markers
    assert "Suspend irrigation" in pkg.user


def test_to_messages_roles(package):
    msgs = package.to_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == package.system
    assert msgs[1]["content"] == package.user


# --------------------------------------------------------------------------
# structured output: no-dosage-field invariant + schema strictness
# --------------------------------------------------------------------------
def test_no_dosage_field_invariant_structural():
    """The renderer schema has no numeric-dosage field — dosage is impossible."""
    import krishisethu.renderer.structured_output as so

    for model_cls in (so.Advisory, so.ActionItem, so.EvidenceRef, so.ConfidenceBreakdown):
        fields = model_cls.model_fields
        for name in ("dosage", "dose", "rate_ml", "quantity"):
            assert name not in fields, f"{model_cls.__name__} must not define a {name!r} field"
    # bonus: the mapping target (domain ActionItem) also has none.
    assert "dosage" not in DomainActionItem.model_fields


def test_parse_advisory_strips_json_fence():
    raw = '```json\n{"actions": [], "confidence_breakdown": {"R": 0.0, "S_RAG": 0.0, "P_vision": 0.0, "has_image": false, "cci": 0.0}}\n```'
    adv = parse_advisory(raw)
    assert adv.actions == ()


def test_parse_advisory_rejects_non_json_prose():
    with pytest.raises(RendererValidationError):
        parse_advisory("The field is waterlogged. Clear the outlets and suspend irrigation.")
    with pytest.raises(RendererValidationError):
        parse_advisory("")
    with pytest.raises(RendererValidationError):
        parse_advisory("[1, 2, 3]")  # JSON but not an object


def test_dosage_field_injection_fails_schema():
    """A prompt that coerces the model to emit a dosage FIELD fails the schema."""
    with pytest.raises(RendererValidationError):
        parse_advisory(
            '{"actions": [{"text": "spray", "evidence": [], "dosage": "2.5 ml/L chlorpyrifos"}],'
            ' "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": false, "cci": 0.625}}'
        )


def test_dosage_in_action_text_caught_by_matching(package):
    """... and a dosage smuggled into the text field is caught by the registry."""
    raw = json.dumps(
        {
            "actions": [
                {
                    "text": "Spray 2.5 ml/L chlorpyrifos on the tillers.",
                    "input_class": None,
                    "consult_expert": False,
                    "evidence": [{"source_id": T1_SUSPEND[0], "tier": "tier1"}],
                }
            ],
            "confidence_breakdown": {
                "R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": False, "cci": 0.625,
            },
        }
    )
    adv = parse_advisory(raw)  # schema-valid (free-text field)
    problems = validate_advisory(adv, allowed_markers=set(package.allowed_markers),
                                 expected_breakdown=package.expected_breakdown)
    assert any("active ingredient" in p for p in problems)
    assert any("numeric dosage" in p for p in problems)


def test_input_class_cannot_smuggle_ingredient_or_dose():
    """BLOCKING-fix: input_class is constrained to registry class labels, so a
    dose or product name placed there fails the schema (no free-text bypass)."""
    base = {
        "text": "Apply to control the pest.",
        "evidence": [{"source_id": "WEATHER:x", "tier": "tier1"}],
    }
    with pytest.raises(RendererValidationError):
        parse_advisory(json.dumps(
            {"actions": [{**base, "input_class": "chlorpyrifos 2.5 ml/L"}],
             "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": False, "cci": 0.625}}
        ))
    with pytest.raises(RendererValidationError):
        parse_advisory(json.dumps(
            {"actions": [{**base, "input_class": "2.5 ml/L"}],
             "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": False, "cci": 0.625}}
        ))
    # a genuine registry class label parses fine
    ok = parse_advisory(json.dumps(
        {"actions": [{**base, "input_class": "organophosphate insecticide"}],
         "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": False, "cci": 0.625}}
    ))
    assert ok.actions[0].input_class == "organophosphate insecticide"


def test_empty_evidence_rejected_at_schema_and_render(package, fake_ollama):
    """BLOCKING-fix: a claim without any evidence ref is not a rendered claim —
    rejected by the schema, so render() can never accept an untraceable action."""
    no_evidence = json.dumps(
        {"actions": [{"text": "Stop irrigation for two weeks and harvest.", "input_class": None,
                      "consult_expert": False, "evidence": []}],
         "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": False, "cci": 0.625}}
    )
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: no_evidence
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: no_evidence
    res = _renderer(max_retries=1).render(package)
    assert not res.ok
    assert res.advisory is None
    assert "evidence" in (res.error or "")  # schema rejection surfaces the reason


def test_render_pii_guard_spaced_phone():
    from krishisethu.renderer.ollama_client import _PII_RE

    assert _PII_RE.search("my mobile is 987 654 3210")
    assert _PII_RE.search("9876-5432-10 for follow-up")
    assert _PII_RE.search("Aadhaar 1234 5678 9012 needed")
    assert not _PII_RE.search("my area is 5 hectares in 2026")
    assert not _PII_RE.search("the plot number is 42")


# --------------------------------------------------------------------------
# render loop: happy path + bounded retries + no prose leakage
# --------------------------------------------------------------------------
def test_render_happy_path_schema_valid(package, fake_ollama):
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: _valid_json(package)
    res = _renderer().render(package)
    assert res.ok
    assert isinstance(res.advisory, Advisory)
    assert res.advisory.actions
    assert res.model_used == "qwen3.5:4b"
    assert res.attempts == 1
    assert res.error is None
    # the chat handle was asked for json format
    assert any(inst.kwargs.get("format") == "json" for inst in fake_ollama["instances"])


def test_render_rejects_after_bounded_retries_no_prose_leak(package, fake_ollama):
    """Always-garbage model output -> bounded reject; raw prose never returned."""
    garbage = "The field is flooded. You should apply fertilizer immediately and drain."
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: garbage
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: garbage
    res = _renderer(max_retries=2).render(package)
    assert not res.ok
    assert res.advisory is None
    assert res.error is not None
    assert garbage not in (res.error or "")  # the prose itself never leaks out
    # bounded: 2 models x (2 retries + 1 first) = 6 calls max
    assert res.attempts <= 6
    assert sum(i.calls for i in fake_ollama["instances"]) == res.attempts


def test_render_recovers_from_transient_garbage_within_budget(package, fake_ollama):
    """First attempt garbage, second valid -> accepted within the retry budget."""
    state = {"n": 0}
    def behavior(attempt):
        state["n"] += 1
        return "not json at all" if state["n"] == 1 else _valid_json(package)
    fake_ollama["script"]["qwen3.5:4b"] = behavior
    res = _renderer(max_retries=2).render(package)
    assert res.ok
    assert res.attempts == 2
    assert res.model_used == "qwen3.5:4b"


def test_render_out_of_set_evidence_rejected(package, fake_ollama):
    """A claim citing a source not in the provided material is never accepted."""
    injected = _valid_json(package)
    injected = injected.replace(
        f'"{T1_SUSPEND[0]}"', '"totally_invented_doc_99::0"', 1
    )
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: injected
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: injected
    res = _renderer(max_retries=1).render(package)
    assert not res.ok
    assert "out-of-set source" in (res.error or "")
    assert res.advisory is None


def test_render_confidence_override_rejected(package, fake_ollama):
    """The LLM cannot override CCI: a wrong echo fails validation."""
    expected_cci = package.expected_breakdown.cci
    tampered = _valid_json(package).replace(f'"cci": {expected_cci}', '"cci": 1.0', 1)
    assert '"cci": 1.0' in tampered  # the tamper is actually in the payload
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: tampered
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: tampered
    res = _renderer(max_retries=1).render(package)
    assert not res.ok
    assert "confidence_breakdown.cci echo" in (res.error or "")
    assert res.advisory is None


def test_render_falls_back_to_2b_when_4b_unavailable(package, fake_ollama):
    """Primary mocked unavailable -> fallback model renders the advisory."""
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: (_ for _ in ()).throw(ConnectionError("4b down"))
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: _valid_json(package)
    res = _renderer().render(package)
    assert res.ok
    assert res.model_used == "qwen3.5:2b"


def test_render_falls_back_when_4b_slow_or_garbage(package, fake_ollama):
    """Primary persistently invalid -> fallback 2b still delivers."""
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: "valid but prose, not JSON" * 3
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: _valid_json(package)
    res = _renderer(max_retries=1).render(package)
    assert res.ok
    assert res.model_used == "qwen3.5:2b"


def test_render_pii_guard_refuses_phone_in_query(package):
    pii_pkg = build_prompt(
        _actions(T1_SUSPEND),
        query="my mobile number is 9876543210, what should I do?",
        breakdown=_breakdown(),
    )
    with pytest.raises(RendererValidationError):
        _renderer().render(pii_pkg)


def test_render_never_calls_fallback_on_success(package, fake_ollama):
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: _valid_json(package)
    res = _renderer().render(package)
    assert res.ok
    assert not any(i.model == "qwen3.5:2b" and i.calls for i in fake_ollama["instances"])


def test_render_empty_output_abandons_model_for_fallback(package, fake_ollama):
    """Deterministic empty output (thinking swallowed the budget) must NOT burn
    max_retries on the same model — abandon it and let the fallback try once."""
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: ""
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: _valid_json(package)
    res = _renderer(max_retries=2).render(package)
    assert res.ok
    assert res.model_used == "qwen3.5:2b"
    assert res.attempts == 2, f"expected 1 (4b empty) + 1 (2b valid), got {res.attempts}"
    calls_4b = [i for i in fake_ollama["instances"] if i.model == "qwen3.5:4b"]
    assert calls_4b and calls_4b[0].calls == 1


def test_render_empty_output_on_both_models_bounded_reject(package, fake_ollama):
    fake_ollama["script"]["qwen3.5:4b"] = lambda a: ""
    fake_ollama["script"]["qwen3.5:2b"] = lambda a: ""
    res = _renderer(max_retries=2).render(package)
    assert not res.ok
    assert "empty model output" in (res.error or "")
    # 1 attempt per model (empty abandons immediately), never the full 6
    assert res.attempts == 2
    assert res.advisory is None


# --------------------------------------------------------------------------
# golden set: 100% schema-valid with bounded retries (stubbed LLM)
# --------------------------------------------------------------------------
def _golden_deliver_material():
    """(case, traceable_actions, resolution, breakdown) for each golden DELIVER
    case, using only the validator-kept (traceable) actions as material."""
    import sys

    sys.path.insert(0, sys_path_golden)
    from harness import FIXTURES_DIR, build_decision, load_golden_cases, resolve_tier1

    from krishisethu.confidence.composite_index import cci as _cci, component_r
    from krishisethu.domain import ConfidenceScores

    for case in load_golden_cases(FIXTURES_DIR):
        if case.expected_verdict != "DELIVER":
            continue
        tier1_actions, rm = resolve_tier1(case)
        resolution = Resolution(case.resolution) if case.resolution else (rm.resolution if rm else Resolution.UNRESOLVED)
        decision = build_decision(case, tier1_actions, rm, resolution)
        allowed = {(p.tier, p.source_id) for p in decision.provenance}
        traceable = [a for a in decision.actions if (a.tier, a.source_id) in allowed]
        if not traceable:
            continue
        r = rm.score if rm else component_r(resolution, case.partial_score)
        breakdown = _breakdown(
            r=r, s=0.0, p=case.p_vision, image=case.has_image,
            cci=_cci(ConfidenceScores(R=r, S_RAG=0.0, P_vision=case.p_vision, has_image=case.has_image)),
        )
        yield case, traceable, breakdown


def test_golden_deliver_cases_render_schema_valid():
    """Every golden DELIVER case's *traceable* material renders to schema-valid
    JSON that parses and passes all post-parse checks — the stubbed LLM never
    fails. Only the validator-kept actions are material (the pipeline hands the
    renderer the traceable decision, never injected out-of-set claims)."""
    failures = []
    checked = 0
    for case, traceable, breakdown in _golden_deliver_material():
        pkg = build_prompt(traceable, query=case.query, breakdown=breakdown)
        raw = _valid_json(pkg)
        advisory = parse_advisory(raw)
        problems = validate_advisory(
            advisory, allowed_markers=set(pkg.allowed_markers), expected_breakdown=breakdown
        )
        checked += 1
        if problems:
            failures.append(f"{case.id}: {problems}")
    assert checked >= 3, "expected several golden DELIVER fixtures to render"
    assert not failures, failures


def test_golden_deliver_cases_render_through_client(fake_ollama):
    """ADVISORY-fix: golden DELIVER cases drive the actual render() retry loop,
    not just parse+validate — the full acceptance (schema-valid, bounded retry,
    evidence-in-allowed-set) holds through the client on first attempt."""
    accepted = []
    for case, traceable, breakdown in _golden_deliver_material():
        pkg = build_prompt(traceable, query=case.query, breakdown=breakdown)
        fake_ollama["script"]["qwen3.5:4b"] = lambda a, _p=pkg: _valid_json(_p)
        res = _renderer(max_retries=2).render(pkg)
        assert res.ok, f"[{case.id}] render failed: {res.error}"
        assert res.attempts == 1, f"[{case.id}] expected first-attempt accept, got {res.attempts}"
        # every rendered claim cites an allowed marker and the echo matches truth
        problems = validate_advisory(
            res.advisory, allowed_markers=set(pkg.allowed_markers), expected_breakdown=breakdown
        )
        assert not problems, f"[{case.id}] post-render problems: {problems}"
        accepted.append(case.id)
    assert len(accepted) >= 3, f"expected several golden DELIVER cases, got {accepted}"


# --------------------------------------------------------------------------
# mapping to the safety-core domain decision (gate seam)
# --------------------------------------------------------------------------
def test_to_domain_decision_then_safety_validator_accepts():
    """A validated renderer advisory maps onto the domain contract and passes the
    Prompt-5 SafetyValidator (traceability + no dosage)."""
    pkg = build_prompt(_actions(T1_SUSPEND, T1_CLEAR), breakdown=_breakdown())
    advisory = parse_advisory(_valid_json(pkg))
    decision = to_domain_decision(
        advisory, resolution=Resolution.EXACT, allowed_markers=set(pkg.allowed_markers)
    )
    assert decision.actions and len(decision.actions) == len(pkg.allowed_markers) == 2
    assert all(a.tier is Tier.TIER1 for a in decision.actions)
    assert all(p.tier is Tier.TIER1 for p in decision.provenance)
    # run it through the real validator — must be accepted (rule-certain)
    from krishisethu.domain import ConfidenceScores

    result = SafetyValidator().validate(
        ConfidenceScores(R=1.0, S_RAG=0.72, P_vision=0.0, has_image=False),
        Resolution.EXACT,
        deterministic=True,
        raw="\n".join(a.text for a in decision.actions),
        decision=decision,
    )
    assert result.accepted and not result.escalate


def test_to_domain_decision_rejects_action_without_evidence():
    # An empty evidence list is now schema-invalid: parse_advisory itself rejects
    # the payload (an untraceable claim is not a claim).
    with pytest.raises(RendererValidationError):
        parse_advisory(
            '{"actions": [{"text": "orphan action", "evidence": []}],'
            ' "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": false, "cci": 0.625}}'
        )
    # Same for a payload that omits the evidence key entirely.
    with pytest.raises(RendererValidationError):
        parse_advisory(
            '{"actions": [{"text": "orphan action"}],'
            ' "confidence_breakdown": {"R": 1.0, "S_RAG": 0.72, "P_vision": 0.0, "has_image": false, "cci": 0.625}}'
        )


# --------------------------------------------------------------------------
# translation stub: only verified fields, unknown phrases flagged
# --------------------------------------------------------------------------
def test_translate_known_phrase_hindi():
    hi, missed = translate_text("Suspend irrigation", LANG_HI)
    assert missed is False
    # Devanagari for "stop irrigation" — must be real script, not ASCII passthrough
    assert all(ord(ch) > 0x0900 for ch in hi if not ch.isspace())


def test_translate_unknown_phrase_flagged_not_guessed():
    out, missed = translate_text("Something novel the model said", LANG_HI)
    assert missed is True
    assert out == "Something novel the model said"  # never silently invented


def test_translate_en_passthrough():
    out, missed = translate_text("Suspend irrigation", LANG_EN)
    assert out == "Suspend irrigation" and missed is False


def test_localize_preserves_markers_and_breakdown():
    # A package with exactly the two known Hindi phrases (tier-1 only, so no
    # tier-2/tier-3 markers clutter the action set).
    pkg = build_prompt(
        _actions(T1_CLEAR, T1_SUSPEND),
        breakdown=_breakdown(),
    )
    advisory = parse_advisory(_valid_json(pkg, texts=["Clear furrow outlets", "Suspend irrigation"]))
    loc = localize(advisory, LANG_HI)
    assert len(loc.actions) == len(advisory.actions) == 2
    assert loc.untranslated == ()  # both phrases are in the seed map
    assert loc.confidence_breakdown is advisory.confidence_breakdown
    # the verified text was transformed into Devanagari (at least the script
    # characters dominate; ASCII punctuation inside a phrase is fine)
    hi0 = loc.actions[0]
    assert hi0 != advisory.actions[0].text
    devanagari = sum(0x0900 <= ord(ch) <= 0x097F for ch in hi0)
    assert devanagari >= 5, f"expected Devanagari text, got {hi0!r}"


def test_localize_unknown_phrase_kept_and_flagged():
    pkg = build_prompt(_actions(T1_SUSPEND), breakdown=_breakdown())
    advisory = parse_advisory(_valid_json(pkg, texts=["A brand-new phrasing from the model"]))
    loc = localize(advisory, LANG_HI)
    assert loc.untranslated == (0,)
    assert loc.actions[0] == "A brand-new phrasing from the model"


# --------------------------------------------------------------------------
# live (gated + opt-in): real qwen3.5:4b through Ollama
# --------------------------------------------------------------------------
# Default-suite policy: a real qwen render is a multi-second model call, and on
# CPU-served boxes can run a minute per call, so the live test is OPT-IN via
# KRISHISETU_LIVE_RENDER=1. The permanent CI story is the hermetic suite above
# (stubbed ChatOllama); this test proves schema-validity + latency against the
# real model when the machine is fast enough to care.
def _ollama_models() -> list[str] | None:
    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code != 200:
            return None
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return None


def test_live_render_schema_valid_and_latency(package):
    import os

    models = _ollama_models()
    if not models or "qwen3.5:4b" not in models:
        pytest.skip("Ollama server or qwen3.5:4b not available")
    if os.environ.get("KRISHISETU_LIVE_RENDER") != "1":
        pytest.skip("live render is opt-in: set KRISHISETU_LIVE_RENDER=1 to measure real-model latency")
    # Bounded: one render per measurement, no retry storm on a slow server.
    renderer = OllamaRenderer(max_retries=0)
    t0 = time.perf_counter()
    res = renderer.render(package)
    elapsed = time.perf_counter() - t0
    assert res.ok, f"live render failed: {res.error}"
    assert isinstance(res.advisory, Advisory) and res.advisory.actions
    assert res.advisory.confidence_breakdown.cci == pytest.approx(
        package.expected_breakdown.cci, abs=1e-3
    )
    assert elapsed < 5.0, (
        f"single render latency {elapsed:.2f}s exceeds the ~5s advisory budget "
        "(model serving bound; see PROMPT-06 report)"
    )
