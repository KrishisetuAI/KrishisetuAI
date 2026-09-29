"""LangGraph integration pipeline (Prompt 7 step 1).

A thin state machine over one typed :class:`CompileState` mapping the
white-paper Algorithm 1 branches to conditional edges:

    triage -> context_assembler -> tier1 -> tier2 -> tier3 -> confidence
        -> (gate edge) render -> validate -> (gate edge) DELIVER | ESCALATE

* the Tier-1 EXACT short-circuit: ``pass_gate(cci, deterministic=True)`` lets a
  rule-certain answer through without consulting the 0.85 threshold (it still
  runs the chemical + traceability gates downstream);
* the DELIVER-vs-ESCALATE gate edge after validation.

Only the orchestration uses LangGraph. Tiers 1 and 2 remain framework-free pure
modules — this file only calls them from node functions. The pipeline is capped
behind :class:`TierOrchestrator`.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict

from langgraph.graph import END, START, StateGraph

from krishisethu.confidence import cci, component_r, pass_gate
from krishisethu.domain import (
    AdvisoryDecision,
    ConfidenceScores,
    EscalationTicket,
    Resolution,
    Tier,
    ValidatorResult,
)
from krishisethu.escalation import EscalationService
from krishisethu.knowledge.loader import CanonicalLoader, CanonicalResult
from krishisethu.pipeline.telemetry import e2e_ms as _e2e_ms
from krishisethu.pipeline.telemetry import emit as _emit
from krishisethu.pipeline.telemetry import now_ms as _now_ms
from krishisethu.pipeline.base import TierOrchestrator
from krishisethu.pipeline.context import AssembledContext, assemble
from krishisethu.pipeline.triage import Intent, TriageResult, classify
from krishisethu.rag import PlotContext, RetrievalResult, Retriever
from krishisethu.renderer import (
    Advisory,
    ConfidenceBreakdown,
    ExcerptSource,
    Fragment,
    PromptPackage,
    RenderResult,
    build_prompt,
)
from krishisethu.renderer.structured_output import (
    ActionItem as RenderActionItem,
)
from krishisethu.renderer.structured_output import EvidenceRef, to_domain_decision
from krishisethu.rules import RuleAction, rule_match, weather_to_action
from krishisethu.rules.rule_match import RuleMatch
from krishisethu.rules.weather_action import IMDAlertType, SoilDrainageClass
from krishisethu.safety import SafetyValidator

logger = logging.getLogger("krishisethu.pipeline")


# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------
class CompileState(BaseModel):
    """The one typed state threaded through the whole pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- input -----------------------------------------------------------
    query: str = ""
    plot_id: str = "demo"
    demo: bool = False
    on_date: date = field(default_factory=date.today)

    # -- triage ----------------------------------------------------------
    trace_id: str = ""
    intent: str = ""            # Intent value
    flagged_dosage: bool = False
    run_tier1: bool = True

    # -- context ---------------------------------------------------------
    plot_context: Optional[PlotContext] = None
    crop_stage: Optional[str] = None
    weather_alert: Optional[str] = None
    soil_drainage: Optional[str] = None
    has_image: bool = False
    decision_type: str = "irrigation_rule"

    # -- tier 1 ----------------------------------------------------------
    rule_actions: List[RuleAction] = field(default_factory=list)
    rule_match: Optional[RuleMatch] = None
    deterministic: bool = False

    # -- tier 2 ----------------------------------------------------------
    canonical_result: Optional[CanonicalResult] = None

    # -- tier 3 ----------------------------------------------------------
    retrieval_result: Optional[RetrievalResult] = None

    # -- confidence ------------------------------------------------------
    scores: Optional[ConfidenceScores] = None
    cci_value: Optional[float] = None
    gate_passed: bool = False
    gate_outcome: str = ""      # pass_exact | pass_cci | fail_cci | escalated

    # -- tier 4 ----------------------------------------------------------
    prompt_package: Optional[PromptPackage] = None
    render_result: Optional[RenderResult] = None
    advisory: Optional[Advisory] = None

    # -- validation ------------------------------------------------------
    final_decision: Optional[AdvisoryDecision] = None
    validator_result: Optional[ValidatorResult] = None

    # -- output ----------------------------------------------------------
    escalate: bool = False
    final_output: str = ""
    escalation_ticket: Optional[EscalationTicket] = None

    # -- telemetry -------------------------------------------------------
    latency_ms: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Renderer seam (an OllamaRenderer or a hermetic test stub both satisfy it)
# ---------------------------------------------------------------------------
class Renderer(Protocol):
    def render(self, package: PromptPackage) -> RenderResult:
        ...


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
class _Nodes:
    """Stateless node bodies; bound to deps by :class:`Pipeline`."""

    def __init__(self, deps: "PipelineDependencies"):
        self.deps = deps

    # -- triage ----------------------------------------------------------
    def triage(self, state: CompileState) -> dict:
        t0 = _now_ms()
        triage = classify(state.query)
        state.trace_id = state.trace_id or str(uuid.uuid4())
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.triage", intent=triage.intent.value,
                       flagged_dosage=triage.flagged_dosage, latency_ms=lat)
        return {
            "trace_id": state.trace_id,
            "intent": triage.intent.value,
            "flagged_dosage": triage.flagged_dosage,
            "latency_ms": {**state.latency_ms, "triage": lat},
        }

    # -- context ---------------------------------------------------------
    def context_assembler(self, state: CompileState) -> dict:
        t0 = _now_ms()
        ctx: AssembledContext = assemble(
            query=state.query,
            plot_id=state.plot_id,
            demo=state.demo,
            on_date=state.on_date,
            triage=TriageResult(
                intent=Intent(state.intent) if state.intent else Intent.WEATHER,
                flagged_dosage=state.flagged_dosage,
            ),
        )
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.context_assembler",
                       crop=ctx.plot_context.crop.value,
                       season=ctx.plot_context.season.value,
                       district=ctx.plot_context.district,
                       alert=ctx.weather_alert.value,
                       run_tier1=ctx.run_tier1, latency_ms=lat)
        return {
            "plot_context": ctx.plot_context,
            "crop_stage": ctx.crop_stage.value,
            "weather_alert": ctx.weather_alert.value,
            "soil_drainage": ctx.soil_drainage.value,
            "has_image": ctx.has_image,
            "on_date": ctx.on_date,
            "run_tier1": ctx.run_tier1,
            "decision_type": ctx.decision_type.value,
            "latency_ms": {**state.latency_ms, "context_assembler": lat},
        }

    # -- tier 1 (deterministic rules; framework-free core) ---------------
    def tier1(self, state: CompileState) -> dict:
        t0 = _now_ms()
        if not state.run_tier1:
            rm = rule_match(Resolution.UNRESOLVED)
            lat = round(_now_ms() - t0, 3)
            _emit(state.trace_id, "node.tier1", skipped=True,
                           resolution=rm.resolution.value, latency_ms=lat)
            return {
                "rule_actions": [],
                "rule_match": rm,
                "deterministic": False,
                "latency_ms": {**state.latency_ms, "tier1": lat},
            }

        from krishisethu.domain import Crop, Season

        ctx = state.plot_context
        if ctx is None:
            raise ValueError("plot_context not set before tier1")
        crop = Crop(ctx.crop.value if hasattr(ctx.crop, "value") else ctx.crop)
        stage = Season(state.crop_stage) if state.crop_stage else Season.GRAND_GROWTH
        alert = IMDAlertType(state.weather_alert) if state.weather_alert else IMDAlertType.NO_ALERT
        drainage = SoilDrainageClass.coerce(state.soil_drainage or "M")

        actions = weather_to_action(crop, stage, alert, drainage)
        total = len(actions)
        rm = rule_match(
            Resolution.EXACT if actions else Resolution.UNRESOLVED,
            matched_clauses=len({a.clause for a in actions}),
            total_clauses=total,
            trace_codes=tuple(a.trace for a in actions),
        )
        deterministic = rm.resolution is Resolution.EXACT and bool(actions)
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.tier1", actions=len(actions),
                       resolution=rm.resolution.value, deterministic=deterministic,
                       latency_ms=lat)
        return {
            "rule_actions": actions,
            "rule_match": rm,
            "deterministic": deterministic,
            "latency_ms": {**state.latency_ms, "tier1": lat},
        }

    # -- tier 2 (canonical exact lookup) ---------------------------------
    def tier2(self, state: CompileState) -> dict:
        t0 = _now_ms()
        ctx = state.plot_context
        if ctx is None:
            raise ValueError("plot_context not set before tier2")
        from krishisethu.domain import Crop, DecisionType, Season, State

        crop = Crop(ctx.crop.value if hasattr(ctx.crop, "value") else ctx.crop)
        state_enum = State(ctx.state.value if hasattr(ctx.state, "value") else "haryana")
        season = Season(ctx.season.value if hasattr(ctx.season, "value") else "kharif")
        decision = DecisionType(state.decision_type or "irrigation_rule")

        result = self.deps.loader.read_canonical(
            crop=crop,
            state=state_enum,
            season=season,
            decision_type=decision,
            on_date=state.on_date,
        )
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.tier2", found=result is not None,
                       doc=result.doc.doc_id if result else None, latency_ms=lat)
        return {
            "canonical_result": result,
            "latency_ms": {**state.latency_ms, "tier2": lat},
        }

    # -- tier 3 (domain-governed vector RAG) -----------------------------
    def tier3(self, state: CompileState) -> dict:
        t0 = _now_ms()
        ctx = state.plot_context
        if ctx is None:
            raise ValueError("plot_context not set before tier3")
        result: RetrievalResult = self.deps.retriever.retrieve(state.query, ctx, k=5)
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.tier3", chunks=len(result.chunks),
                       s_rag=round(result.s_rag, 4), latency_ms=lat)
        return {
            "retrieval_result": result,
            "latency_ms": {**state.latency_ms, "tier3": lat},
        }

    # -- confidence (CCI + gate) ------------------------------------------
    def confidence(self, state: CompileState) -> dict:
        t0 = _now_ms()
        rm = state.rule_match
        resolution = rm.resolution if rm else Resolution.UNRESOLVED
        r_value = component_r(resolution, partial_score=rm.score if rm else None)
        s_rag = state.retrieval_result.s_rag if state.retrieval_result else 0.0
        # Clamp to [0,1] – cosine can be slightly negative
        r_value = max(0.0, min(1.0, r_value))
        s_rag = max(0.0, min(1.0, s_rag))
        scores = ConfidenceScores(R=r_value, S_RAG=s_rag, P_vision=0.0,
                                  has_image=state.has_image)
        value = cci(scores)
        passed = pass_gate(value, deterministic=state.deterministic)
        if state.deterministic:
            outcome = "pass_exact"
        elif passed:
            outcome = "pass_cci"
        else:
            outcome = "fail_cci"
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.confidence", cci=round(value, 4),
                       r=round(r_value, 4), s_rag=round(s_rag, 4),
                       deterministic=state.deterministic, passed=passed,
                       gate_outcome=outcome, latency_ms=lat)
        return {
            "scores": scores,
            "cci_value": round(value, 4),
            "gate_passed": passed,
            "gate_outcome": outcome,
            "latency_ms": {**state.latency_ms, "confidence": lat},
        }

    # -- tier 4 render (paraphraser) ---------------------------------------
    def render(self, state: CompileState) -> dict:
        t0 = _now_ms()
        package = self._build_package(state)
        result: RenderResult | None = None
        advisory: Advisory | None = None

        if self.deps.renderer is not None:
            result = self.deps.renderer.render(package)
            if result is not None and result.ok:
                advisory = result.advisory

        if advisory is None and state.deterministic and state.rule_actions:
            # Rule-certain answer with no usable LLM paraphrase: fall back to
            # the exact rule actions as the advisory (still validated by the
            # chemical + traceability gates downstream).
            advisory = _rule_actions_advisory(state)
            result = RenderResult(advisory=advisory, model_used=None, attempts=0,
                                  error="llm unavailable; exact rule text delivered")

        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.render", advisory=advisory is not None,
                       model=result.model_used if result else None,
                       error=result.error if result else None, latency_ms=lat)
        return {
            "prompt_package": package,
            "render_result": result,
            "advisory": advisory,
            "latency_ms": {**state.latency_ms, "render": lat},
        }

    def _build_package(self, state: CompileState) -> PromptPackage:
        canonical = state.canonical_result
        excerpt = (
            ExcerptSource(doc_id=canonical.doc.doc_id, text=canonical.body)
            if canonical is not None
            else None
        )
        fragments: list[Fragment] = []
        retrieval = state.retrieval_result
        if retrieval is not None:
            for chunk in retrieval.chunks:
                source_id, chunk_index, _title = chunk.provenance()
                fragments.append(
                    Fragment(source_id=source_id, chunk_index=chunk_index, text=chunk.text)
                )
        breakdown = ConfidenceBreakdown(
            R=(state.scores.R if state.scores else 0.0),
            S_RAG=(state.scores.S_RAG if state.scores else 0.0),
            P_vision=(state.scores.P_vision if state.scores else 0.0),
            has_image=state.has_image,
            cci=state.cci_value or 0.0,
        )
        return build_prompt(
            actions=list(state.rule_actions),
            canonical_excerpt=excerpt,
            topk_fragments=fragments,
            query=state.query,
            breakdown=breakdown,
        )

    # -- validation (three-gate SafetyValidator) ---------------------------
    def validate(self, state: CompileState) -> dict:
        t0 = _now_ms()
        rm = state.rule_match
        resolution = rm.resolution if rm else Resolution.UNRESOLVED

        decision: AdvisoryDecision
        raw = ""
        advisory = state.advisory
        package = state.prompt_package
        if advisory is not None:
            allowed = package.allowed_markers if package is not None else None
            decision = to_domain_decision(
                advisory, resolution=resolution, allowed_markers=allowed
            )
            raw = advisory.model_dump_json()
        else:
            decision = AdvisoryDecision(actions=(), resolution=resolution, provenance=())

        result = self.deps.validator.validate(
            scores=state.scores or ConfidenceScores(),
            resolution=resolution,
            deterministic=state.deterministic,
            raw=raw,
            decision=decision,
        )
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.validate", accepted=result.accepted,
                       escalate=result.escalate, warnings=list(result.warnings),
                       latency_ms=lat)
        return {
            "final_decision": decision,
            "validator_result": result,
            "latency_ms": {**state.latency_ms, "validate": lat},
        }

    # -- delivery ----------------------------------------------------------
    def deliver(self, state: CompileState) -> dict:
        t0 = _now_ms()
        result = state.validator_result
        text = result.text if result else ""
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.deliver", escalated=False,
                       final=True, latency_ms=lat)
        return {
            "final_output": text,
            "escalate": False,
            "latency_ms": {**state.latency_ms, "deliver": lat},
        }

    def escalate(self, state: CompileState) -> dict:
        t0 = _now_ms()
        chunks: tuple[dict, ...] = ()
        if state.retrieval_result is not None:
            chunks = tuple(
                {"id": c.id, "score": round(c.score, 4), "text": c.text[:400]}
                for c in state.retrieval_result.chunks
            )
        warnings = tuple(state.validator_result.warnings) if state.validator_result else ()
        ticket = self.deps.escalation_service.raise_ticket(
            query=state.query,
            plot_id=state.plot_id,
            date_window=state.on_date.isoformat(),
            context={
                "trace_id": state.trace_id,
                "intent": state.intent,
                "plot_id": state.plot_id,
                "demo": state.demo,
                "weather_alert": state.weather_alert,
                "soil_drainage": state.soil_drainage,
                "cci": state.cci_value,
                "gate_outcome": state.gate_outcome,
                "resolution": state.rule_match.resolution.value if state.rule_match else None,
                "advisory": (state.advisory.model_dump(mode="json")
                             if state.advisory else None),
            },
            cci=state.cci_value,
            retrieved_chunks=chunks,
            validator_warnings=warnings,
        )
        lat = round(_now_ms() - t0, 3)
        _emit(state.trace_id, "node.escalate", ticket=ticket.ticket_id,
                       escalated=True, final=True, latency_ms=lat)
        # Preserve the original gate outcome (fail_cci, etc.) instead of overwriting
        return {
            "escalation_ticket": ticket,
            "final_output": self.deps.escalation_service.neutral_message,
            "escalate": True,
            "latency_ms": {**state.latency_ms, "escalate": lat},
        }


def _rule_actions_advisory(state: CompileState) -> Advisory:
    """Turn EXACT Tier-1 rule actions into a schema-valid renderer Advisory.

    Used when no LLM is available: the rule text itself is the advisory, each
    action carrying its rule trace as evidence so traceability still holds.
    """
    actions = []
    for a in state.rule_actions:
        actions.append(
            RenderActionItem(
                text=a.action,
                input_class=None,
                consult_expert=False,
                evidence=(
                    EvidenceRef(source_id=a.trace, tier=Tier.TIER1),
                ),
            )
        )
    breakdown = ConfidenceBreakdown(
        R=(state.scores.R if state.scores else 0.0),
        S_RAG=(state.scores.S_RAG if state.scores else 0.0),
        P_vision=0.0,
        has_image=state.has_image,
        cci=state.cci_value or 0.0,
    )
    return Advisory(actions=tuple(actions), confidence_breakdown=breakdown)


# ---------------------------------------------------------------------------
# Conditional edge functions (Algorithm 1 branches)
# ---------------------------------------------------------------------------
def _route_after_confidence(state: CompileState) -> str:
    """Gate edge: pass -> Tier-4 render; fail -> escalate (skip render).
    
    Exception: if the query was flagged for chemical dosage (flagged_dosage),
    we still run render + validate so the chemical gate can catch the dosage
    regardless of CCI (per safety spec: chem gate escalates regardless of CCI).
    """
    if state.gate_passed:
        return "render"
    if state.flagged_dosage:
        # Run render + validate so chemical gate can catch dosage
        return "render"
    return "escalate"


def _route_after_validate(state: CompileState) -> str:
    """Gate edge: validator asks escalation -> escalate, else deliver."""
    v = state.validator_result
    return "escalate" if (v is not None and v.escalate) else "deliver"


# ---------------------------------------------------------------------------
# Dependencies + orchestrator
# ---------------------------------------------------------------------------
@dataclass
class PipelineDependencies:
    """Everything the graph needs, injected so tests stay hermetic."""

    loader: CanonicalLoader
    retriever: Retriever
    renderer: Optional[Renderer] = None
    validator: SafetyValidator = field(default_factory=SafetyValidator)
    escalation_service: EscalationService = field(default_factory=EscalationService)


class Pipeline(TierOrchestrator):
    """LangGraph implementation of the KrishiSetu decision pipeline."""

    def __init__(self, deps: PipelineDependencies):
        self.deps = deps
        self._nodes = _Nodes(deps)
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(CompileState)
        g.add_node("triage", self._nodes.triage)
        g.add_node("context_assembler", self._nodes.context_assembler)
        g.add_node("tier1", self._nodes.tier1)
        g.add_node("tier2", self._nodes.tier2)
        g.add_node("tier3", self._nodes.tier3)
        g.add_node("confidence", self._nodes.confidence)
        g.add_node("render", self._nodes.render)
        g.add_node("validate", self._nodes.validate)
        g.add_node("deliver", self._nodes.deliver)
        g.add_node("escalate", self._nodes.escalate)

        g.add_edge(START, "triage")
        g.add_edge("triage", "context_assembler")
        g.add_edge("context_assembler", "tier1")
        g.add_edge("tier1", "tier2")
        g.add_edge("tier2", "tier3")
        g.add_edge("tier3", "confidence")
        g.add_conditional_edges(
            "confidence", _route_after_confidence, {"render": "render", "escalate": "escalate"}
        )
        g.add_edge("render", "validate")
        g.add_conditional_edges(
            "validate", _route_after_validate, {"deliver": "deliver", "escalate": "escalate"}
        )
        g.add_edge("deliver", END)
        g.add_edge("escalate", END)
        return g.compile()

    def run(self, query: str, plot_id: str = "demo", demo: bool = False) -> CompileState:
        """Run one query through the full decision core."""
        initial = CompileState(query=query, plot_id=plot_id, demo=demo)
        final = self._graph.invoke(initial)
        if isinstance(final, CompileState):
            state = final
        else:
            state = CompileState(**final)
        _emit(
            state.trace_id,
            "pipeline.complete",
            escalated=state.escalate,
            gate_outcome=state.gate_outcome,
            cci=state.cci_value,
            e2e_ms=_e2e_ms(state.latency_ms),
        )
        return state
