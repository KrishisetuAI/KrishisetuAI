"""FastAPI service for KrishiSetu AI.

Exposes:
    GET  /health               -> {"status": "ok"}
    GET  /api/v1/status        -> settings summary + live Ollama reachability
    POST /api/v1/advise        -> safety-gated advisory (JSON REST channel)
    POST /ivr/incoming         -> Twilio voice webhook (gathers Hindi speech)
    POST /ivr/gather           -> safety-gated advisory for the spoken query
    GET  /ivr/audio/{call_sid} -> Bhashini TTS audio for one call

IVR safety contract (migrated from the standalone KrishiSetuAI-IVR prototype):
the webhook NEVER answers a farmer directly. The recognized speech is run
through the full 4-tier LangGraph pipeline — Tier 1-3 + CCI confidence gate +
three-gate SafetyValidator — via ``pipeline.run()``. On escalation
(``fail_cci``, dosage violation, or all claims stripped) the caller hears only
the hardcoded neutral Hindi hand-off; on delivery the caller hears the
validated advisory actions. There is no code path from a raw LLM to the voice
channel. The REST ``/api/v1/advise`` route is the same contract: a thin JSON
transport over ``pipeline.run()`` that cannot bypass the CCI gate or the
three-gate SafetyValidator.
"""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from krishisethu.api.services.bhashini import BhashiniService
from krishisethu.config.settings import Settings, get_settings

logger = logging.getLogger("krishisethu.api")

REPO_ROOT = Path(__file__).resolve().parents[2]
IVR_AUDIO_DIR = REPO_ROOT / "data" / "ivr_audio"

#: Plot identity for IVR callers until the real plot-profile store lands.
IVR_PLOT_ID = "demo-ivr"

#: Neutral Hindi hand-off spoken on ANY escalation (fail_cci, dosage
#: violation, untraceable output). The full context goes to the KVK officer
#: via the escalation ticket, never to the caller.
ESCALATION_HANDOFF_HI = (
    "आपकी समस्या का विश्लेषण कर लिया गया है। सटीक जानकारी के लिए इसे "
    "कृषि विज्ञान केंद्र के विशेषज्ञ को भेज दिया गया है।"
)

#: Twilio CallSid is "CA" + 32 hex; also accepts uuid tokens we mint for
#: form posts without a sid. Blocks path traversal in the audio route.
_CALL_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class AdviseRequest(BaseModel):
    """REST advisory request — the same contract the CLI and IVR channels use."""

    query: str
    plot_id: str = "demo"
    demo: bool = False


class AdviseAction(BaseModel):
    """One gate-validated action with its traceability marker."""

    text: str
    source_id: str | None = None


class AdviseResponse(BaseModel):
    """Safety-gated outcome: validated actions, or an escalation hand-off."""

    escalate: bool
    output: str
    actions: list[AdviseAction] = []
    cci: float | None = None
    gate_outcome: str | None = None
    trace_id: str | None = None
    ticket_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "settings"):
        app.state.settings = get_settings()
    yield


def create_app(
    settings: Settings | None = None,
    pipeline: object | None = None,
    bhashini: BhashiniService | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``pipeline`` and ``bhashini`` are injectable so tests stay hermetic; when
    omitted, the pipeline is built lazily on the first IVR request (it is
    heavy: Chroma, embedder, renderer) and Bhashini reads its credentials from
    settings.
    """
    app = FastAPI(title="KrishiSetu AI", version="0.2.0", lifespan=lifespan)
    # Set at construction so tests that inject a custom settings object (e.g. a
    # dead Ollama host) work even when httpx.ASGITransport skips lifespan events.
    app.state.settings = settings or get_settings()
    app.state.pipeline = pipeline
    app.state.bhashini = bhashini or BhashiniService(settings=app.state.settings.bhashini)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/status")
    async def status() -> dict:
        s: Settings = app.state.settings
        result = {
            "settings": s.summary(),
            "ollama_reachable": False,
            "ollama_version": None,
            "models_present": [],
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{s.ollama.embed_base()}/api/version")
                if r.status_code == 200:
                    result["ollama_reachable"] = True
                    result["ollama_version"] = r.json().get("version")
                    tr = await client.get(f"{s.ollama.embed_base()}/api/tags")
                    if tr.status_code == 200:
                        result["models_present"] = [
                            m.get("name") for m in tr.json().get("models", [])
                        ]
        except Exception:
            # Never 500 when Ollama is down; report unreachable and return.
            pass
        return result

    # ------------------------------------------------------------------
    # Twilio IVR webhook (Bhashini voice channel)
    # ------------------------------------------------------------------
    def _twiml(xml: str) -> Response:
        return Response(content=xml, media_type="application/xml")

    def _get_pipeline():
        """Injected test pipeline, or the lazily built default decision core."""
        pipeline = app.state.pipeline
        if pipeline is None:
            from krishisethu.pipeline.factory import build_default_pipeline

            pipeline = build_default_pipeline()
            app.state.pipeline = pipeline
        return pipeline

    @app.post("/api/v1/advise", response_model=AdviseResponse)
    async def advise(payload: AdviseRequest) -> AdviseResponse:
        """Run one farmer query through the same safety-gated pipeline as the
        CLI and the IVR voice channel.

        Escalations return only the neutral hand-off message plus the ticket
        id; the withheld advisory is never serialized to the REST client.
        """
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must be a non-empty string")
        pipeline = _get_pipeline()
        # Same threadpool dispatch as the IVR route: the LangGraph decision
        # core is synchronous and must not block the event loop.
        state = await run_in_threadpool(pipeline.run, query, payload.plot_id, payload.demo)
        actions = (
            [AdviseAction(text=a.text, source_id=a.source_id) for a in state.final_decision.actions]
            if state.final_decision is not None
            else []
        )
        ticket = state.escalation_ticket
        return AdviseResponse(
            escalate=state.escalate,
            output=state.final_output,
            # A withheld advisory never leaves the server on escalation.
            actions=[] if state.escalate else actions,
            cci=state.cci_value,
            gate_outcome=state.gate_outcome,
            trace_id=state.trace_id,
            ticket_id=ticket.ticket_id if ticket is not None else None,
        )

    @app.post("/ivr/incoming")
    async def ivr_incoming(request: Request) -> Response:
        """Answer the call and gather the farmer's spoken question (hi-IN)."""
        gather_url = f"{str(request.base_url).rstrip('/')}/ivr/gather"
        twiml = f"""
<Response>
    <Gather
        input="speech"
        action="{escape(gather_url)}"
        method="POST"
        language="hi-IN"
        speechTimeout="auto"
        actionOnEmptyResult="true">

        <Say language="hi-IN">
            नमस्ते किसान जी। KrishiSetu AI में आपका स्वागत है।
            कृपया बीप के बाद अपनी खेती की समस्या बताएं।
        </Say>
    </Gather>

    <Say language="hi-IN">
        आपकी आवाज़ प्राप्त नहीं हुई।
        कृपया दोबारा कॉल करें। धन्यवाद।
    </Say>
</Response>
"""
        logger.info("IVR incoming call; gather callback %s", gather_url)
        return _twiml(twiml)

    @app.post("/ivr/gather")
    async def ivr_gather(request: Request) -> Response:
        """Run the recognized speech through the safety-gated pipeline.

        This is the voice-channel enforcement of the CCI gate: escalate ->
        neutral Hindi hand-off only; deliver -> validated advisory actions.
        """
        form = await request.form()
        recognized_text = (form.get("SpeechResult") or "").strip()
        call_sid = (form.get("CallSid") or "").strip()

        if not recognized_text:
            twiml = """
<Response>
    <Say language="hi-IN">
        क्षमा करें, आपकी समस्या समझ नहीं आई।
        कृपया दोबारा कॉल करें और अपनी समस्या स्पष्ट रूप से बताएं।
    </Say>
</Response>
"""
            return _twiml(twiml)

        logger.info("IVR call %s query=%r", call_sid or "<no-sid>", recognized_text)

        pipeline = _get_pipeline()
        # The 4-tier decision core (LangGraph + Ollama render) is synchronous;
        # run it off the event loop so other webhooks keep answering.
        state = await run_in_threadpool(
            pipeline.run, recognized_text, IVR_PLOT_ID, False
        )

        if state.escalate:
            # fail_cci / dosage violation / untraceable output: never speak the
            # (withheld) advisory. The ticket context goes to the KVK officer.
            tts_text = ESCALATION_HANDOFF_HI
            logger.info(
                "IVR call %s ESCALATED (gate=%s cci=%s ticket=%s)",
                call_sid,
                getattr(state, "gate_outcome", None),
                getattr(state, "cci_value", None),
                getattr(getattr(state, "escalation_ticket", None), "ticket_id", None),
            )
        else:
            actions = [
                a.text for a in (state.advisory.actions if state.advisory else ())
            ]
            tts_text = " ".join(actions).strip()
            if not tts_text:
                # No validated advisory survived the gates; never speak
                # untraceable content — fall back to the neutral hand-off.
                tts_text = ESCALATION_HANDOFF_HI
            logger.info(
                "IVR call %s DELIVER (gate=%s cci=%s actions=%d)",
                call_sid,
                getattr(state, "gate_outcome", None),
                getattr(state, "cci_value", None),
                len(state.advisory.actions) if state.advisory else 0,
            )

        try:
            audio = await app.state.bhashini.text_to_speech(tts_text, language="hi")
        except Exception as exc:
            logger.warning("IVR TTS failed for call %s: %s", call_sid, exc)
            twiml = """
<Response>
    <Say language="hi-IN">
        क्षमा करें, अभी आपका जवाब तैयार नहीं हो पाया।
        कृपया कुछ समय बाद दोबारा प्रयास करें।
    </Say>
</Response>
"""
            return _twiml(twiml)

        # Per-call audio file (the shared single-file prototype raced on
        # concurrent calls). Invalid/missing CallSid gets a minted token.
        safe_sid = call_sid if _CALL_SID_RE.fullmatch(call_sid) else uuid.uuid4().hex
        IVR_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = IVR_AUDIO_DIR / f"{safe_sid}.wav"
        try:
            audio_path.write_bytes(audio)
        except OSError as exc:
            logger.warning("IVR audio save failed for call %s: %s", call_sid, exc)
            twiml = """
<Response>
    <Say language="hi-IN">
        क्षमा करें, आपका जवाब चलाया नहीं जा सका।
    </Say>
</Response>
"""
            return _twiml(twiml)

        audio_url = f"{str(request.base_url).rstrip('/')}/ivr/audio/{safe_sid}"
        if state.escalate:
            twiml = f"""
<Response>
    <Play>{escape(audio_url)}</Play>
    <Say language="hi-IN">धन्यवाद किसान जी।</Say>
</Response>
"""
        else:
            twiml = f"""
<Response>
    <Say language="hi-IN">
        आपकी समस्या के लिए KrishiSetu AI का सुझाव है।
    </Say>
    <Play>{escape(audio_url)}</Play>
    <Say language="hi-IN">
        धन्यवाद किसान जी। KrishiSetu AI का उपयोग करने के लिए धन्यवाद।
    </Say>
</Response>
"""
        return _twiml(twiml)

    @app.get("/ivr/audio/{call_sid}")
    async def ivr_audio(call_sid: str) -> Response:
        """Serve the Bhashini audio for one call (no-store, path-safe)."""
        if not _CALL_SID_RE.fullmatch(call_sid):
            return Response(
                content="Invalid call id.", status_code=404, media_type="text/plain"
            )
        audio_path = IVR_AUDIO_DIR / f"{call_sid}.wav"
        if not audio_path.is_file():
            return Response(
                content="Audio file not found.",
                status_code=404,
                media_type="text/plain",
            )
        try:
            audio = audio_path.read_bytes()
        except OSError:
            return Response(
                content="Unable to read audio file.",
                status_code=500,
                media_type="text/plain",
            )
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
