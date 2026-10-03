"""Tests for the Twilio/Bhashini IVR webhook integration.

The voice channel must never bypass the safety gates: the gather route is
tested with stub pipelines (DELIVER and ESCALATE) plus a stub TTS so no
network, Ollama, or Bhashini dependency is needed.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

import krishisethu.api.main as api_main
from krishisethu.api.main import ESCALATION_HANDOFF_HI, IVR_PLOT_ID, create_app
from krishisethu.api.services.bhashini import BhashiniError, BhashiniService
from krishisethu.config.settings import BhashiniSettings

DELIVER_QUERY = "इस हफ्ते गन्ने की फसल के लिए क्या करें?"
ESCALATE_QUERY = "Zinc spray karna hai kya?"
VALID_CALL_SID = "CA" + "1" * 32


class StubPipeline:
    """Duck-typed TierOrchestrator returning a canned final state."""

    def __init__(self, *, escalate: bool, action_texts: tuple[str, ...] = ()) -> None:
        self.escalate = escalate
        self.action_texts = action_texts
        self.calls: list[tuple] = []

    def run(self, query: str, plot_id: str = "demo", demo: bool = False):
        self.calls.append((query, plot_id, demo))
        # A poisoned advisory on escalate proves the route checks the gate
        # BEFORE ever reading the advisory: this text must never be spoken.
        actions = tuple(SimpleNamespace(text=t) for t in self.action_texts)
        return SimpleNamespace(
            escalate=self.escalate,
            advisory=SimpleNamespace(actions=actions),
            gate_outcome="fail_cci" if self.escalate else "pass_exact",
            cci_value=0.23 if self.escalate else 1.0,
            escalation_ticket=(
                SimpleNamespace(ticket_id="KVK-TEST") if self.escalate else None
            ),
        )


class StubBhashini:
    def __init__(self, audio: bytes = b"RIFF-stub-wave-audio") -> None:
        self.audio = audio
        self.tts_calls: list[tuple[str, str]] = []

    async def text_to_speech(self, text: str, language: str = "hi") -> bytes:
        self.tts_calls.append((text, language))
        return self.audio


class FailingBhashini:
    async def text_to_speech(self, text: str, language: str = "hi") -> bytes:
        raise BhashiniError("Bhashini is down")


def _client(pipeline=None, bhashini=None) -> httpx.AsyncClient:
    app = create_app(pipeline=pipeline, bhashini=bhashini)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_ivr_incoming_returns_gather_twiml():
    async with _client() as c:
        r = await c.post("/ivr/incoming")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xml")
    assert "<Gather" in r.text
    assert 'language="hi-IN"' in r.text
    assert "/ivr/gather" in r.text
    assert "actionOnEmptyResult" in r.text


async def test_ivr_gather_routes_query_through_pipeline_and_delivers(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "IVR_AUDIO_DIR", tmp_path)
    pipeline = StubPipeline(
        escalate=False,
        action_texts=(
            "Suspend irrigation for the next 72 hours.",
            "Clear furrow outlets to drain excess water.",
        ),
    )
    bhashini = StubBhashini()

    async with _client(pipeline=pipeline, bhashini=bhashini) as c:
        r = await c.post(
            "/ivr/gather",
            data={"SpeechResult": DELIVER_QUERY, "CallSid": VALID_CALL_SID},
        )
        assert r.status_code == 200
        assert "<Play>" in r.text
        assert f"/ivr/audio/{VALID_CALL_SID}" in r.text
        assert "का सुझाव है" in r.text  # delivery intro, not the escalation path

        audio = await c.get(f"/ivr/audio/{VALID_CALL_SID}")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content == b"RIFF-stub-wave-audio"

    # The spoken text is exactly the pipeline's validated advisory actions.
    assert pipeline.calls == [(DELIVER_QUERY, IVR_PLOT_ID, False)]
    assert len(bhashini.tts_calls) == 1
    spoken_text, language = bhashini.tts_calls[0]
    assert language == "hi"
    assert "Suspend irrigation for the next 72 hours." in spoken_text
    assert "Clear furrow outlets" in spoken_text


async def test_ivr_gather_escalation_never_speaks_the_advisory(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "IVR_AUDIO_DIR", tmp_path)
    pipeline = StubPipeline(
        escalate=True,
        action_texts=("POISON chlorpyrifos 2.5 ml/L never speak this",),
    )
    bhashini = StubBhashini()

    async with _client(pipeline=pipeline, bhashini=bhashini) as c:
        r = await c.post(
            "/ivr/gather",
            data={"SpeechResult": ESCALATE_QUERY, "CallSid": VALID_CALL_SID},
        )
        assert r.status_code == 200
        assert "<Play>" in r.text
        assert "का सुझाव है" not in r.text  # never framed as an advisory

    assert pipeline.calls == [(ESCALATE_QUERY, IVR_PLOT_ID, False)]
    assert len(bhashini.tts_calls) == 1
    spoken_text, language = bhashini.tts_calls[0]
    assert language == "hi"
    # The ONLY thing the caller hears is the neutral hardcoded hand-off.
    assert spoken_text == ESCALATION_HANDOFF_HI
    assert "POISON" not in spoken_text
    assert "2.5" not in spoken_text


async def test_ivr_gather_no_advisory_actions_falls_back_to_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "IVR_AUDIO_DIR", tmp_path)
    pipeline = StubPipeline(escalate=False, action_texts=())
    bhashini = StubBhashini()

    async with _client(pipeline=pipeline, bhashini=bhashini) as c:
        r = await c.post(
            "/ivr/gather",
            data={"SpeechResult": DELIVER_QUERY, "CallSid": VALID_CALL_SID},
        )
        assert r.status_code == 200

    assert pipeline.calls == [(DELIVER_QUERY, IVR_PLOT_ID, False)]
    assert bhashini.tts_calls[0][0] == ESCALATION_HANDOFF_HI


async def test_ivr_gather_empty_speech_retries_without_pipeline():
    pipeline = StubPipeline(escalate=False)
    bhashini = StubBhashini()

    async with _client(pipeline=pipeline, bhashini=bhashini) as c:
        r = await c.post("/ivr/gather", data={"SpeechResult": "", "CallSid": VALID_CALL_SID})

    assert r.status_code == 200
    assert "समझ नहीं आई" in r.text
    assert pipeline.calls == []
    assert bhashini.tts_calls == []


async def test_ivr_gather_tts_failure_returns_spoken_apology():
    pipeline = StubPipeline(escalate=False, action_texts=("Suspend irrigation.",))
    bhashini = FailingBhashini()

    async with _client(pipeline=pipeline, bhashini=bhashini) as c:
        r = await c.post(
            "/ivr/gather",
            data={"SpeechResult": DELIVER_QUERY, "CallSid": VALID_CALL_SID},
        )

    assert r.status_code == 200
    assert "<Play>" not in r.text
    assert "जवाब तैयार नहीं" in r.text
    assert pipeline.calls == [(DELIVER_QUERY, IVR_PLOT_ID, False)]


async def test_ivr_audio_rejects_invalid_ids_and_missing_files():
    async with _client() as c:
        # Too short / wrong charset -> rejected before touching the filesystem.
        assert (await c.get("/ivr/audio/short")).status_code == 404
        assert (await c.get("/ivr/audio/..%2Fsecrets")).status_code == 404
        # Well-formed sid with no audio on disk -> clean 404.
        assert (await c.get(f"/ivr/audio/{VALID_CALL_SID}")).status_code == 404


# ---------------------------------------------------------------------------
# BhashiniService unit tests (hermetic via httpx.MockTransport)
# ---------------------------------------------------------------------------
def _tts_ok_transport(audio: bytes) -> httpx.MockTransport:
    encoded = base64.b64encode(audio).decode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        task = payload["pipelineTasks"][0]
        assert task["taskType"] == "tts"
        assert task["config"]["language"]["sourceLanguage"] == "hi"
        assert payload["inputData"]["input"][0]["source"] == "नमस्ते किसान जी"
        assert request.headers["Authorization"] == "test-key"
        return httpx.Response(
            200, json={"pipelineResponse": [{"audio": [{"audioContent": encoded}]}]}
        )

    return httpx.MockTransport(handler)


async def test_bhashini_tts_returns_decoded_audio():
    svc = BhashiniService(
        settings=BhashiniSettings(inference_api_key="test-key"),
        transport=_tts_ok_transport(b"RIFF-fake-bytes"),
    )
    audio = await svc.text_to_speech("नमस्ते किसान जी", language="hi")
    assert audio == b"RIFF-fake-bytes"


async def test_bhashini_tts_requires_api_key():
    svc = BhashiniService(settings=BhashiniSettings(inference_api_key=""))
    with pytest.raises(BhashiniError):
        await svc.text_to_speech("नमस्ते")


async def test_bhashini_tts_raises_when_no_audio_content():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"pipelineResponse": []})
    )
    svc = BhashiniService(
        settings=BhashiniSettings(inference_api_key="test-key"), transport=transport
    )
    with pytest.raises(BhashiniError):
        await svc.text_to_speech("नमस्ते")


async def test_bhashini_asr_extracts_source_text():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["pipelineTasks"][0]["taskType"] == "asr"
        assert "audioContent" in payload["inputData"]["audio"][0]
        return httpx.Response(
            200, json={"pipelineResponse": [{"output": [{"source": "गेहूं की बुवाई"}]}]}
        )

    svc = BhashiniService(
        settings=BhashiniSettings(inference_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    text = await svc.speech_to_text(b"\x00\x01\x02", language="hi")
    assert text == "गेहूं की बुवाई"
