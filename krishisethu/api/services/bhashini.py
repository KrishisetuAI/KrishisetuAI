"""Bhashini (Dhruva) STT/TTS client for the IVR voice channel.

Migrated from the standalone ``KrishiSetuAI-IVR`` prototype. Two changes on
migration:

* configuration is resolved through :class:`BhashiniSettings` (the settings
  module is the single place the environment is read), and
* the HTTP calls are async (httpx) so the FastAPI webhook never blocks the
  event loop on a slow Bhashini request.

The service is a pure speech pipe: it never generates advisory content. Every
string it speaks is produced upstream by the safety-gated pipeline.
"""
from __future__ import annotations

import base64
from typing import Any

import httpx

from krishisethu.config.settings import BhashiniSettings, get_settings

ASR_SERVICE_ID = "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4"
TTS_SERVICE_ID = "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4"
_TTS_GENDER = "female"
_TTS_SAMPLING_RATE = 16000


class BhashiniError(RuntimeError):
    """Bhashini is unconfigured, unreachable, or returned an unusable payload."""


def _find_audio_content(obj: Any) -> str | None:
    """Depth-first search for ``audioContent`` anywhere in the response."""
    if isinstance(obj, dict):
        content = obj.get("audioContent")
        if content:
            return content
        for value in obj.values():
            found = _find_audio_content(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_audio_content(value)
            if found:
                return found
    return None


class BhashiniService:
    """Async Bhashini inference client (ASR + TTS).

    ``transport`` is injectable for hermetic tests (``httpx.MockTransport``);
    ``None`` uses the default transport.
    """

    def __init__(
        self,
        settings: BhashiniSettings | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings().bhashini
        self._timeout = timeout
        self._transport = transport

    # -- plumbing ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        key = self._settings.inference_api_key
        if not key:
            raise BhashiniError("BHASHINI_INFERENCE_API_KEY is not configured")
        return {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Authorization": key,
        }

    async def _post(self, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    self._settings.inference_url,
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise BhashiniError(f"Bhashini inference request failed: {exc}") from exc

    # -- ASR ----------------------------------------------------------------
    async def speech_to_text(self, audio_bytes: bytes, language: str = "hi") -> str:
        """Transcribe call audio (base64 in, recognized text out)."""
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": language},
                        "serviceId": ASR_SERVICE_ID,
                    },
                }
            ],
            "inputData": {"audio": [{"audioContent": audio_base64}]},
        }
        data = await self._post(payload)
        try:
            text = data["pipelineResponse"][0]["output"][0]["source"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BhashiniError("Bhashini ASR returned no recognized text") from exc
        if not text:
            raise BhashiniError("Bhashini ASR returned empty text")
        return str(text)

    # -- TTS ----------------------------------------------------------------
    async def text_to_speech(self, text: str, language: str = "hi") -> bytes:
        """Synthesize speech for a safety-gated reply string; returns WAV bytes."""
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {"sourceLanguage": language},
                        "serviceId": TTS_SERVICE_ID,
                        "gender": _TTS_GENDER,
                        "samplingRate": _TTS_SAMPLING_RATE,
                    },
                }
            ],
            "inputData": {"input": [{"source": text}]},
        }
        data = await self._post(payload)
        audio_content = _find_audio_content(data)
        if not audio_content:
            raise BhashiniError("Bhashini TTS returned no audioContent")
        try:
            return base64.b64decode(audio_content)
        except (ValueError, TypeError) as exc:
            raise BhashiniError("Bhashini TTS returned invalid base64 audio") from exc
