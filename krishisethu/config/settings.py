"""Single source of truth for KrishiSetu AI configuration.

This is the ONLY module that reads the environment / .env. Every other module
imports ``Settings`` via :func:`get_settings`. Env mapping, by prefix:

    OLLAMA_*      -> OllamaSettings
    CHROMA_*      -> ChromaSettings
    RAG_*         -> RagSettings
    CONFIDENCE_*  -> GateSettings
    APP_*         -> AppSettings
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Connection + model selection for the local Ollama stack."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    host: str = "http://localhost:11434"
    base_url: str | None = None  # derive from host when absent
    llm_model: str = "qwen3.5:4b"  # Tier-4 paraphraser
    llm_fallback: str = "qwen3.5:2b"  # fallback paraphraser
    embedding_model: str = "bge-m3:567m"  # multilingual embedding (1024-d)

    @model_validator(mode="after")
    def _normalize(self) -> "OllamaSettings":
        """Strip a trailing slash; derive base_url from host when it is absent."""
        self.host = self.host.rstrip("/")
        self.base_url = (self.base_url or self.host).rstrip("/")
        return self

    def embed_base(self) -> str:
        return self.base_url or self.host


class ChromaSettings(BaseSettings):
    """Persistent vector store (ChromaDB) configuration."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")

    persist_dir: str = "./data/chroma"
    collection_name: str = "krishisethu_kvk"

    @model_validator(mode="after")
    def _absolute_persist_dir(self) -> "ChromaSettings":
        self.persist_dir = str(Path(self.persist_dir).resolve())
        return self


class RagSettings(BaseSettings):
    """Retrieval tuning (safe-by-construction defaults)."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    top_k: int = 5  # HARD CAP from the white paper

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("RAG_TOP_K must be >= 1")
        return v


class GateSettings(BaseSettings):
    """Composite Confidence Index weights + gate threshold (white paper Sec IV-B)."""

    model_config = SettingsConfigDict(env_prefix="CONFIDENCE_", env_file=".env", extra="ignore")

    confidence_gate: float = 0.85
    w_rule: float = 0.50
    w_rag: float = 0.30
    w_vision: float = 0.20

    @field_validator("w_rule", "w_rag", "w_vision")
    @classmethod
    def _weights_in_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence weights must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "GateSettings":
        total = self.w_rule + self.w_rag + self.w_vision
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"confidence weights must sum to 1.0, got {total:.6f}")
        return self

    def redistribute_without_vision(self) -> tuple[float, float]:
        """Drop w_vision, renormalize w_rule + w_rag to fill 1.0 (keep ratio).

        Returns (w_rule', w_rag'); e.g. (0.50, 0.30, 0.20) -> (0.625, 0.375),
        sum == 1.0, ratio 5/3 preserved. Keeps CCI on the same 0-1 scale so the
        gate threshold stays comparable with no image present.
        """
        total = self.w_rule + self.w_rag
        if total <= 0:
            raise ValueError("w_rule + w_rag must be > 0 to redistribute")
        return (self.w_rule / total, self.w_rag / total)


class AppSettings(BaseSettings):
    """FastAPI server binding."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000


class BhashiniSettings(BaseSettings):
    """Bhashini (Dhruva) inference credentials for the IVR voice channel."""

    model_config = SettingsConfigDict(env_prefix="BHASHINI_", env_file=".env", extra="ignore")

    inference_api_key: str = ""
    inference_url: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    # ULCA model-discovery credentials; not required for direct inference calls.
    ulca_api_key: str = ""
    user_id: str = ""
    config_url: str = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"


class TwilioSettings(BaseSettings):
    """Twilio credentials for the IVR webhook channel."""

    model_config = SettingsConfigDict(env_prefix="TWILIO_", env_file=".env", extra="ignore")

    account_sid: str = ""
    auth_token: str = ""
    phone_number: str = ""


class Settings(BaseSettings):
    """Aggregate configuration. Import this, never read dotenv/os.environ elsewhere."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama: OllamaSettings = OllamaSettings()
    chroma: ChromaSettings = ChromaSettings()
    rag: RagSettings = RagSettings()
    gate: GateSettings = GateSettings()
    app: AppSettings = AppSettings()
    bhashini: BhashiniSettings = BhashiniSettings()
    twilio: TwilioSettings = TwilioSettings()

    def summary(self) -> dict:
        """Plain-dict view for /health and /api/v1/status."""
        return {
            "ollama": {
                "host": self.ollama.host,
                "base_url": self.ollama.embed_base(),
                "llm_model": self.ollama.llm_model,
                "llm_fallback": self.ollama.llm_fallback,
                "embedding_model": self.ollama.embedding_model,
            },
            "chroma": {
                "persist_dir": self.chroma.persist_dir,
                "collection_name": self.chroma.collection_name,
            },
            "rag": {"top_k": self.rag.top_k},
            "gate": {
                "confidence_gate": self.gate.confidence_gate,
                "w_rule": self.gate.w_rule,
                "w_rag": self.gate.w_rag,
                "w_vision": self.gate.w_vision,
            },
            "app": {"host": self.app.host, "port": self.app.port},
            "ivr": {
                "bhashini_configured": bool(self.bhashini.inference_api_key),
                "twilio_configured": bool(self.twilio.account_sid and self.twilio.auth_token),
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Calling twice returns the same object."""
    return Settings()
