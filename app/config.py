"""Central app configuration. Every path/setting other modules need
is read from here — nothing should hardcode a path string elsewhere.
Uses pydantic-settings so values load from environment / .env
automatically and get validated once at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "models/text-embedding-001"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = ""

    # Data paths — relative paths are resolved against project root,
    # so behavior doesn't depend on the working directory `uv run` is
    # invoked from.
    kb_dir: str = "knowledge-base"
    orders_path: str = "data/orders.json"
    faiss_index_path: str = "storage/faiss_index"
    bm25_index_path: str = "storage/bm25_index.pkl"

    @property
    def kb_dir_path(self) -> Path:
        return self._resolve(self.kb_dir)

    @property
    def orders_path_resolved(self) -> Path:
        return self._resolve(self.orders_path)

    @property
    def faiss_index_path_resolved(self) -> Path:
        return self._resolve(self.faiss_index_path)

    @property
    def bm25_index_path_resolved(self) -> Path:
        return self._resolve(self.bm25_index_path)

    @staticmethod
    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else _PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
