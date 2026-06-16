from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "LLM RAG Document Assistant"
    app_version: str = "0.1.0"
    api_prefix: str = ""

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path | None = None
    chroma_collection: str = "documents"

    chunk_size: int = 1_000
    chunk_overlap: int = 150
    max_preview_chars: int = 240

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    use_mock_llm: bool | None = None
    use_mock_embeddings: bool | None = None

    request_timeout_seconds: float = 60.0
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_chroma_dir(self) -> Path:
        return self.chroma_dir or self.data_dir / "chroma"

    @property
    def should_use_mock_llm(self) -> bool:
        if self.use_mock_llm is not None:
            return self.use_mock_llm
        return not bool(self.openai_api_key)

    @property
    def should_use_mock_embeddings(self) -> bool:
        if self.use_mock_embeddings is not None:
            return self.use_mock_embeddings
        return not bool(self.openai_api_key)

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

