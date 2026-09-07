"""HTTP-слой сервиса распознавания сущностей."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Настройки HTTP-сервиса с проверкой переменных окружения."""

    model_config = SettingsConfigDict(
        env_prefix="NER_",
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_batch_size: int = Field(default=64, ge=1)
    max_text_length: int = Field(default=50_000, ge=1)
    max_total_characters: int = Field(default=500_000, ge=1)
    cors_origins: list[str] = Field(default_factory=list)
