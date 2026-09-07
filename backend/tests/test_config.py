"""Проверки HTTP-сервиса."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_read_environment(monkeypatch):
    """Проверяет чтение настроек из окружения."""
    monkeypatch.setenv("NER_MAX_BATCH_SIZE", "8")
    monkeypatch.setenv("NER_CORS_ORIGINS", '["http://localhost:5173"]')
    settings = Settings(_env_file=None)
    assert settings.max_batch_size == 8
    assert settings.cors_origins == ["http://localhost:5173"]


@pytest.mark.parametrize("field", ["max_batch_size", "max_text_length", "max_total_characters"])
def test_limits_must_be_positive(field):
    """Отклоняет неположительные лимиты."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})
