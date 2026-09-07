"""Проверки HTTP-сервиса."""

import pytest

from app.config import Settings
from app.predictor import Predictor
from app.schemas import Entity


class RecordingPredictor(Predictor):
    """Тестовый обработчик для проверки запросов и Unicode-координат."""

    def __init__(self):
        """Инициализирует состояние обработчика."""
        self.calls = []
        self.closed = False

    def predict(self, texts):
        """Возвращает сущности для документов с исходными символьными границами."""
        self.calls.append(texts.copy())
        results = []
        for text in texts:
            entities = []
            for word, label in [("Ali", "NAME"), ("Toshkent", "GEO"), ("Навоий", "NAME")]:
                start = text.find(word)
                if start >= 0:
                    entities.append(Entity(label=label, start=start, end=start + len(word)))
            results.append(entities)
        return results

    def close(self):
        """Освобождает ресурсы обработчика."""
        self.closed = True


@pytest.fixture
def settings():
    """Создаёт настройки теста без чтения файла окружения."""
    return Settings(_env_file=None)


@pytest.fixture
def predictor():
    """Создаёт отдельный тестовый обработчик."""
    return RecordingPredictor()


@pytest.fixture
def anyio_backend():
    """Выбирает asyncio для асинхронных тестов."""
    return "asyncio"
