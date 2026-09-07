"""Подключение автономного ансамбля к проверенному HTTP-контракту."""

import os
from pathlib import Path

from app.main import create_app
from app.predictor import Predictor


class EnsemblePredictor(Predictor):
    """Загружает serving-пакет только при запуске production-приложения."""

    def __init__(self) -> None:
        """Читает конфигурацию; переменная окружения не обязательна в образе."""
        from uzner.serving.config import RuntimeConfig
        from uzner.serving.runtime import ResidentEnsemble

        config_path = Path(os.environ.get("UZNER_SERVICE_CONFIG", "configs/serving/default.json"))
        self.runtime = ResidentEnsemble(RuntimeConfig.read(config_path))

    def predict(self, texts: list[str]):
        """Возвращает общие Entity с исходными Unicode offsets."""
        return self.runtime.predict(texts)


app = create_app(predictor_factory=EnsemblePredictor)
