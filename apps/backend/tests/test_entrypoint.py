"""Проверки HTTP-сервиса."""

import os
import subprocess
import sys
from pathlib import Path


def test_default_entrypoint_starts_without_ml_dependencies_or_checkpoint(tmp_path):
    """Запускает точку входа без ML-библиотек и весов."""
    script = """
import sys

# Проверяем точку входа без доступного ML-окружения.
for module in ("torch", "transformers", "tokenizers", "app.baseline"):
    sys.modules[module] = None

from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    assert client.get("/livez").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/healthz").status_code == 503
    result = client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}])
    assert result.status_code == 503
    assert result.json() == {
        "error": {"code": "model_unavailable", "message": "Model is not connected.", "details": []}
    }
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
