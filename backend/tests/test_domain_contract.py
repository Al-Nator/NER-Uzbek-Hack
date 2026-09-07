"""Проверки структурного контракта результатов модели."""

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.main import create_app


@dataclass(frozen=True)
class ModelEntity:
    """Dataclass с полями общего типа uzner.domain.Entity."""

    label: str
    start: int
    end: int
    score: float | None = None


def test_model_dataclasses_and_tuple_batches_preserve_unicode_offsets(settings, predictor):
    """Принимает доменные spans без зависимости модели от HTTP-схем."""
    predictor.predict = lambda texts: ((ModelEntity("NAME", 1, 4, 0.9),), ())
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post(
            "/api/v1/predict", json=[{"hash": "one", "text": "🙂Ali"}, {"hash": "two", "text": ""}]
        )
    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {"hash": "one", "entities": [{"label": "NAME", "start": 1, "end": 4}]},
            {"hash": "two", "entities": []},
        ]
    }


def test_dataclass_spans_are_validated_atomically(settings, predictor):
    """Не пропускает неверные координаты из объектов модели."""
    predictor.predict = lambda texts: [[ModelEntity("NAME", 0, 3)], [ModelEntity("GEO", 0, 99)]]
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post(
            "/api/v1/predict", json=[{"hash": "one", "text": "Ali"}, {"hash": "two", "text": "joy"}]
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_error"
