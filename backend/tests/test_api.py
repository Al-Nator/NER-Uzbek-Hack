"""Проверки HTTP-сервиса."""

import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import Entity


def test_unconfigured_service_is_alive_but_not_ready(settings):
    """Проверяет служебные маршруты до подключения модели."""
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/livez").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["error"]["code"] == "model_unavailable"
        response = client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}])
        assert response.status_code == 503


def test_load_once_health_does_not_infer_and_shutdown_closes(settings, predictor):
    """Проверяет однократную загрузку, лёгкий healthcheck и освобождение ресурсов."""
    loaded = []

    def factory():
        """Создаёт обработчик в проверяемом сценарии запуска."""
        loaded.append(True)
        return predictor

    app = create_app(settings=settings, predictor_factory=factory)
    assert loaded == []
    with TestClient(app) as client:
        for _ in range(3):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        assert predictor.calls == []
        for _ in range(2):
            assert (
                client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}]).status_code
                == 200
            )
        assert loaded == [True]
    assert predictor.closed
    assert not app.state.service.ready


def test_batch_preserves_original_text_hash_order_and_unicode_offsets(settings, predictor):
    """Проверяет исходный текст, порядок, hash и Unicode-границы."""
    documents = [
        {"hash": "кириллица", "text": "🧑🏽‍💻 Алишер Навоий"},
        {"hash": "latin", "text": "  Ali Toshkent shahrida ishlaydi.\n"},
        {"hash": "empty", "text": ""},
        {"hash": "whitespace", "text": " \n\t "},
        {"hash": "no-entities", "text": "salom dunyo"},
    ]
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=documents)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    results = response.json()["data"]
    assert len(results) == len(documents)
    assert [result["hash"] for result in results] == [document["hash"] for document in documents]
    assert sum(len(result["entities"]) for result in results) == 3
    for result, document in zip(results, documents, strict=True):
        seen = set()
        for entity in result["entities"]:
            assert entity["label"] in {"ORG", "NAME", "GEO"}
            assert type(entity["start"]) is int and type(entity["end"]) is int
            assert 0 <= entity["start"] < entity["end"] <= len(document["text"])
            key = (entity["label"], entity["start"], entity["end"])
            assert key not in seen
            seen.add(key)
    assert predictor.calls == [[document["text"] for document in documents]]
    assert results[1]["entities"] == [
        {"label": "NAME", "start": 2, "end": 5},
        {"label": "GEO", "start": 6, "end": 14},
    ]
    entity = results[0]["entities"][0]
    assert documents[0]["text"][entity["start"] : entity["end"]] == "Навоий"
    assert all(result["entities"] == [] for result in results[2:])


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        None,
        "text",
        {"data": [{"hash": "1", "text": "Ali"}]},
        [{"hash": "1"}],
        [{"text": "Ali"}],
        [{"hash": "", "text": "Ali"}],
        [{"hash": 1, "text": "Ali"}],
        [{"hash": True, "text": "Ali"}],
        [{"hash": "1", "text": 123}],
        [{"hash": "1", "text": None}],
        [{"hash": "1", "text": ["Ali"]}],
        [{"hash": "1", "text": "Ali"}, {"hash": "1", "text": "Toshkent"}],
        [{"hash": "valid", "text": "Ali"}, {"hash": "invalid", "text": False}],
    ],
)
def test_invalid_request_is_atomic_422(settings, predictor, payload):
    """Отклоняет некорректный пакет без частичного инференса."""
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["details"]
    assert predictor.calls == []


@pytest.mark.parametrize(
    "body",
    [b"{not-json", b'[{"hash":"x","text":"\\ud800"}]', b"\xff"],
)
def test_invalid_encoding_and_json_are_client_errors(settings, predictor, body):
    """Проверяет клиентские ошибки кодировки и JSON."""
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post(
            "/api/v1/predict", content=body, headers={"Content-Type": "application/json"}
        )
    assert 400 <= response.status_code < 500
    assert predictor.calls == []


@pytest.mark.parametrize(
    ("overrides", "documents"),
    [
        ({"max_batch_size": 1}, [{"hash": "a", "text": ""}, {"hash": "b", "text": ""}]),
        ({"max_text_length": 2}, [{"hash": "a", "text": "Ali"}]),
        (
            {"max_total_characters": 5},
            [{"hash": "a", "text": "Ali"}, {"hash": "b", "text": "Ali"}],
        ),
    ],
)
def test_limits_reject_without_truncation_or_partial_inference(predictor, overrides, documents):
    """Отклоняет превышение лимитов без обрезки текста."""
    settings = Settings(_env_file=None, **overrides)
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=documents)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert predictor.calls == []


def test_long_text_limit_counts_unicode_characters_and_preserves_tail(settings, predictor):
    """Проверяет Unicode-лимит и сущность в конце длинного текста."""
    text = "🙂" * (settings.max_text_length - 8) + "Toshkent"
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=[{"hash": "long", "text": text}])
    assert response.status_code == 200
    assert response.json()["data"][0]["entities"] == [
        {"label": "GEO", "start": 49_992, "end": 50_000}
    ]
    assert predictor.calls == [[text]]


@pytest.mark.parametrize(
    "result",
    [
        [],
        [[], []],
        None,
        [[{"label": "PER", "start": 0, "end": 3}]],
        [[{"label": "NAME", "start": True, "end": 3}]],
        [[{"label": "NAME", "start": "0", "end": 3}]],
        [[{"label": "NAME", "start": 0.0, "end": 3}]],
        [[{"label": "NAME", "start": -1, "end": 3}]],
        [[{"label": "NAME", "start": 0, "end": 4}]],
        [[{"label": "NAME", "start": 1, "end": 1}]],
        [[{"label": "NAME", "start": 0, "end": 3}] * 2],
        [[Entity.model_construct(label="NAME", start=-1, end=3)]],
    ],
)
def test_invalid_model_output_is_500(settings, predictor, result):
    """Проверяет отказ при некорректных предсказаниях."""
    predictor.predict = lambda texts: result
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}])
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "inference_error", "message": "Prediction failed.", "details": []}
    }


def test_inference_exception_does_not_leak_details_and_next_request_works(settings, predictor):
    """Проверяет скрытие деталей ошибки и последующий успешный запрос."""
    original_predict = predictor.predict

    def fail_once(texts):
        """Имитирует однократный сбой модели."""
        predictor.predict = original_predict
        raise RuntimeError("internal model file /private/model.bin")

    predictor.predict = fail_once
    with TestClient(create_app(settings=settings, predictor_factory=lambda: predictor)) as client:
        response = client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}])
        assert response.status_code == 500
        assert "/private" not in response.text
        assert (
            client.post("/api/v1/predict", json=[{"hash": "1", "text": "Ali"}]).status_code == 200
        )


def test_factory_failure_prevents_startup(settings):
    """Отклоняет запуск при ошибке фабрики."""

    def factory():
        """Создаёт обработчик в проверяемом сценарии запуска."""
        raise RuntimeError("Cannot load model")

    with pytest.raises(RuntimeError, match="Cannot load model"):
        with TestClient(create_app(settings=settings, predictor_factory=factory)):
            pytest.fail("Application must not start with a failed factory")


def test_factory_must_return_a_predictor(settings):
    """Отклоняет фабрику с некорректным результатом."""
    with pytest.raises(TypeError, match="must return a Predictor"):
        with TestClient(create_app(settings=settings, predictor_factory=lambda: None)):
            pytest.fail("Application must not start with an invalid factory")


def test_cors_only_allows_configured_origins():
    """Проверяет ограничения CORS."""
    settings = Settings(_env_file=None, cors_origins=["http://localhost:5173"])
    with TestClient(create_app(settings=settings)) as client:
        allowed = client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
        denied = client.get("/livez", headers={"Origin": "http://example.com"})
        assert "access-control-allow-origin" not in denied.headers


def test_openapi_describes_array_body_and_response_errors(settings):
    """Проверяет контракт запроса и ошибок в OpenAPI."""
    with TestClient(create_app(settings=settings)) as client:
        schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/predict"]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    body_schema = schema["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    assert body_schema["type"] == "array"
    assert body_schema["minItems"] == 1
    assert {"200", "413", "422", "500", "503"} <= operation["responses"].keys()


def test_routing_errors_share_json_format(settings):
    """Проверяет единый формат ошибок маршрутизации."""
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/missing").json()["error"]["code"] == "not_found"
        response = client.get("/api/v1/predict")
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"
        assert "POST" in response.headers["allow"]


@pytest.mark.anyio
async def test_inference_is_serialized_and_health_remains_responsive(settings, predictor):
    """Проверяет последовательный инференс и доступность healthcheck."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_predict(texts):
        """Блокирует инференс до сигнала теста."""
        calls.append(texts)
        started.set()
        if not release.wait(timeout=5):
            raise RuntimeError("Test did not release inference")
        return [[] for _ in texts]

    predictor.predict = slow_predict
    app = create_app(settings=settings, predictor_factory=lambda: predictor)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/api/v1/predict", json=[{"hash": "a", "text": "first"}])
            )
            second = None
            try:
                assert await asyncio.to_thread(started.wait, 2)
                second = asyncio.create_task(
                    client.post("/api/v1/predict", json=[{"hash": "b", "text": "second"}])
                )
                health = await asyncio.wait_for(client.get("/healthz"), timeout=1)
                assert health.status_code == 200
                await asyncio.sleep(0.05)
                assert calls == [["first"]]
            finally:
                release.set()
                pending = [first] + ([second] if second is not None else [])
                responses = await asyncio.gather(*pending)
            assert all(response.status_code == 200 for response in responses)
            assert calls == [["first"], ["second"]]
