"""Регрессии контракта HTTP-команд организаторов без загрузки моделей."""

import json

import httpx
import pytest

from uzner.domain import Document, Entity
from uzner.serving import http_client
from uzner.serving.http_client import HttpSettings, request, wait_ready
from uzner.serving.predict_job import PredictionJob, predict_file


@pytest.mark.parametrize(
    "url",
    [
        "localhost:8000",
        "ftp://host",
        "http://",
        "http://x/predict",
        "http://u:p@x",
        "http://x?q=1",
        "http://x#hash",
        "http://x:99999",
        "http://x:bad",
    ],
)
def test_invalid_service_urls(url):
    """Не принимает endpoint вместо базового URL и некорректный порт."""
    with pytest.raises(ValueError):
        HttpSettings(url)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts(timeout):
    """Не допускает бесконечного ожидания по пользовательскому значению."""
    with pytest.raises(ValueError):
        HttpSettings(timeout=timeout)
    with pytest.raises(ValueError):
        wait_ready(None, timeout)


def test_unicode_request_never_sends_gold():
    """Сохраняет emoji/кириллицу и передаёт только hash/text, не gold."""
    document = Document("x", "🙂 Али Toshkent", (Entity(2, 5, "NAME"),))
    seen = []

    def respond(incoming):
        """Возвращает корректные символьные границы в исходной строке."""
        seen.append(json.loads(incoming.content))
        return httpx.Response(
            200,
            json={"data": [{"hash": "x", "entities": [{"label": "NAME", "start": 2, "end": 5}]}]},
        )

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        result = request(client, (document,))
    assert seen == [[{"hash": "x", "text": document.text}]]
    assert result.predictions[0].entities == document.entities
    assert result.seconds >= 0


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": None},
        {"data": []},
        {"data": [{"hash": "x"}]},
        {"data": [{"hash": "wrong", "entities": []}]},
        {"data": [{"hash": "x", "entities": {}}]},
        {"data": [{"hash": "x", "entities": ["ORG"]}]},
        {"data": [{"hash": "x", "entities": [{"label": "BAD", "start": 0, "end": 1}]}]},
        {"data": [{"hash": "x", "entities": [{"label": "NAME", "start": True, "end": 1}]}]},
        {"data": [{"hash": "x", "entities": [{"label": "NAME", "start": 0, "end": 20}]}]},
        {
            "data": [
                {
                    "hash": "x",
                    "entities": [
                        {"label": "NAME", "start": 0, "end": 2},
                        {"label": "ORG", "start": 1, "end": 3},
                    ],
                }
            ]
        },
    ],
)
def test_malformed_response_is_not_published(payload):
    """Не принимает неполный ответ, неверные классы или потерянные offsets."""
    transport = httpx.MockTransport(lambda incoming: httpx.Response(200, json=payload))
    with (
        httpx.Client(base_url="http://test", transport=transport) as client,
        pytest.raises((ValueError, TypeError)),
    ):
        request(client, (Document("x", "Ali", ()),))


def test_empty_and_duplicate_batch_rejected_before_network():
    """Не посылает неоднозначный пакет в API."""
    with pytest.raises(ValueError):
        request(None, ())
    doc = Document("x", "Ali", ())
    with pytest.raises(ValueError):
        request(None, (doc, doc))


def test_readiness_retries_503_and_transport_error(monkeypatch):
    """Ожидает прогрев сервиса, но проверяет содержимое health JSON."""
    attempts = iter([None, 503, 200])

    def respond(incoming):
        """Имитирует отсутствие сокета, прогрев и готовность."""
        status = next(attempts)
        if status is None:
            raise httpx.ConnectError("starting", request=incoming)
        return httpx.Response(status, json={"status": "ok"})

    monkeypatch.setattr(http_client, "sleep", lambda value: None)
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        wait_ready(client)


@pytest.mark.parametrize("status,payload", [(200, {}), (200, []), (404, {}), (302, {})])
def test_readiness_does_not_accept_wrong_service(status, payload):
    """HTML-прокси и чужой 200 не считаются готовой NER-моделью."""
    transport = httpx.MockTransport(lambda incoming: httpx.Response(status, json=payload))
    with (
        httpx.Client(base_url="http://test", transport=transport) as client,
        pytest.raises((ValueError, httpx.HTTPStatusError)),
    ):
        wait_ready(client)


def test_readiness_has_deadline(monkeypatch):
    """Проверяет конечный deadline без ожидания настоящих секунд."""
    ticks = iter([0.0, 0.1, 1.1, 1.2])
    monkeypatch.setattr(http_client, "perf_counter", lambda: next(ticks))
    transport = httpx.MockTransport(lambda incoming: httpx.Response(503))
    with (
        httpx.Client(base_url="http://test", transport=transport) as client,
        pytest.raises(TimeoutError, match="503"),
    ):
        wait_ready(client, 1.0)


@pytest.mark.parametrize("batch", [True, 0, -1, 9, 2.5, "8"])
def test_predict_batch_validation(tmp_path, batch):
    """Размер API-пакета задаётся целым числом от 1 до 8."""
    with pytest.raises(ValueError):
        PredictionJob(tmp_path / "input", tmp_path / "output", batch_size=batch)


def _write_inputs(path, count=9):
    """Пишет несколько размеченных строк для проверки удаления gold из запросов."""
    path.write_text(
        "".join(
            json.dumps(
                {
                    "hash": str(i),
                    "text": "🙂 Али",
                    "entities": [{"label": "NAME", "start": 2, "end": 5}],
                },
                ensure_ascii=False,
            )
            + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )


def test_predict_file_complete_manifest_and_no_overwrite(tmp_path):
    """Публикует ровно все строки и отказывается затирать предыдущую посылку."""
    source = tmp_path / "inputs.jsonl"
    _write_inputs(source)
    batches = []

    def respond(incoming):
        """Имитирует финальный сервис, не используя annotations входного файла."""
        if incoming.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        rows = json.loads(incoming.content)
        assert all(set(row) == {"hash", "text"} for row in rows)
        batches.append(len(rows))
        return httpx.Response(
            200, json={"data": [{"hash": row["hash"], "entities": []} for row in rows]}
        )

    job = PredictionJob(source, tmp_path / "result")
    output = predict_file(job, transport=httpx.MockTransport(respond))
    assert batches == [8, 1]
    assert [json.loads(line)["hash"] for line in output.read_text().splitlines()] == [
        str(i) for i in range(9)
    ]
    manifest = json.loads((job.output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["documents"] == 9
    assert not manifest["gold_sent_to_service"]
    assert len(manifest["predictions_sha256"]) == 64
    with pytest.raises(FileExistsError):
        predict_file(job)


def test_predict_file_second_batch_failure_never_creates_partial_submission(tmp_path):
    """Падение после первого batch не оставляет готовый неполный predictions.jsonl."""
    source = tmp_path / "inputs.jsonl"
    _write_inputs(source)

    def respond(incoming):
        """Завершает первый batch и роняет второй."""
        if incoming.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        rows = json.loads(incoming.content)
        if len(rows) == 1:
            return httpx.Response(500)
        return httpx.Response(
            200, json={"data": [{"hash": row["hash"], "entities": []} for row in rows]}
        )

    job = PredictionJob(source, tmp_path / "result")
    with pytest.raises(httpx.HTTPStatusError):
        predict_file(job, transport=httpx.MockTransport(respond))
    assert not (job.output / "predictions.jsonl").exists()
    assert not (job.output / "manifest.json").exists()
    assert json.loads((job.output / "failure.json").read_text())["status"] == "failed"


def test_predict_file_empty_input_does_not_create_output(tmp_path):
    """Проверяет данные до создания каталога результата и обращения к сервису."""
    source = tmp_path / "empty.jsonl"
    source.touch()
    with pytest.raises(ValueError, match="пустым"):
        predict_file(PredictionJob(source, tmp_path / "result"))
    assert not (tmp_path / "result").exists()
