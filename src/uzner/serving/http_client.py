"""Проверяемый HTTP-клиент без загрузки модели и без использования gold в запросе."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter, sleep
from urllib.parse import urlsplit

import httpx

from uzner.domain import Document, Prediction


@dataclass(frozen=True)
class HttpSettings:
    """Адрес сервиса и конечные таймауты организаторских команд."""

    url: str = "http://127.0.0.1:8000"
    timeout: float = 120.0
    startup_timeout: float = 180.0

    def __post_init__(self) -> None:
        """Отклоняет не-HTTP адреса, credentials и некорректные таймауты."""
        parsed = urlsplit(self.url)
        _ = parsed.port
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL должен начинаться с http:// или https://")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("URL сервиса не должен содержать credentials, query или fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("Нужен базовый URL сервиса, без пути endpoint")
        for value in (self.timeout, self.startup_timeout):
            if not isfinite(value) or value <= 0:
                raise ValueError("Таймауты должны быть конечными положительными числами")


@dataclass(frozen=True)
class ResponseSample:
    """Полный ответ с проверенными spans и временем HTTP-запроса."""

    seconds: float
    predictions: tuple[Prediction, ...]
    dispatch_lag_seconds: float = 0.0


def wait_ready(client: httpx.Client, timeout: float = 180.0) -> None:
    """Ожидает настоящий health JSON; 200 HTML не считается готовностью."""
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("startup timeout должен быть конечным положительным числом")
    deadline = perf_counter() + timeout
    last_error = "нет ответа"
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        try:
            response = client.get("/healthz", timeout=min(3.0, remaining))
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    raise ValueError("GET /healthz: ожидался JSON {status: ok}")
                return
            if response.status_code != 503:
                response.raise_for_status()
                raise ValueError(f"GET /healthz: неожиданный status {response.status_code}")
            last_error = "HTTP 503"
        except httpx.TransportError as error:
            last_error = str(error)
        remaining = deadline - perf_counter()
        if remaining > 0:
            sleep(min(0.5, remaining))
    raise TimeoutError(f"Сервис не готов за {timeout:g} с: {last_error}")


def request(client: httpx.Client, documents: tuple[Document, ...]) -> ResponseSample:
    """Отправляет только hash/text и строго проверяет весь HTTP-ответ."""
    if not documents or len({d.hash for d in documents}) != len(documents):
        raise ValueError("HTTP batch должен быть непустым и содержать уникальные hash")
    started = perf_counter()
    response = client.post(
        "/api/v1/predict", json=[{"hash": doc.hash, "text": doc.text} for doc in documents]
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("POST /api/v1/predict: ожидался JSON object с data[]")
    rows = payload["data"]
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("entities"), list)
        or any(not isinstance(entity, dict) for entity in row["entities"])
        for row in rows
    ):
        raise ValueError("Каждый ответ должен содержать hash и entities[]")
    predictions = tuple(Prediction.from_mapping(row) for row in rows)
    seconds = perf_counter() - started
    if [p.hash for p in predictions] != [d.hash for d in documents]:
        raise ValueError("Неполный или переставленный HTTP-ответ")
    for doc, prediction in zip(documents, predictions, strict=True):
        Document(doc.hash, doc.text, prediction.entities)
    return ResponseSample(seconds, predictions)
