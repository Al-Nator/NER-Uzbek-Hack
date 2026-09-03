from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LABELS = frozenset({"ORG", "NAME", "GEO"})
PROBE_ITEMS = [
    {"hash": "contract-latin", "text": "Ali Toshkent shahrida ishlaydi."},
    {"hash": "contract-cyrillic", "text": "Алишер Навоий Тошкентда туғилган."},
]
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonObject = dict[str, Any]


class ContractError(ValueError):
    """Ошибка обязательного HTTP-контракта."""


def parse_args() -> argparse.Namespace:
    """Разбирает адрес сервиса и таймауты проверки."""

    parser = argparse.ArgumentParser(description="Check the hackathon HTTP API contract.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def validate_timeouts(startup_timeout: float, request_timeout: float) -> None:
    """Проверяет таймауты запуска и отдельного HTTP-запроса."""

    if startup_timeout <= 0:
        raise ContractError("startup-timeout must be positive")
    if request_timeout <= 0:
        raise ContractError("request-timeout must be positive")


def normalize_base_url(url: str) -> str:
    """Нормализует и проверяет адрес HTTP-сервиса."""

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ContractError("url must start with http:// or https://")
    return base_url


def _decode_json(body: bytes, source: str) -> JsonValue:
    """Декодирует UTF-8 JSON с понятным сообщением об ошибке."""

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{source}: response is not valid UTF-8 JSON: {error}") from error


def _request_json(
    url: str,
    *,
    method: str,
    timeout: float,
    payload: JsonValue = None,
) -> tuple[int, str, JsonValue | None]:
    """Выполняет HTTP-запрос и возвращает статус, content type и JSON."""

    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(  # noqa: S310 - схема URL проверена в _validate_args
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL задан участником
            status = response.status
            content_type = response.headers.get_content_type()
            response_body = response.read()
    except HTTPError as error:
        status = error.code
        content_type = error.headers.get_content_type()
        response_body = error.read()
    except URLError as error:
        raise ContractError(f"{method} {url}: connection failed: {error.reason}") from error
    except TimeoutError as error:
        raise ContractError(f"{method} {url}: request timed out") from error

    decoded = _decode_json(response_body, f"{method} {url}") if response_body else None
    return status, content_type, decoded


def _validate_health(payload: JsonValue | None) -> None:
    """Проверяет успешный ответ `/healthz`."""

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ContractError('GET /healthz: expected JSON object {"status":"ok"}')


def wait_for_health(base_url: str, startup_timeout: float, request_timeout: float) -> None:
    """Ожидает доступности сервиса, допуская connection refused и HTTP 503."""

    url = f"{base_url}/healthz"
    deadline = time.monotonic() + startup_timeout
    last_error = "service did not answer"

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContractError(f"GET /healthz did not become ready: {last_error}")
        try:
            status, content_type, payload = _request_json(
                url,
                method="GET",
                timeout=min(request_timeout, remaining),
            )
        except ContractError as error:
            last_error = str(error)
        else:
            if status == 200:
                if content_type != "application/json":
                    raise ContractError(
                        f"GET /healthz: expected application/json, got {content_type!r}"
                    )
                _validate_health(payload)
                return
            if status != 503:
                raise ContractError(f"GET /healthz: expected 200 or 503, got {status}")
            last_error = "HTTP 503 Service Unavailable"
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _validate_entity(entity: Any, text: str, source: str) -> tuple[str, int, int]:
    """Проверяет обязательные поля и координаты одной сущности."""

    if not isinstance(entity, dict):
        raise ContractError(f"{source}: entity must be an object")
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if label not in LABELS:
        raise ContractError(f"{source}: label must be one of {sorted(LABELS)}")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= len(text)
    ):
        raise ContractError(f"{source}: invalid character offsets")
    return label, start, end


def _validate_predict(
    payload: JsonValue | None,
    inputs: list[dict[str, str]],
) -> tuple[list[JsonObject], int]:
    """Проверяет envelope, порядок документов и exact-span поля."""

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ContractError("POST /api/v1/predict: expected JSON object with data[]")
    results = payload["data"]
    if len(results) != len(inputs):
        raise ContractError("POST /api/v1/predict: data length differs from request batch length")

    entity_count = 0
    for index, (result, item) in enumerate(zip(results, inputs, strict=True)):
        source = f"POST /api/v1/predict/data[{index}]"
        if not isinstance(result, dict):
            raise ContractError(f"{source}: result must be an object")
        if result.get("hash") != item["hash"]:
            raise ContractError(f"{source}: hash or result order differs from request")
        entities = result.get("entities")
        if not isinstance(entities, list):
            raise ContractError(f"{source}: entities must be an array")
        seen: set[tuple[str, int, int]] = set()
        for entity_index, entity in enumerate(entities):
            key = _validate_entity(
                entity,
                item["text"],
                f"{source}/entities[{entity_index}]",
            )
            if key in seen:
                raise ContractError(f"{source}/entities[{entity_index}]: duplicate entity")
            seen.add(key)
        entity_count += len(entities)
    return results, entity_count


def predict_batch(
    base_url: str,
    inputs: list[dict[str, str]],
    request_timeout: float,
) -> tuple[list[JsonObject], int]:
    """Отправляет один батч и возвращает проверенные результаты и число сущностей."""

    if not inputs:
        raise ContractError("predict batch must not be empty")
    status, content_type, payload = _request_json(
        f"{base_url}/api/v1/predict",
        method="POST",
        timeout=request_timeout,
        payload=inputs,
    )
    if status != 200:
        raise ContractError(f"POST /api/v1/predict: expected 200, got {status}")
    if content_type != "application/json":
        raise ContractError(
            f"POST /api/v1/predict: expected application/json, got {content_type!r}"
        )
    return _validate_predict(payload, inputs)


def run(args: argparse.Namespace) -> None:
    """Проверяет health и predict работающего сервиса."""

    validate_timeouts(args.startup_timeout, args.request_timeout)
    base_url = normalize_base_url(args.url)
    wait_for_health(base_url, args.startup_timeout, args.request_timeout)
    print("OK  GET /healthz")

    _, entity_count = predict_batch(base_url, PROBE_ITEMS, args.request_timeout)
    print(
        f"OK  POST /api/v1/predict ({len(PROBE_ITEMS)} documents, {entity_count} returned entities)"
    )
    print("Service contract: OK")


def main() -> int:
    """Запускает проверку с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (ContractError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
