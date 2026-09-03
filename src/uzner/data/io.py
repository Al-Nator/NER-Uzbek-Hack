"""Чтение нескольких источников данных без потери исходного текста."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from uzner.domain import Document, Prediction


def sha256_file(path: Path) -> str:
    """Вычисляет SHA-256 файла потоковым способом."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Читает JSONL и добавляет номер строки в сообщения об ошибках."""
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: пустая строка")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: некорректный JSON: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: ожидается JSON-объект")
            records.append(value)
    return records


def load_documents(sources: Sequence[tuple[str, Path]]) -> list[Document]:
    """Загружает документы из нескольких файлов и проверяет уникальность hash."""
    documents: list[Document] = []
    owners: dict[str, str] = {}
    for source_name, path in sources:
        for raw in read_jsonl(path):
            document = Document.from_mapping(raw, source=source_name)
            previous_source = owners.get(document.hash)
            if previous_source is not None:
                raise ValueError(
                    f"hash {document.hash!r} повторяется в {previous_source!r} и {source_name!r}"
                )
            owners[document.hash] = source_name
            documents.append(document)
    return documents


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> None:
    """Атомарно записывает предсказания в JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(
                json.dumps(prediction.to_mapping(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)
