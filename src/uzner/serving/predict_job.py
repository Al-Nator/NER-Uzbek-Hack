"""Полный JSONL-прогон финального HTTP-сервиса с атомарной публикацией ответа."""

from dataclasses import dataclass
from pathlib import Path

import httpx

from uzner.data.io import load_documents, sha256_file, write_predictions
from uzner.experiments.artifacts import write_json
from uzner.serving.http_client import HttpSettings, request, wait_ready


@dataclass(frozen=True)
class PredictionJob:
    """Вход и новый каталог результата, не перезаписывающий предыдущую посылку."""

    input: Path
    output: Path
    http: HttpSettings = HttpSettings()
    batch_size: int = 8

    def __post_init__(self) -> None:
        """Проверяет размер штатного HTTP-пакета."""
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= 8
        ):
            raise ValueError("batch_size должен быть целым числом от 1 до 8")


def predict_file(job: PredictionJob, *, transport: httpx.BaseTransport | None = None) -> Path:
    """Публикует predictions только после успешной обработки каждого документа."""
    if job.output.exists():
        raise FileExistsError(f"Каталог результата уже существует: {job.output}")
    documents = tuple(load_documents((("predict", job.input),)))
    if not documents:
        raise ValueError("Входной JSONL не должен быть пустым")
    input_hash = sha256_file(job.input)
    job.output.mkdir(parents=True, exist_ok=False)
    output = job.output / "predictions.jsonl"
    try:
        predictions = []
        with httpx.Client(
            base_url=job.http.url, timeout=job.http.timeout, transport=transport, trust_env=False
        ) as client:
            wait_ready(client, job.http.startup_timeout)
            for start in range(0, len(documents), job.batch_size):
                result = request(client, documents[start : start + job.batch_size])
                predictions.extend(result.predictions)
                print(f"Predicted {len(predictions)}/{len(documents)}", flush=True)
        write_predictions(output, predictions)
        write_json(
            job.output / "manifest.json",
            {
                "status": "complete",
                "kind": "http_predictions",
                "url": job.http.url,
                "batch_size": job.batch_size,
                "documents": len(documents),
                "input_sha256": input_hash,
                "predictions_sha256": sha256_file(output),
                "gold_sent_to_service": False,
                "model_identity": "Determined by the running service, not inferred by client",
            },
        )
    except Exception as error:
        write_json(job.output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    return output
