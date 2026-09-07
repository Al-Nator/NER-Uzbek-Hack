"""Строгая файловая оценка с общим exact-span scorer и защитой артефактов."""

from pathlib import Path

from uzner.data.io import load_documents, read_jsonl
from uzner.domain import Prediction
from uzner.evaluation.metrics import EvaluationResult, evaluate_predictions
from uzner.experiments.artifacts import write_json


def evaluate_files(gold: Path, predictions: Path, output: Path | None = None) -> EvaluationResult:
    """Отличает размеченный gold от public inputs и не затирает готовый отчёт."""
    if output is not None and output.exists():
        raise FileExistsError(f"Отчёт уже существует: {output}")
    documents = load_documents((("gold", gold),), require_entities=True)
    rows = read_jsonl(predictions)
    if any(not isinstance(row.get("entities"), list) for row in rows):
        raise ValueError("В каждой строке predictions требуется entities[]")
    result = evaluate_predictions(documents, tuple(Prediction.from_mapping(row) for row in rows))
    if output is not None:
        write_json(output, result.to_mapping())
    return result
