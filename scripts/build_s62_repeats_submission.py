"""Готовит отдельную посылку старого s62 с правилом точных повторов."""

import json
from dataclasses import asdict
from pathlib import Path

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Prediction
from uzner.evaluation.repeated_mentions import RepeatConfig, propagate_repeats
from uzner.experiments.artifacts import write_json
from uzner.training.submission import validate_submission


def main() -> None:
    """Проверяет исходный кэш, добавляет повторы и сохраняет provenance."""
    base = Path("artifacts/submissions/s62_public_v1")
    output = Path("artifacts/submissions/s62_repeats_public_v1")
    input_path = Path("/home/danya/Загрузки/public_test_inputs.jsonl")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((base / "predictions.manifest.json").read_text())
    source = base / "predictions.jsonl"
    if sha256_file(input_path) != manifest["input_sha256"]:
        raise ValueError("Вход отличается от исходного public test")
    if sha256_file(source) != manifest["output_sha256"]:
        raise ValueError("Исходные предсказания изменились")
    documents = tuple(load_documents([("public", input_path)]))
    original = tuple(Prediction.from_mapping(row) for row in read_jsonl(source))
    validate_submission(documents, original)
    config = RepeatConfig()
    predictions = tuple(
        propagate_repeats(doc, pred, config) for doc, pred in zip(documents, original, strict=True)
    )
    validate_submission(documents, predictions)
    target = output / "predictions.jsonl"
    write_predictions(target, predictions)
    write_json(
        output / "predictions.manifest.json",
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(target),
            "rule": "old s62 + exact case-sensitive repeated mentions",
            "config": asdict(config),
            "documents": len(documents),
            "entities": sum(len(p.entities) for p in predictions),
            "added": sum(
                len(p.entities) - len(o.entities)
                for p, o in zip(predictions, original, strict=True)
            ),
            "training": False,
            "gold_evaluation": False,
            "implementation_sha256": sha256_file(Path("src/uzner/evaluation/repeated_mentions.py")),
        },
    )
    print(target)


if __name__ == "__main__":
    main()
