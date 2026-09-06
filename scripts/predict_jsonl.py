"""Инференс лучшего GP-checkpoint в формат лидерборда без обучения и MLflow-run."""

import argparse
from pathlib import Path

from uzner.data.io import load_documents, sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.experiments.artifacts import write_json
from uzner.training.frozen import FrozenPredictor


def validate_predictions(documents: tuple[Document, ...], predictions: tuple[Prediction, ...]):
    """Проверяет полноту, порядок hash и границы в исходных Unicode-строках."""
    if tuple(d.hash for d in documents) != tuple(p.hash for p in predictions):
        raise ValueError("Не совпали количество или порядок hash")
    for document, prediction in zip(documents, predictions, strict=True):
        Document(document.hash, document.text, prediction.entities)


def main() -> None:
    """Сохраняет проверенный JSONL и отдельный manifest происхождения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = tuple(load_documents((("public_test", args.input),)))
    predictor = FrozenPredictor(args.source_run)
    predictions = []
    for start in range(0, len(documents), 128):
        group = documents[start : start + 128]
        result = predictor.predict(group)
        validate_predictions(group, result)
        predictions.extend(result)
        print(f"Predicted {len(predictions)}/{len(documents)}", flush=True)
    validate_predictions(documents, tuple(predictions))
    write_predictions(args.output, predictions)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "source_run": str(args.source_run),
            "checkpoint": "best",
            "input_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
            "documents": len(documents),
            "threshold": predictor.config.model.span_threshold,
            "checkpoint_manifest_sha256": sha256_file(args.source_run / "artifact_manifest.json"),
            "training": False,
        },
    )


if __name__ == "__main__":
    main()
