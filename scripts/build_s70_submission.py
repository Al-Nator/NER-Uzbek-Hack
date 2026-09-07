"""Собирает s70 из проверенных public-предсказаний исходных компонентов."""

import json
from pathlib import Path

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Prediction
from uzner.evaluation.span_vote import SpanVoteConfig, majority_vote
from uzner.experiments.artifacts import write_json
from uzner.training.submission import validate_submission


def build(root: Path, input_path: Path, output: Path) -> Path:
    """Проверяет происхождение, входные хэши и готовит неизменяемую посылку."""
    if output.exists():
        raise FileExistsError(output)
    config_path = root / "runs/s70_s45_vote_swap_v1/resolved_config.json"
    config = json.loads(config_path.read_text())
    sources = SpanVoteConfig(tuple(config["sources"])).sources
    base = root / "artifacts/submissions"
    single_manifest = base / "s45_public_v1/predictions.manifest.json"
    ensemble_manifest = base / "s62_public_v1/predictions.manifest.json"
    single = json.loads(single_manifest.read_text())
    ensemble = json.loads(ensemble_manifest.read_text())
    input_hash = sha256_file(input_path)
    if single["input_sha256"] != input_hash or ensemble["input_sha256"] != input_hash:
        raise ValueError("Кэш относится к другому public test")
    if Path(single["source_run"]).name != sources[0] or single["checkpoint"] != "best":
        raise ValueError("Неверный одиночный компонент")
    if ensemble["rule"] != "exact span 2 of 3":
        raise ValueError("Неизвестный источник ансамблевых компонентов")
    lookup = {item["run_id"]: item for item in ensemble["sources"]}
    paths = [base / "s45_public_v1/predictions.jsonl"]
    expected = [single["output_sha256"]]
    for source in sources[1:]:
        paths.append(base / f"s62_public_v1/components/{source}.jsonl")
        expected.append(lookup[source]["predictions_sha256"])
    documents = tuple(load_documents((("public_test", input_path),)))
    components = []
    for path, digest in zip(paths, expected, strict=True):
        if sha256_file(path) != digest:
            raise ValueError(f"Повреждён кэш: {path}")
        predictions = tuple(Prediction.from_mapping(row) for row in read_jsonl(path))
        validate_submission(documents, predictions)
        components.append(predictions)
    predictions = majority_vote(documents, tuple(components))
    validate_submission(documents, predictions)
    target = output / "predictions.jsonl"
    write_predictions(target, predictions)
    write_json(
        output / "predictions.manifest.json",
        {
            "ensemble_run": config["run_id"],
            "rule": "exact span 2 of 3",
            "sources": list(sources),
            "mode": "verified_public_prediction_cache",
            "component_files": {str(p): h for p, h in zip(paths, expected, strict=True)},
            "source_manifests": {
                str(p): sha256_file(p) for p in (single_manifest, ensemble_manifest)
            },
            "ensemble_config_sha256": sha256_file(config_path),
            "input_sha256": input_hash,
            "output_sha256": sha256_file(target),
            "documents": len(documents),
            "entities": sum(len(p.entities) for p in predictions),
            "training": False,
            "gold_evaluation": False,
        },
    )
    return target


if __name__ == "__main__":
    print(
        build(
            Path.cwd(),
            Path("/home/danya/Загрузки/public_test_inputs.jsonl"),
            Path("artifacts/submissions/s70_public_v1"),
        )
    )
