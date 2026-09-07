"""Перенос проверенного текстового posthoc-рецепта на public без доступа к меткам."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

from uzner.data.io import load_documents, read_jsonl, sha256_file, write_predictions
from uzner.domain import Document, Prediction
from uzner.experiments.artifacts import write_json
from uzner.posthoc.config import Variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.posthoc.rules import boundaries, dictionary_rule, repeat
from uzner.posthoc.study import read_source
from uzner.training.submission import validate_submission


@dataclass(frozen=True)
class PosthocSubmission:
    """Все исходные файлы и выбранный dev-run без неявных модельных вызовов."""

    selected_run: Path
    train: Path
    input_path: Path
    base_predictions: Path
    output_dir: Path
    dictionary_dev: Path | None = None


def build_submission(request: PosthocSubmission) -> Path:
    """Применяет ровно один сохранённый рецепт и проверяет каждую строку ответа."""
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config_path = request.selected_run / "resolved_config.json"
    config = json.loads(config_path.read_text("utf-8"))
    metadata = json.loads((request.selected_run / "metadata.json").read_text("utf-8"))
    train_hash = sha256_file(request.train)
    if metadata["inputs"].get(str(request.train)) != train_hash:
        raise ValueError("Словарь должен строиться по тому же train, что и dev-абляция")
    variant = Variant(
        **{
            f.name: tuple(config[f.name]) if f.name == "labels" else config[f.name]
            for f in fields(Variant)
        }
    )
    if variant.operation not in {"lex_add", "lex_relabel", "repeat", "boundary", "short"}:
        raise ValueError("Для этого рецепта нужны дополнительные модельные входы")
    records = read_jsonl(request.input_path)
    if any(record.get("entities") for record in records):
        raise ValueError("Public inference не принимает gold-разметку")
    documents = tuple(Document(record["hash"], record["text"]) for record in records)
    if not documents or len({d.hash for d in documents}) != len(documents):
        raise ValueError("Пустой public или повторные hash")
    base = read_source(request.base_predictions, documents)
    sources = [("train", request.train)]
    if request.dictionary_dev is not None:
        sources.append(("dictionary_dev", request.dictionary_dev))
    train = tuple(load_documents(sources))
    if request.dictionary_dev is not None:
        public_hashes = {d.hash for d in documents}
        public_texts = {d.text for d in documents}
        if any(d.hash in public_hashes or d.text in public_texts for d in train):
            raise ValueError("Словарные источники пересекаются с public по hash или тексту")
    lexicon = build_lexicon(train, variant.normalized)
    predictions = []
    for document, prediction in zip(documents, base, strict=True):
        if variant.operation in {"lex_add", "lex_relabel", "short"}:
            prediction = dictionary_rule(document, prediction, variant, lexicon)
        elif variant.operation == "repeat":
            prediction = repeat(document, prediction, variant)
        elif variant.operation == "boundary":
            prediction = boundaries(document, prediction, variant, lexicon)
        if variant.boundary_after:
            prediction = boundaries(document, prediction, variant, lexicon)
        if variant.repeat_after:
            prediction = repeat(document, prediction, Variant("exact_repeat", "repeat"))
        predictions.append(prediction)
    result: tuple[Prediction, ...] = tuple(predictions)
    validate_submission(documents, result)
    request.output_dir.mkdir(parents=True)
    output = request.output_dir / "predictions.jsonl"
    write_predictions(output, result)
    effective_config = dict(config)
    if request.dictionary_dev is not None:
        effective_config.update(
            dictionary="official train + dev; annotated/all occurrences",
            data_split="public inference; dev included in dictionary, not held out",
        )
    write_json(
        request.output_dir / "predictions.manifest.json",
        {
            "selected_run": str(request.selected_run),
            "config": effective_config,
            "selected_config_sha256": sha256_file(config_path),
            "train_sha256": train_hash,
            "input_sha256": sha256_file(request.input_path),
            "base_predictions_sha256": sha256_file(request.base_predictions),
            "output_sha256": sha256_file(output),
            "documents": len(documents),
            "entities": sum(len(p.entities) for p in result),
            "training": False,
            "dictionary_sources": {str(path): sha256_file(path) for _, path in sources},
            "dictionary_documents": len(train),
            "dictionary_dev_used": request.dictionary_dev is not None,
            "dev_micro_f1": None,
            "implementation_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("submission.py", "lexicon.py", "rules.py", "config.py")
            },
            "public_gold_used": False,
            "public_micro_f1": None,
            "selection_note": (
                "original train-only recipe; dictionary refit on train+dev by explicit "
                "request; original dev is not held out for this variant; "
                "public not evaluated locally"
                if request.dictionary_dev is not None
                else "selected on original dev; public not evaluated locally"
            ),
        },
    )
    return output
