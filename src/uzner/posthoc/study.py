"""Загрузка зафиксированных источников и последовательный запуск posthoc-абляций."""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from uzner.data.io import load_documents, read_jsonl, sha256_file
from uzner.domain import Document, Prediction
from uzner.evaluation.span_vote import majority_vote
from uzner.experiments.artifacts import write_json
from uzner.experiments.research_run import ResearchRun
from uzner.posthoc.config import Variant
from uzner.posthoc.engine import Inputs, apply_variant
from uzner.posthoc.lexicon import build_lexicon
from uzner.posthoc.reporting import save_changes, split_metrics
from uzner.training.submission import validate_submission


@dataclass(frozen=True)
class Study:
    """Пути, provenance и заранее объявленная матрица гипотез."""

    name: str
    train: Path
    dev: Path
    sources: dict[str, Path]
    candidates: Path
    variants: tuple[Variant, ...]
    experiment: str = "uzner-posthoc-decoding"

    @classmethod
    def load(cls, path: Path) -> "Study":
        """Загружает строгие варианты и проверяет уникальность имён запусков."""
        payload = yaml.safe_load(path.read_text("utf-8"))
        variants = tuple(
            Variant(**{**v, "labels": tuple(v.get("labels", ("ORG", "NAME", "GEO")))})
            for v in payload.pop("variants")
        )
        if len({v.name for v in variants}) != len(variants):
            raise ValueError("Повторное имя варианта")
        return cls(
            **{
                **payload,
                "train": Path(payload["train"]),
                "dev": Path(payload["dev"]),
                "candidates": Path(payload["candidates"]),
                "sources": {k: Path(v) for k, v in payload["sources"].items()},
                "variants": variants,
            }
        )


def read_source(
    path: Path, documents: tuple[Document, ...], *, flat: bool = True
) -> tuple[Prediction, ...]:
    """Проверяет полное покрытие hash; порядок исходных файлов не существенен."""
    predictions = tuple(Prediction.from_mapping(value) for value in read_jsonl(path))
    mapping = {p.hash: p for p in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {d.hash for d in documents}:
        raise ValueError(f"Неполное покрытие или повторные hash: {path}")
    result = tuple(mapping[d.hash] for d in documents)
    if flat:
        validate_submission(documents, result)
    else:
        for document, prediction in zip(documents, result, strict=True):
            if any(e.end > len(document.text) or e.score is None for e in prediction.entities):
                raise ValueError("Некорректный scored cache")
    return result


def load_inputs(study: Study) -> tuple[Inputs, tuple[Document, ...], tuple[Document, ...]]:
    """Читает dev gold только для evaluator, не передавая его правилам."""
    train = tuple(load_documents((("train", study.train),)))
    gold = tuple(load_documents((("dev", study.dev),)))
    if {d.hash for d in train} & {d.hash for d in gold}:
        raise ValueError("Пересечение train/dev по hash")
    documents = tuple(Document(d.hash, d.text) for d in gold)
    sources = {key: read_source(path, documents) for key, path in study.sources.items()}
    replay = majority_vote(documents, tuple(sources[k] for k in ("s33", "s21", "s31")))
    if [p.to_mapping() for p in replay] != [p.to_mapping() for p in sources["s62"]]:
        raise ValueError("Компоненты не воспроизвели эталон s62")
    manifest = json.loads(study.candidates.with_suffix(".manifest.json").read_text("utf-8"))
    if manifest["input_sha256"] != sha256_file(study.dev):
        raise ValueError("Кэш рассчитан на другом наборе документов")
    if manifest["output_sha256"] != sha256_file(study.candidates):
        raise ValueError("Кэш не совпадает с манифестом")
    print("Построение словарей: original train, без dev/silver", flush=True)
    exact, normalized = build_lexicon(train), build_lexicon(train, True)
    inputs = Inputs(
        documents,
        sources,
        read_source(study.candidates, documents, flat=False),
        exact,
        normalized,
        manifest["floor"],
    )
    return inputs, gold, train


def run_study(config_path: Path, *, skip_complete: bool = False) -> Path:
    """Сохраняет каждый запуск отдельно и логирует metrics, изменения и источник в MLflow."""
    study = Study.load(config_path)
    inputs, gold, train = load_inputs(study)
    paths = (
        config_path,
        study.train,
        study.dev,
        *study.sources.values(),
        study.candidates,
        study.candidates.with_suffix(".manifest.json"),
    )
    os.environ["UZNER_MLFLOW_EXPERIMENT"] = study.experiment
    rows = []
    for variant in study.variants:
        root = Path("runs") / f"{study.name}_{variant.name}"
        config = {
            **asdict(variant),
            "study": study.name,
            "seed": 42,
            "training": False,
            "selection": variant.selection,
            "data_split": "original dev; no dev training",
            "sources": {k: str(v) for k, v in study.sources.items()},
            "dictionary": "original train only; annotated/all occurrences",
        }
        if skip_complete and (root / "status.json").exists():
            status = json.loads((root / "status.json").read_text("utf-8"))
            stored = json.loads((root / "resolved_config.json").read_text("utf-8"))
            if status["status"] == "complete" and stored == json.loads(json.dumps(config)):
                if json.loads((root / "metadata.json").read_text("utf-8"))["inputs"] != {
                    str(p): sha256_file(p) for p in paths
                }:
                    raise ValueError("Входы завершённого запуска изменились")
                rows.append(json.loads((root / "summary.json").read_text("utf-8")))
                continue
        with ResearchRun(root, config, paths, "posthoc_no_training") as run:
            started = time.perf_counter()
            predictions = apply_variant(inputs, variant)
            duration = time.perf_counter() - started
            score = run.evaluate(gold, predictions, train, "dev")
            halves = split_metrics(gold, predictions)
            write_json(root / "metrics/halves.json", halves)
            changes = save_changes(root / "diagnostics", gold, inputs.sources["s62"], predictions)
            row = {
                "name": variant.name,
                "operation": variant.operation,
                "f1": score,
                "run": str(root),
                "mlflow_run_id": run.run_id,
                "half_a_f1": halves["half_a"]["micro"]["f1"],
                "half_b_f1": halves["half_b"]["micro"]["f1"],
                "postprocess_ms_per_document": duration * 1000 / len(gold),
                **asdict(changes),
            }
            write_json(root / "summary.json", row)
            run.log(
                {
                    "postprocess/ms_per_document": row["postprocess_ms_per_document"],
                    **{f"changes/{k}": v for k, v in asdict(changes).items()},
                    "halves/a_f1": row["half_a_f1"],
                    "halves/b_f1": row["half_b_f1"],
                }
            )
            rows.append(row)
    report = Path("artifacts") / study.name / "comparison.json"
    write_json(
        report,
        {
            "study": study.name,
            "runs": sorted(rows, key=lambda r: -r["f1"]),
            "dev_selection_not_independent": True,
        },
    )
    return report
