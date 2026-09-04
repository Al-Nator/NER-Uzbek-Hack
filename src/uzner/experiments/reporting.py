"""Локальные графики, Markdown-отчёт и сводная CSV без внешнего tracker-а."""

from __future__ import annotations

import csv
from pathlib import Path

from uzner.config import ExperimentConfig
from uzner.evaluation.slices import DetailedEvaluation
from uzner.evaluation.tokenizer_audit import TokenizerAudit
from uzner.experiments.logging import EpochRecord

COMPARISON_FIELDS = (
    "run_id",
    "pipeline",
    "encoder",
    "revision",
    "max_length",
    "tag_scheme",
    "head",
    "decoder",
    "seed",
    "best_epoch",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_f1",
    "org_f1",
    "name_f1",
    "geo_f1",
    "eval_documents_per_second",
    "peak_gpu_memory_gib",
    "status",
    "run_path",
)


def _points(values: list[float], width: int, height: int, maximum: float) -> str:
    """Преобразует ряд чисел в SVG-polyline."""
    if not values:
        return ""
    denominator = max(1, len(values) - 1)
    return " ".join(
        f"{40 + index * width / denominator:.1f},{20 + height * (1 - value / maximum):.1f}"
        for index, value in enumerate(values)
    )


def render_training_svg(records: tuple[EpochRecord, ...], path: Path) -> None:
    """Рисует лёгкий SVG с dev F1 и train/dev loss без plot-зависимостей."""
    if not records:
        raise ValueError("Для графика нужна хотя бы одна эпоха")
    width, height = 720, 220
    losses = [record.train_loss for record in records] + [record.dev_loss for record in records]
    loss_max = max(losses) or 1.0
    f1_points = _points([record.micro_f1 for record in records], width, height, 1.0)
    train_points = _points([record.train_loss for record in records], width, height, loss_max)
    dev_points = _points([record.dev_loss for record in records], width, height, loss_max)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="560"
 viewBox="0 0 800 560">
<style>
text{{font:14px sans-serif;fill:#172033}} .axis{{stroke:#8390a5}}
.f1{{stroke:#15a36d;fill:none;stroke-width:3}}
.train{{stroke:#3478f6;fill:none;stroke-width:3}}
.dev{{stroke:#ef6c54;fill:none;stroke-width:3}}
</style>
<rect width="800" height="560" fill="#ffffff"/>
<text x="40" y="20" font-weight="bold">Exact micro-F1</text>
<line class="axis" x1="40" y1="240" x2="760" y2="240"/>
<line class="axis" x1="40" y1="20" x2="40" y2="240"/>
<polyline class="f1" points="{f1_points}"/>
<text x="40" y="290" font-weight="bold">Loss · blue=train · red=dev</text>
<line class="axis" x1="40" y1="510" x2="760" y2="510"/>
<line class="axis" x1="40" y1="290" x2="40" y2="510"/>
<polyline class="train" points="{train_points}" transform="translate(0 270)"/>
<polyline class="dev" points="{dev_points}" transform="translate(0 270)"/>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_run_report(
    path: Path,
    config: ExperimentConfig,
    best: EpochRecord,
    evaluation: DetailedEvaluation,
    audit: TokenizerAudit,
) -> None:
    """Записывает короткий человекочитаемый Markdown-отчёт run-а."""
    rows = [
        ("ORG", evaluation.overall.by_label["ORG"].f1),
        ("NAME", evaluation.overall.by_label["NAME"].f1),
        ("GEO", evaluation.overall.by_label["GEO"].f1),
        ("micro", evaluation.overall.micro.f1),
        ("macro", evaluation.overall.macro_f1),
    ]
    metric_rows = "\n".join(f"| {name} | {value:.4f} |" for name, value in rows)
    content = f"""# Run `{config.run_id}`

## Конфигурация

- Encoder: `{config.encoder.name}@{config.encoder.revision}`
- Head/decoder: `{config.model.head}` / `{config.model.decoder}`
- Tags: `{config.model.tag_scheme.upper()}`
- Context/stride: `{config.tokenization.max_length}` / `{config.tokenization.stride}`
- Seed: `{config.training.seed}`
- Best epoch: `{best.epoch}`

## Качество

| Scope | Exact F1 |
|---|---:|
{metric_rows}

Boundary-only F1: `{evaluation.errors.boundary.f1:.4f}`.  
Document exact match: `{evaluation.errors.document_exact_match:.4f}`.  
Tokenizer representability: `{audit.overall.representability:.4%}`.

## Скорость

- Dev throughput: `{best.eval_documents_per_second:.2f}` documents/s
- Peak GPU memory: `{best.peak_gpu_memory_gib:.2f}` GiB
- Train/eval time at best epoch: `{best.train_seconds:.1f}` / `{best.eval_seconds:.1f}` s

## Артефакты

Machine-readable exact, slice and error metrics are in `metrics/`; predictions are in
`predictions/dev.jsonl`; checkpoints are in `checkpoints/best` and `checkpoints/last`;
the complete chronology is in `logs/events.jsonl`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def update_comparison_csv(
    path: Path,
    config: ExperimentConfig,
    best: EpochRecord,
    run_path: Path,
) -> None:
    """Атомарно добавляет или заменяет фактическую строку run-а."""
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [
                {field: row.get(field, "") for field in COMPARISON_FIELDS}
                for row in csv.DictReader(stream)
            ]
    payload = {
        "run_id": config.run_id,
        "pipeline": config.pipeline,
        "encoder": config.encoder.name,
        "revision": config.encoder.revision,
        "max_length": config.tokenization.max_length,
        "tag_scheme": config.model.tag_scheme,
        "head": config.model.head,
        "decoder": config.model.decoder,
        "seed": config.training.seed,
        "best_epoch": best.epoch,
        "micro_precision": best.micro_precision,
        "micro_recall": best.micro_recall,
        "micro_f1": best.micro_f1,
        "macro_f1": best.macro_f1,
        "org_f1": best.org_f1,
        "name_f1": best.name_f1,
        "geo_f1": best.geo_f1,
        "eval_documents_per_second": best.eval_documents_per_second,
        "peak_gpu_memory_gib": best.peak_gpu_memory_gib,
        "status": "complete",
        "run_path": str(run_path),
    }
    serialized = {key: str(value) for key, value in payload.items()}
    rows = [row for row in rows if row.get("run_id") != config.run_id]
    rows.append(serialized)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
