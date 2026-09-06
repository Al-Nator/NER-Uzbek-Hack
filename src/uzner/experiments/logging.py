"""Двойное логирование: читаемая консоль и машинные JSONL/CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from uzner.evaluation.slices import DetailedEvaluation
from uzner.experiments.artifacts import RunPaths
from uzner.experiments.mlflow_tracking import MlflowTracker


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Компактная строка истории одной эпохи."""

    epoch: int
    train_loss: float
    dev_loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    org_f1: float
    name_f1: float
    geo_f1: float
    learning_rate: float
    mean_gradient_norm: float
    train_seconds: float
    eval_seconds: float
    train_tokens_per_second: float
    eval_documents_per_second: float
    peak_gpu_memory_gib: float


class RunLogger:
    """Пишет append-only events и красивую сводку в консоль."""

    def __init__(
        self,
        run_id: str,
        paths: RunPaths,
        tracker: MlflowTracker | None = None,
    ) -> None:
        """Создаёт логи и очищает их только для нового run-каталога."""
        self.run_id = run_id
        self.paths = paths
        self.console = Console(record=True)
        self.tracker = tracker
        self._history: list[EpochRecord] = []
        if not paths.events.exists():
            paths.events.touch()
        if not paths.history.exists():
            with paths.history.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(EpochRecord.__dataclass_fields__))
                writer.writeheader()
        else:
            self._history.extend(self._read_history())

    def _read_history(self) -> tuple[EpochRecord, ...]:
        """Восстанавливает типизированную историю при resume."""
        with self.paths.history.open(encoding="utf-8", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
        return tuple(
            EpochRecord(
                **{
                    name: int(row[name]) if name == "epoch" else float(row[name])
                    for name in EpochRecord.__dataclass_fields__
                }
            )
            for row in rows
        )

    @property
    def history(self) -> tuple[EpochRecord, ...]:
        """Возвращает историю текущего процесса."""
        return tuple(self._history)

    def attach_tracker(self, tracker: MlflowTracker | None) -> None:
        """Подключает MLflow после безопасного создания локальных логов."""
        self.tracker = tracker

    def event(
        self,
        name: str,
        values: dict[str, Any],
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Добавляет одно timestamped-событие в JSONL."""
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": name,
            "epoch": epoch,
            "step": step,
            "values": values,
        }
        with self.paths.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def start(self, description: dict[str, object]) -> None:
        """Печатает однозначную шапку запуска."""
        table = Table(title=f"NER run: {self.run_id}", show_header=False)
        table.add_column("field", style="cyan")
        table.add_column("value", style="white")
        for key, value in description.items():
            table.add_row(str(key), str(value))
        self.console.print(table)
        self.event("run_started", description)
        self.flush_console()

    def train_step(
        self,
        *,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: float | None,
        tokens_per_second: float,
        gpu_memory_gib: float,
        loss_components: dict[str, float] | None = None,
        head_learning_rate: float | None = None,
    ) -> None:
        """Логирует один интервал обучения."""
        values = {
            "loss": loss,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "tokens_per_second": tokens_per_second,
            "gpu_memory_gib": gpu_memory_gib,
        }
        values.update(
            {f"loss_component/{name}": value for name, value in (loss_components or {}).items()}
        )
        if head_learning_rate is not None:
            values["head_learning_rate"] = head_learning_rate
        self.event("train_step", values, epoch=epoch, step=step)
        if self.tracker is not None:
            self.tracker.log_train_step(values, step=step)
        gradient = "—" if gradient_norm is None else f"{gradient_norm:.3f}"
        self.console.print(
            f"[blue]train[/] epoch={epoch} step={step} loss={loss:.4f} "
            f"lr={learning_rate:.2e} grad={gradient} tok/s={tokens_per_second:.0f} "
            f"gpu={gpu_memory_gib:.2f} GiB"
        )
        self.flush_console()

    def epoch(self, record: EpochRecord, evaluation: DetailedEvaluation, *, best: bool) -> None:
        """Показывает таблицу exact-span метрик и обновляет CSV."""
        self._history.append(record)
        with self.paths.history.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(EpochRecord.__dataclass_fields__))
            writer.writerow(asdict(record))
        table = Table(title=f"Epoch {record.epoch}{' · new best' if best else ''}")
        table.add_column("scope")
        table.add_column("precision", justify="right")
        table.add_column("recall", justify="right")
        table.add_column("F1", justify="right", style="bold green")
        for label, scores in evaluation.overall.by_label.items():
            table.add_row(
                label,
                f"{scores.precision:.4f}",
                f"{scores.recall:.4f}",
                f"{scores.f1:.4f}",
            )
        micro = evaluation.overall.micro
        table.add_row("micro", f"{micro.precision:.4f}", f"{micro.recall:.4f}", f"{micro.f1:.4f}")
        table.add_row("macro", "—", "—", f"{evaluation.overall.macro_f1:.4f}")
        self.console.print(table)
        self.console.print(
            f"loss train/dev={record.train_loss:.4f}/{record.dev_loss:.4f} · "
            f"time train/eval={record.train_seconds:.1f}/{record.eval_seconds:.1f}s · "
            f"peak GPU={record.peak_gpu_memory_gib:.2f} GiB"
        )
        self.event("epoch_completed", {**asdict(record), "is_best": best}, epoch=record.epoch)
        if self.tracker is not None:
            self.tracker.log_epoch(record, evaluation, best=best)
        self.flush_console()

    def flush_console(self) -> None:
        """Сохраняет накопленный текст Rich-консоли."""
        self.paths.console_log.write_text(self.console.export_text(clear=False), encoding="utf-8")
