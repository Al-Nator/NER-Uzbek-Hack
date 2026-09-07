"""Фиксированное повторение рецепта лидербордного s62 на train+dev."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from uzner.config import ExperimentConfig, load_experiment_config
from uzner.experiments.artifacts import write_json
from uzner.experiments.remote_sync import verify_run_manifest


@dataclass(frozen=True)
class FinalStage:
    """Одна стадия обучения с заранее выбранной эпохой и ролью в ансамбле."""

    run_id: str
    selected_epoch: int
    ensemble_member: bool = True

    @property
    def config_path(self) -> Path:
        """Возвращает зафиксированный конфиг данной стадии."""
        return Path("configs/final") / f"{self.run_id}.yaml"


STAGES = (
    FinalStage("f10_s32_train_dev_epoch5", 5, False),
    FinalStage("f11_s33_train_dev_low_lr_epoch1", 1),
    FinalStage("f12_s21_train_dev_epoch5", 5),
    FinalStage("f13_s31_train_dev_epoch5", 5),
)
ENSEMBLE_ID = "f14_s62_train_dev_vote_v1"


def run_queue(root: Path) -> None:
    """Запускает стадии в отдельных процессах, не выбирая checkpoint по dev."""
    for stage in STAGES:
        run = root / "runs" / stage.run_id
        if run.exists():
            status = json.loads((run / "status.json").read_text())
            if (
                status.get("status") != "complete"
                or status.get("selected_epoch") != stage.selected_epoch
            ):
                raise RuntimeError(f"Незавершённый или несовместимый run: {stage.run_id}")
            verify_run_manifest(run)
            saved = ExperimentConfig.from_mapping(
                yaml.safe_load((run / "resolved_config.yaml").read_text())
            )
            if saved != load_experiment_config(root / stage.config_path):
                raise ValueError("Конфиг завершённой стадии изменился")
            continue
        print(f"Starting {stage.run_id}; fixed epoch={stage.selected_epoch}", flush=True)
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/train_final.py",
                "--config",
                str(stage.config_path),
                "--selected-epoch",
                str(stage.selected_epoch),
            ],
            cwd=root,
            check=True,
        )
    write_json(
        root / "runs" / ENSEMBLE_ID / "resolved_config.json",
        {
            "run_id": ENSEMBLE_ID,
            "rule": "exact span 2 of 3",
            "sources": [stage.run_id for stage in STAGES if stage.ensemble_member],
            "reference": "s62_span_majority_full-targeted-v1",
            "data_usage": "official train + official dev; no independent validation",
        },
    )
    write_json(
        root / "runs" / ENSEMBLE_ID / "status.json",
        {
            "kind": "final_ensemble",
            "status": "complete",
            "independent_validation": False,
        },
    )


if __name__ == "__main__":
    run_queue(Path.cwd())
