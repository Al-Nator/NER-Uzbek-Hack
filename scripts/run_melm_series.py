"""Локальная очередь: генератор MELM-inspired → filtered train → полный s42."""

import argparse
from dataclasses import replace
from pathlib import Path

from uzner.training.melm_generator import MelmConfig
from uzner.training.melm_pipeline import prepare_melm
from uzner.training.requests import TrainRequest
from uzner.training.runner import run_experiment


def main() -> None:
    """Запускает новую data-ветку или изолированный генератор smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/melm/s42_melm_v1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = MelmConfig()
    if args.smoke:
        if any(
            args.output.resolve().is_relative_to(Path(p).resolve()) for p in ("runs", "artifacts")
        ):
            raise ValueError("Smoke output должен быть отдельным временным каталогом")
        config = replace(config, epochs=1, rounds=1, max_candidates=8, max_accepted=4)
    elif args.output.resolve() != Path("artifacts/melm/s42_melm_v1").resolve():
        raise ValueError("Полный s42 использует только релиз artifacts/melm/s42_melm_v1")
    source = Path("runs/s32_bge_m3_retromae_global_pointer_a100-continuation-v1")
    prepare_melm(args.output, source, config, smoke=args.smoke)
    if not args.smoke:
        run_experiment(
            TrainRequest(
                config_path=Path("configs/experiments/s42_bge_gp_melm.yaml"),
                project_root=Path.cwd(),
                run_id_suffix="s4-melm-v1",
            )
        )


if __name__ == "__main__":
    main()
