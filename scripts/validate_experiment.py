"""Быстрая проверка эксперимента без загрузки pretrained-модели."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uzner.config import load_data_config, load_experiment_config, resolve_sources
from uzner.data.io import sha256_file


def parse_args() -> argparse.Namespace:
    """Разбирает путь к experiment YAML и корню проекта."""
    parser = argparse.ArgumentParser(description="Проверить конфигурацию NER-эксперимента")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    """Проверяет конфиг, pinned revision и доступность всех данных."""
    args = parse_args()
    project_root = args.project_root.resolve()
    experiment = load_experiment_config(args.config.resolve())
    if experiment.encoder.revision == "main" or len(experiment.encoder.revision) < 7:
        raise ValueError("Encoder revision должна быть закреплена commit hash")
    data_path = (project_root / experiment.data_config).resolve()
    data = load_data_config(data_path)
    files: dict[str, dict[str, str]] = {}
    for split in ("train", "dev"):
        for name, path in resolve_sources(data, split=split, project_root=project_root):
            if not path.is_file():
                raise FileNotFoundError(path)
            files[name] = {
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
            }
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": experiment.run_id,
                "encoder": experiment.encoder.name,
                "revision": experiment.encoder.revision,
                "tag_scheme": experiment.model.tag_scheme,
                "head": experiment.model.head,
                "decoder": experiment.model.decoder,
                "require_gpu": experiment.training.require_gpu,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
