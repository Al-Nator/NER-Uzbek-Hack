"""Защищённый от перезаписи снимок победившей системы."""

import json
import shutil
import subprocess
from pathlib import Path

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import write_json
from uzner.experiments.source_snapshot import capture_source


def verify_champion(root: Path) -> None:
    """Проверяет содержимое релиза, включая реальные файлы весов."""
    manifest = json.loads((root / "release.json").read_text("utf-8"))
    for relative, digest in manifest["files"].items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"Изменён файл эталона: {relative}")


def freeze_champion(project: Path, submission: Path, target: Path) -> Path:
    """Копирует точные веса без optimizer, проверяет хэши и запрещает запись."""
    if target.exists():
        raise FileExistsError(target)
    source_manifest = json.loads((submission / "predictions.manifest.json").read_text("utf-8"))
    if sha256_file(submission / "predictions.jsonl") != source_manifest["output_sha256"]:
        raise ValueError("Посылка изменилась после инференса")
    target.mkdir(parents=True)
    shutil.copytree(submission, target / "submission")
    for source in source_manifest["sources"]:
        checkpoint = project / "runs" / source["run_id"] / "checkpoints/best"
        for relative, digest in source["checkpoint_sha256"].items():
            path = checkpoint / relative
            if sha256_file(path) != digest:
                raise ValueError(f"Исходный checkpoint изменён: {path}")
            output = target / "models" / source["run_id"] / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "--", str(path), str(output)], check=True)
    ensemble = project / "runs" / source_manifest["ensemble_run"]
    for directory in ("metrics", "predictions"):
        shutil.copytree(ensemble / directory, target / "dev" / directory)
    capture_source(project, target / "environment")
    shutil.copyfile(project / "uv.lock", target / "environment/uv.lock")
    write_json(
        target / "release.json",
        {
            "name": target.name,
            "ensemble": source_manifest["ensemble_run"],
            "rule": "exact span 2 of 3",
            "public_micro_f1": 0.8861,
            "public_score_source": "user supplied leaderboard screenshot, 2026-09-06",
            "protection": "read-only snapshot plus SHA-256; not privileged filesystem immutability",
            "files": {
                str(p.relative_to(target)): sha256_file(p)
                for p in sorted(target.rglob("*"))
                if p.is_file()
            },
        },
    )
    verify_champion(target)
    for path in target.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    target.chmod(0o555)
    return target
