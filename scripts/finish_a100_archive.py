"""Доводит перенос до двойной проверки и безопасного удаления удалённых run-копий."""

import argparse
import inspect
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from uzner.experiments.artifacts import write_json
from uzner.experiments.remote_archive import REMOTE_ROOT, SSH, inventory


def validate_targets(files: dict) -> list[str]:
    """Разрешает только непустой список отдельных experiment-каталогов."""
    names = sorted({Path(name).parts[0] for name in files})
    if not names or any(not re.fullmatch(r"[sf][0-9][A-Za-z0-9_.-]*", n) for n in names):
        raise ValueError("Небезопасные имена удаляемых run")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in files):
        raise ValueError("Выход за каталог runs")
    return names


def remove_verified_remote(expected: dict, names: list[str]) -> None:
    """Ещё раз хэширует remote и удаляет только exact-копии сохранённых каталогов."""
    source = "import hashlib,json,shutil,os\nfrom pathlib import Path\n"
    source += inspect.getsource(inventory)
    source += f"\nroot=Path({REMOTE_ROOT!r})\nexpected={expected!r}\nnames={names!r}\n"
    source += """
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cwd = (proc / 'cwd').resolve()
        command = (proc / 'cmdline').read_bytes().replace(b'\\0', b' ').decode()
    except (OSError, UnicodeError):
        continue
    if cwd == root.parent and any(token in command for token in (
        'scripts/train.py', 'scripts/train_final.py', 'scripts/run_series.py',
        'scripts/run_span_rerankers.py', 'scripts/train_reranker_proposers.py',
    )):
        raise RuntimeError('Активное обучение: удаление запрещено')
if inventory(root) != expected:
    raise RuntimeError('Remote изменился после двух проверок')
for name in names:
    target = root / name
    if target.is_symlink() or target.resolve().parent != root.resolve():
        raise RuntimeError('Небезопасная цель')
    shutil.rmtree(target)
    print('Removed verified remote copy:', name, flush=True)
"""
    subprocess.run(
        [*SSH, "cd /home/danya/NER-Uzbek-Hack && uv run python -"],
        input=source,
        text=True,
        check=True,
    )


def main() -> None:
    """Ждёт копирование, проверяет и освобождает A100 без удаления локальных весов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--copy-pid", type=int, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    command_path = Path(f"/proc/{args.copy_pid}/cmdline")
    while command_path.exists():
        try:
            command = command_path.read_bytes()
        except FileNotFoundError:
            break
        if b"archive_a100_runs.py" not in command:
            raise RuntimeError("PID не относится к нашему копированию")
        print("Ожидаю завершения checksum-rsync", flush=True)
        time.sleep(60)
    # Старый watcher должен закончить импорт до удаления его remote-источника.
    while (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "uzner-return-s47.service"],
            check=False,
        ).returncode
        == 0
    ):
        print("Ожидаю завершения возврата s47", flush=True)
        time.sleep(60)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/archive_a100_runs.py",
            "--archive",
            str(archive),
            "--verify",
        ],
        check=True,
    )
    first = json.loads((archive / "verification_1.json").read_text())
    second = json.loads((archive / "verification_2.json").read_text())
    if first != second or not first["identical"]:
        raise RuntimeError("Две независимые проверки не совпали")
    names = validate_targets(second["files"])
    # Отсутствующие канонические runs публикуются без дублирования файлов архива.
    published = []
    for name in names:
        destination = Path("runs") / name
        if not destination.exists():
            shutil.copytree(archive / "runs" / name, destination, copy_function=os.link)
            published.append(name)
    write_json(archive / "canonical_publication.json", {"created": published})
    remove_verified_remote(second["files"], names)
    write_json(
        archive / "deletion_complete.json",
        {
            "removed_remote_runs": names,
            "remote_root": REMOTE_ROOT,
            "bytes_preserved": second["bytes"],
            "files_preserved": second["count"],
            "local_archive": str(archive / "runs"),
            "verification_passes": 2,
            "additional_remote_hash_check_before_deletion": True,
            "mlflow_backend_removed": False,
        },
    )
    print("Все проверенные run-копии удалены с A100; локальный архив сохранён", flush=True)


if __name__ == "__main__":
    main()
