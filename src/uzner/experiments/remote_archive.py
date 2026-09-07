"""Точный архив A100 runs с независимыми SHA-256 сверками."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from uzner.experiments.artifacts import write_json

REMOTE_ROOT = "/home/danya/NER-Uzbek-Hack/runs"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "alnator"]


def inventory(root: Path) -> dict:
    """Хэширует каждый обычный файл; ссылки и изменяющиеся файлы запрещены."""
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Симлинк требует отдельной проверки: {path}")
        if not path.is_file():
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Файл изменяется: {path}")
        entries[str(path.relative_to(root))] = [after.st_size, digest.hexdigest()]
    return entries


def remote_inventory() -> dict:
    """Выполняет тот же потоковый аудит на удалённом диске, без передачи весов."""
    import inspect

    source = "import hashlib,json\nfrom pathlib import Path\n" + inspect.getsource(inventory)
    source += f"\nprint(json.dumps(inventory(Path({REMOTE_ROOT!r}))))\n"
    result = subprocess.run(
        [*SSH, "cd /home/danya/NER-Uzbek-Hack && uv run python -"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def assert_equal(remote: dict, local: dict) -> None:
    """Запрещает удаление при пропущенном, лишнем или отличающемся файле."""
    if remote != local:
        bad = [name for name in remote.keys() | local.keys() if remote.get(name) != local.get(name)]
        raise ValueError(f"Архив не совпадает: {bad[:20]}; всего {len(bad)}")


def main() -> None:
    """Копирует архив либо дважды проверяет и удаляет только перечисленные run-каталоги."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if archive == Path.cwd() or archive == Path("/"):
        raise ValueError("Небезопасный архивный каталог")
    target = archive / "runs"
    if args.copy:
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-aHc",
                "--partial",
                "--info=progress2",
                f"--link-dest={Path.cwd() / 'runs'}",
                f"--link-dest={Path.cwd() / 'runs/.remote-incoming'}",
                f"alnator:{REMOTE_ROOT}/",
                f"{target}/",
            ],
            check=True,
        )
    if args.verify or args.delete:
        previous = None
        for number in (1, 2):
            print(f"SHA-256 pass {number}: remote + local", flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                remote_job = pool.submit(remote_inventory)
                local_job = pool.submit(inventory, target)
                remote, local = remote_job.result(), local_job.result()
            assert_equal(remote, local)
            if previous is not None:
                assert_equal(previous, remote)
            write_json(
                archive / f"verification_{number}.json",
                {
                    "remote_root": REMOTE_ROOT,
                    "files": remote,
                    "count": len(remote),
                    "bytes": sum(v[0] for v in remote.values()),
                    "identical": True,
                },
            )
            previous = remote
            print(f"Pass {number} OK: {len(remote)} files", flush=True)
        if args.delete:
            raise RuntimeError(
                "Удаление выполняется отдельным явным вызовом после проверки отчётов"
            )


if __name__ == "__main__":
    main()
