"""Проверка автономного non-root Docker: без сети, mounts и обязательных env."""

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter, sleep

from uzner.experiments.artifacts import write_json

HEALTH = (
    "import urllib.request; "
    "urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"
)
PROBE = '''
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError

def send(items):
    """Отправляет настоящие Unicode-строки без доступа к внешней сети."""
    request = Request(
        "http://127.0.0.1:8000/api/v1/predict",
        data=json.dumps(items, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)["data"]

assert os.getuid() != 0, "Контейнер не должен требовать root"
texts = ["", " \\t\\n", "🙂 Ali Toshkentda ishlaydi.",
         "Алишер Навоий Тошкентда туғилган.", "İstanbul Oʻzbekiston O’zbekiston",
         "Ali Toshkentda ishlaydi. " * 500]
items = [{"hash": str(i), "text": text} for i, text in enumerate(texts)]
result = send(items)
assert [x["hash"] for x in result] == [x["hash"] for x in items]
counts = []
for source, prediction in zip(items, result):
    keys = set()
    for entity in prediction["entities"]:
        start, end, label = entity["start"], entity["end"], entity["label"]
        assert type(start) is int and type(end) is int
        assert label in {"ORG", "NAME", "GEO"}
        assert 0 <= start < end <= len(source["text"])
        assert (label, start, end) not in keys
        keys.add((label, start, end))
    counts.append(len(keys))
assert counts[:2] == [0, 0]
assert send([{"hash": "0", "text": ""}])[0]["entities"] == []
try:
    send([items[2], items[2]])
except HTTPError as error:
    assert error.code == 422
else:
    raise AssertionError("Повторный hash не отклонён")
print(json.dumps({"uid": os.getuid(), "documents": len(items),
                  "entity_counts": counts, "max_characters": max(map(len, texts)),
                  "empty_unicode_long_input": "passed", "duplicate_hash_status": 422}))
'''


def execute(container: str, script: str) -> subprocess.CompletedProcess:
    """Исполняет stdlib-проверку внутри контейнера по его loopback HTTP."""
    return subprocess.run(
        ["docker", "exec", "-i", container, "/opt/uzner/.venv/bin/python", "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=180,
    )


def main() -> None:
    """Создаёт только свой временный контейнер и сохраняет все результаты."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container", default="uzner-offline-contract-check")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    existing = subprocess.run(["docker", "inspect", args.container], capture_output=True)
    if existing.returncode == 0:
        raise ValueError("Контейнер с таким именем уже существует; он не будет изменён")
    started = perf_counter()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            args.container,
            "--network",
            "none",
            "--gpus",
            "all",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m",
            args.image,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        deadline = perf_counter() + 240
        while perf_counter() < deadline:
            if execute(args.container, HEALTH).returncode == 0:
                break
            sleep(1)
        else:
            raise TimeoutError("Модель не загрузилась за 240 секунд")
        ready_seconds = perf_counter() - started
        probe = execute(args.container, PROBE)
        if probe.returncode:
            raise RuntimeError(probe.stderr)
        result = json.loads(probe.stdout)
        result.update(
            {
                "image": args.image,
                "network": "none",
                "read_only": True,
                "mounts": [],
                "readiness_seconds": ready_seconds,
            }
        )
        write_json(args.output / "result.json", result)
        print(json.dumps(result))
    finally:
        logs = subprocess.run(["docker", "logs", args.container], capture_output=True, text=True)
        (args.output / "container.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        subprocess.run(["docker", "stop", args.container], check=False, capture_output=True)


if __name__ == "__main__":
    main()
