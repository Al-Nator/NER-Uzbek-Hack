"""Последовательная локальная очередь после освобождения GPU веткой MELM/s42."""

import subprocess
import sys
import time


def main() -> int:
    """Ожидает локальную зависимость и запускает два независимых опыта без наложения GPU."""
    print("Ожидаю завершения uzner-melm-s42.service, затем s57 → s58", flush=True)
    while (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "uzner-melm-s42.service"], check=False
        ).returncode
        == 0
    ):
        time.sleep(30)
    failed = []
    for name in ("s57_bge_gp_memory_knn", "s58_bge_gp_char_boundary"):
        print(f"Запускаю {name}", flush=True)
        result = subprocess.run(
            [sys.executable, "scripts/run_frozen_research.py", f"configs/research/{name}.yaml"],
            check=False,
        )
        if result.returncode:
            failed.append(name)
            print(f"FAILED: {name}; следующий независимый опыт не отменяется", flush=True)
    if failed:
        print(f"Не завершились: {failed}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
