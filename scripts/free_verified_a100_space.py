"""Освобождает только дважды проверенные старые runs для раннего старта."""

import inspect
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from uzner.experiments.artifacts import write_json
from uzner.experiments.remote_archive import SSH, assert_equal, inventory

ARCHIVE = Path("artifacts/remote_archive/alnator_20260906_v1")
NAMES = (
    "s32_bge_m3_retromae_global_pointer_a100-continuation-v1",
    "s33_bge_global_pointer_low_lr_a100-low-lr-v1",
    "s40_bge_global_pointer_translit_converter_matched_s4-translit-matched-v1",
    "s41_bge_global_pointer_translit_luna_matched_s4-translit-matched-v1",
    "s44_bge_gp_translit_luna_full_full-targeted-v1",
)


def remote_scan(name: str) -> dict:
    """Получает полные SHA-256 одного фиксированного удалённого run."""
    source = "import hashlib,json\nfrom pathlib import Path\n" + inspect.getsource(inventory)
    source += f"\nprint(json.dumps(inventory(Path('/home/danya/NER-Uzbek-Hack/runs') / {name!r})))"
    result = subprocess.run(
        [*SSH, "cd /home/danya/NER-Uzbek-Hack && uv run python -"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    """Сверяет каждый байт дважды до удаления конкретных удалённых копий."""
    reports = {}
    for name in NAMES:
        previous = None
        for number in (1, 2):
            print(f"{name}: SHA-256 pass {number}", flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                remote = pool.submit(remote_scan, name)
                local = pool.submit(inventory, ARCHIVE / "runs" / name)
                result = remote.result()
                assert_equal(result, local.result())
            if not result:
                raise ValueError("Пустой run нельзя удалять")
            if previous is not None:
                assert_equal(previous, result)
            previous = result
            write_json(ARCHIVE / f"early_{name}_pass{number}.json", result)
        reports[name] = previous
    # Цели перечислены явно; новые final-fit каталоги в список не входят.
    source = "import shutil\nfrom pathlib import Path\n"
    source += f"root=Path('/home/danya/NER-Uzbek-Hack/runs')\nnames={NAMES!r}\n"
    source += (
        "for name in names:\n p=root/name\n"
        " if p.is_symlink() or p.resolve().parent != root: raise ValueError(name)\n"
        " shutil.rmtree(p)\n"
    )
    subprocess.run(
        [*SSH, "cd /home/danya/NER-Uzbek-Hack && uv run python -"],
        input=source,
        text=True,
        check=True,
    )
    write_json(
        ARCHIVE / "early_space_ready.json",
        {
            "verification_passes": 2,
            "removed_remote_runs": list(NAMES),
            "bytes_preserved": sum(v[0] for run in reports.values() for v in run.values()),
        },
    )


if __name__ == "__main__":
    main()
