"""Сборка public predictions выбранным лёгким posthoc-рецептом."""

import argparse
from pathlib import Path

from uzner.posthoc.submission import PosthocSubmission, build_submission


def main() -> int:
    """Принимает явные входы и не отправляет файл на лидерборд."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("selected-run", "train", "input-path", "base-predictions", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument(
        "--dictionary-dev",
        type=Path,
        help="Явно добавить dev в финальный словарь; dev больше не held-out оценка",
    )
    args = parser.parse_args()
    print(build_submission(PosthocSubmission(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
