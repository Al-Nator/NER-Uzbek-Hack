"""Запуск зафиксированной серии декодеров на готовых dev-предсказаниях."""

import argparse
from pathlib import Path

from uzner.posthoc.study import run_study


def main() -> int:
    """Запускает матрицу и запрещает неявное перезаписывание результатов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    print(run_study(args.config, skip_complete=args.skip_complete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
