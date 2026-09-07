"""Фиксация автономного serving-bundle исходного ансамбля 0.8922."""

import argparse
from pathlib import Path

from uzner.serving.bundle import prepare_bundle


def main() -> None:
    """Создаёт новый bundle, не перезаписывая предыдущий."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/serving/s62-c02-v1"))
    args = parser.parse_args()
    print(prepare_bundle(args.root, args.output))


if __name__ == "__main__":
    main()
