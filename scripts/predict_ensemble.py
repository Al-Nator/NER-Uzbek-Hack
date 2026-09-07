"""Генерация predictions.jsonl проверенного ансамбля для лидерборда."""

import argparse
from pathlib import Path

from uzner.training.submission import EnsembleSubmission, build_ensemble_submission


def main() -> None:
    """Читает пути и запускает последовательный GPU-инференс трёх источников."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output = build_ensemble_submission(
        EnsembleSubmission(args.ensemble_run, args.input, args.output_dir, args.batch_size)
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
