"""JSONL → финальный HTTP-ансамбль → проверенный predictions.jsonl."""

import argparse
import sys
from pathlib import Path

import httpx

from uzner.serving.http_client import HttpSettings
from uzner.serving.predict_job import PredictionJob, predict_file


def main(argv: list[str] | None = None) -> int:
    """Предсказывает все документы либо завершается ошибкой без частичной посылки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    try:
        job = PredictionJob(
            args.input,
            args.output_dir,
            HttpSettings(args.url, args.timeout, args.startup_timeout),
            args.batch_size,
        )
        print(predict_file(job))
    except (ValueError, TypeError, OSError, httpx.HTTPError) as error:
        print(f"predict: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
