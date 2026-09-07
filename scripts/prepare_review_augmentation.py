"""Явно подключает reviewed-кандидатов к эксперименту, не к gold-корпусу."""

import argparse
import json
from pathlib import Path

from uzner.data.augmentation_review import prepare_review


def main() -> None:
    """Готовит отдельный immutable-релиз train-аугментации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train", type=Path, default=Path("ner_uz_hackathon_participant/data/train.jsonl")
    )
    parser.add_argument(
        "--dev", type=Path, default=Path("ner_uz_hackathon_participant/data/dev.jsonl")
    )
    args = parser.parse_args()
    print(json.dumps(prepare_review(args.input, args.train, args.dev, args.output), indent=2))


if __name__ == "__main__":
    main()
