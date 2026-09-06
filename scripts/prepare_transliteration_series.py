"""CLI воспроизводимой подготовки train-аугментации серии 4."""

import argparse
import json
from pathlib import Path

from uzner.data.augmentation import prepare
from uzner.data.augmentation_full import prepare_full


def main() -> None:
    """Создаёт release из явно указанных derived-файлов, не меняя official train/dev."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full-combined", action="store_true")
    parser.add_argument(
        "--train", type=Path, default=Path("ner_uz_hackathon_participant/data/train.jsonl")
    )
    parser.add_argument(
        "--dev", type=Path, default=Path("ner_uz_hackathon_participant/data/dev.jsonl")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            (prepare_full if args.full_combined else prepare)(
                args.derived, args.train, args.dev, args.output
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
