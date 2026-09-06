"""Готовит проверенную исследовательскую выборку из согласованного LLM-релиза."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from uzner.data.silver_experiment import SilverPreparation, prepare_silver


def main() -> None:
    """Разбирает явное разрешение и сохраняет выборку с аудитом."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--official-root", type=Path, default=Path("ner_uz_hackathon_participant/data")
    )
    parser.add_argument("--research-authorized", action="store_true")
    args = parser.parse_args()
    counts = prepare_silver(
        SilverPreparation(
            args.release,
            args.official_root,
            args.output,
            args.research_authorized,
        )
    )
    print(json.dumps(asdict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
