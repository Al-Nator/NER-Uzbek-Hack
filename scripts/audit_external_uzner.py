"""CLI аудита приложенного UzNER-100K без подключения к обучению."""

import argparse
from pathlib import Path

from uzner.data.external_audit import audit_directory
from uzner.experiments.artifacts import write_json


def main() -> None:
    """Записывает проверяемый отчёт о происхождении, offsets и дублях."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--official-root", type=Path, default=Path("ner_uz_hackathon_participant/data")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_directory(args.source, args.official_root)
    write_json(args.output, result)
    for item in result["files"]:
        print(
            item["file"],
            "records=",
            item["records"],
            "synthetic_original=",
            item["synthetic_original"],
            "promoted=",
            item["promoted_synthetic"],
            "surface_mismatch=",
            item["surface_mismatch"],
        )


if __name__ == "__main__":
    main()
