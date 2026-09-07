"""Обучение финальной модели на train+dev с фиксированным последним checkpoint."""

import argparse
from pathlib import Path

from uzner.training.final_fit import FinalFitRequest, run_final_fit


def main() -> None:
    """Запускает отдельный final-fit, не публикуя фиктивных validation-метрик."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--selected-epoch", type=int)
    parser.add_argument("--allow-research-initial", action="store_true")
    args = parser.parse_args()
    print(
        run_final_fit(
            FinalFitRequest(
                args.config.resolve(),
                Path.cwd(),
                args.smoke_output,
                args.document_limit,
                args.selected_epoch,
                args.allow_research_initial,
            )
        )
    )


if __name__ == "__main__":
    main()
