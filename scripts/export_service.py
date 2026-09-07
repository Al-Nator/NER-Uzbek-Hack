"""Экспорт выбранных encoder-ов ансамбля без изменения checkpoint-ов."""

import argparse
from pathlib import Path

from uzner.serving.export import build_engine, export_encoder


def main() -> None:
    """Собирает отдельные engine; ошибки не заменяются скрытым fallback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/serving/s62-c02-v1"))
    parser.add_argument("--models", nargs="+", default=["s33", "s31", "s21"])
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx-only", action="store_true")
    args = parser.parse_args()
    for name in args.models:
        print(f"Export {name}", flush=True)
        onnx_path = export_encoder(
            args.bundle / "models" / name, args.bundle.parent / "onnx" / name
        )
        if not args.onnx_only:
            print(build_engine(onnx_path, args.output / f"{name}.plan", args.precision), flush=True)


if __name__ == "__main__":
    main()
