.PHONY: sync sync-train mlflow test lint format-check validate-data check

sync:
	uv sync --group dev

sync-train:
	uv sync --extra train --group dev

mlflow:
	uv run --extra train mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

test:
	uv run --extra train pytest

lint:
	uv run --extra train ruff check .

format-check:
	uv run --extra train ruff format --check .

validate-data:
	uv run python scripts/validate_data.py --config configs/data/official.yaml

check: test lint format-check validate-data
