.PHONY: sync sync-train test lint format-check validate-data check

sync:
	uv sync --group dev

sync-train:
	uv sync --extra train --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

validate-data:
	uv run python scripts/validate_data.py --config configs/data/official.yaml

check: test lint format-check validate-data
