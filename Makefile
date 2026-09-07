.DEFAULT_GOAL := help
.PHONY: sync sync-train mlflow test lint format-check validate-data check
.PHONY: help setup-organizers setup-research serve predict eval benchmark train train-smoke train-resume test-organizers

URL ?= http://127.0.0.1:8000
BATCH ?= 8
ROUNDS ?= 2
CONCURRENCY ?= 1
MEMORY ?= auto
PROFILE ?= configs/serving/default.json
CONFIG ?= configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml
RUN_SUFFIX ?= organizers-v1

help:
	@echo 'Команды организаторов (подробно: docs/ORGANIZERS.md):'
	@echo '  make setup-organizers       — зафиксированное окружение uv'
	@echo '  make serve                  — финальный GPU Docker-сервис, localhost:8000'
	@echo '  make predict INPUT=... OUTPUT=... [URL=...]'
	@echo '  make eval GOLD=... PREDICTIONS=... OUTPUT=...'
	@echo '  make benchmark INPUT=... OUTPUT=... [LABELED=1 BATCH=8 MEMORY=required]'
	@echo '  make train [CONFIG=... RUN_SUFFIX=organizers-v1]'
	@echo '  make train-smoke OUTPUT=... [CONFIG=...]'
	@echo '  make train-resume CONFIG=... RUN_SUFFIX=...'
	@echo 'OUTPUT predict/benchmark/smoke — новый каталог; OUTPUT eval — новый JSON-файл.'

setup-organizers:
	uv sync --frozen --extra train --extra serve --group dev

setup-research:
	uv sync --frozen --extra train --extra serve --extra research --group dev

serve:
	docker compose up -d --wait backend

predict:
	@test -n "$(INPUT)" -a -n "$(OUTPUT)" || { echo 'Нужны INPUT и OUTPUT'; exit 2; }
	uv run --no-sync python scripts/predict_service.py --input "$(INPUT)" --output-dir "$(OUTPUT)" --url "$(URL)" --batch-size "$(BATCH)"

eval:
	@test -n "$(GOLD)" -a -n "$(PREDICTIONS)" -a -n "$(OUTPUT)" || { echo 'Нужны GOLD, PREDICTIONS и OUTPUT'; exit 2; }
	uv run --no-sync python scripts/evaluate.py --gold "$(GOLD)" --predictions "$(PREDICTIONS)" --output "$(OUTPUT)"

benchmark:
	@test -n "$(INPUT)" -a -n "$(OUTPUT)" || { echo 'Нужны INPUT и OUTPUT'; exit 2; }
	uv run --no-sync python scripts/benchmark_service.py --input "$(INPUT)" --output "$(OUTPUT)" --url "$(URL)" --config "$(PROFILE)" --batch "$(BATCH)" --rounds "$(ROUNDS)" --concurrency "$(CONCURRENCY)" --gpu-memory "$(MEMORY)" $(if $(filter 1,$(LABELED)),--gold,) $(if $(RATE),--arrival-rate "$(RATE)",) $(if $(REFERENCE),--reference "$(REFERENCE)",)

train:
	uv run --no-sync python scripts/train.py --config "$(CONFIG)" --run-suffix "$(RUN_SUFFIX)"

train-smoke:
	@test -n "$(OUTPUT)" || { echo 'Нужен OUTPUT — отдельный каталог smoke'; exit 2; }
	uv run --no-sync python scripts/train.py --config "$(CONFIG)" --run-suffix "$(RUN_SUFFIX)" --smoke-output "$(OUTPUT)" --max-train-documents 8 --max-dev-documents 4 --max-epochs 1

train-resume:
	uv run --no-sync python scripts/train.py --config "$(CONFIG)" --run-suffix "$(RUN_SUFFIX)" --resume

test-organizers:
	uv run --no-sync pytest --no-cov tests/test_organizer_http.py tests/test_organizer_tools.py tests/test_training_safety.py tests/test_serving_runtime.py tests/test_serving_bundle.py tests/test_tensorrt_contract.py tests/test_organizer_gpu_smoke.py tests/test_serving_contract.py tests/test_service_benchmark.py tests/test_training_pipeline.py tests/test_official_runner.py tests/test_metrics.py tests/test_tagging.py tests/test_windows_and_models.py

sync:
	uv sync --group dev

sync-train:
	uv sync --extra train --group dev

mlflow:
	uv run --extra train mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

test:
	uv run --frozen --extra train --extra serve --extra research --group dev pytest

lint:
	uv run --extra train ruff check .

format-check:
	uv run --extra train ruff format --check .

validate-data:
	uv run python scripts/validate_data.py --config configs/data/official.yaml

check: test lint format-check validate-data
