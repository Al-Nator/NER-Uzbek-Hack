FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv
WORKDIR /opt/uzner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    UV_CACHE_DIR=/tmp/uv-cache \
    HF_HOME=/tmp/huggingface \
    PYTHONPATH=/opt/uzner/src:/opt/uzner/apps/backend \
    PATH=/opt/uzner/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra serve --extra export --no-dev --no-install-project --no-cache
COPY --chown=10001:10001 artifacts/serving/ ./artifacts/serving/
COPY src/ ./src/
COPY apps/backend/app/ ./apps/backend/app/
COPY configs/serving/ ./configs/serving/
ARG RUNTIME_PROFILE=configs/serving/default.json
COPY ${RUNTIME_PROFILE} ./configs/serving/default.json

RUN useradd --uid 10001 --create-home ner
USER ner
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=5 \
  CMD uv run --no-sync python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2)" || exit 1
CMD ["uv", "run", "--no-sync", "python", "-m", "uvicorn", "app.ensemble:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
