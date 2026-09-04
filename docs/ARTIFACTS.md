# Артефакты запуска

`runs/<run_id>/` — единица воспроизводимости. Каталог не перезаписывается;
продолжение возможно только через `--resume` из `checkpoints/last`.

```text
runs/<run_id>/
├── resolved_config.yaml
├── metadata.json
├── status.json
├── artifact_manifest.json
├── environment/
│   ├── runtime.json
│   ├── process.json
│   └── uv.lock
├── checkpoints/
│   ├── best/
│   └── last/
├── logs/
│   ├── events.jsonl
│   ├── history.csv
│   ├── mlflow_run_id.txt
│   └── console.log
├── predictions/
│   └── dev.jsonl
├── metrics/
│   ├── dev.json
│   ├── slices.json
│   ├── error_summary.json
│   ├── errors.jsonl
│   ├── tokenizer_audit.json
│   └── epochs/epoch_XX.json
└── reports/
    ├── report.md
    └── training.svg
```

`events.jsonl` — полная хронология с loss, learning rate, gradient norm, throughput и
GPU memory. `history.csv` — одна строка на эпоху. `artifact_manifest.json` содержит
размер и SHA-256 каждого файла.

`mlflow_run_id.txt` связывает локальный каталог с MLflow и нужен для `--resume`.
MLflow хранит интерактивные ряды и лёгкие копии артефактов в `mlruns/`, но не
копирует `checkpoints/`: источником весов остаётся только `runs/<run_id>/`.

Урезанный smoke-run обязан использовать `--smoke-output` вне `runs/`. Код
блокирует его публикацию в `reports/experiments.csv`. Автотесты используют
только pytest temporary directories; их результаты не являются экспериментами.
