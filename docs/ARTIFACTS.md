# Артефакты запуска

`runs/<run_id>/` — единица воспроизводимости. Каталог не перезаписывается;
продолжение возможно только через `--resume` из `checkpoints/last`.

Новые runs также сохраняют `environment/source.tar.gz` и `source_manifest.json`
с фактически исполняемыми Python/YAML и pyproject, включая untracked-код.
Это дополняет git commit/dirty и uv.lock; данные и веса в snapshot не входят.
Span-модели добавляют `metrics/span_coverage.json`: достижимость gold-границ
во всех окнах до обучения. Оба вида лёгких артефактов публикуются в MLflow.

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

## Frozen research s57/s58

Постобработка фиксированного encoder-а имеет отдельную схему, не выдаётся за
обычный полный TrainRequest. `runs/s57…` / `runs/s58…` содержат:

- `resolved_config.json`, `metadata.json` с hashes source best и original train/dev;
- `environment/source.tar.gz`, `source_manifest.json`, `status.json`;
- `environment/uv.lock`, `artifact_manifest.json` с SHA-256 всех собственных артефактов;
- `events.jsonl`, `mlflow.json` с ID локального MLflow run;
- `reference/` и `dev/`: общий metrics JSON, exact predictions, error JSONL;
- s57: `memory.pt` и `train_groups.json`; encoder не копируется, обучения/эпох нет;
- s58: `checkpoints/best.pt`, `last.pt` только char-head/optimizer/RNG/алфавит;
  `epochs/<n>/` хранит dev-оценки каждой эпохи. Автоматический CLI resume для этой
  исследовательской ветки пока не реализован; состояние last сохранено.

Это независимые frozen-эксперименты с видом `frozen_encoder_research` в MLflow.
Глобальные/срезовые метрики считаются общим evaluator; без chunk-edge cache срез
границ окон не рассчитывается, а не подменяется произвольными границами.
Память и веса не загружаются в MLflow, только лёгкие JSON/логи/source snapshot.
Полная скорость HTTP-сервиса отдельно ещё не измерена.
