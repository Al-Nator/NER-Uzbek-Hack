# Артефакты запуска

## Serving, отдельно от обучающих runs

`artifacts/serving/s62-c02-v1/` — проверяемый SHA-256 bundle трёх исходных
checkpoint-ов и train-only словаря. Optimizer/training state исключены.
`reference/` содержит архивные predictions исключительно для offline parity;
runtime их не читает для ответа. Локальные веса по возможности hard-link,
их нельзя изменять на месте.

`artifacts/serving/engines-bf16/` — производные A100 TensorRT plans и metadata;
они не подменяют исходные веса. Engine проверяется по SHA-256 и версии TensorRT.
Промежуточный ONNX можно пересоздать из bundle и не хранить постоянно.
Основной профиль использует все три engine: s33, s21, s31.

Финальная поставка — `uzner-a100-s21-resources-v1.tar.zst`
(6 627 462 514 байт) и соседний `.sha256`. Архив сохраняет пути относительно
корня репозитория и содержит bundle, engines и `RELEASE-SHA256SUMS`.
Проверены распаковка, SHA-256 всех 31 файлов ресурсов и отдельно 24 файла
по исходному bundle manifest. [Состав и команды сборки](DELIVERY.md).
Архив передаётся отдельно: Git не содержит весов и не заменяет эту поставку.

`artifacts/serving/benchmarks/<variant>/` — полный `summary.json`, predictions,
parity diff и, для HTTP, индивидуальные latency и проверка повторных прогонов.
Тяжёлые файлы не идут в Git или MLflow; компактная проверяемая сводка —
[`reports/serving_s21_20260907.md`](../reports/serving_s21_20260907.md).
Первый этап оптимизации сохранён в
[`reports/serving_a100_20260907.md`](../reports/serving_a100_20260907.md) как история.

## Исследовательские runs

[Posthoc-проверки 07.09.2026](POSTHOC_DECODING.md) сохраняются в `runs/posthoc_…/`:
resolved config, hashes, source snapshot, полный dev evaluator, изменения spans и MLflow.
В них нет новых checkpoint-ов и обучающих loss-ов: это явно помеченные проверки
готовых предсказаний. Сводная таблица — `reports/posthoc_decoding_20260907.md`.

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
Для самих s57/s58 полный HTTP benchmark не проводился.
Скорость другого, финального ансамбля s62+c02 впоследствии измерена отдельно:
[all-TensorRT HTTP-отчёт](../reports/serving_s21_20260907.md).

## Посылка проверенного ансамбля

`scripts/predict_ensemble.py` применяет три `best`-checkpoint-а из
`resolved_config.json` выбранного ensemble run последовательно на GPU, затем
использует тот же `majority_vote`, что и dev-эксперимент. Обучение, подбор
порогов, оценка по gold и отправка на лидерборд не выполняются.

```bash
uv run python scripts/predict_ensemble.py \
  --ensemble-run runs/s62_span_majority_full-targeted-v1 \
  --input /path/to/public_test_inputs.jsonl \
  --output-dir artifacts/submissions/s62_public_v1
```

Новый каталог содержит `predictions.jsonl` для загрузки, `components/*.jsonl`
с отдельными голосами и `predictions.manifest.json` с SHA-256 входа, выхода,
весов и tokenizer-ов. Повторная запись существующего каталога запрещена.
Эти файлы не являются обучающим run и не создают фиктивные метрики в MLflow.
