# Трекинг экспериментов

## Решение

Используется гибрид `runs/` + MLflow. Каноническая воспроизводимая единица —
`runs/<run_id>/`; MLflow даёт интерактивные графики и сравнение запусков. Локальный
backend — `mlruns/mlflow.db`, artifact store — `mlruns/artifacts/`, experiment —
`uzner-first-series`. Сервер не нужен для записи: обучение пишет прямо в SQLite.

Checkpoint-ы не копируются в MLflow, чтобы не удваивать десятки гигабайт. Вместо
этого логируются их размеры и хэши из manifest; config, metadata, predictions,
metrics, logs, environment и reports копируются как лёгкие артефакты.

Smoke/test-запуски с `--smoke-output` не создают MLflow run и не загрязняют
сравнение. Полный run без MLflow можно явно запустить только с
`UZNER_MLFLOW_ENABLED=false`.

## Что логируется

- на optimizer step: loss, learning rate, gradient norm, tokens/s, CUDA memory;
- на каждой эпохе: train/dev loss, exact micro/macro и классы, TP/FP/FN,
  boundary/error-метрики и все slice-метрики;
- финально: tokenizer ceiling, subwords, chars/subword, окна, объёмы артефактов;
- каждые 10 секунд: CPU, RAM, disk/network и NVIDIA GPU utilization/memory/power;
- tags/params: encoder revision, BIO/BIOES, head/decoder, seed, гиперпараметры,
  data hashes, размеры выборок, runtime и число параметров.

## Запуск UI

```bash
uv sync --extra train --group dev
make mlflow
# открыть http://127.0.0.1:5000
```

Для общего сервера достаточно задать `MLFLOW_TRACKING_URI`; имя experiment
меняется через `UZNER_MLFLOW_EXPERIMENT`, системные метрики — через
`UZNER_MLFLOW_SYSTEM_METRICS`.

## Варианты

| Вариант | Когда выбирать | Компромисс |
|---|---|---|
| `runs/` + MLflow | Сейчас, одна GPU/малая команда | Нужно хранить SQLite и artifact store |
| Только `runs/` | Аварийный offline-режим | Нет интерактивного multi-run UI |
| ClearML | Нужны ещё remote agents, queues, datasets и orchestration | Более тяжёлая платформа и server setup |
| W&B | Важны самый быстрый hosted UI и collaboration | Cloud/privacy нужно решить заранее; self-managed значительно тяжелее |

При росте команды текущий MLflow переносится на общий tracking server. ClearML
имеет смысл брать, когда понадобится не только сравнение, но и управление GPU-очередью.

## Временный A100-хост

На `alnator` используется отдельный SQLite backend и experiment
`uzner-second-series-a100`. Его UI доступен через SSH tunnel на
`http://127.0.0.1:5001`. Обучение не пишет по сети в локальный backend,
поэтому обрыв SSH не теряет метрики.

Для новой общей очереди s24/s25/s30/s31 задан experiment
`uzner-continuation-a100`, suffix `a100-continuation-v1`.
Инструкции запуска и возврата: [`A100_QUEUE.md`](A100_QUEUE.md).

После `status=complete` весь `runs/<run_id>` возвращается через `rsync`
и проверяется по artifact manifest. Params, tags, полная metric history
и лёгкие артефакты импортируются в основной experiment
`uzner-first-series`. Checkpoint-ы остаются только в локальном `runs/`.
Команды и аварийный повторный импорт описаны в
[`REMOTE_EXPERIMENTS.md`](REMOTE_EXPERIMENTS.md).

Ссылки: [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking),
[ClearML Tasks](https://clear.ml/docs/latest/docs/fundamentals/task/),
[ClearML architecture](https://clear.ml/docs/latest/docs/getting_started/architecture/),
[W&B deployment options](https://docs.wandb.ai/platform/hosting).
