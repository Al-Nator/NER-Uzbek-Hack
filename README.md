<a id="top"></a>

# Uzbek exact-span NER

Рабочий репозиторий для экспериментов по распознаванию сущностей `ORG`, `NAME`
и `GEO` в узбекских, русских и смешанных текстах. Итоговая единица оценки —
точное совпадение `(label, start, end)` в исходной Unicode-строке.

Сейчас в репозитории есть исходный комплект организаторов, единые доменные
типы, загрузка и проверка нескольких источников данных, BIO/BIOES, softmax/CRF,
constrained decoding, exact-span и диагностические метрики, полный train/resume-контур,
MLflow и конфигурации первых двух серий. Все запуски первой серии выполнены; результаты
после исправления whitespace-offsets сохранены в канонических run-каталогах с
suffix `s1-offsetfix-mlflow-r2`. Вторая серия sequence-моделей подготовлена, но
полные результаты ещё не получены.

## Оглавление

- [Состояние проекта](#status)
- [Структура](#layout)
- [Данные](#data)
- [Эксперименты](#experiments)
- [Запуск обучения](#training)
- [Метрики и артефакты](#artifacts)
- [Установка через uv](#setup)
- [Проверки](#checks)
- [Документация](#docs)
- [Исходный комплект организаторов](#organizer-kit)

<a id="status"></a>

## Состояние проекта

- Исходные `train` и `dev` находятся в комплекте организаторов.
- Подготовлен полный путь `train → exact eval → best/last → report`, включая resume.
- Первая серия завершена: девять полных запусков имеют статус `complete` и
  соответствующие завершённые MLflow run-ы.
- Лучший encoder — mDeBERTa-v3-base: exact micro-F1 `0.8804`, best epoch 5.
- Лучший decoding на XLM-R-base — BIOES + CRF: exact micro-F1 `0.8756`,
  `+0.0206` к BIO/softmax/greedy.
- Вторая серия переносит BIOES constrained и BIOES + CRF на mDeBERTa-v3-base
  и XLM-R-large. Её результаты пока не заявляются.
- Все новые эксперименты обязаны сохранять разрешённый конфиг, data hashes,
  метрики, предсказания, MLflow run и два checkpoint: `best` и `last`.
- Полноценные модельные эксперименты выполняются на GPU. CPU используется для
  быстрых проверок данных, конфигов и unit-тестов.

[Наверх](#top)

<a id="layout"></a>

## Структура

```text
configs/                       данные, эксперименты и манифесты серий
docs/                          архитектура, протокол и журнал экспериментов
ner_uz_hackathon_participant/  неизменённый комплект организаторов
reports/                       компактные результаты для просмотра
scripts/                       CLI для проверок, аудита и запуска обучения
src/uzner/                     переиспользуемый код проекта
tests/                         unit-тесты контрактов проекта
runs/                          локальные артефакты запусков, не коммитятся
mlruns/                        SQLite и лёгкие MLflow-артефакты, не коммитятся
```

Код модели не должен знать, из какого файла пришли данные. Оценка получает
только финальные символьные spans, поэтому будущие span-level и set-prediction
архитектуры смогут использовать тот же evaluator.

[Наверх](#top)

<a id="data"></a>

## Данные

Текущие источники перечислены в
[`configs/data/official.yaml`](configs/data/official.yaml). Новый источник
добавляется отдельной записью с собственными `name`, `path`, `split`, `kind` и
`weight`. Загрузчик проверяет уникальность `hash` между всеми включёнными
файлами.

Исходная строка никогда не нормализуется на месте. Любые дополнительные
представления должны хранить обратимое отображение к исходным character offsets.

[Наверх](#top)

<a id="experiments"></a>

## Эксперименты

Первая серия конфигов находится в `configs/experiments/`:

| ID | Назначение | Micro-F1 |
|---|---|---:|
| `b00_official` | исходный DistilmBERT + BIO + softmax | 0.7897 |
| `b01_reference` | тот же состав в новом воспроизводимом контуре | 0.8175 |
| `e10_xlmr_base_bio` | XLM-R-base, BIO, greedy | 0.8550 |
| `e11_mdeberta_v3_base_bio` | лучший encoder первой серии | **0.8804** |
| `e12_mmbert_base_bio` | современный multilingual-кандидат | 0.8718 |
| `a20_xlmr_base_bio_constrained` | эффект constrained decoding | 0.8673 |
| `a21_xlmr_base_bioes_constrained` | эффект BIOES | 0.8734 |
| `a22_xlmr_base_bio_crf` | эффект CRF | 0.8707 |
| `a23_xlmr_base_bioes_crf` | совместный эффект BIOES и CRF | 0.8756 |

Перед запуском конфиг и все пути можно проверить без загрузки модели:

```bash
uv run python scripts/validate_experiment.py \
  --config configs/experiments/e10_xlmr_base_bio.yaml
```

Правила честного сравнения описаны в
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

Вторая серия sequence-моделей описана в
[`docs/SECOND_SERIES.md`](docs/SECOND_SERIES.md). Её четыре run-а не включают
span-based архитектуры.

[Наверх](#top)

<a id="training"></a>

## Запуск обучения

Один полный run:

```bash
uv run python scripts/train.py \
  --config configs/experiments/e10_xlmr_base_bio.yaml
```

Первая серия задана в `configs/series/first.yaml`. Проверить порядок без
обучения или запустить один этап:

```bash
uv run python scripts/run_series.py --stage all --dry-run
uv run python scripts/run_series.py --stage encoders
```

Повторный запуск не перезаписывает старые артефакты: задайте общий suffix,
который попадёт и в каталог, и в MLflow run name:

```bash
uv run python scripts/run_series.py --stage encoders --run-suffix rerun-01
uv run python scripts/run_series.py --stage decoding --run-suffix rerun-01
```

Аудит tokenizer-а до обучения:

```bash
uv run python scripts/audit_tokenizer.py \
  --config configs/experiments/e10_xlmr_base_bio.yaml

# pinned revision, fast tokenizer и один BF16 forward на GPU
uv run python scripts/preflight_models.py --stage encoders

# все train/dev документы проходят token alignment без запуска обучения
uv run python scripts/preflight_alignment.py --stage all
```

Урезанный smoke-run обязан идти в отдельный каталог и не публикуется в сводку:

```bash
uv run python scripts/train.py \
  --config configs/experiments/e10_xlmr_base_bio.yaml \
  --max-train-documents 16 --max-dev-documents 16 \
  --max-epochs 1 \
  --smoke-output /tmp/uzner-smoke
```

Порядок и гипотезы: [`docs/FIRST_SERIES.md`](docs/FIRST_SERIES.md).

Вторая серия запускается тем же CLI с отдельным манифестом:

```bash
uv run python scripts/run_series.py \
  --series configs/series/second.yaml \
  --stage all \
  --run-suffix s2-sequence-v1
```

[Наверх](#top)

<a id="artifacts"></a>

## Метрики и артефакты

Каждая эпоха пишет step/epoch loss, learning rate, gradient norm, throughput,
train/eval time, latency, GPU/CPU/RAM, exact micro/macro P/R/F1, TP/FP/FN и
P/R/F1 по `ORG`/`NAME`/`GEO`. В MLflow по каждой эпохе также попадают все срезы:
script, seen/unseen, span/document length, suffix, apostrophe/quotes, chunk-edge,
boundary/error и tokenizer representability/fragmentation/chars-per-subword.

Точная схема `runs/<run_id>/`: [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md). Решение по
ClearML, MLflow, W&B и локальному сравнению:
[`docs/EXPERIMENT_TRACKING.md`](docs/EXPERIMENT_TRACKING.md).

MLflow пишет metadata в `mlruns/mlflow.db`, а лёгкие копии отчётов — в
`mlruns/artifacts/`. Checkpoint-ы остаются только в каноническом `runs/`.
Локальный UI запускается отдельно и не нужен самому обучению:

```bash
make mlflow
# http://127.0.0.1:5000
```

[Наверх](#top)

<a id="setup"></a>

## Установка через uv

Минимальная среда для работы с данными и конфигами:

```bash
uv sync --group dev
```

Среда для полных тестов и GPU-экспериментов с PyTorch CUDA 12.4:

```bash
uv sync --extra train --group dev
```

После синхронизации команды запускаются только через `uv run`, чтобы не
зависеть от глобального Python.

[Наверх](#top)

<a id="checks"></a>

## Проверки

```bash
uv run --extra train pytest
uv run --extra train ruff check .
uv run --extra train ruff format --check .
uv run python scripts/validate_data.py \
  --config configs/data/official.yaml
```

Или одной командой:

```bash
make check
```

[Наверх](#top)

<a id="docs"></a>

## Документация

- [Архитектура](docs/ARCHITECTURE.md)
- [Протокол экспериментов](docs/EXPERIMENT_PROTOCOL.md)
- [Первая серия](docs/FIRST_SERIES.md)
- [Вторая серия](docs/SECOND_SERIES.md)
- [Артефакты запуска](docs/ARTIFACTS.md)
- [Трекинг экспериментов](docs/EXPERIMENT_TRACKING.md)
- [Журнал экспериментов](docs/EXPERIMENTS.md)
- [Карточка данных](docs/DATA_CARD.md)
- [Шаблон анализа ошибок](docs/ERROR_ANALYSIS.md)
- [Документы задачи](docs/task/README.md)

[Наверх](#top)

<a id="organizer-kit"></a>

## Исходный комплект организаторов

Каталог [`ner_uz_hackathon_participant/`](ner_uz_hackathon_participant/README.md)
содержит предоставленные данные, baseline, evaluator и проверки HTTP API. Он
используется как неизменяемый reference. Дополнительные FAQ и описание кейса
лежат в корне и перечислены в [индексе документов задачи](docs/task/README.md).

[Наверх](#top)
