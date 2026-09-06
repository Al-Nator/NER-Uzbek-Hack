<a id="top"></a>

# Uzbek exact-span NER

Рабочий репозиторий для экспериментов по распознаванию сущностей `ORG`, `NAME`
и `GEO` в узбекских, русских и смешанных текстах. Итоговая единица оценки —
точное совпадение `(label, start, end)` в исходной Unicode-строке.

Сейчас в репозитории есть исходный комплект организаторов, единые доменные
типы, загрузка и проверка нескольких источников данных, BIO/BIOES, softmax/CRF,
constrained decoding, exact-span и диагностические метрики, полный train/resume-контур,
MLflow, Biaffine/GlobalPointer, MELM-inspired подготовка данных и research-ветки
обучающих целей, kNN и символьных границ. Все запуски первой серии выполнены; результаты
после исправления whitespace-offsets сохранены в канонических run-каталогах с
suffix `s1-offsetfix-mlflow-r2`. Вторая серия sequence-моделей завершена;
лучший результат — XLM-R-large BIOES constrained, exact micro-F1 `0.90101`.
Серии 2B и 3 выполнены на A100. Текущий номинальный лучший среди одиночных моделей —
BGE-M3-RetroMAE + GlobalPointer на train + silver (s45), exact micro-F1
`0.90958`; [результаты s45/s46](docs/FOURTH_SILVER.md).
Фиксированный ансамбль s33+s21+s31 (s62)
получил **`0.91240` на dev**; его HTTP-скорость ещё не проверена.
s30 прерван и сохранён отдельно как неудачный частичный результат.
Подробные результаты: [журнал экспериментов](docs/EXPERIMENTS.md).

## Оглавление

- [Состояние проекта](#status)
- [Структура](#layout)
- [Данные](#data)
- [Эксперименты](#experiments)
- [Запуск обучения](#training)
- [Удалённый A100](#remote-a100)
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
- Лучший encoder первой серии — mDeBERTa-v3-base: exact micro-F1 `0.8804`, best epoch 5.
- Лучший decoding на XLM-R-base — BIOES + CRF: exact micro-F1 `0.8756`,
  `+0.0206` к BIO/softmax/greedy.
- Вторая серия переносит BIOES constrained и BIOES + CRF на mDeBERTa-v3-base
  и XLM-R-large. `s20` достиг `0.8963`, `s21` — `0.8984`, `s22` — **`0.90101`**,
  `s23` — `0.89827`. Два последних run-а выполнены на A100.
- [Общая очередь A100](docs/A100_QUEUE.md): s24 → s25 → s30 → s31, по одному обучению.
- [Серия 2B](docs/ENCODER_CONTINUATION.md): BGE и XLM-V; на A100 оба используют обычный AdamW.
- [Серия 3](docs/THIRD_SERIES.md): XLM-R-large + Biaffine и GlobalPointer на A100 после 2B.
- [Серия 4](docs/FOURTH_SERIES.md): [s40/s41 — парная абляция транслитераций](docs/FOURTH_TRANSLITERATION.md), затем отдельные эксперименты с внешними данными.
- s40/s41 завершены: `0.905255 / 0.901283`, улучшения к s32 нет; веса и MLflow возвращены.
- [s42 MELM-inspired](docs/FOURTH_MELM.md): завершён с 4803 копиями, F1 `0.903436`.
- [Серия 5](docs/FIFTH_SERIES.md): s53–s56, kNN/char-head завершены; auxiliary-головы ещё на A100.
- [Полные транслитерации s43/s44](docs/FOURTH_FULL_TRANSLITERATION.md): +15872 / +9827 копий, поставлены в очередь A100.
- [Серия 6](docs/SIXTH_SERIES.md): mDeBERTa+GP, отдельный LR головы и измеренное span-голосование.
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
только финальные символьные spans; token/span модели уже используют один evaluator.
Будущие set-prediction архитектуры сохранят этот контракт.

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

Основной запуск продолжения encoder-ов и span-head на A100 (из локального терминала):

```bash
uv run python scripts/remote_experiments.py --remote-experiment uzner-continuation-a100 start \
  --series configs/series/continuation_a100.yaml --stage all \
  --run-suffix a100-continuation-v1 --session uzner-s24-s31
```

Проверки, логи и возврат в MLflow: [`A100_QUEUE.md`](docs/A100_QUEUE.md).

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

<a id="remote-a100"></a>

## Удалённый A100

`s22` и `s23` можно запустить на A100-хосте `alnator` из локального
корня проекта:

```bash
uv run --extra train python scripts/remote_experiments.py prepare --preflight
uv run --extra train python scripts/remote_experiments.py start
```

Обучение продолжается в remote zellij после закрытия SSH. Remote
MLflow открывается через tunnel на `http://127.0.0.1:5001`, а готовые
run-ы возвращаются через проверяемый `rsync` и импорт MLflow.
Команды для status, live log, tunnel и sync:
[`docs/REMOTE_EXPERIMENTS.md`](docs/REMOTE_EXPERIMENTS.md).

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
- [Удалённые эксперименты на A100](docs/REMOTE_EXPERIMENTS.md)
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
