<a id="top"></a>

# Uzbek exact-span NER

Рабочий репозиторий для экспериментов по распознаванию сущностей `ORG`, `NAME`
и `GEO` в узбекских, русских и смешанных текстах. Итоговая единица оценки —
точное совпадение `(label, start, end)` в исходной Unicode-строке.

Сейчас в репозитории есть исходный комплект организаторов, единые доменные
типы, загрузка и проверка нескольких источников данных, BIO/BIOES-преобразования,
exact-span метрики, конфигурации первой серии экспериментов и тесты этих
контрактов. Результаты ещё не заявлены: таблица запусков заполняется только
после фактического обучения и оценки.

## Оглавление

- [Состояние проекта](#status)
- [Структура](#layout)
- [Данные](#data)
- [Эксперименты](#experiments)
- [Установка через uv](#setup)
- [Проверки](#checks)
- [Документация](#docs)
- [Исходный комплект организаторов](#organizer-kit)

<a id="status"></a>

## Состояние проекта

- Исходные `train` и `dev` находятся в комплекте организаторов.
- Конфиги подготовлены для baseline, XLM-R, mDeBERTa и mmBERT, а также для
  сравнений BIO/BIOES и softmax/CRF.
- Все новые эксперименты обязаны сохранять разрешённый конфиг, data hashes,
  метрики, предсказания и два checkpoint: `best` и `last`.
- Полноценные модельные эксперименты выполняются на GPU. CPU используется для
  быстрых проверок данных, конфигов и unit-тестов.

[Наверх](#top)

<a id="layout"></a>

## Структура

```text
configs/                       конфигурации данных и экспериментов
docs/                          архитектура, протокол и журнал экспериментов
ner_uz_hackathon_participant/  неизменённый комплект организаторов
reports/                       компактные результаты для просмотра
scripts/                       CLI для проверки данных, конфигов и метрик
src/uzner/                     переиспользуемый код проекта
tests/                         unit-тесты контрактов проекта
runs/                          локальные артефакты запусков, не коммитятся
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

| ID | Назначение |
|---|---|
| `b00_official` | исходный DistilmBERT + BIO + softmax |
| `b01_reference` | тот же состав в новом воспроизводимом контуре |
| `e10_xlmr_base_bio` | сильная multilingual-база |
| `e11_mdeberta_v3_base_bio` | quality-кандидат |
| `e12_mmbert_base_bio` | современный multilingual-кандидат |
| `a20_xlmr_base_bio_constrained` | эффект constrained decoding |
| `a21_xlmr_base_bioes_constrained` | эффект BIOES |
| `a22_xlmr_base_bio_crf` | эффект CRF |
| `a23_xlmr_base_bioes_crf` | совместный эффект BIOES и CRF |

Перед запуском конфиг и все пути можно проверить без загрузки модели:

```bash
uv run python scripts/validate_experiment.py \
  --config configs/experiments/e10_xlmr_base_bio.yaml
```

Правила честного сравнения описаны в
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

[Наверх](#top)

<a id="setup"></a>

## Установка через uv

Минимальная среда для разработки и тестов:

```bash
uv sync --group dev
```

Среда для GPU-экспериментов с PyTorch CUDA 12.4:

```bash
uv sync --extra train --group dev
```

После синхронизации команды запускаются только через `uv run`, чтобы не
зависеть от глобального Python.

[Наверх](#top)

<a id="checks"></a>

## Проверки

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
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
