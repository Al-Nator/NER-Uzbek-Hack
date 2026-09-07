# Uzbek NER · exact spans, three complementary models

Распознавание **организаций, людей и географических объектов** в узбекских,
русских и смешанных текстах. Латиница и кириллица, длинные документы,
точные границы в исходной Unicode-строке.

**Public micro-F1: 0.8922** · **Dev micro-F1: 0.91745** · **3 модели, один API**

[Команды организаторам](docs/ORGANIZERS.md) · [Запуск и A100](docs/SERVING.md) ·
[Измерения](reports/serving_a100_20260907.md) ·
[Эксперименты](docs/EXPERIMENTS.md) · [Документация](docs/README.md)

## Решение

```text
Исходный текст → окна 512 / overlap 128
                      ├─ BGE-M3-RetroMAE + GlobalPointer  (s33)
                      ├─ mDeBERTa-v3-base + BIOES / CRF   (s21)
                      └─ XLM-R-large + GlobalPointer     (s31)
                → exact-span большинство 2 из 3
                → проверенный train-словарь → точные повторы
                → ORG / NAME / GEO, исходные [start, end)
```

Это фиксированный **s62 + c02**, не новая обученная модель. Все три encoder-а
резидентны на GPU. Токенизация, объединение окон и декодирование используют
общую исследовательскую реализацию. Словарь построен только по original train;
исходный текст не переписывается. Нет LLM-запросов, сети или кэша готовых ответов
в пути предсказания.

## Качество

| Система | Dev exact micro-F1 | Public exact micro-F1 |
|---|---:|---:|
| s45, лучшая одиночная модель исследования | 0.90958 | 0.8768 |
| s62, большинство трёх моделей | 0.91240 | 0.8861 |
| s62 + точные повторы | 0.91525 | 0.8897 |
| **s62 + train-словарь → повторы** | **0.91745** | **0.8922** |

Public — результаты посылок, сообщённые пользователем; скрытых меток у сервиса
нет. Dev: исходные 1 500 документов, строгое совпадение `hash/label/start/end`.
Dev многократно использовался для отбора; это не независимая оценка привата.
Состав и веса заморожены в SHA-256 manifest. Свежий BF16-прогон на A100 немного
отличается от архивных ответов; [отчёт о parity](reports/serving_a100_20260907.md)
разделяет это численное отличие и эффект оптимизаций.

## Запуск

Нужны Docker с NVIDIA Container Toolkit и GPU. Финальная цель — **A100 80 GB,
driver 550.90.07 / CUDA 12.4**. Веса не хранятся в Git: перед сборкой нужны
проверенный bundle `artifacts/serving/s62-c02-v1/` и два TensorRT plan с metadata
в `artifacts/serving/engines-bf16/`. Здесь они уже подготовлены.
[Получение bundle и plans](docs/SERVING.md#bundle).
Есть и [загрузка готового автономного образа](docs/SERVING.md#перенос-готового-образа)
без повторной сборки; архив передаётся отдельно от Git.

```bash
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```

Модели и словарь находятся внутри образа; runtime не требует скачивания,
переменных окружения или внешних файлов. `GET /healthz` — готовность модели,
`POST /api/v1/predict` — пакетное предсказание. Первый запуск загружает веса.

```bash
curl http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  --data '[{"hash":"example-001","text":"Ali Toshkent shahrida ishlaydi."}]'
```

Форма ответа; конкретные сущности определяет модель:

```json
{"data":[{"hash":"example-001","entities":[{"label":"NAME","start":0,"end":3}]}]}
```

`end` не включается; индексы — символы Python Unicode, не байты и не UTF-16.
Пустой ответ — `"entities": []`. [Контракт организаторов](ner_uz_hackathon_participant/API.md).

### Интерфейс

```bash
docker compose up --build
```

Открыть **http://localhost:3000**. Frontend: React / TypeScript, ввод текста,
JSONL-пакеты, подсветка сущностей, экспорт ответов. Backend: FastAPI, проверка
входов и offsets, одна резидентная модельная служба. [Код приложений](apps/README.md)
отделён от обучения и исходного reference-комплекта.

## Производительность

Default — TensorRT BF16 для двух encoder-ов, PyTorch BF16 для mDeBERTa,
CPU TorchScript CRF. Полный HTTP, включая пред- и постобработку, на A100 80 GB:

| Входы | По одному, сообщений/с | Пакетами по 8, сообщений/с |
|---|---:|---:|
| Dev | 32,36 | 61,35 |
| Public | 31,07 | 56,44 |

Пик GPU **6,26 GiB**. Поток 30 одиночных запросов/с прошёл без ошибок:
public p95 **1,34 с**. Dev-F1 batch=8: **0,917458 → 0,917410** при +20,6%
throughput; одиночный режим ускорился на 47,5%. Измерительный драйвер **580.173.02**,
не целевой 550.90.07. Условия, небольшой запас одиночного режима и ограничения:
[полный отчёт](reports/serving_a100_20260907.md).

## Разработка и воспроизводимость

```bash
uv sync --extra train --extra serve --group dev
uv run pytest --no-cov tests/test_serving_contract.py tests/test_posthoc_rules.py
uv run python scripts/train.py --config configs/experiments/e10_xlmr_base_bio.yaml
make mlflow
```

Для изолированного сервиса достаточно `uv sync --extra serve`; TensorRT-экспорт
добавляет `--extra export`. Полные модельные эксперименты выполняются на GPU.
Каждый run сохраняет config, seed, data hashes, исходники, метрики, predictions
и запись MLflow. Веса не дублируются в MLflow.

```text
apps/                    frontend и HTTP-адаптер
src/uzner/               данные, модели, evaluation, posthoc, serving
configs/                 воспроизводимые эксперименты и runtime
scripts/                 обучение, экспорт, проверка, HTTP benchmark
tests/                   offsets, decoder, метрики, API и serving
docs/ · reports/         протоколы, решения и проверенные результаты
runs/ · artifacts/       локальные веса и артефакты; не Git
ner_uz_hackathon_participant/  неизменённый reference организаторов
```

История отрицательных результатов сохранена: увеличение данных, смена heads
и обучение на train+dev не объявляются улучшениями без измерений.
[Навигация по сериям и отчётам](docs/README.md).
