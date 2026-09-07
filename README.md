# Uzbek NER · exact spans, three complementary models

Распознавание **организаций, людей и географических объектов** в узбекских,
русских и смешанных текстах. Латиница и кириллица, длинные документы,
точные границы в исходной Unicode-строке.

**Public micro-F1: 0.8922** · **Dev micro-F1: 0.91745** · **3 модели, один API**

[Команды организаторам](docs/ORGANIZERS.md) · [Запуск и A100](docs/SERVING.md) ·
[Комплект поставки](docs/DELIVERY.md) · [Измерения](reports/serving_s21_20260907.md) ·
[Эксперименты](docs/EXPERIMENTS.md) · [Документация](#документация)

## Решение

```nushell
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
проверенный bundle `artifacts/serving/s62-c02-v1/` и три TensorRT plan с metadata
в `artifacts/serving/engines-bf16/`. Они передаются единым архивом
`uzner-a100-s21-resources-v1.tar.zst` отдельно от Git.
[Распаковка, проверка и сборка](docs/ORGANIZERS.md#1-окружение-и-сервис).

```nushell
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```

Модели и словарь находятся внутри образа; runtime не требует скачивания,
переменных окружения или внешних файлов. `GET /healthz` — готовность модели,
`POST /api/v1/predict` — пакетное предсказание. Первый запуск загружает веса.

```nushell
curl http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  --data '[{"hash":"example-001","text":"Ali Toshkent shahrida ishlaydi."}]'
```

Форма ответа; конкретные сущности определяет модель:

```nushell
{"data":[{"hash":"example-001","entities":[{"label":"NAME","start":0,"end":3}]}]}
```

`end` не включается; индексы — символы Python Unicode, не байты и не UTF-16.
Пустой ответ — `"entities": []`. [Контракт организаторов](ner_uz_hackathon_participant/API.md).

### Интерфейс

```nushell
docker compose up --build
```

Открыть **http://localhost:3000**. Frontend: React / TypeScript, ввод текста,
JSONL-пакеты, подсветка сущностей, экспорт ответов. Backend: FastAPI, проверка
входов и offsets, одна резидентная модельная служба. [Код приложений](apps/README.md)
отделён от обучения и исходного reference-комплекта.

## Производительность

Default — TensorRT BF16 для всех трёх encoder-ов,
CPU TorchScript CRF. Полный HTTP, включая пред- и постобработку, на A100 80 GB:

| Входы | По одному, сообщений/с | p50 / p95, мс |
|---|---:|---:|
| Dev | 44,21 | 16,24 / 51,57 |
| Public | 40,25 | 16,64 / 52,20 |

Два прохода, batch=1, concurrency=1, загрузка и warmup исключены.
Прирост к прежнему гибриду: **+35,1% dev / +28,4% public**.
Поток 30 запросов/с прошёл без ошибок: public p95 **592 мс** с учётом очереди.
GPU-память процесса: **4936 MiB** (снимок, не пик).
Dev-F1 batch=1: **0,917114**; batch=8: **0,917291** против **0,917410**
у гибрида. Это измерения runtime, а не новые результаты public-лидерборда.
Измерительный драйвер **580.173.02**, не целевой 550.90.07.
[Условия и полный отчёт](reports/serving_s21_20260907.md).
Откат: `configs/serving/hybrid_bf16.json`; CRF в обоих вариантах остаётся CPU TorchScript.

## Разработка и воспроизводимость

```nushell
uv sync --extra train --extra serve --group dev
uv run pytest --no-cov tests/test_serving_contract.py tests/test_posthoc_rules.py
uv run python scripts/train.py --config configs/experiments/e10_xlmr_base_bio.yaml
make mlflow
```

Для изолированного сервиса достаточно `uv sync --extra serve`; TensorRT-экспорт
добавляет `--extra export`. Полные модельные эксперименты выполняются на GPU.
Каждый run сохраняет config, seed, data hashes, исходники, метрики, predictions
и запись MLflow. Веса не дублируются в MLflow.

## 📑 Структура репозитория

```nushell
NER-Uzbek-Hack/
├── apps/                           # Приложения, отдельно от исследований
│   ├── frontend/                   # React / TypeScript: тексты, spans, JSONL
│   └── backend/app/ensemble.py     # FastAPI → финальный ансамбль
├── src/uzner/                      # Общая воспроизводимая логика
│   ├── data/                       # Данные, токенизация, окна и offsets
│   ├── models/                     # Encoder-ы, BIOES / CRF, span-головы
│   ├── training/                   # Train, resume, checkpoint и inference
│   ├── evaluation/                 # Exact-span метрики и срезы ошибок
│   ├── posthoc/                    # Голосование, словарь и повторы
│   ├── serving/                    # Bundle, TensorRT, HTTP-клиент, benchmark
│   └── experiments/                # Артефакты, provenance, MLflow, A100
├── configs/                        # Зафиксированные рецепты и параметры
│   ├── experiments/                # Обучающие эксперименты
│   ├── posthoc/                    # Абляции без переобучения
│   └── serving/default.json        # Финальный runtime s62 + c02
├── scripts/                        # Командные точки входа
│   ├── train.py                    # Обучение и продолжение
│   ├── predict_service.py          # HTTP → predictions.jsonl
│   ├── evaluate.py                 # Оценка по gold-разметке
│   ├── benchmark_service.py        # Скорость, latency, GPU-память
│   └── export_service.py           # Encoder → ONNX → TensorRT
├── tests/                          # Контракты, offsets, модели и регрессии
├── docs/ORGANIZERS.md               # Команды и состав поставки
├── reports/                        # Результаты, абляции и отчёты QA
├── ner_uz_hackathon_participant/    # Неизменённый комплект организаторов
├── runs/                           # Локальные runs и checkpoint-ы, вне Git
├── artifacts/serving/              # Локальные bundle и engines, вне Git
├── output/                         # Локальные предсказания и Docker-архив
├── Dockerfile                      # Автономный GPU-сервис с весами
├── compose.yaml                    # API + веб-интерфейс
├── Makefile                        # Единые команды организаторам
├── pyproject.toml                  # Зависимости и настройки инструментов
└── uv.lock                         # Зафиксированные версии окружения
```

Показаны основные каталоги. Большие модельные артефакты передаются отдельно;
готовый образ уже содержит все ресурсы для автономного инференса.

История отрицательных результатов сохранена: увеличение данных, смена heads
и обучение на train+dev не объявляются улучшениями без измерений.

## Документация

### Финальная система

- [Команды организаторам: predict, eval, benchmark, обучение](docs/ORGANIZERS.md)
- [Serving: запуск, экспорт, совместимость и benchmark](docs/SERVING.md)
- [Актуальная производительность A100](reports/serving_s21_20260907.md)
- [Архив ресурсов и передача организаторам](docs/DELIVERY.md)
- [Словарь, повторы и 71 абляция без переобучения](docs/POSTHOC_DECODING.md)
- [Архитектура](docs/ARCHITECTURE.md), [артефакты](docs/ARTIFACTS.md), [MLflow](docs/EXPERIMENT_TRACKING.md)
- [Frontend и backend](apps/README.md)

### Эксперименты — история, не очередь автозапуска

[Общий журнал и результаты](docs/EXPERIMENTS.md) · [Протокол](docs/EXPERIMENT_PROTOCOL.md)

| Серия | Что менялось |
|---|---|
| [1](docs/FIRST_SERIES.md) | Базовые encoder-ы и корректный offset pipeline |
| [2](docs/SECOND_SERIES.md), [2B](docs/ENCODER_CONTINUATION.md) | Sequence-модели, большие encoder-ы, BGE / XLM-V |
| [3](docs/THIRD_SERIES.md) | Biaffine / GlobalPointer, пороги, короткое продолжение |
| [4](docs/FOURTH_SERIES.md) | Транслитерации, MELM-inspired, silver |
| [5](docs/FIFTH_SERIES.md) | Вспомогательные цели, kNN, символьные границы |
| [6](docs/SIXTH_SERIES.md) | Дополнительные модели и exact-span ансамбль |
| [7](docs/SEVENTH_SERIES.md) | Защищённый эталон и span-reranker |

[s48](docs/S48_SILVER_INTERIM.md) · [train+dev ансамбль](docs/FINAL_ENSEMBLE.md) ·
[Sol review](docs/FOURTH_SOL_REVIEW.md). Рецепты из этих документов не заменяют
подтверждённый s62+c02 автоматически.

### Работа с проектом

- [Удалённые runs, логи, возврат артефактов](docs/REMOTE_EXPERIMENTS.md)
- [Карточка данных](docs/DATA_CARD.md), [анализ ошибок](docs/ERROR_ANALYSIS.md)
- [Условия задачи и FAQ](docs/task/README.md)
- [API](ner_uz_hackathon_participant/API.md) и
  [правила разметки](ner_uz_hackathon_participant/LABELING_GUIDE.md)
