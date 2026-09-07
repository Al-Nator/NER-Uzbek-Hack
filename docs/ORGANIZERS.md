# Команды для организаторов

Все команды ниже — **из корня репозитория**, Linux x86_64. `make help` выводит
краткую памятку. Сервис — замороженный **s62+c02**, три модели, словарь и повторы.
Git содержит код, но **не веса**; для inference нужен отдельно переданный
архив ресурсов `uzner-a100-s21-resources-v1.tar.zst` и его `.sha256`.
Это не Docker-образ: образ собирается из репозитория. [Состав поставки](SERVING.md#bundle).

## 1. Окружение и сервис

Для host-команд: Python 3.10–3.12, `uv`, GNU Make. Для сервера и обучения:
NVIDIA GPU, для готовых plans — **A100**, Docker с NVIDIA Container Toolkit.
Обучение из pretrained впервые скачивает encoder; inference внутри образа офлайн.

```bash
# Положить архив и checksum в корень чистого checkout.
sha256sum -c uzner-a100-s21-resources-v1.tar.zst.sha256
tar --zstd -xf uzner-a100-s21-resources-v1.tar.zst
sha256sum -c artifacts/serving/RELEASE-SHA256SUMS
docker build -t ner-uz-solution:a100-s21 .
make serve
curl -f http://127.0.0.1:8000/healthz
# Окружение для predict/eval/benchmark, не требуется для docker run:
make setup-organizers
```

`make serve` использует образ `ner-uz-solution:a100-s21`; `--wait` ждёт Docker
healthcheck. При отсутствии образа Compose собирает корневой Dockerfile — для
этого нужны bundle/plans и интернет для зависимостей. Веса, словарь и metadata
включены в образ; mounts и переменные окружения для default не нужны.
Логи: `docker compose logs -f backend`. Остановка: `docker compose stop backend`.

На целевом driver **550.90.07** обязательно выполнить этот smoke и официальный
checker: наши A100-замеры проводились на **580.173.02**, runtime CUDA 12.4.

```bash
uv run --no-sync python ner_uz_hackathon_participant/scripts/check_service.py \
  --url http://127.0.0.1:8000
```

## 2. Predict — весь финальный ансамбль

```bash
make predict INPUT=/path/to/public_test_inputs.jsonl \
  OUTPUT=output/public-v1 BATCH=8
```

Результат: **`output/public-v1/predictions.jsonl`**, рядом `manifest.json`
с SHA-256 входа/ответа, числом документов, URL и размером HTTP-пакета.
Вход: по одному `{"hash":"...","text":"..."}` на строку. Если во входе есть
`entities`, они **не отправляются** в сервис. Выход строго:

```json
{"hash":"example-001","entities":[{"label":"NAME","start":0,"end":3}]}
```

Проверяются все строки, уникальность/порядок hash, классы, типы и Unicode offsets.
Предсказания публикуются после обработки **всего** файла; при HTTP-ошибке создаётся
`failure.json`, неполная посылка не выдаётся за готовую. Существующий OUTPUT
не перезаписывается — выберите новый каталог. `text` никогда не нормализуется.

Удалённый сервер: добавьте `URL=http://server:8000`. Через SSH tunnel это тоже
работает; CPU-клиенту не нужна доступная CUDA. Это не старый
`predict_ensemble.py`: тот считает raw majority без финальной c02-постобработки.

## 3. Eval — strict exact-span

```bash
make predict INPUT=ner_uz_hackathon_participant/data/dev.jsonl \
  OUTPUT=output/dev-v1
make eval GOLD=ner_uz_hackathon_participant/data/dev.jsonl \
  PREDICTIONS=output/dev-v1/predictions.jsonl OUTPUT=output/dev-v1/metrics.json
```

Нужны gold `hash/text/entities` и полный набор predictions с теми же hash.
Оценка работает на CPU, без сервера и без загрузки весов. Совпадение — одновременно
`hash/label/start/end`; выводятся micro/macro и ORG/NAME/GEO, TP/FP/FN.
Пустые `entities: []` допустимы; **отсутствующие** gold entities — ошибка,
поэтому public inputs нельзя случайно оценить как пустой gold.
Готовый metrics.json не перезаписывается.

## 4. Benchmark — полный HTTP, не только encoder

Запускать на **том же хосте, где работает GPU-сервис**, без параллельного обучения.
Перед измерением дождаться readiness. Каждая строка dev обрабатывается заново;
reference/prediction-cache для ответа не используются.

```bash
make benchmark INPUT=ner_uz_hackathon_participant/data/dev.jsonl \
  OUTPUT=output/bench-dev-b8 BATCH=8 ROUNDS=2 LABELED=1 MEMORY=required \
  REFERENCE=output/dev-v1/predictions.jsonl
make benchmark INPUT=ner_uz_hackathon_participant/data/dev.jsonl \
  OUTPUT=output/bench-dev-b1 BATCH=1 ROUNDS=2 LABELED=1 MEMORY=required
```

В новом каталоге: `summary.json`, все `latencies.json`, `predictions.jsonl`,
parity повторов; при REFERENCE — сравнение всех exact spans с этим файлом.
В summary: messages/s, requests/s, latency mean/p50/p95/p99/max, NVML-память,
GPU/driver/CPU клиентского хоста, hashes входа/config и качество при LABELED=1.
Три warmup-запроса и загрузка весов **не входят** в измерение.

**BATCH — документы в HTTP-запросе**, не внутренний batch окон (он равен 16).
Latency относится к запросу. NVML опрашивается каждые 100 мс; device-used включает
TensorRT и CUDA context, а также чужие процессы, если они есть. Это наблюдаемый пик.
Config в отчёте — ожидаемый профиль клиента, не handshake с моделью сервера.

Поток **30 одиночных запросов/с**, включая очередь в latency:

```bash
make benchmark INPUT=ner_uz_hackathon_participant/data/dev.jsonl \
  OUTPUT=output/bench-dev-30rps BATCH=1 ROUNDS=1 CONCURRENCY=128 RATE=30 \
  LABELED=1 MEMORY=required
```

Для public файла уберите `LABELED=1`: скрытый F1 скрипт не знает. Для замера
через удалённый URL допустимо `MEMORY=off` или `auto`, но тогда это latency
с сетью, а оборудование/VRAM клиента **не являются** оборудованием сервера.
Отказ обязательного NVML, неправильный ответ или неповторяемые spans дают
ненулевой exit code; отсутствие summary не означает успешный benchmark.

## 5. Обучение и resume

```bash
# Быстрая проверка полного контура: 8 train, 4 dev, 1 эпоха; нужен GPU.
make train-smoke OUTPUT=artifacts/smoke/organizers-v1

# Полноценный одиночный BGE-M3-RetroMAE + GlobalPointer на original train.
UZNER_MLFLOW_EXPERIMENT=organizers make train RUN_SUFFIX=organizers-v1

# Продолжение того же прерванного run с optimizer/scheduler/RNG.
UZNER_MLFLOW_EXPERIMENT=organizers make train-resume RUN_SUFFIX=organizers-v1
```

Default CONFIG — `configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml`:
до 5 эпох, BF16, исходные train/dev, отбор best по dev exact micro-F1.
На A100 сначала проверьте свободные GPU-память и диск: сохраняются best и last
вместе с состоянием optimizer; это **не** компактный inference-bundle.
GPU smoke с полным encoder тоже требует памяти под эти checkpoint-ы.

```bash
make train CONFIG=configs/experiments/a100/s31_xlmr_large_global_pointer.yaml \
  RUN_SUFFIX=organizers-v1
```

Другая модель задаётся CONFIG; параметры обучения/данные фиксируются YAML,
не скрыты в Makefile. Полный run пишет `runs/<run_id>_<suffix>/`, resolved config,
data hashes, версии, seed, метрики, predictions, best/last и MLflow.
По умолчанию MLflow использует локальную SQLite `mlruns/mlflow.db`; сервер заранее
поднимать не нужно. `make mlflow` открывает UI на 5000. `MLFLOW_TRACKING_URI`
позволяет явно выбрать другой backend.

Smoke-output изолирован, **не публикуется в MLflow и сводную таблицу**.
Повторный train с тем же именем запрещён; resume явно включается отдельной командой.
При появлении `.backup` после аварии сохранение остановится с просьбой проверить
резервную копию, вместо её удаления.

**Обучение одного CONFIG не пересобирает замороженный ансамбль.** Default s32 —
база для последующего s33; s21 и s31 обучаются отдельно. s33 требует исходный
checkpoint s32 по `training.initial_checkpoint` в его YAML. Для воспроизведения
сданной системы используйте переданный образ; веса нового обучения автоматически
не подменяют эталон. [История компонентов](SIXTH_SERIES.md).

## Проверки и диагностика

`make test-organizers` — локальные контрактные и интеграционные проверки без
настоящего крупного обучения. Дополнительно на GPU:

```bash
uv run --no-sync pytest --no-cov tests/test_organizer_gpu_smoke.py
```

Это маленький локально создаваемый BERT: проверяет CUDA/BF16, три типа голов,
checkpoint и resume без скачивания pretrained. Он не измеряет качество ансамбля.
Для полного набора исследовательских модулей есть `make setup-research`
(добавляет pyahocorasick, rapidfuzz и UzTransliterator), затем `make test`.
У `make test` общий coverage gate **90%**, он не подменяется выборкой serving-тестов.
[Отчёт о проверке поставки и оставшихся ограничениях](../reports/organizer_qa_20260907.md).

При ошибке: сначала stderr команды и `docker compose logs backend`; для обучения —
`runs/<run_id>/logs/`, status и metadata. Не удаляйте checkpoints/backup и не
переобучайте модель ради устранения ошибки HTTP-клиента.
