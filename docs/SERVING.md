# Serving: фиксированный ансамбль на A100

Финальный рецепт — **s62 + train-only c02**: s33 BGE-M3-RetroMAE/GP,
s21 mDeBERTa-v3-base/BIOES-CRF, s31 XLM-R-large/GP. Большинство 2/3,
нормализованный train-словарь, затем точные повторы. Никакого переобучения,
включения dev в словарь, смены порогов или обращения к LLM при оптимизации.

[Производительность и quality gate](../reports/serving_a100_20260907.md).
[Отдельная памятка команд организаторам](ORGANIZERS.md).

Default: **s33 и s31 в TensorRT BF16, s21 в PyTorch BF16**, CPU TorchScript CRF,
до 16 окон за forward. HTTP-пакет и batch окон — разные величины.
Резерв без TensorRT: `configs/serving/torch_bf16.json`; его можно выбрать
необязательной переменной `UZNER_SERVICE_CONFIG` при запуске того же образа.
Небольшие численные изменения TensorRT приняты пользователем только ради
существенного измеренного ускорения; побитная эквивалентность не заявляется.

## Окружение и совместимость

- Цель: NVIDIA A100 80 GB, Linux x86_64, driver **550.90.07**.
- PyTorch **2.6.0+cu124**, CUDA runtime **12.4**, Transformers **5.14.1**.
- TensorRT **10.3.0**: версии Python bindings и native libs закреплены вместе.
- Docker: Python 3.11, один Uvicorn worker, офлайн-модели, непривилегированный UID.

У измерительного A100-хоста другой драйвер — **580.173.02**. Драйвер на нём
не менялся. Совместимость с целевым 550.90.07 основана на CUDA 12.4 и матрице
NVIDIA, **не на фактическом тесте с этим драйвером**. Перед финальной сдачей
обязателен readiness и короткий контрактный прогон на целевой машине.

NVIDIA указывает драйвер 550.54.14 для CUDA 12.4 GA;
[CUDA release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html).
TensorRT 10.3 поддерживает CUDA 12.4 update 1;
[TensorRT release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html).
Надпись CUDA 13.0 в `nvidia-smi` измерительного хоста — возможность драйвера,
а не версия runtime внутри нашего образа.

<a id="bundle"></a>

## Автономный bundle

Из корня, если исходные checkpoint-ы находятся в `runs/`:

```bash
uv sync --extra train --extra serve
uv run python scripts/prepare_service_bundle.py
```

Создаётся `artifacts/serving/s62-c02-v1/`: три checkpoint-а без optimizer,
tokenizer-ы, конфиги, словарь и SHA-256 manifest. Повторная запись запрещена.
В текущем рабочем каталоге bundle уже подготовлен. Это **не содержимое Git**:
на новую машину перенесите весь каталог или готовый Docker-образ.
Hard links локальных весов нельзя редактировать на месте.

Для default TensorRT также нужны `s33.plan`, `s31.plan` и их JSON metadata в
`artifacts/serving/engines-bf16/`; они уже перенесены локально с A100 и проверены
по SHA-256. При подготовке с нуля после bundle выполните
[экспорт на A100](#воспроизведение-tensorrt), затем `docker build`.
Одна копия Git без этих ресурсов не является готовой модельной посылкой.

`reference/dev.jsonl` и `reference/public.jsonl` нужны только offline-проверкам.
При обработке HTTP ни reference predictions, ни входной hash не используются
для выбора сущностей. В runtime проверяются hashes ресурсов перед загрузкой.

## Docker и интерфейс

Из корня репозитория:

```bash
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```

Сборке нужен интернет для зависимостей, **работе готовой модели — нет**.
Ни mounts, ни env-переменные для стандартного запуска не требуются.
`GET /healthz` становится доступен после загрузки модели. Один HTTP worker
сериализует использование GPU; несколько workers загрузят несколько копий весов.

```bash
curl -f http://localhost:8000/healthz
curl http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  --data '[{"hash":"a","text":"🙂 Ali Toshkentda."},{"hash":"empty","text":""}]'
docker compose up --build
```

Compose поднимает FastAPI на 8000 и frontend на 3000. Nginx проксирует API
к backend по внутренней сети. Frontend не требует CORS при этом запуске.
Swagger UI может запрашивать CDN; сам inference и OpenAPI JSON автономны.

`apps/backend/Dockerfile` предназначен только для тестирования HTTP-каркаса
без модели. Для сдачи используется **корневой Dockerfile**, `app.ensemble:app`.

## Оптимизации без смены рецепта

| Механизм | Что именно меняется |
|---|---|
| Резидентный ансамбль | Все три модели загружаются один раз, не на каждый запрос |
| BF16 autocast | Encoder/головы используют BF16 там, где поддерживается; master weights FP32 |
| CPU CRF | Маленькая рекурсия Viterbi выполняется без сотен мелких CUDA launches |
| TorchScript CRF | Компилируется тот же общий Viterbi-цикл, без второй реализации правил |
| TensorRT BF16 | Экспорт только выбранных encoder-ов; исходные heads и decoder остаются общими |
| GPU binding | Прямые адреса PyTorch-тензоров и отдельный CUDA stream, без CPU-копии hidden states |

«Zero-copy» относится к границе **TensorRT encoder → PyTorch head**, не ко всему
сервису: токенизация на CPU и перенос logits для общего decoder остаются.
Размер окон и stride не урезаются. Нет response cache, отсечения длинных текстов
или исключения словаря/повторов из тайминга.

INT8/FP8 не включены: без калибровки и полного quality gate обещать отсутствие
потерь нельзя. BF16 flag TensorRT не означает, что каждый слой исполняется в BF16.

## Воспроизведение TensorRT

На A100, отдельно от измерения производительности:

```bash
uv sync --extra serve --extra export
uv run python scripts/export_service.py --models s33 s31 \
  --precision bf16 --output artifacts/serving/engines-bf16
```

ONNX opset 17; profile: min `[1,2]`, opt `[8,128]`, max `[16,512]`;
workspace 4 GiB, TF32 выключен. Plans и metadata сохраняются отдельно от весов.
Runtime требует ровно перечисленные `engine_models`, сверяет SHA-256 и TensorRT
version. Повторная сборка поверх готового plan запрещена.
Plans платформозависимы: при смене GPU/версии TensorRT пересоберите и перепроверьте.

Измеренный финальный образ: `ner-uz-solution:a100-final`; его image ID и hashes
ресурсов зафиксированы в [отчёте](../reports/serving_a100_20260907.md).

## Перенос готового образа

Архив образа передаётся отдельно от Git. Загрузка на целевой **A100**:

```bash
sha256sum -c output/ner-uz-solution-a100-final.tar.zst.sha256
zstd -dc output/ner-uz-solution-a100-final.tar.zst | docker load
docker run --rm --gpus all -p 8000:8000 ner-uz-solution:a100-final
```

Команды выполняются из корня проекта. SHA-256 архива и его 20 OCI blobs уже
проверены после переноса локально. После загрузки дождитесь `/healthz` и выполните
официальный `check_service.py`. OCI index/config digests сохранены в отчёте;
формат image ID у разных Docker image stores может различаться.
На другом семействе GPU готовые A100 plans
не считаются переносимыми: нужна новая сборка plans и quality/performance gate.

Тестовые контейнеры оставлены на `alnator`: API на loopback **18000**, frontend
на **13000**. Доступ с рабочей машины, без публичного открытия этих портов:

```bash
ssh -N -L 8000:127.0.0.1:18000 -L 3000:127.0.0.1:13000 alnator
```

После открытия туннеля UI доступен на `http://localhost:3000`, API — на 8000.
Если эти локальные порты заняты, поменяйте только левый номер в `-L`.

## Автономная контрактная проверка

```bash
uv run python scripts/check_service_container.py --image ner-uz-solution \
  --output artifacts/serving/offline-contract
uv run python ner_uz_hackathon_participant/scripts/check_service.py \
  --url http://localhost:8000
```

Первый скрипт создаёт временный контейнер с `--network none`, read-only rootfs
и временным `/tmp`, без внешних данных и env. Он проверяет non-root UID,
readiness, пустые строки, Unicode, длинный документ и отклонение duplicate hash.
Существующий контейнер с тем же именем не перезаписывается.

## Как измерять

Сервис должен быть уже запущен на том же A100. Пример для исходного dev:

```bash
uv run --extra serve python scripts/benchmark_service.py \
  --url http://127.0.0.1:8000 --config configs/serving/default.json \
  --input ner_uz_hackathon_participant/data/dev.jsonl --gold \
  --reference artifacts/serving/benchmarks/cpu-crf-dev/predictions.jsonl \
  --output artifacts/serving/benchmarks/my-dev-b8 \
  --batch 8 --concurrency 1 --rounds 2
```

Пакетный и одиночный режимы проверяются раздельно (`--batch 8` / `--batch 1`).
Concurrent closed-loop тест задаётся `--concurrency 4`; это не тест устойчивости
под заданным open-loop входным потоком. Три warmup-запроса и загрузка весов
исключены. Учитываются HTTP/JSON, валидация, все три модели, декодирование и правила.
Процентили считаются по **запросам**; messages/s учитывает документы внутри пакета.

NVML опрашивается каждые 100 мс: измеряется полный device-used VRAM, включая
контекст CUDA, TensorRT и allocator cache. Это наблюдаемый пик, не доказанный
максимум для любого допустимого входа. GPU должна быть свободна от чужих процессов.
PyTorch `max_memory_allocated` из внутренних профилей не включает память TensorRT.
Для обязательного замера задайте `--gpu-memory required`. Режим `auto` допускает
клиент без NVIDIA, но записывает отсутствие замера явно; `off` отключает его.
Оборудование в отчёте относится к HTTP-клиенту, а не определяется по URL сервера.

Для отдельного open-loop теста используйте `--batch 1 --concurrency 128
--arrival-rate 30 --rounds 1`. Запросы планируются через 1/30 секунды;
latency включает ожидание от запланированного прихода, а dispatch lag клиента
сохраняется отдельно. В elapsed входит дренирование очереди после последнего
отправленного запроса. Это измерение заданной нагрузки, а не умножение batch-RPS.

Все документы, hash, labels, offsets и повторные ответы проверяются. Public
прогон не имеет gold и не выдаёт новый public F1. Без заданного `--reference`
скрипт проверяет контракт и повторяемость, но не эквивалентность прежнему сервису.

```bash
uv run --extra train python scripts/publish_service_benchmarks.py
```

Полные измерения публикуются в `uzner-serving-a100` локального MLflow; pilot
пропускаются, weights/engines туда не копируются. Повторная публикация идемпотентна.
