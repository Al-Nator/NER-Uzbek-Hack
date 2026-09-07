# Финальная поставка: A100, все три encoder-а TensorRT BF16

Передать организаторам:

1. Репозиторий `https://github.com/Al-Nator/NER-Uzbek-Hack` с зафиксированным commit.
2. `uzner-a100-s21-resources-v1.tar.zst` — около 6,2 GiB, отдельно от Git.
3. `uzner-a100-s21-resources-v1.tar.zst.sha256`.

SHA-256 архива:

```text
e522129b0dd16d4ff9b15d6cb62d2656614d5f531385ed65f6dca1286c44a252
```

Архив содержит три исходных checkpoint-а (s33/s21/s31), tokenizer-ы, train-only
словарь, bundle manifest, три TensorRT plan и metadata. Optimizer и обучающие
состояния не входят. `reference/` — только архивные предсказания для offline parity;
при инференсе они не используются. Внешних LLM/API и поиска ответов по hash нет.

Распаковать в **чистый checkout** и собрать:

```bash
sha256sum -c uzner-a100-s21-resources-v1.tar.zst.sha256
tar --zstd -xf uzner-a100-s21-resources-v1.tar.zst
sha256sum -c artifacts/serving/RELEASE-SHA256SUMS
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```

Во время сборки нужен интернет для зависимостей. После сборки сеть, mounts,
секреты и обязательные переменные окружения не нужны. Готовность: `GET /healthz`.
Команды predict/eval/benchmark/train: [ORGANIZERS.md](ORGANIZERS.md).

Целевая GPU — A100, runtime CUDA 12.4 / TensorRT 10.3. На A100 driver 580.173.02
проверены автономный образ, HTTP-контракт и производительность. Driver 550.90.07
на стенде недоступен: нужен короткий smoke на машине организаторов.
Полная повторная сборка корневого Dockerfile локально заблокирована правами
Docker; проверенный A100-образ получен добавлением s21 plan и профиля к прежнему
автономному образу. Это ограничение проверки, а не требование к запуску.

Прежний `ner-uz-solution-a100-final.tar.zst` — **резервный гибрид**, не эта поставка.
Откат в новом образе: `UZNER_SERVICE_CONFIG=configs/serving/hybrid_bf16.json`.
Веса и словарь не менялись; BF16 не даёт побитной эквивалентности.
[Качество, скорость и память](../reports/serving_s21_20260907.md).
