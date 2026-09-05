# Общая очередь на A100: 2B → серия 3

Историческая инструкция запуска: s24/s25/s31 завершены, s30 остановлен
пользователем. Затем отдельно выполнены s32/s33 и подбор порогов.
Актуальные результаты: [серия 2B](ENCODER_CONTINUATION.md), [серия 3](THIRD_SERIES.md).
Команды ниже не следует повторять под уже существующими run ID.

## Оглавление

- [Распределение](#распределение)
- [Запуск](#запуск)
- [Логи и возврат](#логи-и-возврат)
- [Проверки и время](#проверки-и-время)

## Распределение

Основной вариант — все четыре обучения последовательно на alnator:
**s24 → s25 → s30 → s31**. Локально — анализ данных, тесты, основной MLflow и
приём артефактов. Локальные конфиги 2B остаются только резервным вариантом.

Манифест: [`continuation_a100.yaml`](../configs/series/continuation_a100.yaml).
Этап `encoders` выбирает s24/s25, `spans` — s30/s31, `all` — всю очередь.
Это очередь вычислений, а не объединение научных серий: encoder-абляции относятся
к 2B, span-head — к серии 3. Данные остаются official train/dev.

Все четыре: train/eval batch 8, accumulation 1, BF16 autocast, FP32 weights,
обычный AdamW, **без gradient checkpointing**. Effective batch 8, LR 2e-5,
seed 42, context 512/stride 128 и бюджет до пяти эпох не меняются.
У A100-конфига XLM-V нет отличия optimizer-а от s22; локальный AdamW8bit
не используется. Результаты разных конфигов нельзя объединять под одним run ID.

## Запуск

Команда выполняется **локально из корня проекта**:

```bash
uv run python scripts/remote_experiments.py --remote-experiment uzner-continuation-a100 start \
  --series configs/series/continuation_a100.yaml --stage all \
  --run-suffix a100-continuation-v1 --session uzner-s24-s31
```

Запуск идёт в detached zellij на `/home/danya/NER-Uzbek-Hack`.
Закрытие SSH не останавливает обучение. Одновременно отдельные команды старой
2B/серии 3 не запускать. CLI откажется перезаписывать существующий run/session.

## Логи и возврат

```bash
uv run python scripts/remote_experiments.py attach --session uzner-s24-s31
uv run python scripts/remote_experiments.py tunnel
# remote MLflow: http://127.0.0.1:5001, experiment uzner-continuation-a100
```

В отдельном локальном терминале оставить возврат после завершения run-ов:

```bash
uv run python scripts/remote_experiments.py watch \
  --series configs/series/continuation_a100.yaml --stage all \
  --run-suffix a100-continuation-v1
```

`watch` не нужен для обучения, но нужен для автоматического возврата.
При выключенном локальном компьютере результаты остаются на alnator.
Позднее вызвать `sync` с теми же параметрами; `sync --metrics-only` позволяет
получить метрики раньше весов. Основной локальный MLflow остаётся на порту 5000,
experiment `uzner-first-series`. Фильтровать новые runs по `a100-continuation-v1`.
Checkpoint-ы не дублируются в MLflow; полный rsync проверяет SHA-256 manifest.

## Проверки и время

BF16 backward/обычный AdamW проверены на полном batch 8×512:

| Run | Два optimizer step | Peak allocated |
|---|---:|---:|
| s24 BGE | 0,67 с | 10,97 GiB |
| s25 XLM-V | 0,56 с | 14,55 GiB |

Изолированный smoke на 128 train/32 dev документах: s24 train 2,3–2,5 с,
eval 0,7–0,8 с; s25 train 1,9 с, eval 0,6–0,7 с. Обе модели также прошли
загрузку last checkpoint и продолжение обучения с обычным AdamW.
Это проверка работоспособности,
не эксперимент по качеству. Метрики smoke не публикуются в MLflow и CSV.
s30/s31 ранее прошли A100 train/eval/resume; их настройки не меняются.
Конфиги и порядок очереди покрыты тестами; вся suite: 134 passed, coverage 92,66%.

Ориентир на всю очередь — **2–3,5 часа**, без rsync тяжёлых checkpoint-ов.
Это экстраполяция коротких проверок, не измеренный полный run. Модели уже
скачаны на A100; длины текстов, evaluation и дисковая нагрузка меняют время.
