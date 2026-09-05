# Серия 2B: продолжение сравнения encoder-ов

**Основной запуск перенесён на A100:** [общая очередь s24 → s25 → s30 → s31](A100_QUEUE.md).
Для неё используются конфиги `configs/experiments/a100/`, batch 8, обычный AdamW
и отключённый checkpointing. Локальные команды и ограничения ниже — резервный
вариант, а не рекомендуемый текущий запуск.

## Оглавление

- [Состав](#состав)
- [Результаты A100](#результаты-a100)
- [Протокол](#протокол)
- [Локальный запуск](#локальный-запуск)
- [Проверки и стоимость](#проверки-и-стоимость)

## Состав

Манифест: [`second_b.yaml`](../configs/series/second_b.yaml).
Контроль — завершённый `s22_xlmr_large_bioes_constrained_s2-sequence-a100-v2`,
exact micro-F1 **0.9010106245**. `s23` с CRF получил **0.8982668642**.

| Run | Encoder | Гипотеза |
|---|---|---|
| `s24_bge_m3_retromae_bioes_constrained` | `BAAI/bge-m3-retromae` | продолженное RetroMAE pretraining |
| `s25_xlm_v_base_bioes_constrained` | `facebook/xlm-v-base` | расширенный multilingual vocabulary |

Обе ревизии закреплены SHA в конфиге. Репозитории публикуют `pytorch_model.bin`,
поэтому `use_safetensors: false` задан явно. Загружается только корневой encoder;
реконструкционный decoder из каталога BGE `bge-m3-with-deocder/` не используется.
Jina в серию не входит.

## Протокол

Official train/dev, BIOES + softmax + constrained Viterbi, context 512, stride 128,
seed 42, до пяти эпох, LR `2e-5`, warmup 10%, weight decay 0.01,
gradient clipping 1.0. Best выбирается по exact micro-F1; patience 2.
Данные и правила decoding совпадают с s22. Поддержка BGE контекста 8192 здесь
не используется: расширение контекста требует самостоятельной абляции.

Локальная цель — RTX 4070 Ti SUPER 16 GB: BF16 autocast, FP32 параметры,
physical batch 1, gradient accumulation 8, effective batch 8,
gradient checkpointing. Разница GPU/microbatch относительно A100 документируется;
побитовая идентичность обучения не предполагается.

**Исключение для XLM-V:** `training.optimizer: adamw_8bit`, bitsandbytes 0.45.5.
Обычный и fused AdamW не помещаются при accumulation на локальных 16 GB:
временный буфер градиента большой embedding-матрицы вызывает OOM.
Квантуются состояния optimizer-а, не веса encoder-а; все параметры обучаются.
Это memory-adapted сравнение, а не чистая абляция encoder-а относительно s22.
BGE использует обычный AdamW. Метод:
[документация bitsandbytes](https://huggingface.co/docs/bitsandbytes/optimizers).

## Локальный запуск

```bash
uv run python scripts/run_series.py \
  --series configs/series/second_b.yaml \
  --stage all \
  --run-suffix s2b-encoders-v1
```

Этапы `--stage bge` и `--stage xlm-v` можно запускать отдельно. Для пропуска
уже завершённого run добавить `--skip-complete`; частичный run продолжается
через `scripts/train.py --resume` с тем же config и suffix.

Логи: `runs/<run_id>_s2b-encoders-v1/logs/console.log`, `events.jsonl`, `history.csv`.
MLflow: основной локальный experiment `uzner-first-series`; фильтр run name
`s2b-encoders-v1`. Наследуются все step/epoch/system и slice-метрики.
Checkpoint-ы сохраняются только в `runs/`, без второй копии в MLflow.

## Проверки и стоимость

```bash
uv run python scripts/run_series.py --series configs/series/second_b.yaml --stage all --dry-run
uv run python scripts/preflight_models.py --series configs/series/second_b.yaml --stage all
uv run python scripts/preflight_alignment.py --series configs/series/second_b.yaml --stage all
```

Проверено на RTX 4070 Ti SUPER: pinned encoder forward, весь official train/dev
alignment, два optimizer step на полном окне 512 с accumulation 8,
изолированный train/eval/save/load; для XLM-V также продолжение обучения после resume.

| Run | Окна train/dev | Два step на 512 | Peak allocated |
|---|---:|---:|---:|
| s24 | 14 840 / 1 682 | 1,53 с | 10,58 GiB |
| s25 | 14 619 / 1 661 | 1,15 с | 9,87 GiB |

Грубый ориентир на полные пять эпох: **s24 2–3 ч, s25 1,5–2,5 ч**,
вместе 3,5–5,5 ч. Это экстраполяция короткой проверки, не измеренный полный run:
длины документов, evaluation, сохранение весов и фоновые процессы меняют время.
Модели уже скачаны локально. Полные s24/s25 завершены на A100; оценки выше
относятся к резервному локальному исполнению, не фактическому времени A100.
Smoke-артефакты изолированы в `/tmp`, не входят в MLflow и таблицу результатов.

## Результаты A100

Suffix `a100-continuation-v1`, official dev, best по exact micro-F1:

| Run | Best epoch | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|---:|
| s24 BGE-M3-RetroMAE | 4 | 0.89439 | 0.90985 | **0.90205** |
| s25 XLM-V-base | 5 | 0.86781 | 0.88348 | 0.87557 |

BGE выше s22 лишь на 0.10 п.п.; это не доказательство статистически значимого
преимущества. Его перенос на GlobalPointer и low-LR продолжение оформлены
отдельными s32/s33 в [серии 3](THIRD_SERIES.md).
Повторный capacity-test: `scripts/preflight_training.py --config <experiment.yaml>`.
