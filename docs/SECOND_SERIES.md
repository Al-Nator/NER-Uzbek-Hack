# Вторая серия

Манифест запуска: `configs/series/second.yaml`. Вторая серия полностью посвящена
sequence-моделям. Span-based архитектуры в неё не входят и будут оформлены
отдельной следующей серией после выбора sequence-финалистов.

## Цель

1. Перенести два лучших варианта decoding первой серии на её лучший encoder —
   mDeBERTa-v3-base.
2. Проверить масштабирование XLM-R-base до XLM-R-large при тех же BIOES и
   decoder-ах.
3. Получить сильные и воспроизводимые sequence-кандидаты для последующего
   сравнения со span-based NER и ансамблирования.

## Состав и порядок

| Run ID | Encoder | Head / decoder | Назначение |
|---|---|---|---|
| `s20_mdeberta_v3_base_bioes_constrained` | mDeBERTa-v3-base | softmax / constrained | быстрый перенос BIOES + Viterbi |
| `s21_mdeberta_v3_base_bioes_crf` | mDeBERTa-v3-base | CRF / CRF | перенос абсолютного победителя decoding |
| `s22_xlmr_large_bioes_constrained` | XLM-R-large | softmax / constrained | эффект масштаба encoder-а |
| `s23_xlmr_large_bioes_crf` | XLM-R-large | CRF / CRF | quality-кандидат, самый дорогой run |

Все run-ы используют official train/dev, context `512`, stride `128`, seed `42`,
пять эпох, learning rate `2e-5`, BF16 и effective batch `8`. Для XLM-R-large
используются per-device batch `1`, gradient accumulation `8` и gradient
checkpointing, чтобы обучение помещалось в 16 ГБ VRAM.

## Предварительная оценка времени

Оценка относится к NVIDIA GeForce RTX 4070 Ti SUPER 16 ГБ и включает пять эпох
с полной dev-оценкой. Она рассчитана по одноэпоховым GPU smoke-run на 512 train
и 128 dev документах, числу полных окон и времени первой серии.

| Run | Ожидаемое время |
|---|---:|
| `s20` | 35–45 минут |
| `s21` | 60–80 минут |
| `s22` | 1.7–2.2 часа |
| `s23` | 2.2–2.8 часа |
| **Вся серия** | **5.5–7 часов** |

CRF первой серии был лишь на `0.0021` micro-F1 выше BIOES constrained, но занял
`60.1` вместо `19.1` минуты. Поэтому `s21` и `s23` запускаются после соответствующего
constrained-run; результаты не считаются заранее гарантированными.

Все четыре конфигурации прошли BF16 forward и одноэпоховый train/eval/checkpoint
smoke на целевой GPU. mDeBERTa использовала до `6.99 GiB`, XLM-R-large — до
`10.43 GiB`; OOM не возник. Полный alignment также проверен: mDeBERTa создаёт
`14 858 / 1 689`, XLM-R-large — `14 840 / 1 682` train/dev окон.

На четыре канонических run-каталога нужно ориентировочно `38–45 ГБ`: каждый
содержит собственные `best` и `last`, а checkpoint-ы не копируются в MLflow.

## Проверка до обучения

```bash
uv run python scripts/run_series.py \
  --series configs/series/second.yaml \
  --stage all \
  --dry-run

uv run python scripts/preflight_models.py \
  --series configs/series/second.yaml \
  --stage all

uv run python scripts/preflight_alignment.py \
  --series configs/series/second.yaml \
  --stage all
```

Preflight и smoke-run-ы не создают MLflow run и не записываются в `runs/`.
Для короткой проверки одного train/eval/checkpoint-прохода используется
`--max-epochs 1` вместе с `--smoke-output`.

## Полный запуск

```bash
uv run python scripts/run_series.py \
  --series configs/series/second.yaml \
  --stage all \
  --run-suffix s2-sequence-v1
```

При необходимости этапы запускаются независимо:

```bash
uv run python scripts/run_series.py \
  --series configs/series/second.yaml \
  --stage mdeberta \
  --run-suffix s2-sequence-v1

uv run python scripts/run_series.py \
  --series configs/series/second.yaml \
  --stage xlmr-large \
  --run-suffix s2-sequence-v1
```
