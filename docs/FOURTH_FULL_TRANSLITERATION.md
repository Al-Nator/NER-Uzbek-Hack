# Серия 4: полные транслитерации s43/s44

## Оглавление

- [Новые данные](#новые-данные)
- [Честность сравнения](#честность-сравнения)
- [Воспроизведение и очередь](#воспроизведение-и-очередь)

## Новые данные

Источник: полный релиз
`artifacts/transliteration/full_train_20260905_v3/comparison_batched_v1_finished_1788641202139/derived/`.
Использованы **объединённые** `corrected_converter.jsonl` и `luna_high.jsonl`,
а не старый частичный release s40/s41.

| Ветка | Входных строк | Review | Неизменённых после review | Дублей | Добавлено | Всего train |
|---|---:|---:|---:|---:|---:|---:|
| s43, конвертер | 26000 | 0 | 10115 | 13 | **15872** | **28872** |
| s44, Luna high | 25354 | 1464 | 14050 | 13 | **9827** | **22827** |

Конвертер: 10724 Latin→Cyrillic и 5148 Cyrillic→Latin; Luna: 6780 и 3047.
Смешанные тексты могут изменяться в обоих направлениях, поэтому у конвертера
копий больше исходных 13000 документов. Это не ошибка подсчёта.
Исходный dev 1500 документов неизменён. Языковое качество **не утверждено**:
структурная валидность offsets не доказывает правильность транслитерации.

Отбор: train membership/source hash, исходная разметка, непрерывное alignment,
точные Unicode-границы и surfaces; только status=ok и text != source_text;
нормализованные дубли train/dev/копий исключены. Gold dev не используется для отбора.
Оригиналы остаются как есть, копии сохраняют source_hash, direction, method и input_hash.

## Честность сравнения

s43/s44 — **независимые полные наборы**, не matched-сравнение двух методов:
метод и объём здесь меняются вместе. Контроль чистого метода остаётся s40/s41,
где у каждой ветки были одинаковые 4803 пары. Не смешиваем Luna и конвертер.

Оба опыта повторяют pretrained BGE-M3-RetroMAE + GP из s32: 512/128, batch 8,
AdamW 2e-5, BF16, максимум пять эпох, без gradient checkpointing.
Никакого head-LR, multihead или warm-start в этих конфигурациях нет.
Пять эпох расширенного train означают больше optimizer steps; это не сравнение
при одинаковом compute budget. Причинный вывод «только данные улучшили качество»
потребует отдельного step-matched контроля, который сейчас не запущен.

## Воспроизведение и очередь

```bash
uv run python scripts/prepare_transliteration_series.py --full-combined \
  --derived artifacts/transliteration/full_train_20260905_v3/comparison_batched_v1_finished_1788641202139/derived \
  --output artifacts/series4/transliteration_full_v2
```

Существующий выходной каталог не перезаписывается. Готовый manifest содержит
SHA-256 входов/выходов, причины исключений, размеры, направления и число групп.
Release уже передан на A100; hashes выходных JSONL сверены с локальными.

Конфиги `configs/data/s4_converter_full_v2.yaml`, `s4_luna_full_v2.yaml`;
опыты `configs/experiments/a100/s43_bge_gp_translit_converter_full.yaml`,
`s44_bge_gp_translit_luna_full.yaml`. Suffix `full-targeted-v1`.
Оба завершены и возвращены с A100: s43 **0.905058** (best epoch 2),
s44 **0.907456** (best epoch 5). Оба имеют локальные MLflow-связи.
Luna-full дала небольшой номинальный прирост к s32, конвертер-full — нет.
Следующая абляция [silver s45/s46](FOURTH_SILVER.md) показала:
silver отдельно лучше (0.909576), чем silver + Luna (0.905626).
Порядок и автоматический возврат: [новая очередь A100](SIXTH_SERIES.md#очередь-и-a100).
