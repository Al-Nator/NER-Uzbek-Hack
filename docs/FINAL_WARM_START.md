# Короткое продолжение исходного s33 на train+dev

## f15 — запущен локально 7 сентября 2026

Run: `f15_s33_train_dev_warm_epoch1`. Исходный checkpoint:
`runs/s33_bge_global_pointer_low_lr_a100-low-lr-v1/checkpoints/best`.
SHA-256 encoder, head, tokenizer, config и trainer state сверены с
`artifacts/releases/s62_public_08861_v1/release.json` перед запуском.
Это исходный голос победившего s62, не повторно обученный f11 из f14.

- 13 000 original train + 1 500 бывшего dev; всего 16 522 окна.
- Одна фиксированная эпоха, LR `1e-6 → 0`, новый AdamW/scheduler, без warmup.
- BGE-M3-RetroMAE + GlobalPointer, BF16, batch 8, 512/128, без checkpointing.
- Validation и early stopping отсутствуют. `best` — alias последней эпохи,
  а не выбранный по F1 checkpoint. Независимой dev-метрики у f15 нет.
- Остальные голоса s21/s31, пороги, словарь и повторы не меняются.
- Исходные checkpoint-ы не перезаписываются. Новое обучение Astra не запускалось.

Конфиг: `configs/final/f15_s33_train_dev_warm_epoch1.yaml`.
CLI требует явного `--allow-research-initial`; без флага сохраняется прежний
запрет на research warm-start. Проверяются завершённость источника,
хэши/split-ы данных и совместимость encoder/model/tokenization.
При переносе с A100 пути файлов могут отличаться, хэши должны совпадать.

```bash
journalctl --user -u uzner-f15-warm.service -f
```

MLflow: локальный `mlruns/mlflow.db`, experiment `uzner-final-fit-local`.
Статус: `runs/f15_s33_train_dev_warm_epoch1/status.json`.
Первые 75 optimizer steps прошли на RTX 4070 Ti SUPER, GPU memory 11.21 GiB.
Предварительная оценка по первым шагам: около 9 минут обучения плюс сохранение;
это прогноз, не измеренное полное время. Итог пока не получен.

Проверки: Ruff; 12 тестов final-fit/final-ensemble/research-warm, включая
отсутствие dev-F1, reset optimizer/scheduler, несовпадение хэшей и незавершённый
checkpoint. Исходные правила offsets и декодирования не менялись.
