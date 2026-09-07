# Финальная модель: s45 на train + dev + silver

Повторение победившего трёхмодельного **s62 на train+dev** описано отдельно:
[FINAL_ENSEMBLE.md](FINAL_ENSEMBLE.md). Оно не использует s45/silver.

[Рецепт](#рецепт) · [Артефакты](#артефакты) · [Запуск](#запуск)

## Рецепт

`f01_s45_train_dev_silver_epoch5`: BGE-M3-RetroMAE + GlobalPointer,
pretrained start, **ровно 5 эпох**. Не продолжение checkpoint s45.
13 000 train + 1 500 бывший dev + 5 583 silver = **20 083 документа**.
Luna, public test и другие новые данные не добавляются.
LR 2e-5, warmup 10%, seed 42, BF16, batch 8, окно 512/128, threshold 0.5.

Специальный `scripts/train_final.py` объединяет оба split-а исходного s45 data-config.
Их исходные имена сохранены для provenance, но фактическое использование обоих — train.
Validation отсутствует: нет dev loss/F1, early stopping, выбора по public или dev.
Итог всегда после пятой эпохи. Run не добавляется в сравнительную F1-таблицу.
Метрика исходной s45 (0.909576 на dev) не является оценкой новой модели.

## Артефакты

Стандартный `runs/f01_s45_train_dev_silver_epoch5/`: config, data hashes,
source snapshot, train step/epoch loss, LR, gradient norm, throughput, GPU и MLflow.
`checkpoints/last` сохраняется после каждой эпохи с optimizer/RNG состоянием.
`checkpoints/best` создаётся только после пятой эпохи как совместимое имя
фиксированного финального checkpoint для `predict_jsonl.py`; это **не best по F1**.
`TrainerState.best_micro_f1=-1` — служебный sentinel, не измеренная метрика.
Статус содержит `kind=final_fit`, `selected_epoch=5`, без `best_micro_f1`.
Автоматический CLI resume этого отдельного final-fit пока не реализован.

Изолированный CPU smoke проверяет все 5 эпох на tiny-модели, объединение split-ов,
отсутствие dev-метрик и выбор пятой эпохи. Remote import не публикует фиктивную F1.
Веса не копируются в MLflow. Оригинальные s45 и лидербордный файл не меняются.

## Запуск

```bash
uv run python scripts/train_final.py --config configs/final/f01_s45_train_dev_silver.yaml
```

Только этот CLI для final-fit; не `run_series.py`/обычный `train.py`.
Серия `configs/series/final_fit_a100.yaml` нужна для выбора run-а при remote watch/sync,
suffix пустой. A100 MLflow experiment: `uzner-final-fit-a100`.
Локальный watcher возвращает полный run с проверкой SHA-256 и импортом метрик.
При текущей скорости ориентир 35–45 минут обучения с сохранением, плюс перенос весов.
