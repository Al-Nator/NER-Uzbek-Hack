# Серия 4: Sol-reviewed транслитерации — s47

Итог: run завершён, exact micro-F1 dev **0.905105**, P=0.912028,
R=0.898285. Ниже сохранены условия и история запуска.

## Оглавление

- [Данные](#данные)
- [Протокол](#протокол)
- [Запуск и артефакты](#запуск-и-артефакты)
- [Проверки](#проверки)

## Данные

Источник: `artifacts/transliteration/sol_review_complete_20260907_v1/releases/1788717691042035853/sol_review_candidates.jsonl`.
SHA-256: `95db23b6a942560000a3b44cbea0562d0fc043d509cb92c355b426a0a6d333ce`.
Это Luna-транслитерации после Sol review, не отдельная генерация с нуля.
Кандидаты не проверены человеком; пользователь явно разрешил экспериментальное
использование. Они остаются synthetic, не gold. Карантин (1 942 записи) не подключён.

| Этап | Записей |
|---|---:|
| Входные кандидаты | 13 310 |
| Неизменённые тексты, исключены | 3 965 |
| Дубли/совпадения с original train или dev, исключены | 8 |
| Добавлено | **9 337** |
| Из них Latin → Cyrillic / Cyrillic → Latin | 6 563 / 2 774 |
| Train вместе с оригиналами | **22 337** |
| Исходный dev без изменений | **1 500** |

Проверены происхождение каждого source_hash из official train, исходный текст,
полное непрерывное отображение offsets, сохранение классов и числа сущностей.
Дедупликация использует существующий `normalize_surface`; исходный текст не меняется.
Dev labels не используются для отбора. При новом split оригинал и копии должны
оставаться вместе по source_hash.

Подготовленный JSONL и manifest: `artifacts/series4/sol_review_v1/`.
SHA-256 JSONL: `f5c1580bd9dd56440d350473dd47472e44bfa9c98a1db1cf0ab9fbca104468c3`.

## Протокол

[`s47_bge_gp_translit_sol_review.yaml`](../configs/experiments/a100/s47_bge_gp_translit_sol_review.yaml)
повторяет модельный рецепт s44: pretrained BGE-M3-RetroMAE + GlobalPointer,
512/128, threshold 0.5, seed 42, BF16, batch 8, AdamW LR 2e-5,
до 5 эпох, early stopping patience 2. Начало от pretrained, не от обученного s44.
Silver не добавляется. Best выбирается по exact-span micro-F1 исходного dev.

Контроль s44: 13 000 + 9 827 Luna, best F1 **0.907456**. Здесь объём копий
другой: сравнение отражает эффект нового релиза в целом, а не чистую matched-абляцию
исправлений Sol. s47 получил 0.905105 и не превзошёл s44.

## Запуск и артефакты

Запущен на A100: `s47_bge_gp_translit_sol_review_sol-review-v1`.
[Remote MLflow](http://127.0.0.1:5001/#/experiments/3/runs/f34a8eb0d8844ab297e39d7ab0cae05e).
Первый подтверждённый шаг: epoch 1, step 25, finite loss 7.6183;
25 321 train windows, GPU 91%, память около 14 GiB (моментальный снимок).
Ориентир полного запуска **40–50 минут**, не гарантия; у s44 одна эпоха
занимала около 425 с обучения + 47–49 с оценки, дополнительно сохраняются веса.

Терминал:

```bash
ssh -t alnator zellij attach uzner-s47-sol
```

Лог на сервере:
`/home/danya/NER-Uzbek-Hack/runs/s47_bge_gp_translit_sol_review_sol-review-v1/logs/console.log`.
Resolved config, data hashes, source snapshot, predictions, best/last и подробные
метрики сохраняет стандартный runner. MLflow хранит метрики и лёгкие артефакты, не веса.

Локальная служба `uzner-return-s47.service` ждёт завершения, затем переносит полный
run и импортирует историю метрик в `uzner-first-series` (localhost:5000).
Для возврата локальный компьютер должен оставаться включённым с доступом по SSH.

```bash
journalctl --user -u uzner-return-s47.service -f
```

## Проверки

- 46 тестов passed, 3 неприменимые комбинации span-целей skipped.
- 11 новых проверок: provenance, offsets, повреждения, дубли, dev leakage,
  неизменённые копии, запрет перезаписи и равенство рецепту s44.
- Ruff новых модулей и тестов: passed.
- GPU smoke: 8 train / 4 dev, одна эпоха, успешно выполнены forward/backward,
  оценка и сохранение. Smoke не публиковался в MLflow и был удалён из `/tmp`.
- SHA-256 подготовленных данных и original train/dev совпали локально и на A100.
