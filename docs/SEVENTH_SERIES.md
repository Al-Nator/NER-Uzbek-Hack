# Серия 7: защищённый эталон, inference-абляции и span-reranker

Актуализация **07.09.2026**: три reranker-варианта завершены, метрики находятся
локально. Ни один не превзошёл исходный s62 на official dev:

| Вариант | Exact micro-F1 dev |
|---|---:|
| s73, linear | 0.904699 |
| s74, MLP | 0.904366 |
| s75, char-context | 0.911569 |
| Исходный s62 | **0.912400** |

Результаты: `runs/s7[345]*/metrics/dev.json`. В финальный сервис reranker
не включён. Позже s62 + train-словарь → повторы получил public **0.8922**
по сообщению пользователя; [posthoc-отчёт](POSTHOC_DECODING.md),
[текущий serving](SERVING.md). Ниже сохранены рецепт и история запуска серии.

## Оглавление

- [Эталон s62](#эталон-s62)
- [Дешёвые проверки: результаты](#дешёвые-проверки-результаты)
- [Честное обучение reranker](#честное-обучение-reranker)
- [Варианты](#варианты)
- [Очередь, время и логи](#очередь-время-и-логи)
- [Артефакты и возврат](#артефакты-и-возврат)
- [Проверки и ограничения](#проверки-и-ограничения)

## Эталон s62

Фиксированное большинство **2 из 3** по exact `(hash, label, start, end)`:

1. `s33_bge_global_pointer_low_lr_a100-low-lr-v1`;
2. `s21_mdeberta_v3_base_bioes_crf_s2-sequence-v1`;
3. `s31_xlmr_large_global_pointer_a100-continuation-v1`.

Dev micro-F1: **0.912399713**. Public micro-F1: **0.8861** — значение
из предоставленного пользователем скриншота лидерборда, не локальная оценка.

Независимый снимок: [`artifacts/releases/s62_public_08861_v1/`](../artifacts/releases/s62_public_08861_v1/).
Внутри реальные inference-веса, tokenizer/config каждой модели, исходная посылка
и компоненты, dev-предсказания/метрики, source snapshot и `uv.lock`.
Optimizer state не дублируется. `release.json` содержит SHA-256 каждого файла.
Каталоги имеют права `0555`, файлы `0444`; повторное создание запрещено.
Это read-only snapshot с проверкой целостности, **не привилегированная WORM-защита**:
владелец ОС технически может изменить права. Оригинальные runs не изменены.

SHA-256 посылки `submission/predictions.jsonl`:
`7f4b8e87aa6827942ba88af95e686ff6fa51118c990c759e50a308b76faca98d`.

SHA-256 самого `release.json`:
`82bfaffe17049ece157da99cc256e2ff8703058f2f215a033de350fbb7db5ab8`.

Проверка без изменения релиза:

```bash
uv run python -c 'from pathlib import Path; from uzner.experiments.champion import verify_champion; verify_champion(Path("artifacts/releases/s62_public_08861_v1"))'
```

## Дешёвые проверки: результаты

Конфиг: [`configs/seventh/cheap.yaml`](../configs/seventh/cheap.yaml).
Все результаты — **тот же official dev**, не новый public test.

| Run ID | Изменение | Exact micro-F1 | TP / FP / FN |
|---|---|---:|---:|
| Канонический s62 | Сохранённые исходные компоненты | 0.912400 | 6994 / 639 / 704 |
| `s70_s45_vote_swap_v1` | s45 вместо s33, остальные голоса прежние | **0.913529** | 6999 / 626 / 699 |
| `s7c0_fresh_uniform_control_v1` | Свежий GPU-инференс двух GP, обычная склейка | 0.912068 | 6991 / 641 / 707 |
| `s71_gp_center_weighting_v1` | Больший вес середины окна у двух GP | 0.911924 | 6994 / 647 / 704 |
| `s72_gp_word_windows_v1` | Сдвиг границ окон к пробелам у двух GP | 0.911667 | 6982 / 637 / 716 |

s70 даёт **+0.113 процентного пункта** к сохранённому s62 на dev.
Это небольшой кандидат на последующую проверку, не подтверждённое улучшение
на public/private. Эталон и файл победившей посылки не заменены.

Свежий uniform-инференс немного отличается от канонического: **−0.033 п.п.**
Причина численной/инференсной разницы пока не установлена; битовая
воспроизводимость не заявляется. Поэтому s71/s72 сравниваются прежде всего
с новым контрольным s7c0, а не только со старым s62. Оба изменения хуже контроля.
При центрированном взвешивании учитываются и отрицательные голоса перекрывающихся окон.
CRF-компонент s21 в этих трёх запусках фиксирован из cache.

Время всего run с оценкой/артефактами: s70 ≈3 с, остальные ≈32 с каждый
на локальной GPU. Это **не HTTP-бенчмарк полного ансамбля**: s21 закэширован.

Метрики/ошибки/компоненты лежат в `runs/<run_id>/` и
[локальном MLflow — uzner-seventh-series](http://127.0.0.1:5000/#/experiments/4/runs).

## Честное обучение reranker

Нельзя обучать reranker на train-предсказаниях моделей, которые уже видели
gold этих документов: такой reranker получает нереалистично лёгкие примеры.
Используется **grouped train-holdout pilot**, а не полный OOF по всем фолдам.

Исходные 13 000 train-документов разбиты с seed 42:

| Split | Документов | Назначение |
|---|---:|---|
| `proposer_train` | 9 328 | Обучение трёх предлагающих моделей |
| `proposer_valid` | 1 058 | Выбор checkpoint только этих моделей |
| `meta_train` | 2 084 | Обучение reranker на не виденных proposer-ами текстах |
| `meta_valid` | 526 | Выбор эпохи и порога reranker |

Ещё 4 train-документа исключены из-за группового пересечения с official dev.
Группировка связывает `source_hash`, нормализованные совпадения и близкие
word-shingle-дубликаты. Это эвристика near-duplicate, **не доказательство
отсутствия всех семантических или межалфавитных дублей**. Текст/offsets не меняются.
Silver и транслитерации сюда не добавляются.

Три новых proposer-а инициализируются из pretrained encoder, не из s33/s21/s31
NER checkpoint. Конфиги: `configs/seventh/s7p*.yaml`; отдельная серия
[`seventh_proposers.yaml`](../configs/series/seventh_proposers.yaml).
До 5 эпох, LR 2e-5, batch 8, BF16, без gradient checkpointing.
Early stopping patience 2 использует только `proposer_valid`.
BGE-proposer — базовый GP-рецепт; дополнительное low-LR продолжение s33
здесь не воспроизводится. Это подготовка честных кандидатов, не новый эталон.

Перед инференсом сверяются split SHA-256, фактические training inputs
из metadata и отсутствие NER warm-start. Кандидаты и признаки строятся
без gold; разметка добавляется только как supervised target.

После выбора reranker оценивается на:

- `heldout_dev`: official dev с тремя новыми, уменьшенными proposer-ами;
- `dev`: official dev с сохранёнными компонентами канонического s62.

Для обоих сохраняется соответствующее majority-baseline сравнение.
Второй срез проверяет перенос reranker на более сильные full-train модели;
распределение их ошибок может отличаться. Dev многократно использовался
в предыдущих сериях, поэтому он не становится независимым финальным тестом.

## Варианты

Конфиг: [`configs/seventh/rerankers.yaml`](../configs/seventh/rerankers.yaml).

| Run ID | Голова | Проверяемая гипотеза |
|---|---|---|
| `s73_span_reranker_linear_holdout_v1` | Линейный logistic classifier | Достаточно обучить доверие к сочетаниям голосов и простым признакам |
| `s74_span_reranker_mlp_holdout_v1` | MLP, 64 hidden | Полезны нелинейные взаимодействия признаков |
| `s75_span_reranker_char_context_holdout_v1` | MLP + небольшой char-CNN | Локальный текст помогает отличить правильный спан/класс от ложного |

Общие признаки: тип сущности, три бинарных голоса, длина/позиция спана,
письменность, регистр, цифры, соседние символы, повторы в документе.
Сырые logits CRF и GP **не усредняются**, так как их шкалы несопоставимы.
Char-вариант получает локальные Unicode-символы с маркерами границ;
второй Transformer для него не запускается.

Кандидаты — **объединение exact-спанов трёх proposer-ов**. Reranker может
сохранить одиночный полезный голос или отвергнуть большинство, но не создаёт
новые границы и не восстанавливает сущность, пропущенную всеми моделями.
Это ещё не semantic boundary-reranker и не прежняя char-offset head серии 5.

Общий рецепт: BCE, AdamW LR 1e-3, weight decay 0.01, batch 512, 12 эпох,
gradient clipping 1.0, seed 42. Выбор best epoch и порога из
`0.3 / 0.4 / 0.5 / 0.6 / 0.7` — **только exact micro-F1 meta_valid**.
Все варианты имеют одинаковые кандидаты, split и сетку порогов.
Final `ORG/NAME/GEO` и плоские Unicode spans возвращаются общим decoder/evaluator.

## Очередь, время и логи — исторический снимок 06.09

Запущено на `alnator` в `/home/danya/NER-Uzbek-Hack`:

1. `uzner-s7-proposers`: s7p0 → s7p1 → s7p2, по одной модели на A100;
2. `uzner-s7-rerankers`: ждёт три `complete` с корректным `kind`,
   создаёт prediction cache, затем s73 → s74 → s75.

Снимок около 20:13 МСК: s7p0 завершён, best epoch 5,
**внутренняя** proposer-valid F1 0.895097 (не official dev).
Provenance его фактических данных проверен; rsync уже возвращает полный run.
s7p1 начал обучение, очередь reranker находится в ожидании зависимостей.

При ошибке proposer-а reranker не стартует. Ожидание ограничено шестью часами.
При неполном cache/run повторный запуск запрещён до явной проверки;
существующие результаты не перезаписываются.

Измеренная эпоха BGE-proposer: **175–178 с train + 7–9 с eval**,
пиковая память 11.21 GiB, плюс запись checkpoint. Оценка всей очереди:
**1–2 часа**, не SLA; основное время — три proposer-а, затем инференс cache.
Маленькие reranker-головы требуют значительно меньше времени.
Перед каждой новой большой моделью проверяется минимум 25 GiB свободного диска.
Локальная GPU освобождена после дешёвых проверок.

```bash
ssh -t alnator zellij attach uzner-s7-proposers
ssh -t alnator zellij attach uzner-s7-rerankers
journalctl --user -u uzner-return-s7 -f
```

Proposer-логи: `runs/s7p*/logs/events.jsonl`, `console.log`, `history.csv`.
Reranker-логи: `runs/s7[345]*/logs/events.jsonl` и терминал очереди.
В MLflow пишутся step loss/LR/gradient norm, epoch loss/F1/time,
все пять meta-valid порогов, выбранная эпоха/порог, подробные финальные срезы,
ошибки и время только reranker-инференса.

[A100 MLflow — uzner-seventh-proposers](http://127.0.0.1:5001/#/experiments/7/runs)
содержит **внутреннюю** validation, не official dev.
Эксперимент `uzner-seventh-series` на A100 появится при старте первой головы.

## Артефакты и возврат

- `artifacts/reranker/s7_holdout_v1/`: split JSONL, group assignments/hashes,
  canonical dev predictions и проверяемый общий prediction cache;
- `runs/s7p*/`: обычные encoder/head best/last, полный optimizer state,
  `holdout_protocol.json` и `kind=reranker_proposer`;
- `runs/s73*/`, `s74*/`, `s75*/`: `resolved_config.json`, input hashes,
  `checkpoints/best.pt`, `last.pt` с optimizer/RNG, predictions,
  exact/slice/error JSON, events, source snapshot, lock и artifact manifest.

Работает локальная `uzner-return-s7.service`: после каждого завершения
rsync возвращает **весь run с checkpoint-ами**, проверяет SHA-256 и импортирует
всю историю metrics/params/tags в основной локальный MLflow. Общий cache
возвращается вместе с готовыми reranker-ами. Checkpoint-ы в MLflow не копируются.
Повторный импорт идемпотентен. Сбой связи перезапускает службу через 90 секунд.
Для возврата локальный компьютер должен быть включён и иметь SSH-доступ;
текущая служба transient, после перезагрузки можно продолжить:

```bash
uv run python scripts/return_seventh.py
```

Proposer-ы импортируются в `uzner-seventh-proposers`, reranker-ы — в
`uzner-seventh-series`. Исследовательские runs не добавляют внутреннюю F1
в общую CSV таблицу official-dev моделей.

## Проверки и ограничения

Проверены: независимость/read-only/hash эталона, запрет перезаписи,
group split и dev-пересечения, gold-free признаки, provenance rejection,
Unicode offsets/длинные окна/отрицательные голоса,
все головы на CPU и CUDA, неизменность выбора при подмене official dev gold,
MLflow metric history/idempotency и отсутствие весов в MLflow.
Smoke запуск BGE на A100 прошёл в `/tmp`; его собственные временные
веса/логи удалены, в таблицах экспериментов и MLflow его нет.

Полный pytest после реализации: **476 passed, 21 skipped**; Ruff пройден.
Пропуски — необязательные `UzTransliterator`/`ahocorasick` и неприменимые
комбинации auxiliary head; CPU/CUDA reranker-тесты выполнены.

Скорость полного ансамбля **через HTTP ещё не измерена**. Эти reranker-ы
не добавляют Transformer forward, но не устраняют стоимость трёх encoder-ов.
Ни рост public F1, ни выполнение HTTP-лимита пока не заявляются.
