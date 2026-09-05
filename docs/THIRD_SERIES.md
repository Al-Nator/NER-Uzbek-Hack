# Серия 3: span-based NER

## Оглавление

- [Состав и сравнение](#состав-и-сравнение)
- [Результаты](#результаты)
- [Реализация](#реализация)
- [Запуск на A100](#запуск-на-a100)
- [Проверки](#проверки)
- [Перенос GlobalPointer на BGE — s32](#перенос-globalpointer-на-bge--s32)
- [Порог и low-LR продолжение](#порог-и-low-lr-продолжение)
- [Проверка публикации](#проверка-публикации)

## Состав и сравнение

Две span-head реализованы для **A100**. Исходное распределение —
[единая последовательная очередь s24 → s25 → s30 → s31](A100_QUEUE.md),
а не параллельное обучение 2B локально. Отдельные команды серии 3 ниже
сохраняются для независимого запуска; одновременно с общей очередью их не запускать.

| Run | Encoder | Head |
|---|---|---|
| `s30_xlmr_large_biaffine` | XLM-R-large | Biaffine |
| `s31_xlmr_large_global_pointer` | XLM-R-large | GlobalPointer с RoPE |
| `s32_bge_m3_retromae_global_pointer` | BGE-M3-RetroMAE | GlobalPointer с RoPE |
| `s33_bge_global_pointer_low_lr` | s32 checkpoint эпохи 5 | GlobalPointer, low-LR продолжение |

Reference — `s22_xlmr_large_bioes_constrained_s2-sequence-a100-v2`: **0.90101**.
Для s30/s31 инициализация — pretrained XLM-R-large, не NER checkpoint s22.
BIOES/CRF-голова в эти модели не переносится.

Official train/dev, context 512, stride 128, seed 42, effective batch 8,
LR 2e-5, warmup 10%, weight decay 0.01, до пяти эпох, patience 2.
Меняется семейство головы и необходимый ему loss; значения loss между
архитектурами напрямую не сравниваются. Best — exact micro-F1.

Новые encoder-ы 2B и внешние данные не являются зависимостями серии 3.
Перенос лучшей головы на победителя 2B затем оформляется отдельным run.
Efficient GlobalPointer в текущий состав не входит.

## Результаты

Канонический suffix s30–s32 — `a100-continuation-v1`, s33 — `a100-low-lr-v1`.
Источник: сохранённые `logs/history.csv` и `status.json` на A100.
Полные результаты — best checkpoint по exact-span micro-F1 на official dev.

| Run | Статус | Best epoch | Precision | Recall | Micro-F1 | ORG | NAME | GEO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s30 | Прерван пользователем на эпохе 3 | 1 из двух завершённых | 0.84176 | 0.52793 | 0.64889* | — | — | — |
| s31 | Завершён, 5 эпох | 5 | 0.90542 | 0.89790 | 0.90164 | 0.88703 | 0.91229 | 0.90670 |
| **s32** | **Завершён, 5 эпох** | **5** | **0.91161** | **0.90036** | **0.90595** | 0.88184 | 0.91571 | 0.92133 |
| **s33** | **Завершён, 2 дополнительные эпохи** | **1 дополнительная** | 0.90889 | 0.90452 | **0.90670** | 0.88538 | 0.91649 | 0.91927 |

\* s30 — частичный результат, не финальная оценка полного бюджета и не участник
рейтинга завершённых запусков. Вторая эпоха: F1 0.59784, recall 0.46752.
Сохранён как отрицательный результат; незавершённый MLflow run помечается KILLED.
Старый исходный `status=running` после остановки не означает живой процесс.
Локальный архив сохраняет его в `environment/status_before_archive.json`,
а локальный `status.json` помечен `interrupted`. Время завершения в MLflow
для s30 — время последнего записанного события, не точное время Ctrl+C.
Частичный архив не содержит гарантированного финального набора reports/predictions.

s32 выше s31 на 0.43 п.п.; выше sequence-reference s24 (0.90205) на 0.39 п.п.
Относительно s24 вырос precision и GEO F1, снизились recall и NAME F1.
Статистическая значимость не проверена. s32 — reference до low-LR продолжения;
s33 — номинальный лидер с приростом всего 0.075 п.п.
Локальный MLflow s31: `22a77ee9369747528ed31383ecfb9ef2`.
Локальный MLflow s30: `d7d9a321729b4e4f80fc3d70e9d187ff`, статус KILLED;
лёгкие артефакты и частичная история импортированы. Возврат весов s30/s31
завершён; соответствующие фоновые сервисы успешно завершились.
У s31/s32 лучшая пятая эпоха, но dev loss растёт после второй: продление обучения
не считается автоматически обоснованным.

Неудача s30 не опровергает Biaffine: текущая CE усредняется по всем span-парам.
На диагностической выборке 128 train-документов около 4086 отрицательных пар
на положительную. Дисбаланс и threshold — гипотезы для отдельной диагностики,
не доказанная единственная причина. В текущем recipe нет boundary smoothing.

Следующий шаг — анализ дополняемости s24/s32, а не обязательное внедрение ансамбля.
Ограничения скорости и варианты ускорения: [качество и стоимость инференса](INFERENCE_TRADEOFFS.md).

## Реализация

- `models/span_heads.py`: Biaffine с двумя MLP по 64 и bias, CE по
  NONE/ORG/NAME/GEO; полный GlobalPointer с head size 64, RoPE и multilabel
  categorical crossentropy в FP32. Это наши head-абляции поверх XLM-R,
  не буквальное воспроизведение полного recipe исходных статей.
- `data/spans.py`: разреженные gold-targets, плотная матрица только на текущий
  batch. Padding/special tokens, обратные пары и непредставимые/обрезанные
  gold маскируются, включая охватывающие их пары. Длина span не ограничивается
  дополнительно к окну; BIOES не используется при построении targets.
- `training/span_prediction.py`: среднее вероятностей точного character-span
  во всех покрывающих окнах, включая отрицательные голоса. Порог **0.5**;
  затем greedy non-overlap по confidence, при равенстве — длинный span.
  Декодирование не использует gold masks. Матрицы хранятся только для текущего
  документа, не всего dev.
- Общие train loop, evaluator, detailed metrics, source snapshot, best/last и
  MLflow. `metrics/span_coverage.json` содержит oracle-покрытие границ и также
  попадает в MLflow. Отдельная проверка threshold s32 приведена ниже.

Основа: [Biaffine NER, ACL 2020](https://aclanthology.org/2020.acl-main.577/),
[авторский GlobalPointer](https://github.com/bojone/GlobalPointer).

## Запуск на A100

Код и uv-среда подготовлены в `/home/danya/NER-Uzbek-Hack` на alnator.
Из локального корня проекта:

```bash
uv run python scripts/remote_experiments.py --remote-experiment uzner-third-series-a100 start \
  --series configs/series/third_a100.yaml --stage all \
  --run-suffix s3-spans-a100-v1 --session uzner-s30-s31
```

Train/eval batch 8, accumulation 1, BF16, **без gradient checkpointing**.
Команда оставляет обучение в detached zellij. Живой лог и MLflow:

```bash
uv run python scripts/remote_experiments.py attach --session uzner-s30-s31
uv run python scripts/remote_experiments.py tunnel
# http://127.0.0.1:5001 — experiment uzner-third-series-a100
```

Возврат результатов (оставить в отдельном локальном терминале):

```bash
uv run python scripts/remote_experiments.py watch \
  --series configs/series/third_a100.yaml --stage all --run-suffix s3-spans-a100-v1
```

После завершения можно вызвать `sync` с теми же параметрами вместо `watch`.
`sync --metrics-only` импортирует историю без ожидания больших checkpoint-ов.
Основной локальный MLflow — experiment `uzner-first-series` на порту 5000;
фильтр по suffix `s3-spans-a100-v1`.

## Проверки

Весь official train/dev: точно представимы **66 067 / 66 083** train gold и
**7 697 / 7 698** dev gold. Это oracle-покрытие токенизации/окон, не F1 модели.
Общие targets для s30/s31 проверяются один раз.

На A100 два полных optimizer step 8×512: s30 **0,61 с / 10,99 GiB**,
s31 **0,64 с / 11,03 GiB** peak allocated. Тесты проверяют математику голов,
Unicode offsets, повторные границы, длинные spans, masks, отсутствие gold leakage,
усреднение окон, API-формат и полный CPU train/eval/checkpoint/resume.
GPU train/eval/save/load/продолжение обучения прошли для обеих голов на 128 train
и 32 dev документах. s30: train 2,4–2,6 с / eval 0,8–1,1 с;
s31: train 2,5–2,9 с / eval 1,1–1,2 с. Peak allocated 11,08/11,12 GiB.
Ориентир на пять полных эпох: **30–45 мин s30, 35–55 мин s31**;
вместе примерно **1–2 часа**, без передачи checkpoint-ов по SSH.
Это экстраполяция короткого smoke, не замер полного обучения. Рост числа
кандидатов и длинных документов может увеличить время evaluation.
Полные результаты приведены выше; smoke не публикуется в MLflow/сводке.
Отдельный overfit-тест подтверждает обучение положительных spans и negatives.

Для дальнейшего отбора использовать ошибки unseen ORG, кириллицы и длинных
сущностей, а также пересечение ошибок с sequence-reference для будущего ансамбля.

## Перенос GlobalPointer на BGE — s32

`s32_bge_m3_retromae_global_pointer` проверяет замену только encoder-а s31
на исходный `BAAI/bge-m3-retromae` с revision из s24, не на NER checkpoint.
GlobalPointer, threshold 0.5, official train/dev и весь training budget s31
сохранены: 512/128, batch 8, AdamW 2e-5, BF16, до пяти эпох, seed 42,
без gradient checkpointing. Новая модельная логика не нужна.

Отдельный манифест `configs/series/third_bge_a100.yaml` не меняет старые очереди.
Run suffix: `a100-continuation-v1`; zellij: `uzner-s32`;
remote MLflow experiment: `uzner-continuation-a100`.

```bash
uv run python scripts/remote_experiments.py --remote-experiment uzner-continuation-a100 start \
  --series configs/series/third_bge_a100.yaml --stage all \
  --run-suffix a100-continuation-v1 --session uzner-s32
uv run python scripts/remote_experiments.py watch \
  --series configs/series/third_bge_a100.yaml --stage all \
  --run-suffix a100-continuation-v1
```

Watch возвращает полный run и импортирует метрики в локальный MLflow;
локальный компьютер должен оставаться включённым и иметь SSH-доступ.
Результат s32: 0.90595, best epoch 5; полный run возвращён локально и импортирован
в MLflow, local run `56dfaf992874433f8b63ff79ba37664e`.

Запущен 2026-09-05 на A100; remote MLflow run
`e10821fd6d694db08a786d3635d64046`. Локальный возврат работает как
`uzner-return-s32.service` успешно завершил возврат; его журнал доступен через journalctl.
Перед запуском: 135 тестов, coverage 92,79%, Ruff; GPU preflight — два
optimizer step 8×512 за 0,62 с, peak allocated 11,12 GiB.
Preflight не создавал experiment-артефактов или MLflow run.

## Порог и low-LR продолжение

Порог: `scripts/sweep_span_threshold.py --source-run runs/s32_bge_m3_retromae_global_pointer_a100-continuation-v1`.
Фиксированная сетка 0.35/0.40/0.45/0.50/0.55, веса best epoch 5 не изменяются.
Каждый порог проходит полный dev inference: фильтрация готового predictions
не восстановила бы кандидатов ниже исходного threshold. Runs
`s32_threshold_*_v1` помечаются `dev_threshold_selection`, не обучением.
Сохраняются resolved config, хэши весов/данных, predictions и подробные метрики;
checkpoint-ы не копируются. Подбор на dev не является независимой test-оценкой.

Полный sweep 2026-09-05, 1500 official dev документов / 7698 gold-сущностей:

| Порог | Precision | Recall | Exact micro-F1 |
|---|---:|---:|---:|
| 0.35 | 0.906315 | 0.906080 | **0.906197** |
| 0.40 | 0.908035 | 0.904261 | 0.906144 |
| 0.45 | 0.909746 | 0.902182 | 0.905948 |
| 0.50 | 0.911614 | 0.900364 | 0.905954 |
| 0.55 | 0.912837 | 0.897896 | 0.905305 |

0.50 точно воспроизвёл исходные TP/FP/FN: 6931/672/767.
Выигрыш выбранного на dev порога 0.35 — только 0.024 п.п.; 0.91 не достигнут.
Для s33 сохраняется 0.50, чтобы не смешивать threshold и обучение.

`s33_bge_global_pointer_low_lr`: отдельный новый run из **всех весов** s32 best
(encoder и GlobalPointer). Новый AdamW, новый linear schedule 2e-6 → 0,
без warmup, две эпохи; threshold 0.5 и остальные настройки сохранены.
Optimizer moments, RNG и счётчики исходного s32 не восстанавливаются.
`training.initial_checkpoint` и SHA-256 исходных весов сохраняются в артефактах.
Это warm-start с reset optimizer, не точный resume s32.
`--resume` допустим только для продолжения самого s33 при его прерывании.

Из локального репозитория, после завершения sweep и при свободной A100:

```bash
uv run python scripts/remote_experiments.py --remote-experiment uzner-continuation-a100 start \
  --series configs/series/third_low_lr_a100.yaml --stage all \
  --run-suffix a100-low-lr-v1 --session uzner-s33
```

Best s33 выбирается среди новых эпох; исходный s32 остаётся отдельным reference.
Если обе новые эпохи хуже s32, в сервис не переносим s33.
Полное обучение s33 не запускается автоматически вместе с threshold sweep.

### Результат s33

Обе дополнительные эпохи завершены. F1: 0.906700 → 0.905865; dev loss:
1.33956 → 1.40587 (у исходного s32: 1.16529). Best — первая дополнительная
эпоха, то есть шестая суммарно; вторая ниже исходного s32.
Локальный MLflow: `f3731e491bc44d4b8a0309b899ccc97b`; импортировано 4741
точка метрик. Веса возвращаются отдельно с проверкой manifest; наличие
`.checkpoint-transfer-pending` означает, что передача ещё не завершена.
Прирост к s32: 0.0746 п.п.; 0.91 не достигнут. Продолжать ту же рецептуру
без новой гипотезы не планируется. Следующий шаг — анализ дополняемости
сохранённых predictions, не автоматическое внедрение ансамбля.

## Проверка публикации

Снимок коммита серий 2/3: 139 тестов пройдено, coverage 92,77%; Ruff для всех
37 добавленных/изменённых Python-файлов пройден. Незавершённые модули curation
и transliteration в этот снимок не входят. Глобальный Ruff по всему дереву
выявляет старые замечания в EDA-ноутбуке и EDA-скриптах из предыдущего коммита;
они не исправлялись в рамках модельных экспериментов.
