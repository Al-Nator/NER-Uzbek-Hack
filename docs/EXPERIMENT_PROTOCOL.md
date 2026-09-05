# Протокол экспериментов

## Инварианты

1. Основная метрика — exact-span micro-F1.
2. В новом контуре `best` выбирается по exact micro-F1. Единственное
   исключение — неизменённый official baseline, где checkpoint выбирается по dev loss.
3. Исходный raw text не изменяется.
4. Encoder revision, data hashes, seed и зависимости закрепляются.
5. Один run меняет одну проверяемую гипотезу.

## Первая серия

1. Воспроизвести исходный baseline без изменений.
2. Проверить эквивалентность нового контура на том же encoder/BIO/softmax.
3. Сравнить XLM-R-base, mDeBERTa-v3-base и mmBERT-base при одинаковых голове,
   context 512 и effective batch 8.
4. На стабильной XLM-R-base отдельно проверить constrained decoding, BIOES, CRF
   и BIOES + CRF. Лучший механизм перенести на лучший encoder в следующей
   серии.

## Вторая серия

Вторая серия содержит только sequence-модели. BIOES constrained и BIOES + CRF
переносятся на mDeBERTa-v3-base и XLM-R-large. Span-based головы не смешиваются
с этой серией и проверяются после выбора sequence-финалистов. Точный порядок и
команды: [`SECOND_SERIES.md`](SECOND_SERIES.md).

## Справедливость сравнения

- одинаковые train/dev и preprocessing;
- одинаковый seed на первичном отборе;
- одинаковое effective-число окон на optimizer step и training budget;
- одинаковый способ объединения окон;
- один и тот же exact-span evaluator;
- текущие серии используют один зафиксированный seed 42; обязательного multi-seed этапа нет.

## Обязательные отчёты

Следующие этапы: [2B — encoder continuation](ENCODER_CONTINUATION.md),
[3 — Biaffine и GlobalPointer](THIRD_SERIES.md),
[4 — данные и словари](FOURTH_SERIES.md).

Кроме micro/macro и F1 по классам сохраняются:

- Latin / Cyrillic / mixed;
- exact-seen / normalized-seen / unseen surface;
- длина span;
- attached suffix;
- длина документа;
- chunk edge;
- boundary-short, boundary-long, boundary-shift, wrong-label, missed и spurious;
- boundary-only F1, document exact match, empty-document FP rate и mean matched IoU;
- tokenizer representability, mean/p95 subwords, chars/subword и число окон;
- apostrophe/quotes и entity-level Latin/Cyrillic/mixed;
- step/epoch loss, learning rate, gradient norm, throughput, latency, train/eval
  time, CPU/RAM и GPU utilization/memory/power.

## Запреты

- Не выбирать модель по впечатлению от отдельных примеров.
- Не смешивать смену encoder-а, головы и данных в одном run.
- Не перезаписывать существующий run_id.
- Не заносить результат в сводную таблицу без predictions и resolved config.
- Не заносить smoke/test-run в `runs/` и общую таблицу.
- Не создавать MLflow run для smoke/test-run и не копировать туда checkpoint-ы.

Точный порядок и гипотезы: [`FIRST_SERIES.md`](FIRST_SERIES.md). Схема файлов run-а:
[`ARTIFACTS.md`](ARTIFACTS.md).
