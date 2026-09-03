# Протокол экспериментов

## Инварианты

1. Основная метрика — exact-span micro-F1.
2. `best` выбирается по этой метрике, не по token loss.
3. Исходный raw text не изменяется.
4. Encoder revision, data hashes, seed и зависимости закрепляются.
5. Один run меняет одну проверяемую гипотезу.

## Первая серия

1. Воспроизвести исходный baseline без изменений.
2. Проверить эквивалентность нового контура на том же encoder/BIO/softmax.
3. Сравнить XLM-R-base, mDeBERTa-v3-base и mmBERT-base при одинаковой голове.
4. На выбранном encoder отдельно проверить constrained decoding, BIOES и CRF.
5. Только после этого запускать сочетание BIOES + CRF.

## Справедливость сравнения

- одинаковые train/dev и preprocessing;
- одинаковый seed на первичном отборе;
- одинаковое effective-число окон на optimizer step и training budget;
- одинаковый способ объединения окон;
- один и тот же exact-span evaluator;
- для двух лучших конфигураций — минимум три seed с `mean ± std`.

## Обязательные отчёты

Кроме micro/macro и F1 по классам сохраняются:

- Latin / Cyrillic / mixed;
- seen / unseen surface;
- длина span;
- attached suffix;
- длина документа;
- chunk edge;
- boundary-short, boundary-long, wrong-label, missed и spurious.

## Запреты

- Не выбирать модель по впечатлению от отдельных примеров.
- Не смешивать смену encoder-а, головы и данных в одном run.
- Не перезаписывать существующий run_id.
- Не заносить результат в сводную таблицу без predictions и resolved config.
