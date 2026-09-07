# Серия 4: данные и словари

## Оглавление

- [Порядок](#порядок)
- [Транслитерации: s40 и s41](FOURTH_TRANSLITERATION.md)
- [MELM-inspired: s42](FOURTH_MELM.md)
- [Полные транслитерации: s43 и s44](FOURTH_FULL_TRANSLITERATION.md)
- [Согласованная silver-разметка: s45 и s46](FOURTH_SILVER.md)
- [Sol-reviewed транслитерации: s47](FOURTH_SOL_REVIEW.md)
- [UzNER-100K](#uzner-100k)
- [Условия подключения](#условия-подключения)

## Порядок

Первый фактический этап серии — [парная абляция транслитераций s40/s41](FOURTH_TRANSLITERATION.md):
исправленный конвертер → Luna high, каждый добавляет ровно 4 803 записи к official
train. Reference — рецептура BGE-M3-RetroMAE + GlobalPointer из s32.
Оба завершены без улучшения общего F1: s40 `0.905255`, s41 `0.901283`.
Следующий фактически запущенный этап — [s42 MELM-inspired](FOURTH_MELM.md):
сначала генератор/фильтр локально, затем NER на original train + принятых копиях.
Внешние корпуса в него не добавляются.

Ниже сохранён **план следующих экспериментов с внешними данными и словарями**;
эти источники не включены в s40/s41 и их запуск не выполнен. Перенос лучшей
data-рецептуры на другую архитектуру затем оформляется отдельным запуском.

Исходный аудит: [`data_analysis.ipynb`](../data_analysis.ipynb), разделы 4–5,
и [`DATA_ANALYSIS.md`](DATA_ANALYSIS.md), commit `8d32e57`.

| Порядок | Изменение относительно фиксированного reference |
|---|---|
| 4A.1 | + проверенный Uzbek NER Gold (4 176 предложений по EDA) |
| 4A.2 | + проверенный 25K-UzNER-5Style отдельно от 4A.1 |
| 4A.3 | + исходно несинтетическая часть UzNER-100K, после согласования разметки |
| 4B.1 | + отфильтрованная внешняя синтетика, отдельно от real/weak-real |
| 4B.2 | + наша целевая синтетика: кириллица, новые ORG, суффиксы, короткие сообщения |
| 4C | train-only gazetteer и затем внешние словари как отдельные inference-абляции |

Комбинация источников допускается после измерения каждого по отдельности.
Помимо F1 считать бюджет optimizer steps/tokens и объём каждого источника:
пять эпох по расширенному train не эквивалентны пяти эпохам official train.
Сравнить смешанное обучение с external pretraining → official fine-tuning;
это разные рецептуры и отдельные runs.

## UzNER-100K

Приложенный релиз `UzNER-100K_paper_aligned_70k_real_30k_synthetic_v1`
проверен скриптом [`audit_external_uzner.py`](../scripts/audit_external_uzner.py).
Полный отчёт с hashes: [`uzner100k_audit.json`](../reports/data/uzner100k_audit.json).

| Split | Строк | `is_synthetic_original=true` | Synthetic переименованы в real | Совпадения с official dev |
|---|---:|---:|---:|---:|
| train | 100 000 | 64 548 | 34 548 | 4 |
| dev | 2 000 | 900 | 900 | 4 |
| test | 2 000 | 900 | 900 | 3 |
| gold_candidate | 10 000 | 3 000 | 3 000 | 2 |
| hard_eval | 269 | 269 | 269 | 1 |

35 452 train-записи имеют `is_synthetic_original=false`; это не означает
35 452 независимо проверенных gold-примеров. Первичные `is_synthetic` и
`source_tier` в релизе намеренно изменены под пропорцию 70K/30K, что прямо
описано поставщиком в `PAPER_ALIGNMENT_NOTES.md`. Использовать original-поля,
сохранять source, origin_file и обе версии provenance.

Совпадения выше — равенство полного текста после NFKC/casefold/апострофов/пробелов;
внешний train также совпадает с 39 official train-документами. Есть по одному
повтору нормализованного текста в train и gold_candidate. Это нижняя оценка:
предложения внутри длинных документов и near-duplicates ещё не проверены.
Ошибок диапазонов и несовпадений `text[start:end]` с entity.text не найдено;
согласование BIOES с entities и семантика аннотаций требуют отдельного аудита.

По метаданным все записи латинские. Корпус не закрывает непосредственно
кириллическую слабость s22. Внешние dev/test не заменяют official dev и
автоматически в train не добавляются.

## Условия подключения

1. Общая дедупликация внешних источников между собой, с official train/dev,
   включая предложения, near-duplicates и шаблоны. Из пересечений с dev
   удаляется внешний кандидат, а не official dev.
2. `PER → NAME`, `ORG → ORG`, `GPE/LOC → GEO` после проверки границ.
   `FAC` нельзя вслепую отбросить: многие объекты относятся у нас к GEO,
   храмы — к ORG. `PRODUCT` может включать бренд ORG, `POSITION` не входит в NAME.
   Неоднозначные примеры требуют accept/fix/drop или маски частичной разметки;
   иначе скрытые целевые сущности ошибочно обучаются как O.
3. Согласовать суффиксы, кавычки, маршруты и составные организации с
   [`LABELING_GUIDE.md`](../ner_uz_hackathon_participant/LABELING_GUIDE.md).
4. Новый versioned data manifest, неизменяемые raw-файлы, hashes, provenance,
   отдельные gold/weak/pseudo/synthetic наборы и отчёт отбраковки.
5. Реализовать и проверить реальное применение source weights: текущий
   `load_split` объединяет записи, но `DataSourceConfig.weight` не влияет на sampling/loss.
   Одной записи weight в YAML недостаточно.
6. GeoNames из EDA добавляет лишь 31 покрытое gold-упоминание dev; это не прирост
   F1. Gazetteer требует контекста и не должен автоматически назначать GEO.
   Dev-примеры из review-таблиц не добавляются в train или словарь.

Повторить read-only аудит:

```bash
uv run python scripts/audit_external_uzner.py \
  --source /путь/к/UzNER-100K_paper_aligned_70k_real_30k_synthetic_v1 \
  --output reports/data/uzner100k_audit.json
```
