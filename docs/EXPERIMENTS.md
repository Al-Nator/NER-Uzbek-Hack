# Журнал экспериментов

07.09.2026: [71 posthoc-проверка без обучения](POSTHOC_DECODING.md),
[полная таблица](../reports/posthoc_decoding_20260907.md).
Подтверждённый пользователем public: **s62 + train-словарь → повторы = 0.8922**,
архивный dev 0.917448. Предыдущая версия только с повторами: public 0.8897,
dev 0.915248. Почти весь дополнительный dev-прирост сосредоточен на одном документе.
Все 71 runs сверены с локальным MLflow (experiment 7).
Состав ансамбля заморожен; [отдельный serving-отчёт A100](../reports/serving_a100_20260907.md)
содержит проверки численного воспроизведения и производительности, не новые обучения.
[s48 завершён](S48_SILVER_INTERIM.md): dev 0.906604, public 0.8778; не заменяет эталон.

[Серия 7](SEVENTH_SERIES.md): s62 сохранён отдельным read-only эталоном;
s70 (s45+s21+s31) дал **0.913529** на dev. Изменения веса/границ окон
не улучшили свежий контроль. Три span-reranker-а завершены: s73 — 0.904699,
s74 — 0.904366, s75 — 0.911569; исходный s62 не превзойдён.

Измеренные продолжения (см. отдельные отчёты):
[2B: encoder continuation](ENCODER_CONTINUATION.md),
[3: Biaffine/GlobalPointer на A100](THIRD_SERIES.md).
[Серия 4](FOURTH_TRANSLITERATION.md): s40 — **0.905255**, s41 — **0.901283**,
оба завершены и возвращены с A100 вместе с checkpoint-ами и MLflow.
По 13 000 оригиналов + 4 803 транслитерации; прироста к s32 не получено.
[Продолжение серии 4 — s42 MELM-inspired](FOURTH_MELM.md): завершён,
4803 принятые копии, best epoch 5, exact micro-F1 **0.903436**.
[Полные s43/s44](FOURTH_FULL_TRANSLITERATION.md) завершены:
**0.905058 / 0.907456**, best epoch 2/5.
[Sol-reviewed s47](FOURTH_SOL_REVIEW.md) завершён на A100:
13 000 оригиналов + 9 337 копий, рецепт s44; dev **0.905105**.
[Silver s45/s46](FOURTH_SILVER.md) завершены: **0.909576 / 0.905626**, best epoch 5/5.
Лучший одиночный run — **s45, original train + 5 583 silver**;
добавление 9 827 Luna-копий в s46 снизило F1 на 0.395 п.п.
[Серия 5](FIFTH_SERIES.md): s53–s56 завершены на A100 с F1
`0.902589 / 0.900020 / 0.902856 / 0.904849`; локальные runs и MLflow-связи сохранены.
s57 kNN — `0.905549`, s58 char — `0.905574`: завершены локально,
ошибка публикации MLflow исправлена без повторного обучения.
[Серия 6](SIXTH_SERIES.md): s61 завершён с **0.905655**, s60 — **0.898954**;
**s62, фиксированное голосование s33+s21+s31: 0.912400**, P=0.916285, R=0.908548.
Это ансамбль на том же dev, не новая одиночная модель. Его финальный serving с
постобработкой проверяется отдельно: [Docker и измерения](SERVING.md).
Smoke-проверки не включаются в таблицы F1 ниже.
В серии 3 s31 завершился с 0.90164, s32 — 0.90595, s33 — **0.90670**
(исторический лучший одиночный run серии 3; теперь ниже s45).
s30 прерван: частичный best 0.64889 не включается в рейтинг полных runs.
Историческая очередь 2B/3 — [A100_QUEUE.md](A100_QUEUE.md);
история расширений — [FOURTH_MELM.md](FOURTH_MELM.md) и [FIFTH_SERIES.md](FIFTH_SERIES.md).

Фактические результаты добавляются только после появления полного набора артефактов.
Ниже приведена каноническая исправленная серия `s1-offsetfix-mlflow-r2`; все
метрики рассчитаны на одинаковом dev по exact совпадению `(label, start, end)`.

| Run ID | Encoder / decoding | P | R | Micro-F1 | Macro-F1 | ORG | NAME | GEO | Epoch | Мин. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `b00_official` | DistilmBERT, BIO greedy | 0.7868 | 0.7925 | 0.7897 | 0.7902 | 0.7515 | 0.8009 | 0.8180 | 2 | 7.5 |
| `b01_reference` | DistilmBERT, BIO greedy | 0.8060 | 0.8293 | 0.8175 | 0.8180 | 0.7782 | 0.8294 | 0.8464 | 5 | 8.7 |
| `e10_xlmr_base_bio` | XLM-R-base, BIO greedy | 0.8420 | 0.8683 | 0.8550 | 0.8558 | 0.8214 | 0.8723 | 0.8739 | 5 | 18.1 |
| `e11_mdeberta_v3_base_bio` | mDeBERTa-v3-base, BIO greedy | 0.8685 | 0.8926 | **0.8804** | **0.8815** | 0.8478 | **0.9059** | 0.8908 | 5 | 28.0 |
| `e12_mmbert_base_bio` | mmBERT-base, BIO greedy | 0.8680 | 0.8756 | 0.8718 | 0.8728 | 0.8374 | 0.8956 | 0.8856 | 5 | 23.8 |
| `a20_xlmr_base_bio_constrained` | XLM-R-base, BIO constrained | 0.8628 | 0.8718 | 0.8673 | 0.8679 | 0.8364 | 0.8802 | 0.8870 | 5 | 17.5 |
| `a21_xlmr_base_bioes_constrained` | XLM-R-base, BIOES constrained | 0.8698 | 0.8771 | 0.8734 | 0.8739 | 0.8438 | 0.8858 | **0.8922** | 5 | 19.1 |
| `a22_xlmr_base_bio_crf` | XLM-R-base, BIO CRF | 0.8658 | 0.8757 | 0.8707 | 0.8712 | 0.8492 | 0.8810 | 0.8834 | 5 | 58.7 |
| `a23_xlmr_base_bioes_crf` | XLM-R-base, BIOES CRF | **0.8727** | 0.8784 | 0.8756 | 0.8762 | **0.8505** | 0.8884 | 0.8897 | 5 | 60.1 |

## Вывод первой итерации

- Новый reference-контур выше official baseline на `+0.0278` micro-F1.
- Лучший encoder — mDeBERTa-v3-base, `0.8804`: `+0.0254` к XLM-R-base и
  `+0.0086` к mmBERT-base при одинаковом основном протоколе.
- Constrained BIO даёт XLM-R `+0.0123`, BIOES поверх него — ещё `+0.0061`.
- CRF с BIO даёт `+0.0158` к XLM-R greedy; BIOES + CRF достигает `0.8756`,
  суммарно `+0.0206` к greedy.
- BIOES + CRF выше BIOES constrained только на `+0.0021`, но полный run
  занимает `60.1` против `19.1` минуты. Механизм переносится на лучший encoder
  как quality-кандидат; constrained остаётся быстрым кандидатом.
- Все числа относятся только к исправленной серии с suffix
  `s1-offsetfix-mlflow-r2`; девять run-ов завершены и сохранены в MLflow.
- Полная последовательная серия заняла `241.6` минуты.

## Вторая серия: завершена

| Run ID | Encoder / decoding | P | R | Micro-F1 | Epoch | Мин. |
|---|---|---:|---:|---:|---:|---:|
| `s20_mdeberta_v3_base_bioes_constrained_s2-sequence-v1` | mDeBERTa, BIOES constrained | 0.8907 | 0.9019 | 0.8963 | 5 | 62.1 |
| `s21_mdeberta_v3_base_bioes_crf_s2-sequence-v1` | mDeBERTa, BIOES CRF | 0.8967 | 0.9001 | **0.8984** | 5 | 43.1 |
| `s22_xlmr_large_bioes_constrained_s2-sequence-a100-v2` | XLM-R-large, BIOES constrained | 0.89868 | 0.90335 | **0.90101** | 5 | 23.6 (A100) |
| `s23_xlmr_large_bioes_crf_s2-sequence-a100-v2` | XLM-R-large, BIOES CRF | 0.89109 | 0.90556 | 0.89827 | 4 | 90.4 (A100) |

Победитель — `s22`: CRF в `s23` уступил ему `0.00274` micro-F1.
Числа относятся к best checkpoint; s23 закончил пять эпох, best — четвёртая.
Продолжение: [`ENCODER_CONTINUATION.md`](ENCODER_CONTINUATION.md).

## Серии 2B и 3: актуальные результаты

| Run | Изменение | Best exact micro-F1 | Best epoch | Статус |
|---|---|---:|---:|---|
| s24 | BGE-M3-RetroMAE, BIOES constrained | 0.90205 | 4 | complete |
| s25 | XLM-V-base, BIOES constrained | 0.87557 | 5 | complete |
| s30 | XLM-R-large, Biaffine | 0.64889* | 1 | interrupted / KILLED |
| s31 | XLM-R-large, GlobalPointer | 0.90164 | 5 | complete |
| s32 | BGE-M3-RetroMAE, GlobalPointer | 0.90595 | 5 | complete |
| s33 | s32 epoch 5 → две эпохи low-LR, reset AdamW | **0.90670** | 1 дополнительная | complete |

\* s30 остановлен пользователем на третьей эпохе; число относится к лучшей из
двух завершённых эпох, не участвует в рейтинге полных запусков.
s24–s32 используют suffix `a100-continuation-v1`, s33 — `a100-low-lr-v1`.
Вторая дополнительная эпоха s33: 0.90586, то есть ниже исходного s32.

Проверка порогов фиксированного s32: лучший выбранный на dev порог 0.35 дал
0.90620 против исходных 0.90595 при 0.50. Это eval-only подбор, не новый
обученный encoder и не независимая оценка. Полная таблица и происхождение
checkpoint-ов: [`THIRD_SERIES.md`](THIRD_SERIES.md).
