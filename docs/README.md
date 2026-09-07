# Документация

## Финальная система

- [Команды организаторам: predict, eval, benchmark, обучение](ORGANIZERS.md)
- [Serving: запуск, экспорт, совместимость и benchmark](SERVING.md)
- [Измеренная производительность A100](../reports/serving_a100_20260907.md)
- [Словарь, повторы и 71 абляция без переобучения](POSTHOC_DECODING.md)
- [Архитектура](ARCHITECTURE.md), [артефакты](ARTIFACTS.md), [MLflow](EXPERIMENT_TRACKING.md)
- [Frontend и backend](../apps/README.md)

## Эксперименты — история, не очередь автозапуска

[Общий журнал и результаты](EXPERIMENTS.md) · [Протокол](EXPERIMENT_PROTOCOL.md)

| Серия | Что менялось |
|---|---|
| [1](FIRST_SERIES.md) | Базовые encoder-ы и корректный offset pipeline |
| [2](SECOND_SERIES.md), [2B](ENCODER_CONTINUATION.md) | Sequence-модели, большие encoder-ы, BGE / XLM-V |
| [3](THIRD_SERIES.md) | Biaffine / GlobalPointer, пороги, короткое продолжение |
| [4](FOURTH_SERIES.md) | Транслитерации, MELM-inspired, silver |
| [5](FIFTH_SERIES.md) | Вспомогательные цели, kNN, символьные границы |
| [6](SIXTH_SERIES.md) | Дополнительные модели и exact-span ансамбль |
| [7](SEVENTH_SERIES.md) | Защищённый эталон и span-reranker |

[s48](S48_SILVER_INTERIM.md) · [train+dev ансамбль](FINAL_ENSEMBLE.md) ·
[Sol review](FOURTH_SOL_REVIEW.md). Рецепты из этих документов не заменяют
подтверждённый s62+c02 автоматически.

## Работа с проектом

- [Удалённые runs, логи, возврат артефактов](REMOTE_EXPERIMENTS.md)
- [Карточка данных](DATA_CARD.md), [анализ ошибок](ERROR_ANALYSIS.md)
- [Условия задачи и FAQ](task/README.md)
- [API](../ner_uz_hackathon_participant/API.md) и
  [правила разметки](../ner_uz_hackathon_participant/LABELING_GUIDE.md)
