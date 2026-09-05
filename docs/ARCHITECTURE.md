# Архитектура проекта

## Главный контракт

Любая модель должна преобразовать `Document` в набор
`Entity(label, start, end, score)`. Координаты относятся к исходной строке, а
не к нормализованному тексту или токенам.

```text
sources -> validation -> tokenization/model -> character spans -> exact scorer
```

## Границы модулей

- `domain.py` содержит общие dataclass-типы.
- `data/` отвечает за источники, offsets и BIO/BIOES.
- `models/` содержит архитектурные компоненты, но не читает файлы.
- `evaluation/` не зависит от tokenizer-а или модели.
- `experiments/` задаёт стандартную раскладку артефактов.
- `training/` связывает окна, модель, checkpoint, inference и отчёт, но не содержит
  архитектурную логику.
- `scripts/` остаётся тонким CLI-слоем без бизнес-логики.

## Расширение данных

Каждый источник объявляется в data YAML. Поддерживаются `gold`, `synthetic` и
`pseudo`; у источника есть отдельный вес и split. Повторяющийся `hash` между
включёнными файлами считается ошибкой. Dev нельзя смешивать с обучающими
источниками.

## Расширение моделей

Реализованы `token_tagging` и `span` (Biaffine, GlobalPointer). Фабрика моделей
выбирает head; span targets/collator живут в `data/spans.py`, объединение окон —
в `training/span_prediction.py`. Обучение, checkpoint/resume и evaluator общие.
Будущая `set_prediction` сохраняет общие типы и формат артефактов.
Архитектурная логика не должна попадать в data loader.

Pretrained encoder всегда загружается с FP32-параметрами. BF16 включается
только через CUDA autocast, чтобы optimizer обновлял численно устойчивую
копию весов.
Основная A100-очередь использует обычный AdamW для всех моделей.
В резервном локальном конфиге XLM-V задан AdamW8bit: квантуются состояния
optimizer-а, не параметры модели; отличие зафиксировано в конфиге и MLflow.

## Run-артефакты

Каждый `runs/<run_id>/` содержит:

- `resolved_config.yaml`;
- `metadata.json` с data hashes и версиями;
- `checkpoints/best` по exact micro-F1;
- `checkpoints/last` для resume;
- `predictions/dev.jsonl`;
- `metrics/dev.json` и slice-метрики;
- машинно-читаемый журнал обучения.

Полная схема и правила изоляции smoke-run-ов описаны в
[`ARTIFACTS.md`](ARTIFACTS.md).
