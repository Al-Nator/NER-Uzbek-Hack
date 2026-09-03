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
- `scripts/` остаётся тонким CLI-слоем без бизнес-логики.

## Расширение данных

Каждый источник объявляется в data YAML. Поддерживаются `gold`, `synthetic` и
`pseudo`; у источника есть отдельный вес и split. Повторяющийся `hash` между
включёнными файлами считается ошибкой. Dev нельзя смешивать с обучающими
источниками.

## Расширение моделей

Текущая постановка — `token_tagging`. Будущие `span` и `set_prediction`
реализации добавляются отдельными пакетами и используют общие типы, evaluator и
формат run-артефактов. Архитектурная логика не должна попадать в data loader.

## Run-артефакты

Каждый `runs/<run_id>/` содержит:

- `resolved_config.yaml`;
- `metadata.json` с data hashes и версиями;
- `checkpoints/best` по exact micro-F1;
- `checkpoints/last` для resume;
- `predictions/dev.jsonl`;
- `metrics/dev.json` и slice-метрики;
- машинно-читаемый журнал обучения.
