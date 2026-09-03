# Data card

## Текущие данные

Используется официальный комплект:

- train: 13 000 документов, 66 083 сущности;
- dev: 1 500 документов, 7 698 сущностей;
- классы: `ORG`, `NAME`, `GEO`;
- координаты: Python Unicode character offsets, `[start, end)`.

Точные hashes и статистика находятся в
`ner_uz_hackathon_participant/data/dataset_manifest.json`.

## Политика добавления источников

Новый файл получает уникальное имя, фиксированный путь, split, provenance,
`kind` (`gold`, `synthetic`, `pseudo`) и SHA-256. Источник сначала проходит
общую валидацию offsets, классов, пересечений и повторяющихся hash.

Synthetic и pseudo-labeled данные не объединяются с gold без возможности
отдельно отключить их конфигом и провести абляцию.
