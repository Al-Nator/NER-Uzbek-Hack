"""Проверка всех включённых источников данных."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from uzner.config import load_data_config, resolve_sources
from uzner.data.io import load_documents, sha256_file
from uzner.domain import Document


def parse_args() -> argparse.Namespace:
    """Разбирает путь к конфигурации данных."""
    parser = argparse.ArgumentParser(description="Проверить NER-источники и их статистику")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def summarize(documents: list[Document]) -> dict[str, object]:
    """Собирает компактную статистику документов и сущностей."""
    labels = Counter(entity.label for document in documents for entity in document.entities)
    sources = Counter(document.source for document in documents)
    return {
        "documents": len(documents),
        "documents_with_entities": sum(bool(document.entities) for document in documents),
        "entities": sum(len(document.entities) for document in documents),
        "entities_by_label": dict(sorted(labels.items())),
        "documents_by_source": dict(sorted(sources.items())),
    }


def main() -> int:
    """Проверяет пути, hashes и содержимое train/dev источников."""
    args = parse_args()
    project_root = args.project_root.resolve()
    data = load_data_config(args.config.resolve())
    report: dict[str, object] = {"config": str(args.config.resolve()), "splits": {}}
    for split in ("train", "dev"):
        sources = resolve_sources(data, split=split, project_root=project_root)
        missing = [str(path) for _, path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Не найдены файлы {split}: {missing}")
        documents = load_documents(sources)
        split_report = summarize(documents)
        split_report["files"] = {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sources
        }
        report["splits"][split] = split_report
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
