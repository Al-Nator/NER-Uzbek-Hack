"""Проверки Sol-релиза, offsets и неизменности модельного рецепта."""

import json
from pathlib import Path

import pytest
import yaml

from uzner.data.augmentation_review import prepare_review
from uzner.experiments.artifacts import write_jsonl


def inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Создаёт минимальный кириллический reviewed-релиз с provenance."""
    train, dev, source, output = (tmp_path / name for name in ("train", "dev", "input", "out"))
    write_jsonl(
        train, [{"hash": "a", "text": "Ali", "entities": [{"label": "NAME", "start": 0, "end": 3}]}]
    )
    write_jsonl(dev, [{"hash": "d", "text": "dev", "entities": []}])
    write_jsonl(
        source,
        [
            {
                "hash": "t",
                "text": "Али",
                "source_text": "Ali",
                "source_hash": "a",
                "entities": [{"label": "NAME", "start": 0, "end": 3}],
                "variant": "luna_sol_medium_review",
                "direction": "latn_to_cyrl",
                "train_eligible": False,
                "human_validated": False,
                "alignments": [
                    {"source_start": 0, "source_end": 3, "target_start": 0, "target_end": 3}
                ],
            }
        ],
    )
    return source, train, dev, output


def test_review_valid_offsets_and_no_overwrite(tmp_path: Path) -> None:
    """Сохраняет Unicode offsets и не повышает synthetic до gold."""
    paths = inputs(tmp_path)
    manifest = prepare_review(*paths)
    assert manifest["rows"] == 1 and manifest["train_total"] == 2
    row = json.loads((paths[-1] / "sol_review.jsonl").read_text())
    assert row["text"][row["entities"][0]["start"] : row["entities"][0]["end"]] == "Али"
    assert row["source_hash"] == "a" and row["language_approved"] is False
    with pytest.raises(FileExistsError):
        prepare_review(*paths)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_hash", "unknown"),
        ("source_text", "other"),
        ("direction", "other"),
        ("variant", "other"),
        ("entities", [{"label": "NAME", "start": 1, "end": 3}]),
        ("alignments", [{"source_start": 0, "source_end": 2, "target_start": 0, "target_end": 3}]),
    ],
)
def test_review_rejects_corruption(tmp_path: Path, field: str, value: object) -> None:
    """Повреждённые источники и границы останавливают подготовку."""
    paths = inputs(tmp_path)
    row = json.loads(paths[0].read_text())
    row[field] = value
    write_jsonl(paths[0], [row])
    with pytest.raises(ValueError):
        prepare_review(*paths)


@pytest.mark.parametrize("mode", ["unchanged", "dev_duplicate", "duplicate_key"])
def test_review_excludes_duplicates(tmp_path: Path, mode: str) -> None:
    """Не допускает копии исходников, dev и повторное происхождение."""
    paths = inputs(tmp_path)
    row = json.loads(paths[0].read_text())
    if mode == "unchanged":
        row["text"] = "Ali"
    if mode == "dev_duplicate":
        write_jsonl(paths[2], [{"hash": "d", "text": "Али", "entities": []}])
    write_jsonl(paths[0], [row, row] if mode == "duplicate_key" else [row])
    with pytest.raises(ValueError):
        prepare_review(*paths)


def test_sol_uses_unchanged_s44_recipe() -> None:
    """В новом эксперименте меняются только идентификатор и данные."""
    root = Path(__file__).resolve().parents[1] / "configs/experiments/a100"
    reference = yaml.safe_load((root / "s44_bge_gp_translit_luna_full.yaml").read_text())[
        "experiment"
    ]
    candidate = yaml.safe_load((root / "s47_bge_gp_translit_sol_review.yaml").read_text())[
        "experiment"
    ]
    for field in ("run_id", "data_config"):
        reference.pop(field)
        candidate.pop(field)
    assert reference == candidate
