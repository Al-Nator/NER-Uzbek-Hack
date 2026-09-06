"""Проверки отбора аугментаций, provenance и парной серии 4."""

from copy import deepcopy
from pathlib import Path

import pytest

from uzner.config import load_experiment_config
from uzner.data.augmentation import (
    DIRECTIONS,
    METHODS,
    Candidate,
    load_candidates,
    prepare,
    select_unique,
    validate_candidate,
)
from uzner.data.io import read_jsonl
from uzner.domain import Document, Entity
from uzner.experiments.artifacts import write_jsonl
from uzner.experiments.series import load_series


def _row(method="corrected_converter", source="a", target="Али") -> dict:
    """Создаёт минимальную транслитерацию с точным отображением границ."""
    entity = {"label": "NAME", "start": 0, "end": 3}
    return {
        "hash": f"copy-{source}",
        "source_hash": source,
        "direction": "latn_to_cyrl",
        "variant": method,
        "source_text": "Ali",
        "text": target,
        "target_text": target,
        "entities": [entity],
        "source_entities": [entity],
        "target_entities": [{**entity, "surface": target}],
        "status": "ok",
        "alignments": [{"source_start": 0, "source_end": 3, "target_start": 0, "target_end": 3}],
    }


def test_candidate_preserves_offsets_and_provenance() -> None:
    """Копия имеет отдельный hash и сохраняет группу, направление и класс."""
    original = Document("a", "Ali", (Entity(0, 3, "NAME"),))
    candidate = validate_candidate(_row(), original, METHODS[0])
    assert candidate.document.hash != original.hash
    assert candidate.key == ("a", "latn_to_cyrl")
    assert candidate.to_mapping()["language_approved"] is False
    assert candidate.document.entities == original.entities


@pytest.mark.parametrize("damage", ["source", "label", "alignment", "surface", "end"])
def test_bad_offsets_and_source_are_rejected(damage) -> None:
    """Подмена исходника и повреждение координат не проходят в обучение."""
    row = deepcopy(_row())
    if damage == "source":
        row["source_text"] = "Bob"
    elif damage == "label":
        row["entities"][0]["label"] = "ORG"
    elif damage == "alignment":
        row["alignments"][0]["source_start"] = 1
    elif damage == "surface":
        row["target_entities"][0]["surface"] = "wrong"
    else:
        row["alignments"][0]["target_end"] = 9
    with pytest.raises(ValueError):
        validate_candidate(row, Document("a", "Ali", (Entity(0, 3, "NAME"),)), METHODS[0])


def test_pairwise_dedup_drops_both_sides() -> None:
    """Если дубль найден лишь у Luna, конвертер исключает тот же ключ."""
    pools = []
    for method, texts in zip(METHODS, [("AA", "BB", "dev"), ("CC", "CC", "DD")], strict=True):
        pool = {}
        for source, text in zip(("a", "b", "c"), texts, strict=True):
            c = Candidate(Document(source, text), source, DIRECTIONS[0], method, source)
            pool[c.key] = c
        pools.append(pool)
    selected, counts = select_unique(tuple(pools), {"dev"})
    assert selected == [("a", DIRECTIONS[0])]
    assert counts["duplicate_or_original_or_dev"] == 2


def test_release_excludes_review_unchanged_and_keeps_dev(tmp_path) -> None:
    """Фиксированный релиз даёт одинаковые ключи и не допускает перезапись."""
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    write_jsonl(
        train,
        ({"hash": key, "text": "Ali", "entities": _row()["entities"]} for key in ("a", "b", "c")),
    )
    write_jsonl(dev, [{"hash": "dev", "text": "Dev", "entities": []}])
    original_bytes = train.read_bytes(), dev.read_bytes()
    for method in METHODS:
        rows = [_row(method), _row(method, "b", "Ali"), _row(method, "c", "Боб")]
        rows[2]["status"] = "review"
        write_jsonl(tmp_path / f"{method}.{DIRECTIONS[0]}.jsonl", rows)
        write_jsonl(tmp_path / f"{method}.{DIRECTIONS[1]}.jsonl", [])
    output = tmp_path / "release"
    report = prepare(tmp_path, train, dev, output)
    assert report["matched"]["selected"] == 1
    assert report["source_stats"][METHODS[1]]["review_or_failed"] == 1
    assert report["source_stats"][METHODS[1]]["unchanged"] == 1
    assert (train.read_bytes(), dev.read_bytes()) == original_bytes
    assert read_jsonl(output / "luna_matched.jsonl")[0]["source_hash"] == "a"
    with pytest.raises(FileExistsError):
        prepare(tmp_path, train, dev, output)
    # Повторная запись одного направления не должна молча заменять кандидата.
    write_jsonl(tmp_path / f"{METHODS[0]}.{DIRECTIONS[0]}.jsonl", [_row(), _row()])
    with pytest.raises(ValueError, match="Дубликат"):
        load_candidates(tmp_path, {"a": Document("a", "Ali", (Entity(0, 3, "NAME"),))}, METHODS[0])


def test_fourth_series_changes_only_data_and_run_id() -> None:
    """Оба опыта повторяют s32 и отличаются только источником аугментаций."""
    root = Path("configs/experiments/a100")
    reference = load_experiment_config(root / "s32_bge_m3_retromae_global_pointer.yaml")
    series = load_series(Path("configs/series/fourth_a100.yaml"))
    configs = [load_experiment_config(Path(path)) for path in series.select("all")]
    assert [c.run_id[:3] for c in configs] == ["s40", "s41"]
    for config in configs:
        assert config.encoder == reference.encoder
        assert config.model == reference.model
        assert config.tokenization == reference.tokenization
        assert config.training == reference.training
        assert config.training.initial_checkpoint is None
    assert configs[0].data_config != configs[1].data_config
