"""Контракты silver-абляции: происхождение, offsets и отсутствие утечек."""

from dataclasses import replace
from pathlib import Path

import pytest

from uzner.config import load_experiment_config
from uzner.data.curation.leakage import digest
from uzner.data.io import read_jsonl, sha256_file
from uzner.data.silver_experiment import SilverPreparation, prepare_silver
from uzner.experiments.artifacts import write_json, write_jsonl


def candidate(identifier="a", text="Али", entities=None):
    """Создаёт пример с исходными флагами непроверенной разметки."""
    return {
        "hash": identifier,
        "source_hash": "ext:" + identifier,
        "text": text,
        "entities": entities if entities is not None else [],
        "annotation_status": "two_pass_exact_agreement",
        "kind": "pseudo",
        "train_eligible": False,
        "release_gate": "pilot_and_rights_review_required",
        "human_validated": False,
        "meta": {"source": "test", "content_sha256": digest(text), "rights_approved": False},
    }


def fixture_request(tmp_path, rows):
    """Создаёт изолированный релиз и неизменяемые official split-ы."""
    release = tmp_path / "release"
    write_jsonl(release / "silver_candidates.jsonl", rows)
    write_json(
        release / "summary.json",
        {
            "silver_candidates": len(rows),
            "output_sha256": {
                "silver_candidates.jsonl": sha256_file(release / "silver_candidates.jsonl")
            },
        },
    )
    official = tmp_path / "official"
    for split in ("train", "dev"):
        write_jsonl(
            official / f"{split}.jsonl",
            [
                {
                    "hash": split,
                    "text": split + " official document",
                    "entities": [],
                }
            ],
        )
    return SilverPreparation(release, official, tmp_path / "output", True)


def test_filter_and_preserve(tmp_path):
    """Исключает утечки и дубли, сохраняя Unicode, пустую разметку и gate."""
    valid = candidate(entities=[{"start": 0, "end": 3, "label": "NAME"}])
    rows = [
        valid,
        candidate("b", "Пустой пример"),
        candidate("copy"),
        candidate("leak", "dev official document"),
        candidate("train", "Новое"),
        candidate("empty", " "),
    ]
    request = fixture_request(tmp_path, rows)
    original = (request.official_root / "dev.jsonl").read_bytes()
    result = prepare_silver(request)
    assert result.accepted == 2 and result.empty == 1
    assert result.excluded == {
        "duplicate": 1,
        "official_dev_exact": 1,
        "official_identifier": 1,
        "empty_text": 1,
    }
    assert read_jsonl(request.output / "silver.jsonl") == rows[:2]
    assert original == (request.official_root / "dev.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_silver(request)


@pytest.mark.parametrize("damage", ["checksum", "offset", "text", "status", "count", "empty"])
def test_invalid_release(tmp_path, damage):
    """Отказывает при повреждении, отсутствии согласования и пустой выборке."""
    row = candidate()
    if damage == "offset":
        row["entities"] = [{"start": 0, "end": 99, "label": "NAME"}]
    elif damage == "text":
        row["meta"]["content_sha256"] = "incorrect"
    elif damage == "status":
        row["annotation_status"] = "disagreement"
    request = fixture_request(tmp_path, [] if damage == "empty" else [row])
    if damage == "checksum":
        write_jsonl(request.release / "silver_candidates.jsonl", [candidate("changed")])
    elif damage == "count":
        write_json(
            request.release / "summary.json",
            {
                "silver_candidates": 99,
                "output_sha256": {
                    "silver_candidates.jsonl": sha256_file(
                        request.release / "silver_candidates.jsonl"
                    )
                },
            },
        )
    with pytest.raises(ValueError):
        prepare_silver(request)
    assert not request.output.exists()


def test_requires_explicit_research_authorization(tmp_path):
    """Автоматический импорт gated-релиза запрещён без исследовательского разрешения."""
    request = fixture_request(tmp_path, [candidate()])
    with pytest.raises(ValueError, match="разрешение"):
        prepare_silver(replace(request, research_authorized=False))


@pytest.mark.parametrize("name", ["s45_bge_gp_external_silver", "s46_bge_gp_silver_luna"])
def test_only_data_factor_changes(name):
    """Модель, оптимизация и токенизация полностью совпадают с s32."""
    base = load_experiment_config(
        Path("configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml")
    )
    new = load_experiment_config(Path(f"configs/experiments/a100/{name}.yaml"))
    assert new.encoder == base.encoder
    assert new.model == base.model
    assert new.training == base.training
    assert new.tokenization == base.tokenization
