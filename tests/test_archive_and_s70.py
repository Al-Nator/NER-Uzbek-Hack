"""Проверки точности архивирования и повторной сборки s70 из кэша."""

import json
from pathlib import Path

import pytest

from scripts.build_s70_submission import build
from scripts.finish_a100_archive import validate_targets
from uzner.data.io import sha256_file
from uzner.experiments.remote_archive import assert_equal, inventory


def test_inventory_detects_mutation_and_missing_file(tmp_path: Path) -> None:
    """Две сверки не принимают изменённый или пропавший файл."""
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    first = inventory(tmp_path)
    assert_equal(first, inventory(tmp_path))
    path.write_bytes(b"abd")
    with pytest.raises(ValueError):
        assert_equal(first, inventory(tmp_path))
    path.unlink()
    with pytest.raises(ValueError):
        assert_equal(first, inventory(tmp_path))


def test_inventory_rejects_symlink(tmp_path: Path) -> None:
    """Не позволяет незаметно архивировать ссылку вместо реальных весов."""
    (tmp_path / "link").symlink_to("/etc/hostname")
    with pytest.raises(ValueError):
        inventory(tmp_path)


@pytest.mark.parametrize("files", [{}, {"../x": []}, {"/tmp/x": []}, {"other/x": []}])
def test_delete_target_guards(files: dict) -> None:
    """Удаление не принимает корни, traversal и не-experiment каталоги."""
    with pytest.raises(ValueError):
        validate_targets(files)


def test_delete_targets_are_explicit() -> None:
    """Для очистки разрешаются только отдельные проверенные experiment имена."""
    assert validate_targets({"s30_old/a": [], "f01_final/b": []}) == ["f01_final", "s30_old"]


def test_s70_cache_reproducible(tmp_path: Path) -> None:
    """Реальный сохранённый кэш даёт ровно тот же leaderboard JSONL."""
    root = Path(__file__).resolve().parents[1]
    input_path = Path("/home/danya/Загрузки/public_test_inputs.jsonl")
    reference = root / "artifacts/submissions/s70_public_v1/predictions.jsonl"
    if not input_path.exists() or not reference.exists():
        pytest.skip("Локальные артефакты public test отсутствуют")
    result = build(root, input_path, tmp_path / "submission")
    assert sha256_file(result) == sha256_file(reference)
    manifest = json.loads(result.with_suffix(".manifest.json").read_text())
    assert manifest["documents"] == 1000 and manifest["entities"] == 4375
    with pytest.raises(FileExistsError):
        build(root, input_path, tmp_path / "submission")
    invalid = tmp_path / "other.jsonl"
    invalid.write_text("{}\n")
    with pytest.raises(ValueError, match="другому public test"):
        build(root, invalid, tmp_path / "bad")
