"""Проверки безопасного отбора checkpoint-ов для удаления."""

import json

from scripts.prune_low_checkpoints import candidates


def test_only_completed_low_scores(tmp_path):
    """Сильные, активные, неизвестные и защищённые runs остаются нетронутыми."""
    for name, score, status in [
        ("low", 0.89, "complete"),
        ("high", 0.9, "complete"),
        ("active", 0.1, "running"),
        ("unknown", None, "complete"),
        ("protected", 0.8, "complete"),
    ]:
        run = tmp_path / name
        (run / "checkpoints").mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"best_micro_f1": score, "status": status}))
    result = candidates(tmp_path, {"protected"})
    assert len(result) == 1
    assert result[0].path == str(tmp_path / "low/checkpoints")
