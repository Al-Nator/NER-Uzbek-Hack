"""Проверки переноса выбранного posthoc-рецепта на неразмеченный public."""

import json
from dataclasses import asdict, replace

import pytest

from uzner.data.io import read_jsonl, sha256_file, write_predictions
from uzner.domain import Entity, Prediction
from uzner.experiments.artifacts import write_json, write_jsonl
from uzner.posthoc.config import Variant
from uzner.posthoc.submission import PosthocSubmission, build_submission


@pytest.fixture
def submission_request(tmp_path):
    """Создаёт независимый public с маленьким train-словарём."""
    train, public, base = (tmp_path / n for n in ("train.jsonl", "public.jsonl", "base.jsonl"))
    write_jsonl(
        train,
        [
            {"hash": str(i), "text": "Toshkent", "entities": [Entity(0, 8, "GEO").to_mapping()]}
            for i in range(3)
        ],
    )
    write_jsonl(public, [{"hash": "p", "text": "Toshkent. Toshkent"}])
    write_predictions(base, (Prediction("p"),))
    selected = tmp_path / "selected"
    variant = Variant("lex", "lex_add", normalized=True, repeat_after=True)
    write_json(selected / "resolved_config.json", asdict(variant))
    write_json(selected / "metadata.json", {"inputs": {str(train): sha256_file(train)}})
    return PosthocSubmission(selected, train, public, base, tmp_path / "output")


def test_public_contract_and_no_overwrite(submission_request):
    """Все публичные строки содержат только hash и валидные spans; повторная запись запрещена."""
    request = submission_request
    output = build_submission(request)
    records = read_jsonl(output)
    assert records[0]["hash"] == "p" and len(records[0]["entities"]) == 2
    assert set(records[0]) == {"hash", "entities"}
    assert records[0]["entities"][1]["start"] == 10
    with pytest.raises(FileExistsError):
        build_submission(request)


def test_changed_train_rejected(submission_request):
    """Замена train после выбора рецепта не допускается."""
    request = submission_request
    write_jsonl(request.train, [{"hash": "new", "text": "X"}])
    with pytest.raises(ValueError, match="тому же train"):
        build_submission(request)


def test_explicit_dev_dictionary_and_unicode(submission_request):
    """Dev расширяет словарь только явно, без обучения и фиктивной dev-метрики."""
    request = submission_request
    dev = request.train.with_name("dev.jsonl")
    write_jsonl(
        dev,
        [
            {"hash": f"d{i}", "text": "Қўқон", "entities": [Entity(0, 5, "GEO").to_mapping()]}
            for i in range(3)
        ],
    )
    write_jsonl(request.input_path, [{"hash": "p", "text": "Toshkent. Қўқон Қўқон"}])
    control = read_jsonl(build_submission(request))[0]
    assert len(control["entities"]) == 1
    extended = replace(request, dictionary_dev=dev, output_dir=dev.parent / "extended")
    output = build_submission(extended)
    entities = read_jsonl(output)[0]["entities"]
    assert [(e["start"], e["end"]) for e in entities] == [(0, 8), (10, 15), (16, 21)]
    manifest = json.loads(output.with_name("predictions.manifest.json").read_text("utf-8"))
    assert manifest["dictionary_sources"][str(dev)] == sha256_file(dev)
    assert manifest["dictionary_documents"] == 6
    assert manifest["dictionary_dev_used"] is True
    assert manifest["training"] is False
    assert manifest["public_gold_used"] is False
    assert manifest["dev_micro_f1"] is None
    assert manifest["public_micro_f1"] is None


@pytest.mark.parametrize("collision", ["hash", "text", "train_hash"])
def test_dev_dictionary_leakage_rejected(submission_request, collision):
    """Запрещает public в словаре и повторные train/dev идентификаторы."""
    request = submission_request
    dev = request.train.with_name("dev.jsonl")
    row = {"hash": "d", "text": "Ali", "entities": []}
    if collision == "hash":
        row["hash"] = "p"
    elif collision == "text":
        row["text"] = "Toshkent. Toshkent"
    else:
        row["hash"] = "0"
    write_jsonl(dev, [row])
    with pytest.raises(ValueError, match="пересекаются|повторяется"):
        build_submission(replace(request, dictionary_dev=dev))
    assert not request.output_dir.exists()


def test_gold_and_empty_public_rejected(submission_request):
    """Нельзя случайно использовать gold или пустой файл как публичный тест."""
    request = submission_request
    write_jsonl(
        request.input_path,
        [{"hash": "p", "text": "Ali", "entities": [Entity(0, 3, "NAME").to_mapping()]}],
    )
    with pytest.raises(ValueError, match="gold"):
        build_submission(request)
    write_jsonl(request.input_path, [])
    with pytest.raises(ValueError, match="Пустой"):
        build_submission(request)


@pytest.mark.parametrize("operation", ["repeat", "boundary", "short", "lex_relabel", "confirm"])
def test_operation_guard(submission_request, operation):
    """Модельные правила требуют отдельного пути, а текстовые исполняются без модели."""
    request = submission_request
    variant = Variant("v", operation)
    write_json(request.selected_run / "resolved_config.json", asdict(variant))
    if operation == "confirm":
        with pytest.raises(ValueError, match="модельные"):
            build_submission(request)
    else:
        assert build_submission(
            replace(request, output_dir=request.output_dir / operation)
        ).is_file()
