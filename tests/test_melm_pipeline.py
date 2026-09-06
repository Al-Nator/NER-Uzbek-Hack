"""Проверки генератора/релиза MELM без скачивания моделей и использования GPU."""

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from uzner.data.melm import make_examples, replace_entity
from uzner.domain import Document, Entity, Prediction
from uzner.training import melm_generator, melm_pipeline
from uzner.training.melm_generator import MelmConfig


class TinyTokenizer:
    """Имитирует необходимые операции tokenizer-а на небольшом словаре."""

    bos_token_id, eos_token_id, pad_token_id, mask_token_id = 0, 2, 1, 18

    def __init__(self):
        """Создаёт словарь тестовых слов и будущих маркеров."""
        self.vocab = {"Ali": 3, "Vali": 4, "said": 5, "person": 6, "organization": 7, "location": 8}

    def encode(self, text, *, add_special_tokens=False):
        """Разбивает по словам без внешнего tokenizer-а."""
        return [self.vocab.get(w, 9) for w in text.split()]

    def add_tokens(self, markers):
        """Добавляет шесть классовых маркеров."""
        self.vocab.update({m: i + 10 for i, m in enumerate(markers)})

    def convert_tokens_to_ids(self, marker):
        """Возвращает ID маркера."""
        return self.vocab.get(marker, 10)

    def __len__(self):
        """Возвращает размер допустимого словаря."""
        return 32

    def decode(self, ids, *, skip_special_tokens=False):
        """Возвращает новое имя для проверки генерации кандидатов."""
        return "Jamshid"

    def save_pretrained(self, path):
        """Пишет тестовый tokenizer рядом с генератором."""
        path.mkdir(exist_ok=True)
        (path / "tokenizer.json").write_text("{}")


class TinyMlm(nn.Module):
    """Маленькая обучаемая модель для настоящего backward в unit-тесте."""

    def __init__(self):
        """Создаёт embedding и LM-проекцию."""
        super().__init__()
        self.emb = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)

    def get_input_embeddings(self):
        """Открывает embedding для semantic initialization маркеров."""
        return self.emb

    def resize_token_embeddings(self, size):
        """Проверяет заранее выделенный размер тестового словаря."""
        assert size == 32

    def forward(self, input_ids, attention_mask, labels=None):
        """Считает MLM loss с игнорированием контекста."""
        logits = self.head(self.emb(input_ids))
        loss = (
            None
            if labels is None
            else nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        )
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, path, **kwargs):
        """Сохраняет веса тестового генератора."""
        path.mkdir()
        torch.save(self.state_dict(), path / "model.pt")


def test_train_and_generate_melm_on_cpu(tmp_path, monkeypatch):
    """Проверяет markers, MLM backward, checkpoint и generation с новыми offsets."""
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(
        melm_generator, "resolve_pretrained_snapshot", lambda config: SimpleNamespace(path=tmp_path)
    )
    monkeypatch.setattr(melm_generator.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(
        melm_generator.AutoModelForMaskedLM, "from_pretrained", lambda *a, **kw: TinyMlm()
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    docs = (Document("x", "Ali Vali said", (Entity(0, 3, "NAME"), Entity(4, 8, "GEO"))),)
    config = replace(MelmConfig(), epochs=1, batch_size=1, max_candidates=1)
    model, tok, examples = melm_generator.train_generator(
        docs, tmp_path, config, melm_pipeline.SilentTracker()
    )
    assert len(examples) == 2
    assert (tmp_path / "generator/model.pt").is_file()
    assert all(
        tok.convert_tokens_to_ids("<B-NAME>") not in [e.input_ids[p] for p in e.entity_positions]
        for e in examples
    )
    generated = melm_generator.generate(model, tok, examples, docs, config)
    assert len(generated) == 1
    assert "Jamshid" in generated[0][0].text
    assert generated[0][1] == "x"
    assert make_examples((Document("empty", "text"),), tok) == ()


class AgreementPredictor:
    """Имитирует согласие frozen NER только с изменённым упоминанием."""

    def __init__(self, source):
        """Не читает реальные веса."""

    def predict(self, docs):
        """Возвращает точные candidate spans для проверки фильтра."""
        return tuple(Prediction(d.hash, d.entities) for d in docs)


@pytest.mark.parametrize("smoke,count", [(True, 4), (False, 501), (False, 1)])
def test_melm_release_filter_and_minimum(tmp_path, monkeypatch, smoke, count):
    """Фиксирует provenance и блокирует NER при недостаточном числе новых копий."""
    source = tmp_path / "source"
    head = source / "checkpoints/best/head.safetensors"
    head.parent.mkdir(parents=True)
    head.write_text("test")
    original = Document("train", "Ali came", (Entity(0, 3, "NAME"),))
    candidates = [
        (replace_entity(original, 0, f"Jamshid{i}", variant=str(i)), original.hash, 0)
        for i in range(count)
    ]
    monkeypatch.setattr(melm_pipeline, "load_documents", lambda sources: (original,))
    monkeypatch.setattr(melm_pipeline, "train_generator", lambda *a, **kw: (object(), object(), ()))
    monkeypatch.setattr(melm_pipeline, "generate", lambda *a, **kw: candidates)
    monkeypatch.setattr(melm_pipeline, "FrozenPredictor", AgreementPredictor)
    ended = []
    for name in (
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "log_metrics",
        "log_artifact",
    ):
        monkeypatch.setattr(melm_pipeline.mlflow, name, lambda *a, **kw: None)
    monkeypatch.setattr(
        melm_pipeline.mlflow, "end_run", lambda **kw: ended.append(kw.get("status", "FINISHED"))
    )
    output = tmp_path / "release"
    if not smoke and count < 500:
        with pytest.raises(RuntimeError):
            melm_pipeline.prepare_melm(output, source, MelmConfig(), smoke=smoke)
        assert ended == ["FAILED"]
        assert not (output / "train_augmented.jsonl").exists()
    else:
        path = melm_pipeline.prepare_melm(output, source, MelmConfig(), smoke=smoke)
        records = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(records) == count
        assert all(r["source_hash"] == "train" and not r["language_approved"] for r in records)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "complete" and not manifest["selection_uses_dev_labels"]
        assert ended == ([] if smoke else ["FINISHED"])
    with pytest.raises(FileExistsError):
        melm_pipeline.prepare_melm(output, source, MelmConfig(), smoke=smoke)
