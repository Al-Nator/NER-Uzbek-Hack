"""Интеграционные тесты train, inference, checkpoint и resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from uzner.config import load_experiment_config
from uzner.models.token_tagger import TokenTagger
from uzner.training.checkpoint import (
    TrainerState,
    load_model_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from uzner.training.engine import TrainRequest, train_experiment


def _save_tiny_hf_assets(path: Path) -> None:
    """Создаёт крошечные BERT и fast tokenizer без сети."""
    vocabulary = {
        token: index
        for index, token in enumerate(
            (
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "Ali",
                "ACME",
                "Toshkent",
                "Samarqand",
                "bordi",
            )
        )
    }
    backend = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=(("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )
    model = BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    path.mkdir(parents=True)
    tokenizer.save_pretrained(path)
    model.save_pretrained(path, safe_serialization=True)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Пишет тестовые документы по одному на строку."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _tiny_project(root: Path, *, epochs: int = 1) -> Path:
    """Собирает самодостаточный локальный experiment project."""
    model_path = root / "tiny_model"
    _save_tiny_hf_assets(model_path)
    train = [
        {
            "hash": "train-1",
            "text": "Ali ACME bordi",
            "entities": [
                {"label": "NAME", "start": 0, "end": 3},
                {"label": "ORG", "start": 4, "end": 8},
            ],
        },
        {
            "hash": "train-2",
            "text": "Toshkent Samarqand",
            "entities": [
                {"label": "GEO", "start": 0, "end": 8},
                {"label": "GEO", "start": 9, "end": 18},
            ],
        },
    ]
    dev = [
        {
            "hash": "dev-1",
            "text": "Ali Toshkent",
            "entities": [
                {"label": "NAME", "start": 0, "end": 3},
                {"label": "GEO", "start": 4, "end": 12},
            ],
        },
        {
            "hash": "dev-2",
            "text": "ACME Samarqand",
            "entities": [
                {"label": "ORG", "start": 0, "end": 4},
                {"label": "GEO", "start": 5, "end": 14},
            ],
        },
    ]
    _write_jsonl(root / "data/train.jsonl", train)
    _write_jsonl(root / "data/dev.jsonl", dev)
    data = {
        "data": {
            "sources": [
                {"name": "train", "path": "data/train.jsonl", "split": "train"},
                {"name": "dev", "path": "data/dev.jsonl", "split": "dev"},
            ]
        }
    }
    experiment = {
        "experiment": {
            "run_id": "tiny_run",
            "pipeline": "uzner",
            "data_config": "configs/data.yaml",
            "output_root": "runs",
            "primary_metric": "exact_micro_f1",
            "encoder": {"name": str(model_path), "revision": "local"},
            "model": {
                "architecture": "token_tagging",
                "tag_scheme": "bio",
                "head": "softmax",
                "decoder": "constrained",
                "dropout": 0.0,
            },
            "tokenization": {"max_length": 16, "stride": 4},
            "training": {
                "seed": 42,
                "epochs": epochs,
                "batch_size": 2,
                "eval_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_ratio": 0.0,
                "max_grad_norm": 1.0,
                "bf16": False,
                "num_workers": 0,
                "require_gpu": False,
                "gradient_checkpointing": False,
                "log_every_steps": 1,
                "early_stopping_patience": 0,
            },
        }
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    config_path = root / "configs/tiny.yaml"
    config_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    (root / "uv.lock").write_text("test lock", encoding="utf-8")
    return config_path


@pytest.mark.train
def test_full_train_artifacts_and_resume_are_isolated(tmp_path: Path) -> None:
    """Полный CPU-smoke создаёт run, checkpoint и продолжается."""
    config_path = _tiny_project(tmp_path)
    smoke_root = tmp_path / "isolated_smoke"
    request = TrainRequest(
        config_path=config_path,
        project_root=tmp_path,
        publish_summary=False,
        output_root_override=smoke_root,
    )
    first = train_experiment(request)

    assert first.paths.status.is_file()
    assert first.paths.best_checkpoint.is_dir()
    assert first.paths.last_checkpoint.is_dir()
    assert first.paths.manifest.is_file()
    assert first.paths.predictions.is_file()
    assert not (tmp_path / "reports/experiments.csv").exists()

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["training"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    resumed = train_experiment(
        TrainRequest(
            config_path=config_path,
            project_root=tmp_path,
            resume=True,
            publish_summary=False,
            output_root_override=smoke_root,
        )
    )
    history = list(csv_line for csv_line in resumed.paths.history.read_text().splitlines())
    assert len(history) == 3
    assert json.loads(resumed.paths.status.read_text(encoding="utf-8"))["status"] == "complete"


@pytest.mark.train
def test_checkpoint_roundtrip_and_state_restore(tmp_path: Path) -> None:
    """Checkpoint дважды атомарно пишется и восстанавлиает state."""
    config_path = _tiny_project(tmp_path)
    config = load_experiment_config(config_path)
    encoder = BertModel.from_pretrained(tmp_path / "tiny_model", local_files_only=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tmp_path / "tiny_model", local_files_only=True
    )
    model = TokenTagger(encoder, config.model)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    state = TrainerState(1, 2, 0.5, 1, 0)
    checkpoint = tmp_path / "checkpoint"

    save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, state, config)
    save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, state, config)
    loaded = load_model_checkpoint(checkpoint, config, torch.device("cpu"))
    restore_training_state(checkpoint, optimizer, scheduler)

    assert loaded.state == state
    assert loaded.model.tags == model.tags
    assert loaded.tokenizer.is_fast
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(tmp_path / "missing", config, torch.device("cpu"))
