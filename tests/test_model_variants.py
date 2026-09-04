"""Одношаговые smoke-тесты всех голов первой серии."""

from dataclasses import replace
from pathlib import Path

import pytest
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tests.helpers import FakeTokenizer, TinyEncoder, sample_documents
from uzner.config import ModelConfig, TokenizationConfig, TrainingConfig, load_experiment_config
from uzner.data.windows import build_window_features
from uzner.experiments.artifacts import prepare_run_paths
from uzner.experiments.logging import RunLogger
from uzner.models.token_tagger import TokenTagger
from uzner.training.data_setup import make_loader
from uzner.training.inference import run_inference
from uzner.training.loop import train_epoch
from uzner.training.runtime import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.train
@pytest.mark.parametrize(
    ("scheme", "head", "decoder"),
    (
        ("bio", "softmax", "greedy"),
        ("bioes", "softmax", "constrained"),
        ("bio", "crf", "crf"),
        ("bioes", "crf", "crf"),
    ),
)
def test_model_variant_trains_and_evaluates_one_step(
    tmp_path: Path,
    scheme: str,
    head: str,
    decoder: str,
) -> None:
    """BIO, BIOES, softmax и CRF проходят train и exact evaluation."""
    model_config = ModelConfig("token_tagging", scheme, head, decoder, 0.0)
    training = TrainingConfig(
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        bf16=True,
        num_workers=0,
        require_gpu=False,
        log_every_steps=1,
        early_stopping_patience=0,
    )
    documents = sample_documents()
    tokenizer = FakeTokenizer()
    features = build_window_features(
        documents,
        tokenizer,
        TokenizationConfig(max_length=8, stride=2),
        model_config.tag_scheme,
        with_labels=True,
    )
    loader = make_loader(
        features,
        tokenizer,
        batch_size=2,
        shuffle=False,
        seed=42,
        num_workers=0,
    )
    device = resolve_device(False)
    model = TokenTagger(TinyEncoder(), model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=training.learning_rate)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    base = load_experiment_config(PROJECT_ROOT / "configs/experiments/e10_xlmr_base_bio.yaml")
    config = replace(
        base,
        run_id=f"smoke_{scheme}_{head}",
        output_root=str(tmp_path),
        model=model_config,
        training=training,
    )
    paths = prepare_run_paths(config, project_root=tmp_path)
    logger = RunLogger(config.run_id, paths)

    trained = train_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        device,
        training,
        logger,
        epoch=1,
        global_step=0,
    )
    inference = run_inference(
        model,
        loader,
        features,
        documents,
        documents,
        device,
        bf16=training.bf16,
    )

    assert trained.global_step == 1
    assert trained.loss >= 0
    assert inference.evaluation.overall.records == len(documents)
    assert len(inference.predictions) == len(documents)
