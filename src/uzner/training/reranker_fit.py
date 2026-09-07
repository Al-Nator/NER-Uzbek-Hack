"""Обучение reranker на holdout-кандидатах с отдельным выбором по meta-valid."""

import copy
import time
from dataclasses import asdict, dataclass

import torch

from uzner.data.reranker_candidates import SpanCandidate, candidate_targets, candidates_from_sources
from uzner.domain import Document, Entity, Prediction
from uzner.evaluation.metrics import evaluate_predictions
from uzner.evaluation.span_vote import majority_vote
from uzner.experiments.research_run import ResearchRun
from uzner.models.span_reranker import RerankerConfig, SpanReranker
from uzner.training.runtime import set_reproducible_seed
from uzner.training.span_prediction import select_flat_entities


@dataclass(frozen=True)
class RerankerDataset:
    """Кандидаты и цели одного split; признаки не содержат gold."""

    documents: tuple[Document, ...]
    sources: tuple[tuple[Prediction, ...], ...]
    candidates: tuple[SpanCandidate, ...]
    features: torch.Tensor
    characters: torch.Tensor
    targets: torch.Tensor


def make_reranker_dataset(
    documents: tuple[Document, ...], sources: tuple[tuple[Prediction, ...], ...]
) -> RerankerDataset:
    """Отделяет построение gold-free кандидатов от разметки supervised-целей."""
    unlabeled = tuple(Document(d.hash, d.text) for d in documents)
    candidates = candidates_from_sources(unlabeled, sources)
    if not candidates:
        raise ValueError("Нет кандидатов для reranker")
    return RerankerDataset(
        documents,
        sources,
        candidates,
        torch.tensor([c.features for c in candidates], dtype=torch.float32),
        torch.tensor([c.characters for c in candidates], dtype=torch.long),
        torch.tensor(candidate_targets(candidates, documents), dtype=torch.float32),
    )


def decode_candidates(
    dataset: RerankerDataset, probabilities: torch.Tensor, threshold: float
) -> tuple[Prediction, ...]:
    """Отбирает exact-кандидатов и применяет общий плоский decoder."""
    grouped = [[] for _ in dataset.documents]
    for candidate, score in zip(dataset.candidates, probabilities.tolist(), strict=True):
        if score > threshold:
            e = candidate.entity
            grouped[candidate.document_index].append(Entity(e.start, e.end, e.label, score))
    return tuple(
        Prediction(d.hash, select_flat_entities(entities))
        for d, entities in zip(dataset.documents, grouped, strict=True)
    )


@torch.inference_mode()
def score_candidates(
    model: SpanReranker, dataset: RerankerDataset, batch_size: int
) -> torch.Tensor:
    """Возвращает вероятности без обучения и без обращения к targets."""
    model.eval()
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(dataset.candidates), batch_size):
        result.append(
            model(
                dataset.features[start : start + batch_size].to(device),
                dataset.characters[start : start + batch_size].to(device),
            )
            .sigmoid()
            .cpu()
        )
    return torch.cat(result)


def fit_reranker(
    config: RerankerConfig,
    train: RerankerDataset,
    valid: RerankerDataset,
    evaluations: dict[str, RerankerDataset],
    reference_train: tuple[Document, ...],
    run: ResearchRun,
    device: torch.device,
) -> None:
    """Выбирает эпоху/порог на meta-valid, после чего единожды оценивает official dev."""
    set_reproducible_seed(config.seed)
    model = SpanReranker(train.features.shape[1], config.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    checkpoints = run.root / "checkpoints"
    checkpoints.mkdir()
    best, selected_threshold, best_state, best_epoch, step = -1.0, 0.5, None, 0, 0
    for name, dataset in {"meta_valid": valid, **evaluations}.items():
        run.evaluate(
            dataset.documents,
            majority_vote(dataset.documents, dataset.sources),
            reference_train,
            f"reference_{name}",
        )
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train.candidates))
        total_loss = 0.0
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.features[indices].to(device), train.characters[indices].to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train.targets[indices].to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Неконечный reranker loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            step += 1
            if step % 10 == 0:
                run.log(
                    {
                        "train/loss": float(loss.detach()),
                        "train/gradient_norm": float(norm),
                        "train/lr": config.learning_rate,
                        "train/epoch": epoch,
                    },
                    step,
                )
        probabilities = score_candidates(model, valid, config.batch_size)
        options = []
        for threshold in config.thresholds:
            predictions = decode_candidates(valid, probabilities, threshold)
            score = evaluate_predictions(valid.documents, predictions).micro.f1
            options.append((score, -abs(threshold - 0.5), threshold))
            run.log({f"meta_valid/f1_threshold_{threshold}": score}, epoch)
        score, _, threshold = max(options)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "threshold": threshold,
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "selection": "meta_valid exact micro F1; official dev not used",
        }
        torch.save(payload, checkpoints / "last.pt")
        if score > best:
            best, selected_threshold, best_epoch = score, threshold, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(payload, checkpoints / "best.pt")
        run.log(
            {
                "epoch/train_loss": total_loss / len(train.candidates),
                "epoch/meta_valid_f1": score,
                "epoch/seconds": time.perf_counter() - started,
                "epoch/threshold": threshold,
            },
            epoch,
        )
    model.load_state_dict(best_state)
    run.log(
        {
            "selected/epoch": best_epoch,
            "selected/threshold": selected_threshold,
            "selected/meta_valid_f1": best,
        }
    )
    for name, dataset in {"meta_valid": valid, **evaluations}.items():
        started = time.perf_counter()
        predictions = decode_candidates(
            dataset, score_candidates(model, dataset, config.batch_size), selected_threshold
        )
        run.log({f"timing/{name}_reranker_only_seconds": time.perf_counter() - started})
        run.evaluate(dataset.documents, predictions, reference_train, name)
