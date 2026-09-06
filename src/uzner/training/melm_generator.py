"""Обучение и генерация MELM-inspired без доступа к dev-разметке."""

import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from uzner.config import EncoderConfig
from uzner.data.melm import MelmExample, make_examples, preserve_suffix, replace_entity
from uzner.domain import Document
from uzner.models.pretrained import resolve_pretrained_snapshot


@dataclass(frozen=True)
class MelmConfig:
    """Фиксированный пилот entity-only MLM; это не буквальное воспроизведение статьи."""

    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    max_length: int = 256
    learning_rate: float = 1e-5
    mask_rate: float = 0.7
    rounds: int = 3
    topk: int = 5
    max_candidates: int = 12000
    max_accepted: int = 4803
    encoder: str = "FacebookAI/xlm-roberta-base"
    revision: str = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"


def masked_batch(
    examples: list[MelmExample], tokenizer: object, rate: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Маскирует только entity-позиции, гарантируя хотя бы одну цель на пример."""
    width = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), width), tokenizer.pad_token_id, dtype=torch.long)
    target = torch.full_like(ids, -100)
    for row, example in enumerate(examples):
        ids[row, : len(example.input_ids)] = torch.tensor(example.input_ids)
        positions = torch.tensor(example.entity_positions)
        chosen = positions[torch.rand(len(positions)) < rate]
        if not len(chosen):
            chosen = positions[torch.randint(len(positions), (1,))]
        target[row, chosen] = ids[row, chosen]
        ids[row, chosen] = tokenizer.mask_token_id
    return ids, ids != tokenizer.pad_token_id, target


def train_generator(
    documents: tuple[Document, ...], output: Path, config: MelmConfig, tracker: object
) -> tuple[object, object, tuple[MelmExample, ...]]:
    """Обучает MLM на выбранных train-упоминаниях и сохраняет единственный checkpoint."""
    snapshot = resolve_pretrained_snapshot(EncoderConfig(config.encoder, config.revision))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path, local_files_only=True, fix_mistral_regex=False
    )
    model = AutoModelForMaskedLM.from_pretrained(snapshot.path, local_files_only=True).cuda()
    markers = [f"<{prefix}-{label}>" for label in ("ORG", "NAME", "GEO") for prefix in ("B", "I")]
    semantic = {
        label: tokenizer.encode(word, add_special_tokens=False)
        for label, word in [("ORG", "organization"), ("NAME", "person"), ("GEO", "location")]
    }
    tokenizer.add_tokens(markers)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        for marker in markers:
            emb[tokenizer.convert_tokens_to_ids(marker)] = emb[semantic[marker[3:-1]]].mean(0)
    examples = make_examples(documents, tokenizer, max_length=config.max_length, seed=config.seed)
    if not examples:
        raise ValueError("Нет MELM train-примеров")
    tracker.log_params({"examples": len(examples), "source_documents": len(documents)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps = math.ceil(len(examples) / config.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(config.seed)
    step = 0
    model.train()
    for epoch in range(config.epochs):
        order = list(examples)
        rng.shuffle(order)
        total = 0.0
        for offset in range(0, len(order), config.batch_size):
            ids, mask, labels = (
                x.cuda()
                for x in masked_batch(
                    order[offset : offset + config.batch_size], tokenizer, config.mask_rate
                )
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Неконечный MELM loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            total += loss.item()
            if step % 25 == 0:
                tracker.log_metrics(
                    {"train/mlm_loss": loss.item(), "optimizer/lr": scheduler.get_last_lr()[0]},
                    step=step,
                )
                print(
                    f"MELM epoch={epoch + 1} step={step}/{steps} loss={loss.item():.4f}", flush=True
                )
        tracker.log_metrics(
            {"epoch/mlm_loss": total / math.ceil(len(order) / config.batch_size)}, step=epoch + 1
        )
    model.save_pretrained(output / "generator", safe_serialization=True)
    tokenizer.save_pretrained(output / "generator")
    del optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return model.eval(), tokenizer, examples


@torch.inference_mode()
def generate(
    model: object,
    tokenizer: object,
    examples: tuple[MelmExample, ...],
    documents: tuple[Document, ...],
    config: MelmConfig,
) -> list[tuple[Document, str, int]]:
    """Выбирает top-2…top-k токены и возвращает кандидатов с provenance."""
    originals = {d.hash: d for d in documents}
    candidates, seen = [], set()
    rng = random.Random(config.seed)
    for round_index in range(config.rounds):
        order = list(examples)
        rng.shuffle(order)
        for offset in range(0, len(order), config.batch_size):
            batch = order[offset : offset + config.batch_size]
            ids, mask, labels = (x.cuda() for x in masked_batch(batch, tokenizer, 0.5))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            top = logits.topk(config.topk, dim=-1).indices
            choice = torch.randint(1, config.topk, (*ids.shape, 1), device=ids.device)
            generated = torch.where(labels >= 0, top.gather(-1, choice).squeeze(-1), ids).cpu()
            for row, example in enumerate(batch):
                original = originals[example.source_hash]
                old = original.entities[example.entity_index]
                words = [
                    tokenizer.decode(generated[row, list(ps)], skip_special_tokens=False).strip()
                    for ps in example.word_positions
                ]
                surface = " ".join(words)
                old_surface = original.text[old.start : old.end]
                if surface == old_surface or not preserve_suffix(old_surface, surface):
                    continue
                try:
                    candidate = replace_entity(
                        original, example.entity_index, surface, variant=str(round_index)
                    )
                except ValueError:
                    continue
                if candidate.text not in seen:
                    candidates.append((candidate, original.hash, example.entity_index))
                    seen.add(candidate.text)
                if len(candidates) >= config.max_candidates:
                    return candidates
    return candidates
