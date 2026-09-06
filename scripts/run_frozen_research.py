"""Локальные frozen-encoder эксперименты памяти и символьного уточнения."""

import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from uzner.data.io import load_documents
from uzner.models.char_boundary import CharConfig
from uzner.models.memory_knn import KnnConfig
from uzner.training.char_pipeline import run_character
from uzner.training.memory_pipeline import run_memory
from uzner.training.sidecar_run import SidecarRun


def main() -> None:
    """Исполняет явный конфиг без переиспользования чужих каталогов результатов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text("utf-8"))
    if set(value) != {"run_id", "source", "kind", "parameters"}:
        raise ValueError("Неизвестные или отсутствующие поля frozen config")
    if value["kind"] not in {"memory_knn", "char_boundary"}:
        raise ValueError("Неизвестный тип frozen experiment")
    config = (KnnConfig if value["kind"] == "memory_knn" else CharConfig)(**value["parameters"])
    train, dev = (
        tuple(load_documents(((s, Path(f"ner_uz_hackathon_participant/data/{s}.jsonl")),)))
        for s in ("train", "dev")
    )
    smoke = args.smoke_output is not None
    if smoke:
        train, dev = train[:16], dev[:8]
    root = args.smoke_output if smoke else Path("runs") / value["run_id"]
    source = Path(value["source"])
    with SidecarRun(root, source, {"kind": value["kind"], **asdict(config)}, smoke) as run:
        (run_memory if value["kind"] == "memory_knn" else run_character)(
            source, train, dev, config, run
        )
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
