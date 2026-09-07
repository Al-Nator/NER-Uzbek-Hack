"""Подготовка изолированного train-holdout пилота с нуля."""

import json
from pathlib import Path

import yaml

from uzner.data.reranker_split import RerankerSplit, prepare_reranker_split


def main() -> None:
    """Создаёт разбиение и отдельные конфиги предлагающих моделей."""
    output = Path("artifacts/reranker/s7_holdout_v1")
    manifest = prepare_reranker_split(
        RerankerSplit(
            Path("ner_uz_hackathon_participant/data/train.jsonl"),
            Path("ner_uz_hackathon_participant/data/dev.jsonl"),
            output,
        )
    )
    directory = Path("configs/seventh")
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "holdout_data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sources": [
                        {"name": name, "path": str(output / f"{name}.jsonl"), "split": split}
                        for name, split in (("proposer_train", "train"), ("proposer_valid", "dev"))
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    templates = (
        ("s7p0_bge_holdout_v1", "configs/experiments/a100/s32_bge_m3_retromae_global_pointer.yaml"),
        ("s7p1_mdeberta_holdout_v1", "configs/experiments/s21_mdeberta_v3_base_bioes_crf.yaml"),
        ("s7p2_xlmr_holdout_v1", "configs/experiments/a100/s31_xlmr_large_global_pointer.yaml"),
    )
    paths = []
    for run_id, template in templates:
        raw = yaml.safe_load(Path(template).read_text())
        config = raw["experiment"]
        config.update(run_id=run_id, data_config=str(data_path))
        config["training"].update(batch_size=8, eval_batch_size=8, gradient_accumulation_steps=1)
        path = directory / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        paths.append(str(path))
    series = Path("configs/series/seventh_proposers.yaml")
    series.write_text(
        yaml.safe_dump(
            {
                "series": {
                    "name": "seventh-holdout-proposers",
                    "stages": [{"name": "proposers", "configs": paths}],
                }
            }
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
