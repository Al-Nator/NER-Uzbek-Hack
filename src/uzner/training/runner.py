"""Единая точка входа для experiment pipeline-ов."""

from uzner.config import load_experiment_config
from uzner.training.engine import TrainRequest, TrainResult, train_experiment


def run_experiment(request: TrainRequest) -> TrainResult:
    """Выбирает official или новый контур по конфигу."""
    config = load_experiment_config(request.config_path.resolve())
    if config.pipeline == "official_reference":
        from uzner.training.official import run_official_reference

        return run_official_reference(request)
    return train_experiment(request)
