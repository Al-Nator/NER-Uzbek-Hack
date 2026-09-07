"""Единые проверки полного и изолированного запуска до любых записей на диск."""

from dataclasses import replace

from uzner.config import ExperimentConfig, load_experiment_config, with_run_suffix
from uzner.training.requests import TrainRequest


def resolve_request_config(request: TrainRequest) -> ExperimentConfig:
    """Применяет явные smoke-переопределения одинаково для uzner и official wrapper."""
    config = with_run_suffix(
        load_experiment_config(request.config_path.resolve()), request.run_id_suffix
    )
    limits = {
        "max_epochs": request.max_epochs,
        "max_train_documents": request.max_train_documents,
        "max_dev_documents": request.max_dev_documents,
    }
    for name, value in limits.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} должен быть положительным целым числом")
    if any(value is not None for value in limits.values()) and request.output_root_override is None:
        raise ValueError("Smoke-limit требует отдельный output_root_override")
    if request.max_epochs is not None:
        config = replace(config, training=replace(config.training, epochs=request.max_epochs))
    if request.output_root_override is not None:
        if request.publish_summary:
            raise ValueError("Smoke/output override нельзя публиковать в сводку")
        config = replace(config, output_root=str(request.output_root_override.resolve()))
    return config
