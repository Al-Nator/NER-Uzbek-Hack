"""Тесты локального разрешения pinned Hugging Face snapshot-ов."""

from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from uzner.config import EncoderConfig
from uzner.models.pretrained import resolve_pretrained_snapshot


def test_local_encoder_never_uses_hub(tmp_path: Path) -> None:
    """Явный локальный encoder разрешается без Hugging Face API."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    snapshot = resolve_pretrained_snapshot(EncoderConfig(str(tmp_path), "local"))

    assert snapshot.path == tmp_path
    assert snapshot.weight_file == "model.safetensors"


def test_cached_snapshot_is_preferred_to_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Готовый cache используется даже при недоступной сети или proxy."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    calls: list[bool] = []

    def fake_download(*_args: object, **kwargs: object) -> str:
        """Возвращает cache и фиксирует offline-флаг вызова."""
        calls.append(bool(kwargs.get("local_files_only")))
        return str(tmp_path)

    monkeypatch.setattr("uzner.models.pretrained.snapshot_download", fake_download)

    snapshot = resolve_pretrained_snapshot(EncoderConfig("remote/model", "revision"))

    assert snapshot.path == tmp_path
    assert calls == [True]


def test_missing_cache_downloads_only_preferred_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При cache miss скачивается один явно выбранный формат весов."""
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    def fake_download(*_args: object, **kwargs: object) -> str:
        """Имитирует cache miss, затем успешную сетевую загрузку."""
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("missing")
        return str(tmp_path)

    monkeypatch.setattr("uzner.models.pretrained.snapshot_download", fake_download)
    encoder = EncoderConfig("remote/model", "revision", use_safetensors=False)

    snapshot = resolve_pretrained_snapshot(encoder)

    assert snapshot.weight_file == "pytorch_model.bin"
    assert calls[-1]["allow_patterns"][-1] == "pytorch_model.bin"


def test_local_encoder_without_weights_is_rejected(tmp_path: Path) -> None:
    """Неполный локальный snapshot отклоняется до Transformers."""
    with pytest.raises(FileNotFoundError, match="нет весов"):
        resolve_pretrained_snapshot(EncoderConfig(str(tmp_path), "local"))
