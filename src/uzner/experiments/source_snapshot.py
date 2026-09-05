"""Лёгкий снимок фактически исполняемого кода, включая untracked-модули."""

import tarfile
from pathlib import Path

from uzner.data.io import sha256_file
from uzner.experiments.artifacts import write_json


def capture_source(project_root: Path, environment: Path) -> None:
    """Сохраняет только Python/YAML и pyproject, без данных, весов и секретов."""
    files = sorted(
        path
        for directory in ("src", "scripts", "configs")
        for path in (project_root / directory).rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".yaml", ".yml"}
    )
    project = project_root / "pyproject.toml"
    if project.is_file():
        files.append(project)
    write_json(
        environment / "source_manifest.json",
        {str(path.relative_to(project_root)): sha256_file(path) for path in files},
    )
    with tarfile.open(environment / "source.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(project_root)), recursive=False)
