from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist and return it as Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """
    Create multiple directories and return list of Path objects.
    """
    created = []
    for p in paths:
        created.append(ensure_dir(p))
    return created


def project_root_from_file(file_path: str | Path, levels_up: int = 2) -> Path:
    """
    Resolve project root by walking up from a file path.

    Example:
        src/utils/io.py with levels_up=2 -> project_root
    """
    p = Path(file_path).resolve()
    for _ in range(levels_up):
        p = p.parent
    return p