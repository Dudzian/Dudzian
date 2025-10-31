"""Domyślne profile trenowania AI wykorzystywane przez pipeline."""

from importlib import resources
from pathlib import Path
from typing import Iterator


def iter_training_profiles() -> Iterator[Path]:
    """Zwróć ścieżki do wbudowanych profili trenowania."""

    with resources.as_file(resources.files(__package__)) as base_path:  # type: ignore[arg-type]
        for entry in sorted(base_path.glob("*.yaml")):
            if entry.is_file():
                yield entry


__all__ = ["iter_training_profiles"]
