"""Pakiet testowy projektu Dudzian."""
from __future__ import annotations

# Import pomocniczego modułu zapewniającego obecność katalogu repozytorium na sys.path.
# Dzięki temu pojedyncze uruchomienia modułów testowych działają tak samo jak w pytest.
from . import _pathbootstrap  # noqa: F401  # pylint: disable=unused-import

__all__ = ["_pathbootstrap"]
