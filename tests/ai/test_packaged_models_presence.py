from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.getenv("BOT_CORE_REQUIRE_OEM_MODELS"),
    reason="Wymuszanie obecności modeli OEM nie jest aktywne",
)
def test_packaged_models_present() -> None:
    """Zapewnia, że pakiet zawiera realne modele, gdy CI tego wymaga."""

    searched_modules = ("KryptoLowca.ai_models", "ai_models")
    for module_name in searched_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        repository = getattr(module, "OEM_MODEL_REPOSITORY", None)
        if repository is None:
            continue
        path = Path(repository)
        assert path.exists(), f"Repozytorium modeli {path} nie istnieje"
        assert (path / "manifest.json").exists(), "Brakuje manifestu modeli OEM"
        return
    pytest.fail(
        "Nie znaleziono modułu OEM z zapakowanymi modelami (ustaw BOT_CORE_REQUIRE_OEM_MODELS tylko w buildach z modelami)"
    )
