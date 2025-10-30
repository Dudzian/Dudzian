from __future__ import annotations

import importlib
import warnings

from bot_core.ai import legacy_models


def test_kryptolowca_ai_models_shim_triggers_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("KryptoLowca.ai_models")

    assert module.AIModels is legacy_models.AIModels
    assert any(issubclass(w.category, DeprecationWarning) for w in captured)
