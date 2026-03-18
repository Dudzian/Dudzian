from __future__ import annotations

import ast
from pathlib import Path


def _load_default_adapter_keys() -> set[str]:
    source_path = Path("bot_core/runtime/bootstrap.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    for node in module.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "_DEFAULT_ADAPTERS":
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError("_DEFAULT_ADAPTERS musi być słownikiem literałowym")

        keys: set[str] = set()
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise AssertionError("Klucze _DEFAULT_ADAPTERS muszą być literałami tekstowymi")
            keys.add(key.value)
        return keys

    raise AssertionError("Nie znaleziono definicji _DEFAULT_ADAPTERS w bootstrap.py")


def test_bootstrap_default_adapter_registry_includes_required_futures_entries() -> None:
    adapter_keys = _load_default_adapter_keys()

    assert {"deribit_futures", "bitmex_futures"}.issubset(adapter_keys)
