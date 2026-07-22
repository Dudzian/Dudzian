"""Verify Windows PyInstaller collectors include dynamic backend modules."""

from __future__ import annotations

import ast
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(__file__).resolve().parents[1]


def _ui_backend_export_modules() -> dict[str, str]:
    backend_init = ROOT / "ui/backend/__init__.py"
    tree = ast.parse(backend_init.read_text(encoding="utf-8"), filename=str(backend_init))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_MODULE_BY_EXPORT"
            for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
            if not isinstance(mapping, dict):
                raise TypeError("_MODULE_BY_EXPORT must be a mapping")
            return mapping
    raise ValueError("_MODULE_BY_EXPORT assignment not found")


def main() -> int:
    collected = set(collect_submodules("ui.backend"))
    required = {
        f"ui.backend.{module_path.removeprefix('.')}"
        for module_path in _ui_backend_export_modules().values()
    }
    missing = sorted(required - collected)
    if missing:
        print("Missing PyInstaller ui.backend modules:")
        for module in missing:
            print(f"- {module}")
        return 1
    print(f"PyInstaller collector includes {len(required)} dynamic ui.backend modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
