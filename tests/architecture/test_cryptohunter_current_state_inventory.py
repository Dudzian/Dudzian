from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "architecture" / "cryptohunter_product_architecture" / "current_state_inventory.json"
ALLOWED_STATUSES = [
    "IMPLEMENTED",
    "PARTIAL",
    "PREVIEW_ONLY",
    "PLANNED",
    "CONFLICTING",
    "UNKNOWN",
]
REQUIRED_COMPONENT_FIELDS = {
    "id",
    "name",
    "domain",
    "status",
    "summary",
    "evidence_paths",
    "evidence_symbols",
    "roadmap_impact",
}


def _load_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _assert_safe_existing_relative_path(path_value: str) -> None:
    path = Path(path_value)
    assert not path.is_absolute(), f"absolute evidence path is forbidden: {path_value}"
    assert ".." not in path.parts, f"parent traversal is forbidden: {path_value}"
    assert (ROOT / path).exists(), f"evidence path does not exist: {path_value}"


def test_current_state_inventory_contract() -> None:
    inventory = _load_inventory()

    assert inventory["schema_version"] == "cryptohunter.current_state_inventory.v1"
    assert re.fullmatch(r"[0-9a-f]{40}", inventory["repository_baseline_commit"])
    assert inventory["status_values"] == ALLOWED_STATUSES

    components = inventory["components"]
    assert components
    component_ids = [component["id"] for component in components]
    assert len(component_ids) == len(set(component_ids))

    for component in components:
        assert REQUIRED_COMPONENT_FIELDS <= set(component)
        assert component["status"] in ALLOWED_STATUSES
        assert component["evidence_paths"]
        assert component["evidence_symbols"]
        for evidence_path in component["evidence_paths"]:
            _assert_safe_existing_relative_path(evidence_path)

    current_windows_app = inventory["current_windows_app"]
    _assert_safe_existing_relative_path(current_windows_app["entrypoint"])
    _assert_safe_existing_relative_path(current_windows_app["spec_file"])
