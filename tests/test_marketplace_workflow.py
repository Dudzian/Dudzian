"""Testy workflow publikacji presetów Marketplace."""
from __future__ import annotations

from pathlib import Path

from bot_core.config_marketplace import PresetPublicationWorkflow


def _signing_keys() -> dict[str, bytes]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "dev-hmac": (repo_root / "config" / "marketplace" / "keys" / "dev-hmac.key").read_bytes(),
    }


def test_marketplace_workflow_builds_ui_payload() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = PresetPublicationWorkflow.from_paths(
        catalog_path=repo_root / "config" / "marketplace" / "catalog.json",
        reviews_dir=repo_root / "config" / "marketplace" / "reviews",
        signing_keys=_signing_keys(),
    )
    workflow.validate(minimum_ready=15)
    payload = workflow.build_ui_payload()

    assert payload["total"] >= 15
    first = payload["presets"][0]
    assert first["wizard"]["importable"] is True
    assert first["artifacts"], "Preset powinien zawierać artefakty"


def test_marketplace_workflow_report_rows_include_reviews() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = PresetPublicationWorkflow.from_paths(
        catalog_path=repo_root / "config" / "marketplace" / "catalog.json",
        reviews_dir=repo_root / "config" / "marketplace" / "reviews",
        signing_keys=_signing_keys(),
    )
    rows = workflow.to_report_rows()

    assert rows, "Raport powinien zawierać wpisy"
    assert all(row["signed_artifacts"] for row in rows)
