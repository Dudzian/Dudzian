"""Testy workflow publikacji presetów Marketplace."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

from bot_core.config_marketplace import PresetPublicationWorkflow
from bot_core.config_marketplace.schema import load_catalog
from bot_core.security.signing import canonical_json_bytes


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


def test_marketplace_review_signature_supports_unicode(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = repo_root / "config" / "marketplace" / "catalog.json"
    signing_key = (repo_root / "config" / "marketplace" / "keys" / "dev-hmac.key").read_bytes()

    catalog = load_catalog(catalog_path)
    preset = next(pkg for pkg in catalog.packages if pkg.release.review_status.lower() == "approved")

    review_payload = {
        "preset_id": preset.package_id,
        "rating": 5,
        "author": "Łukasz Żółć",
        "comment": "Zażółć gęślą jaźń",
    }
    signature_value = base64.b64encode(
        hmac.new(signing_key, canonical_json_bytes(review_payload), hashlib.sha256).digest()
    ).decode("ascii")

    review_path = tmp_path / f"{preset.package_id}.json"
    review_path.write_text(
        json.dumps(
            {
                "preset_id": preset.package_id,
                "reviews": [
                    {
                        **review_payload,
                        "signature": {
                            "algorithm": "HMAC-SHA256",
                            "key_id": "dev-hmac",
                            "value": signature_value,
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    workflow = PresetPublicationWorkflow.from_paths(
        catalog_path=catalog_path,
        reviews_dir=tmp_path,
        signing_keys={"dev-hmac": signing_key},
    )

    payload = workflow.build_ui_payload()
    preset_entry = next(item for item in payload["presets"] if item["packageId"] == preset.package_id)

    assert any(review["signatureValid"] for review in preset_entry["reviews"])
