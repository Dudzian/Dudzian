from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bot_core.exchanges.base import Environment
from bot_core.runtime.bootstrap import bootstrap_environment

from tests.test_runtime_bootstrap import (
    _BASE_CONFIG,
    _apply_license_stub,
    _create_signed_document,
    _prepare_manager,
    _prepare_signed_license_bundle,
    _disable_exchange_health,
    _stub_license_validation,
)


def _write_live_config(tmp_path: Path, *, remove_penetration_signature: bool = False) -> Path:
    _prepare_signed_license_bundle(tmp_path)
    data = yaml.safe_load(_BASE_CONFIG)
    _apply_license_stub(data, tmp_path)
    live_env = data["environments"]["binance_paper"]
    live_env = dict(live_env)
    live_env.update(
        {
            "environment": "live",
            "keychain_key": "binance_live_key",
            "alert_audit": {"backend": "file", "directory": "./var/live_alerts"},
        }
    )

    compliance_doc = _create_signed_document(
        tmp_path,
        "compliance/live/binance/kyc_packet.pdf",
        "compliance/live/binance/kyc_packet.sig",
        key_id="compliance-key",
        signed_by=("compliance",),
    )
    risk_doc = _create_signed_document(
        tmp_path,
        "risk/live/binance/risk_profile_alignment.pdf",
        "risk/live/binance/risk_profile_alignment.sig",
        key_id="risk-key",
        signed_by=("risk",),
    )
    penetration_doc = _create_signed_document(
        tmp_path,
        "security/live/binance/penetration_report.pdf",
        "security/live/binance/penetration_report.sig",
        key_id="security-key",
        signed_by=("security",),
    )

    if remove_penetration_signature:
        (tmp_path / penetration_doc["signature_path"]).unlink()

    live_env["live_readiness"] = {
        "checklist_id": "binance-q3",
        "signed": True,
        "signed_by": ["compliance", "security"],
        "signature_path": "compliance/live/binance/checklist.sig",
        "required_documents": [
            "kyc_packet",
            "risk_profile_alignment",
            "penetration_report",
        ],
        "documents": [
            {
                "name": "kyc_packet",
                "path": "compliance/live/binance/kyc_packet.pdf",
                "sha256": compliance_doc["sha256"],
                "signed": True,
                "signed_by": ["compliance"],
                "signature_path": compliance_doc["signature_path"],
            },
            {
                "name": "risk_profile_alignment",
                "path": "risk/live/binance/risk_profile_alignment.pdf",
                "sha256": risk_doc["sha256"],
                "signed": True,
                "signed_by": ["risk"],
                "signature_path": risk_doc["signature_path"],
            },
            {
                "name": "penetration_report",
                "path": "security/live/binance/penetration_report.pdf",
                "sha256": penetration_doc["sha256"],
                "signed": True,
                "signed_by": ["security"],
                "signature_path": penetration_doc["signature_path"],
            },
        ],
    }
    data["environments"]["binance_live"] = live_env

    config_path = tmp_path / "core_live.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _store_live_credentials(manager_storage):
    credentials_payload = {
        "key_id": "live-key",
        "secret": "live-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    manager_storage.set_secret(
        "tests:binance_live_key:trading",
        json.dumps(credentials_payload),
    )


def test_live_guardrails_accept_signed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_license_validation(monkeypatch, tmp_path)
    _disable_exchange_health(monkeypatch)
    config_path = _write_live_config(tmp_path)
    storage, manager = _prepare_manager()
    _store_live_credentials(storage)

    context = bootstrap_environment("binance_live", config_path=config_path, secret_manager=manager)
    verification = context.live_signature_verification
    assert verification is not None
    assert verification["categories"] == {
        "compliance": True,
        "risk": True,
        "penetration": True,
    }


def test_live_guardrails_block_missing_pentest_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_license_validation(monkeypatch, tmp_path)
    _disable_exchange_health(monkeypatch)
    config_path = _write_live_config(tmp_path, remove_penetration_signature=True)
    storage, manager = _prepare_manager()
    _store_live_credentials(storage)

    with pytest.raises(RuntimeError) as exc:
        bootstrap_environment("binance_live", config_path=config_path, secret_manager=manager)

    assert "penetration_report" in str(exc.value)
