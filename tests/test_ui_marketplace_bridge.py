from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from bot_core.security.signing import build_hmac_signature
from scripts import ui_marketplace_bridge as bridge


def _write_preset(path, *, preset_id: str, fingerprint: str, signing_key: bytes) -> None:
    payload = {
        "name": "Automation Pack",
        "strategies": [
            {
                "name": "mean-pack",
                "engine": "mean_reversion",
                "parameters": {"lookback": 50},
                "license_tier": "professional",
                "risk_classes": ["statistical"],
                "required_data": ["ohlcv"],
            }
        ],
        "metadata": {
            "id": preset_id,
            "profile": "ai",
            "license": {
                "module_id": f"module::{preset_id}",
                "fingerprint": fingerprint,
                "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        },
    }
    signature = build_hmac_signature(payload, key=signing_key, key_id="catalog")
    path.write_text(json.dumps({"preset": payload, "signature": signature}, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture()
def preset_dir(tmp_path, monkeypatch):
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.chdir(tmp_path)
    return presets


@pytest.fixture()
def signing_key() -> bytes:
    return b"catalog-secret"


def test_list_activate_deactivate_flow(capsys, preset_dir, signing_key):
    preset_file = preset_dir / "automation.json"
    _write_preset(preset_file, preset_id="automation-ai", fingerprint="original", signing_key=signing_key)

    licenses_path = preset_dir.parent / "licenses.json"

    key_file = preset_dir.parent / "keys.json"
    key_file.write_text(json.dumps({"catalog": signing_key.hex()}), encoding="utf-8")

    list_args = [
        f"--presets-dir={preset_dir}",
        f"--licenses-path={licenses_path}",
        "--fingerprint",
        "local-override",
        f"--signing-key=catalog={signing_key.hex()}",
        f"--signing-key-file={key_file}",
        "list",
    ]
    bridge.main(list_args)
    output = json.loads(capsys.readouterr().out)
    assert output["presets"][0]["license"]["status"] in {"pending", "fingerprint_mismatch"}

    license_payload = preset_dir.parent / "license_payload.json"
    license_payload.write_text(
        json.dumps({"fingerprint": "local-override", "expires_at": "2099-01-01T00:00:00Z"}, ensure_ascii=False),
        encoding="utf-8",
    )

    bridge.main(
        [
            f"--presets-dir={preset_dir}",
            f"--licenses-path={licenses_path}",
            "--fingerprint",
            "local-override",
            f"--signing-key=catalog={signing_key.hex()}",
            f"--signing-key-file={key_file}",
            "activate",
            "--preset-id",
            "automation-ai",
            f"--license-json={license_payload}",
        ]
    )
    activation_output = json.loads(capsys.readouterr().out)
    assert activation_output["preset"]["license"]["status"] == "active"

    bridge.main(list_args)
    refreshed = json.loads(capsys.readouterr().out)
    assert refreshed["presets"][0]["license"]["status"] == "active"

    bridge.main(
        [
            f"--presets-dir={preset_dir}",
            f"--licenses-path={licenses_path}",
            "--fingerprint",
            "local-override",
            f"--signing-key=catalog={signing_key.hex()}",
            f"--signing-key-file={key_file}",
            "deactivate",
            "--preset-id",
            "automation-ai",
        ]
    )
    deactivate_output = json.loads(capsys.readouterr().out)
    assert deactivate_output["preset"]["license"]["status"] in {"pending", "fingerprint_mismatch"}
