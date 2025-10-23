from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bot_core.security.fingerprint import HardwareFingerprintService, RotatingHmacKeyProvider
from bot_core.security.rotation import RotationRegistry
from scripts import oem_provision_license as cli


FIXED_NOW = datetime(2024, 3, 5, 10, 30, tzinfo=timezone.utc)


def _fingerprint_file(tmp_path, *, dongle: str | None = "USB-LOCK-01"):
    rotation_path = tmp_path / "finger_rotation.json"
    registry = RotationRegistry(rotation_path)
    registry.mark_rotated("fp-key", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=7))
    provider = RotatingHmacKeyProvider(
        {"fp-key": b"finger-secret"},
        registry,
        interval_days=180,
    )
    service = HardwareFingerprintService(
        provider,
        cpu_probe=lambda: "Intel(R) Core(TM) i9-13900K",
        tpm_probe=lambda: "IFX TPM 2.0",
        mac_probe=lambda: "aa:bb:cc:dd:ee:ff",
        dongle_probe=lambda: dongle,
        clock=lambda: FIXED_NOW,
    )
    record = service.build()
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False), encoding="utf-8")
    return path, record


def _license_rotation(tmp_path, *, purpose: str) -> str:
    path = tmp_path / "license_rotation.json"
    registry = RotationRegistry(path)
    registry.mark_rotated("lic-2024", purpose, timestamp=FIXED_NOW - timedelta(days=5))
    return str(path)


def _registry_path(tmp_path) -> str:
    return str(tmp_path / "registry.jsonl")


def test_parse_args_accepts_request_yaml(tmp_path):
    request_path = tmp_path / "request.yaml"
    registry_path = tmp_path / "registry.jsonl"
    request_path.write_text(
        "\n".join(
            [
                "fingerprint: ZXCV-ASDF",
                "mode: usb",
                "issuer: FILE-OPS",
                "license_keys:",
                "  lic-2024: hex:aaaa",
                "fingerprint_keys:",
                "  fp-key: hex:bbbb",
                f"registry: {registry_path}",
                "validate_registry: true",
                "emit_qr: true",
            ]
        ),
        encoding="utf-8",
    )

    args = cli._parse_args([str(request_path), "--license-key", "extra=from-cli"])

    assert args.request_path == str(request_path)
    assert args.fingerprint == "ZXCV-ASDF"
    assert args.mode == "usb"
    assert args.issuer == "FILE-OPS"
    assert args.validate_registry is True
    assert args.emit_qr is True
    assert args.output == str(registry_path)
    assert args.license_keys == ["extra=from-cli", "lic-2024=hex:aaaa"]
    assert args.fingerprint_keys == ["fp-key=hex:bbbb"]


def test_provision_license_usb_flow(tmp_path, capsys):
    fingerprint_path, record = _fingerprint_file(tmp_path)
    rotation_log = _license_rotation(tmp_path, purpose="oem-license")
    registry_path = _registry_path(tmp_path)
    usb_dir = tmp_path / "usb"

    args = cli._parse_args(
        [
            "--fingerprint",
            str(fingerprint_path),
            "--mode",
            "usb",
            "--license-key",
            "lic-2024=dev-license",
            "--license-rotation-log",
            rotation_log,
            "--fingerprint-key",
            "fp-key=finger-secret",
            "--registry",
            registry_path,
            "--usb-output",
            str(usb_dir),
        ]
    )

    exit_code = cli._run_provision(args)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Dodano licencję" in captured.out

    registry_content = Path(registry_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(registry_content) == 1
    entry = json.loads(registry_content[0])
    assert entry["signature"]["key_id"] == "lic-2024"
    assert entry["payload"]["fingerprint_signature"]["key_id"] == "fp-key"
    assert entry["payload"]["fingerprint_payload"]["fingerprint"]["value"] == record.payload["fingerprint"]["value"]

    artifacts = list(usb_dir.glob("*.json"))
    assert artifacts, "CLI powinno zapisać artefakt licencji na USB"

    validate_args = cli._parse_args(
        [
            "--registry",
            registry_path,
            "--license-key",
            "lic-2024=dev-license",
            "--fingerprint-key",
            "fp-key=finger-secret",
            "--validate-registry",
        ]
    )
    assert cli._run_validation(validate_args) == 0


def test_provision_rejects_missing_dongle(tmp_path):
    fingerprint_path, _ = _fingerprint_file(tmp_path, dongle=None)
    rotation_log = _license_rotation(tmp_path, purpose="oem-license")
    registry_path = _registry_path(tmp_path)

    args = cli._parse_args(
        [
            "--fingerprint",
            str(fingerprint_path),
            "--mode",
            "usb",
            "--license-key",
            "lic-2024=dev-license",
            "--license-rotation-log",
            rotation_log,
            "--fingerprint-key",
            "fp-key=finger-secret",
            "--registry",
            registry_path,
        ]
    )

    with pytest.raises(cli.ProvisioningError):
        cli._run_provision(args)
