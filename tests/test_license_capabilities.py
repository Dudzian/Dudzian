from __future__ import annotations

import base64
import hashlib
import json
from datetime import date

import pytest
from nacl.signing import SigningKey

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.hwid import HwIdProviderError
from bot_core.security.license_service import (
    LicenseHardwareMismatchError,
    LicenseService,
)


SAMPLE_PAYLOAD = {
    "license_id": "DUD-2025-00123",
    "edition": "pro",
    "environments": ["demo", "paper", "live"],
    "exchanges": {
        "binance_spot": True,
        "binance_futures": True,
        "kraken_spot": True,
        "kraken_futures": False,
        "zonda_spot": False,
    },
    "strategies": {
        "trend_d1": True,
        "mean_reversion": True,
        "volatility_target": True,
        "cross_exchange": True,
    },
    "runtime": {
        "auto_trader": True,
        "multi_strategy_scheduler": True,
        "hypercare": False,
    },
    "modules": {
        "futures": True,
        "walk_forward": True,
        "reporting_pro": True,
        "observability_ui": True,
        "alerts_advanced": True,
        "oem_updater": True,
        "ai_signals": False,
    },
    "limits": {
        "max_paper_controllers": 3,
        "max_live_controllers": 1,
        "max_concurrent_bots": 6,
        "max_alert_channels": 4,
    },
    "holder": {"name": "ACME Sp. z o.o.", "email": "ops@acme.pl"},
    "issued_at": "2025-07-01",
    "maintenance_until": "2026-07-01",
    "trial": {"enabled": False, "expires_at": None},
    "seats": 2,
    "hwid": "OPTIONAL_MACHINE_HASH",
}


class StaticHwIdProvider:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value


class FailingHwIdProvider:
    def read(self) -> str:
        raise HwIdProviderError("hwid unavailable")


def test_build_capabilities_from_payload() -> None:
    capabilities = build_capabilities_from_payload(SAMPLE_PAYLOAD, effective_date=date(2025, 7, 5))
    assert capabilities.edition == "pro"
    assert capabilities.is_module_enabled("futures")
    assert not capabilities.is_module_enabled("ai_signals")
    assert capabilities.is_runtime_enabled("multi_strategy_scheduler")
    assert capabilities.is_exchange_enabled("kraken_spot")
    assert not capabilities.is_exchange_enabled("kraken_futures")
    assert capabilities.limits.max_paper_controllers == 3
    assert capabilities.is_maintenance_active(date(2025, 8, 1))
    assert not capabilities.is_trial_active(date(2025, 8, 1))


def test_license_service_monotonic_effective_date(tmp_path) -> None:
    signing_key = SigningKey.generate()
    payload_bytes = json.dumps(SAMPLE_PAYLOAD, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(payload_bytes).signature
    bundle = {
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    bundle_path = tmp_path / "license.lic"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    state_path = tmp_path / "state.json"

    status_path = tmp_path / "status.json"
    audit_log_path = tmp_path / "admin.log"
    service = LicenseService(
        verify_key_hex=signing_key.verify_key.encode().hex(),
        state_path=state_path,
        status_path=status_path,
        audit_log_path=audit_log_path,
        today_provider=lambda: date(2025, 7, 10),
        hwid_provider=StaticHwIdProvider("OPTIONAL_MACHINE_HASH"),
    )
    snapshot = service.load_from_file(bundle_path)
    assert snapshot.capabilities.edition == "pro"
    assert snapshot.effective_date == date(2025, 7, 10)
    assert snapshot.local_hwid == "OPTIONAL_MACHINE_HASH"

    status_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_document["license_id"] == SAMPLE_PAYLOAD["license_id"]
    assert status_document["edition"] == "pro"
    assert status_document["payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert status_document["trial"]["active"] is False
    assert status_document["maintenance"]["active"] is True

    audit_lines = [line for line in audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert audit_lines, "Audit log should contain at least one entry"
    audit_entry = json.loads(audit_lines[-1])
    assert audit_entry["license_id"] == SAMPLE_PAYLOAD["license_id"]

    # Powtórny odczyt z wcześniejszą datą zachowuje monotoniczność.
    service_late = LicenseService(
        verify_key_hex=signing_key.verify_key.encode().hex(),
        state_path=state_path,
        status_path=status_path,
        audit_log_path=audit_log_path,
        today_provider=lambda: date(2025, 7, 5),
        hwid_provider=StaticHwIdProvider("OPTIONAL_MACHINE_HASH"),
    )
    snapshot_late = service_late.load_from_file(bundle_path)
    assert snapshot_late.effective_date == date(2025, 7, 10)


def test_license_service_hwid_mismatch(tmp_path) -> None:
    signing_key = SigningKey.generate()
    payload = dict(SAMPLE_PAYLOAD)
    payload["hwid"] = "EXPECTED"
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(payload_bytes).signature
    bundle = {
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    bundle_path = tmp_path / "license.lic"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    service = LicenseService(
        verify_key_hex=signing_key.verify_key.encode().hex(),
        state_path=tmp_path / "state.json",
        status_path=tmp_path / "status.json",
        audit_log_path=tmp_path / "admin.log",
        today_provider=lambda: date(2025, 7, 10),
        hwid_provider=StaticHwIdProvider("DIFFERENT"),
    )

    with pytest.raises(LicenseHardwareMismatchError):
        service.load_from_file(bundle_path)


def test_license_service_hwid_provider_failure(tmp_path) -> None:
    signing_key = SigningKey.generate()
    payload = dict(SAMPLE_PAYLOAD)
    payload["hwid"] = "EXPECTED"
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(payload_bytes).signature
    bundle = {
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    bundle_path = tmp_path / "license.lic"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    service = LicenseService(
        verify_key_hex=signing_key.verify_key.encode().hex(),
        state_path=tmp_path / "state.json",
        status_path=tmp_path / "status.json",
        audit_log_path=tmp_path / "admin.log",
        today_provider=lambda: date(2025, 7, 10),
        hwid_provider=FailingHwIdProvider(),
    )

    with pytest.raises(LicenseHardwareMismatchError):
        service.load_from_file(bundle_path)
