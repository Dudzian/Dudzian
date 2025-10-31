from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


from bot_core.security.fingerprint import HardwareFingerprintService, RotatingHmacKeyProvider
from bot_core.security.rotation import RotationRegistry


FIXED_NOW = datetime(2024, 1, 7, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def rotation_path(tmp_path):
    return tmp_path / "rotation.json"


def _service_for(
    rotation_path,
    *,
    cpu: str,
    mac: str,
    tpm: str | None,
    dongle: str | None,
    interval_days: float = 120.0,
) -> HardwareFingerprintService:
    registry = RotationRegistry(rotation_path)
    registry.mark_rotated("key-2023", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=95))
    registry.mark_rotated("key-2024", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=10))
    provider = RotatingHmacKeyProvider(
        {"key-2023": b"legacy-secret", "key-2024": b"fresh-secret"},
        registry,
        interval_days=interval_days,
    )
    return HardwareFingerprintService(
        provider,
        cpu_probe=lambda: cpu,
        tpm_probe=(lambda: tpm) if tpm is not None else (lambda: None),
        mac_probe=lambda: mac,
        dongle_probe=(lambda: dongle) if dongle is not None else (lambda: None),
        clock=lambda: FIXED_NOW,
    )


@pytest.mark.parametrize(
    "label,data",
    [
        (
            "linux",
            {
                "cpu": "Intel(R) Xeon(R) Platinum 8370C",
                "mac": "aa:bb:cc:dd:ee:01",
                "tpm": "IFX TPM2.0",
                "dongle": "USB-XYZ-001",
                "vendor": "intel(r)",
            },
        ),
        (
            "windows",
            {
                "cpu": "AMD Ryzen 9 5900HX",
                "mac": "11-22-33-44-55-66",
                "tpm": "AMD TPM 2.0",
                "dongle": None,
                "vendor": "amd",
            },
        ),
        (
            "macos",
            {
                "cpu": "Apple M2 Pro",
                "mac": "AA-BB-CC-DD-EE-0F",
                "tpm": None,
                "dongle": None,
                "vendor": "apple",
            },
        ),
    ],
)
def test_fingerprint_deterministic_across_platforms(rotation_path, label, data):
    service = _service_for(
        rotation_path,
        cpu=data["cpu"],
        mac=data["mac"],
        tpm=data.get("tpm"),
        dongle=data.get("dongle"),
    )

    record_first = service.build()
    record_second = service.build()

    assert record_first.payload == record_second.payload
    assert record_first.signature == record_second.signature
    assert record_first.signature["key_id"] == "key-2024"

    components = record_first.payload["components"]
    cpu_entry = components["cpu"]
    assert data["vendor"] in cpu_entry["normalized"]
    assert components["mac"]["raw"].lower().replace("-", ":")
    assert (
        record_first.payload["component_digests"]["mac"]
        == components["mac"]["digest"]
    )

    if data.get("tpm") is None:
        assert components["tpm"] is None
    else:
        assert data["tpm"].lower() in components["tpm"]["raw"].lower()


def test_rotating_key_provider_prefers_recent(rotation_path):
    registry = RotationRegistry(rotation_path)
    registry.mark_rotated("old", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=400))
    registry.mark_rotated("fresh", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=5))
    provider = RotatingHmacKeyProvider(
        {"old": b"old-secret", "fresh": b"fresh-secret"},
        registry,
        interval_days=180,
    )

    key_id, key, status = provider.select_active_key(now=FIXED_NOW)
    assert key_id == "fresh"
    assert key == b"fresh-secret"
    assert status.is_due is False


def test_fingerprint_digest_changes_when_component_changes(rotation_path):
    base = _service_for(
        rotation_path,
        cpu="Intel Core",
        mac="00:11:22:33:44:55",
        tpm="TPM",
        dongle=None,
    )
    record_a = base.build()

    modified = _service_for(
        rotation_path,
        cpu="Intel Core",
        mac="00:11:22:33:44:59",
        tpm="TPM",
        dongle=None,
    )
    record_b = modified.build()

    assert record_a.payload["fingerprint"]["value"] != record_b.payload["fingerprint"]["value"]
