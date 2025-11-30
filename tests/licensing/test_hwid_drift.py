from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import subprocess
import sys

import pytest

from bot_core.security.fingerprint import (
    HardwareFingerprintService,
    RotatingHmacKeyProvider,
    evaluate_hwid_drift,
)
from bot_core.security.rotation import RotationRegistry


FIXED_NOW = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def rotation_path(tmp_path: Path) -> Path:
    return tmp_path / "rotation.json"


def _service_for(
    rotation_path: Path,
    *,
    cpu: str,
    mac: str,
    disk: str,
    tpm: str | None,
    interval_days: float = 120.0,
) -> HardwareFingerprintService:
    registry = RotationRegistry(rotation_path)
    registry.mark_rotated("key-2023", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=95))
    registry.mark_rotated("key-2024", "hardware-fingerprint", timestamp=FIXED_NOW - timedelta(days=10))
    provider = RotatingHmacKeyProvider(
        {"key-2023": b"archival-secret", "key-2024": b"fresh-secret"},
        registry,
        interval_days=interval_days,
    )
    return HardwareFingerprintService(
        provider,
        cpu_probe=lambda: cpu,
        tpm_probe=(lambda: tpm) if tpm is not None else (lambda: None),
        mac_probe=lambda: mac,
        disk_probe=lambda: disk,
        clock=lambda: FIXED_NOW,
    )


def _build_record(rotation_path: Path, **kwargs) -> dict:
    service = _service_for(rotation_path, **kwargs)
    return service.build().payload


def test_baseline_matches(rotation_path: Path) -> None:
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )

    result = evaluate_hwid_drift(baseline, baseline)

    assert result["status"] == "match"
    assert result["changed_components"] == []
    assert not result["blocked"]


def test_mac_drift_is_tolerated(rotation_path: Path) -> None:
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )
    drifted = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:10",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )

    result = evaluate_hwid_drift(baseline, drifted)

    assert result["status"] == "degraded"
    assert result["tolerated"] == ["mac"]
    assert result["blocked"] == []


def test_disk_drift_is_tolerated(rotation_path: Path) -> None:
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )
    drifted = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-099",
        tpm="IFX TPM2.0",
    )

    result = evaluate_hwid_drift(baseline, drifted)

    assert result["status"] == "degraded"
    assert result["tolerated"] == ["disk"]
    assert result["blocked"] == []


def test_cpu_drift_requires_rebind(rotation_path: Path) -> None:
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )
    drifted = _build_record(
        rotation_path,
        cpu="AMD Ryzen 9 5900HX",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )

    result = evaluate_hwid_drift(baseline, drifted)

    assert result["status"] == "rebind_required"
    assert "cpu" in result["blocked"]
    assert "mac" not in result["blocked"]


def test_tpm_drift_requires_rebind(rotation_path: Path) -> None:
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
    )
    drifted = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0 patched",
    )

    result = evaluate_hwid_drift(baseline, drifted)

    assert result["status"] == "rebind_required"
    assert "tpm" in result["blocked"]
    assert "disk" not in result["blocked"]


def test_cli_report_produces_json(tmp_path: Path) -> None:
    output = tmp_path / "compat.json"
    command = [sys.executable, "scripts/generate_hwid_drift_report.py", "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)

    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["baseline"]["status"] == "match"
    assert {scenario["name"] for scenario in payload["scenarios"]} == {
        "mac_drift",
        "disk_drift",
        "cpu_drift",
        "tpm_drift",
    }
    degraded = next(item for item in payload["scenarios"] if item["name"] == "mac_drift")
    assert degraded["status"] == "degraded"
