from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.security.fingerprint import (  # noqa: E402
    HardwareFingerprintService,
    RotatingHmacKeyProvider,
    evaluate_hwid_drift,
)
from bot_core.security.rotation import RotationRegistry  # noqa: E402


def _service_for(
    rotation_path: Path,
    *,
    cpu: str,
    mac: str,
    disk: str,
    tpm: str | None,
    now: datetime,
) -> HardwareFingerprintService:
    registry = RotationRegistry(rotation_path)
    provider = RotatingHmacKeyProvider(
        {"compat-key": b"drift-secret"},
        registry,
        interval_days=120.0,
    )
    return HardwareFingerprintService(
        provider,
        cpu_probe=lambda: cpu,
        tpm_probe=(lambda: tpm) if tpm is not None else (lambda: None),
        mac_probe=lambda: mac,
        disk_probe=lambda: disk,
        clock=lambda: now,
    )


def _build_record(rotation_path: Path, *, cpu: str, mac: str, disk: str, tpm: str | None, now: datetime) -> dict[str, Any]:
    service = _service_for(rotation_path, cpu=cpu, mac=mac, disk=disk, tpm=tpm, now=now)
    record = service.build()
    return record.payload


def _build_matrix(now: datetime, output_root: Path) -> dict[str, Any]:
    rotation_path = output_root / "rotation.json"
    baseline = _build_record(
        rotation_path,
        cpu="Intel Xeon Silver 4210R",
        mac="aa:bb:cc:dd:ee:01",
        disk="nvme-SN-001",
        tpm="IFX TPM2.0",
        now=now,
    )

    scenarios = {
        "mac_drift": _build_record(
            rotation_path,
            cpu="Intel Xeon Silver 4210R",
            mac="aa:bb:cc:dd:ee:10",
            disk="nvme-SN-001",
            tpm="IFX TPM2.0",
            now=now,
        ),
        "disk_drift": _build_record(
            rotation_path,
            cpu="Intel Xeon Silver 4210R",
            mac="aa:bb:cc:dd:ee:01",
            disk="nvme-SN-099",
            tpm="IFX TPM2.0",
            now=now,
        ),
        "cpu_drift": _build_record(
            rotation_path,
            cpu="AMD Ryzen 9 5900HX",
            mac="aa:bb:cc:dd:ee:01",
            disk="nvme-SN-001",
            tpm="IFX TPM2.0",
            now=now,
        ),
        "tpm_drift": _build_record(
            rotation_path,
            cpu="Intel Xeon Silver 4210R",
            mac="aa:bb:cc:dd:ee:01",
            disk="nvme-SN-001",
            tpm="IFX TPM2.0 patched",
            now=now,
        ),
    }

    matrix = {
        "generated_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "baseline": evaluate_hwid_drift(baseline, baseline),
        "baseline_components": baseline.get("component_digests", {}),
        "scenarios": [],
    }

    for name, payload in scenarios.items():
        evaluation = evaluate_hwid_drift(baseline, payload)
        evaluation["name"] = name
        matrix["scenarios"].append(evaluation)

    return matrix


def write_report(output_path: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _build_matrix(now, output_path.parent)
    output_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generuje raport dryfu HWID w oparciu o scenariusze tolerancji.")
    parser.add_argument(
        "--output",
        default="reports/ci/licensing_drift/compatibility.json",
        help="Ścieżka pliku raportu (domyślnie reports/ci/licensing_drift/compatibility.json)",
    )
    args = parser.parse_args(argv)

    write_report(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
