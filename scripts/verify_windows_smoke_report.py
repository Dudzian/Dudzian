"""Validate the JSON smoke report emitted by the Windows PyInstaller artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_smoke_report(path: str | Path) -> dict[str, Any]:
    """Load a non-empty JSON smoke report and require status=ok."""

    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Smoke report does not exist: {report_path}")
    if report_path.stat().st_size <= 0:
        raise ValueError(f"Smoke report is empty: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Smoke report JSON must be an object")
    if "status" not in payload:
        raise ValueError("Smoke report JSON is missing the status field")
    if payload["status"] != "ok":
        raise ValueError(f"Smoke report status is not ok: {payload['status']!r}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a CryptoHunter smoke report JSON file.")
    parser.add_argument("report_path")
    args = parser.parse_args(argv)
    load_smoke_report(args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
