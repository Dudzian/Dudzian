from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _write_overrides(path: Path) -> None:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=2)
    payload = {
        "schema": "stage6.observability.alert_overrides",
        "overrides": [
            {
                "alert": "latency",
                "status": "breach",
                "severity": "critical",
                "reason": "Latency przekracza próg",
                "indicator": "latency",
                "created_at": now.isoformat().replace("+00:00", "Z"),
                "expires_at": expires.isoformat().replace("+00:00", "Z"),
                "tags": ["stage6"],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sync_alert_overrides_dashboard_script(tmp_path: Path) -> None:
    overrides_path = tmp_path / "overrides.json"
    dashboard_path = tmp_path / "dashboard.json"
    output_path = tmp_path / "annotations.json"

    _write_overrides(overrides_path)
    dashboard_path.write_text(
        json.dumps({"uid": "stage6-resilience-ops", "title": "Stage6"}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sync_alert_overrides_dashboard.py",
            "--overrides",
            str(overrides_path),
            "--dashboard",
            str(dashboard_path),
            "--output",
            str(output_path),
            "--panel-id",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dashboard_uid"] == "stage6-resilience-ops"
    assert payload["annotations"]
    annotation = payload["annotations"][0]
    assert annotation["panelId"] == 7
    assert annotation["title"].startswith("latency")
