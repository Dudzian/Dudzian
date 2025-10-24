from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_describe_regime_workflow_reports_availability_and_history(tmp_path: Path) -> None:
    availability_payload = [
        {
            "regime": "trend",
            "version": {
                "hash": "abc123",
                "signature": {"key_id": "regime", "value": "deadbeef"},
                "issued_at": "2024-05-25T12:00:00+00:00",
                "metadata": {
                    "name": "trend-core",
                    "license_tiers": ["standard"],
                    "required_data": ["ohlcv", "order_book"],
                },
            },
            "ready": True,
            "blocked_reason": None,
            "missing_data": [],
            "license_issues": [],
            "schedule_blocked": False,
        },
        {
            "regime": None,
            "version": {
                "hash": "fallback456",
                "signature": {"key_id": "regime", "value": "cafebabe"},
                "issued_at": "2024-05-25T12:30:00+00:00",
                "metadata": {
                    "name": "fallback",
                    "license_tiers": ["standard", "pro"],
                    "required_data": ["order_book"],
                },
            },
            "ready": False,
            "blocked_reason": "missing_data",
            "missing_data": ["order_book"],
            "license_issues": ["missing_capability"],
            "schedule_blocked": True,
        },
    ]
    history_payload = [
        {
            "regime": "trend",
            "preset_regime": "trend",
            "activated_at": "2024-05-25T12:05:00+00:00",
            "used_fallback": False,
            "blocked_reason": None,
            "missing_data": [],
            "license_issues": [],
            "recommendation": "trend-core",
            "version": availability_payload[0]["version"],
            "preset": {
                "name": "trend-core",
                "strategies": [
                    {"name": "trend", "engine": "daily_trend_momentum"},
                ],
            },
            "assessment": {
                "confidence": 0.9,
                "risk_score": 0.12,
                "metrics": {"volatility": 0.2},
            },
        },
        {
            "regime": "mean_reversion",
            "preset_regime": None,
            "activated_at": "2024-05-25T12:40:00+00:00",
            "used_fallback": True,
            "blocked_reason": "missing_data",
            "missing_data": ["order_book"],
            "license_issues": ["missing_capability"],
            "version": availability_payload[1]["version"],
            "preset": {"name": "fallback"},
            "assessment": {
                "regime": "mean_reversion",
                "confidence": 0.75,
                "risk_score": 0.5,
            },
        },
    ]

    _write_json(tmp_path / "availability.json", availability_payload)
    _write_json(tmp_path / "activation_history.json", history_payload)

    script_path = Path("scripts/ui_config_bridge.py")
    config_path = Path("config/core.yaml")
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        tuple(filter(None, [str(Path.cwd()), pythonpath]))
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--describe-regime-workflow",
            "--regime-workflow-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    json_start = next(idx for idx, line in enumerate(lines) if line.lstrip().startswith("{"))
    payload = json.loads("\n".join(lines[json_start:]))
    workflow = payload["regime_workflow"]

    availability = workflow["availability"]
    assert len(availability) == 2
    first_entry = availability[0]
    assert first_entry["regime"] == "trend"
    assert first_entry["version"]["hash"] == "abc123"
    assert isinstance(first_entry["version"]["metadata"]["license_tiers"], list)

    fallback_entry = availability[1]
    assert fallback_entry["regime"] is None
    assert fallback_entry["schedule_blocked"] is True
    assert "missing_capability" in fallback_entry["license_issues"]

    history = workflow["history"]
    activations = history["activations"]
    assert len(activations) == 2
    assert activations[1]["used_fallback"] is True
    assert activations[1]["version"]["hash"] == "fallback456"

    stats = history["stats"]["history"]
    assert stats["fallback_count"] == 1
    reasons = {entry["reason"] for entry in stats["blocked_reasons"]}
    assert "missing_data" in reasons
