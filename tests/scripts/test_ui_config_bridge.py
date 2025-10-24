from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path

from bot_core.ai import MarketRegime, RegimeSnapshot, RegimeSummary, RiskLevel
from bot_core.decision.models import DecisionCandidate


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_summary_payload() -> dict[str, object]:
    float_fields = {
        name
        for name in RegimeSummary.__dataclass_fields__.keys()
        if name not in {"regime", "risk_level", "regime_streak", "history"}
    }
    summary_kwargs: dict[str, object] = {
        name: 0.1 for name in float_fields
    }
    summary_kwargs["regime"] = MarketRegime.TREND
    summary_kwargs["risk_level"] = RiskLevel.CALM
    summary_kwargs["regime_streak"] = 7
    summary_kwargs["history"] = (
        RegimeSnapshot(
            regime=MarketRegime.TREND,
            confidence=0.82,
            risk_score=0.27,
            drawdown=0.05,
            volatility=0.18,
            volume_trend=0.33,
            volatility_ratio=0.52,
            return_skew=0.02,
            return_kurtosis=0.11,
            volume_imbalance=0.25,
        ),
    )
    return dict(RegimeSummary(**summary_kwargs).to_dict())


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
                "issued_at": None,
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
            "summary": _build_summary_payload(),
            "decision_candidates": [
                dict(
                    DecisionCandidate(
                        strategy="daily_trend_momentum",
                        action="activate",
                        risk_profile="trend",
                        symbol="BTC/USDT",
                        notional=125000.0,
                        expected_return_bps=15.0,
                        expected_probability=0.73,
                        metadata={
                            "preset": "trend-core",
                            "preset_version": availability_payload[0]["version"]["hash"],
                        },
                    ).to_mapping()
                )
            ],
        },
        {
            "regime": "mean_reversion",
            "preset_regime": None,
            "activated_at": "2024-05-25T12:40:00+00:00",
            "used_fallback": True,
            "blocked_reason": "missing_data",
            "missing_data": ["order_book"],
            "license_issues": ["missing_capability"],
            "version_hash": availability_payload[1]["version"]["hash"],
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
    assert fallback_entry["version"]["issued_at"] == "1970-01-01T00:00:00+00:00"

    availability_stats = workflow["availability_stats"]
    assert availability_stats["total"] == 2
    assert availability_stats["ready"] == 1
    assert availability_stats["blocked"] == 1
    assert availability_stats["schedule_blocked"] == 1
    regimes = {entry["regime"]: entry["count"] for entry in availability_stats["regime_counts"]}
    assert regimes["trend"] == 1
    assert regimes[None] == 1
    blocked_reasons = {entry["reason"]: entry["count"] for entry in availability_stats["blocked_reasons"]}
    assert blocked_reasons["missing_data"] == 1
    missing_data = {entry["name"]: entry["count"] for entry in availability_stats["missing_data"]}
    assert missing_data["order_book"] == 1
    license_issues = {entry["issue"]: entry["count"] for entry in availability_stats["license_issues"]}
    assert license_issues["missing_capability"] == 1

    history = workflow["history"]
    activations = history["activations"]
    assert len(activations) == 2
    assert activations[1]["used_fallback"] is True
    assert activations[1]["version"]["hash"] == "fallback456"
    assert activations[1]["version"]["metadata"]["name"] == "fallback"
    assert activations[0]["summary"]["regime"] == "trend"
    assert activations[0]["summary"]["risk_level"] == "calm"
    assert activations[0]["summary"]["regime_streak"] == 7
    history_points = activations[0]["summary"]["history"]
    assert isinstance(history_points, list) and history_points
    assert history_points[0]["volatility_ratio"] == 0.52
    candidates = activations[0]["decision_candidates"]
    assert len(candidates) == 1
    first_candidate = candidates[0]
    assert first_candidate["strategy"] == "daily_trend_momentum"
    assert first_candidate["expected_probability"] == 0.73
    assert first_candidate["metadata"]["preset_version"] == "abc123"

    stats = history["stats"]["history"]
    assert stats["fallback_count"] == 1
    reasons = {entry["reason"] for entry in stats["blocked_reasons"]}
    assert "missing_data" in reasons

    fallback_entries = history["fallbacks"]
    assert len(fallback_entries) == 1
    fallback_entry = fallback_entries[0]
    assert fallback_entry["regime"] == "mean_reversion"
    assert fallback_entry["blocked_reason"] == "missing_data"
    assert fallback_entry["version"]["hash"] == "fallback456"
    assert "order_book" in fallback_entry["missing_data"]

    versions = workflow["versions"]
    assert set(versions.keys()) == {"abc123", "fallback456"}
    assert versions["abc123"]["signature"]["key_id"] == "regime"
    assert versions["fallback456"]["issued_at"] == "1970-01-01T00:00:00+00:00"
