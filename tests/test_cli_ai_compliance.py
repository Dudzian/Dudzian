import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.ai.data_monitoring import ComplianceSignOffError
from bot_core.cli import show_ai_compliance


def _build_args(**overrides):
    defaults = {
        "output_format": "text",
        "limit": None,
        "audit_root": None,
        "roles": [],
        "enforce": False,
        "since": None,
        "data_quality_category": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_show_ai_compliance_prints_pending_sign_offs(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    dq_data = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "pending"},
                "compliance": {"status": "approved"},
            },
        },
    )
    drift_data = (
        {
            "category": "drift",
            "drift_score": 1.2,
            "threshold": 1.0,
            "sign_off": {
                "risk": {"status": "investigating"},
                "compliance": {"status": "approved"},
            },
            "report_path": "drift.json",
        },
    )

    captured_kwargs: dict[str, dict[str, object]] = {}

    def _fake_load_dq(**kwargs):
        captured_kwargs["dq"] = dict(kwargs)
        return dq_data

    def _fake_load_drift(**kwargs):
        captured_kwargs["drift"] = dict(kwargs)
        return drift_data

    def _fake_collect(*, data_quality_reports, drift_reports, roles):
        assert data_quality_reports is dq_data
        assert drift_reports is drift_data
        assert roles == ("risk",)
        return {
            "risk": (
                {
                    "category": "completeness",
                    "status": "pending",
                    "report_path": "dq.json",
                },
                {
                    "category": "drift",
                    "status": "investigating",
                    "report_path": "drift.json",
                },
            ),
            "compliance": (),
        }

    def _fail_ensure(**_kwargs):
        raise AssertionError("ensure_compliance_sign_offs should not be invoked")

    monkeypatch.setattr(
        "bot_core.cli.load_recent_data_quality_reports", _fake_load_dq
    )
    monkeypatch.setattr(
        "bot_core.cli.load_recent_drift_reports", _fake_load_drift
    )
    monkeypatch.setattr(
        "bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect
    )
    monkeypatch.setattr("bot_core.cli.ensure_compliance_sign_offs", _fail_ensure)

    args = _build_args(limit=5, roles=["risk"], audit_root="/tmp/audit")

    assert show_ai_compliance(args) == 0
    captured = capfd.readouterr().out
    assert "Katalog audytu: /tmp/audit" in captured
    assert "Brakujące podpisy compliance" in captured
    assert "completeness" in captured
    assert captured_kwargs["dq"]["limit"] == 5
    assert captured_kwargs["drift"]["limit"] == 5
    assert captured_kwargs["dq"]["audit_root"] == Path("/tmp/audit")
    assert captured_kwargs["drift"]["audit_root"] == Path("/tmp/audit")
    assert "category" not in captured_kwargs["dq"]


def test_show_ai_compliance_json_enforce(monkeypatch: pytest.MonkeyPatch, capfd) -> None:
    captured_limits: list[int] = []

    def _fake_load_dq(**kwargs):
        captured_limits.append(kwargs["limit"])
        return ()

    def _fake_load_drift(**kwargs):
        captured_limits.append(kwargs["limit"])
        return ()

    def _fake_collect(**_kwargs):
        raise AssertionError("collect_pending_compliance_sign_offs should not be used")

    def _fake_ensure(**_kwargs):
        raise ComplianceSignOffError(
            {
                "risk": ({"category": "dq", "status": "pending"},),
                "compliance": (),
            }
        )

    monkeypatch.setattr(
        "bot_core.cli.load_recent_data_quality_reports", _fake_load_dq
    )
    monkeypatch.setattr(
        "bot_core.cli.load_recent_drift_reports", _fake_load_drift
    )
    monkeypatch.setattr(
        "bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect
    )
    monkeypatch.setattr("bot_core.cli.ensure_compliance_sign_offs", _fake_ensure)

    args = _build_args(output_format="json-pretty", enforce=True)
    exit_code = show_ai_compliance(args)
    assert exit_code == 3

    payload = json.loads(capfd.readouterr().out)
    assert payload["enforced"] is True
    assert payload["limit"] == 20
    assert set(payload["roles"]) == {"risk", "compliance"}
    assert payload["pending_sign_off"]["risk"][0]["category"] == "dq"
    assert payload["missing_sign_off"]["risk"] == 1
    assert captured_limits == [20, 20]


def test_show_ai_compliance_filters_since(monkeypatch: pytest.MonkeyPatch, capfd) -> None:
    dq_reports = (
        {"timestamp": "2024-01-01T00:00:00Z", "category": "completeness"},
        {"timestamp": "2024-02-01T12:00:00Z", "category": "completeness"},
    )
    drift_reports = (
        {"timestamp": "2024-02-02T00:00:00Z", "category": "drift"},
        {"timestamp": "2023-12-31T23:59:59Z", "category": "drift"},
    )

    captured_kwargs: dict[str, dict[str, object]] = {}
    filter_calls: list[tuple[tuple[dict[str, object], ...], datetime]] = []

    def _fake_load_dq(**kwargs):
        captured_kwargs["dq"] = dict(kwargs)
        return dq_reports

    def _fake_load_drift(**kwargs):
        captured_kwargs["drift"] = dict(kwargs)
        return drift_reports

    def _fake_filter(reports, *, since):
        filter_calls.append((tuple(reports), since))
        if reports is dq_reports:
            return (dq_reports[1],)
        return (drift_reports[0],)

    def _fake_collect(*, data_quality_reports, drift_reports, roles):
        assert data_quality_reports == (dq_reports[1],)
        assert drift_reports == (drift_reports[0],)
        assert roles is None
        return {"risk": (), "compliance": ()}

    monkeypatch.setattr("bot_core.cli.load_recent_data_quality_reports", _fake_load_dq)
    monkeypatch.setattr("bot_core.cli.load_recent_drift_reports", _fake_load_drift)
    monkeypatch.setattr("bot_core.cli.filter_audit_reports_since", _fake_filter)
    monkeypatch.setattr("bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect)
    monkeypatch.setattr(
        "bot_core.cli._resolve_since_cutoff",
        lambda value: datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    args = _build_args(since="48h", data_quality_category="completeness")
    exit_code = show_ai_compliance(args)
    assert exit_code == 0

    output = capfd.readouterr().out
    assert "Minimalny znacznik czasu" in output
    assert captured_kwargs["dq"]["category"] == "completeness"
    assert "category" not in captured_kwargs["drift"]
    assert filter_calls
    assert all(call[1] == datetime(2024, 2, 1, tzinfo=timezone.utc) for call in filter_calls)
