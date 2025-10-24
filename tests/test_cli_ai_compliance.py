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
        "include_tags": [],
        "exclude_tags": [],
        "include_statuses": [],
        "exclude_statuses": [],
        "include_report_statuses": [],
        "exclude_report_statuses": [],
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


def test_show_ai_compliance_filters_tags(monkeypatch: pytest.MonkeyPatch, capfd) -> None:
    dq_reports = (
        {"tags": ["pipeline", "nightly"], "category": "completeness"},
        {"tags": ["ignore"], "category": "legacy"},
    )
    drift_reports = (
        {"tags": ["pipeline"], "category": "drift"},
        {"tags": ["ignore", "noise"], "category": "drift"},
    )

    filter_calls: list[tuple[tuple[dict[str, object], ...], tuple[str, ...], tuple[str, ...]]] = []

    def _fake_load_dq(**_kwargs):
        return dq_reports

    def _fake_load_drift(**_kwargs):
        return drift_reports

    def _fake_filter_by_tags(reports, *, include, exclude):
        filter_calls.append((tuple(reports), tuple(include or ()), tuple(exclude or ())))
        if reports is dq_reports:
            return (dq_reports[0],)
        return (drift_reports[0],)

    def _fake_collect(*, data_quality_reports, drift_reports, roles):
        assert data_quality_reports == (dq_reports[0],)
        assert drift_reports == (drift_reports[0],)
        assert roles is None
        return {"risk": (), "compliance": ()}

    monkeypatch.setattr("bot_core.cli.load_recent_data_quality_reports", _fake_load_dq)
    monkeypatch.setattr("bot_core.cli.load_recent_drift_reports", _fake_load_drift)
    monkeypatch.setattr("bot_core.cli.filter_audit_reports_by_tags", _fake_filter_by_tags)
    monkeypatch.setattr("bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect)

    args = _build_args(include_tags=["pipeline"], exclude_tags=["ignore"])
    exit_code = show_ai_compliance(args)

    assert exit_code == 0
    output = capfd.readouterr().out
    assert "Wymagane tagi: pipeline" in output
    assert "Wykluczone tagi: ignore" in output
    assert len(filter_calls) == 2
    for _reports, include, exclude in filter_calls:
        assert include == ("pipeline",)
        assert exclude == ("ignore",)


def test_show_ai_compliance_filters_report_status(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    dq_reports = (
        {"status": "alert", "category": "dq-keep"},
        {"status": "ok", "category": "dq-drop"},
    )
    drift_reports = (
        {"status": "warning", "category": "drift-keep"},
        {"status": "alert", "category": "drift-drop"},
    )

    filter_calls: list[tuple[tuple[dict[str, object], ...], tuple[str, ...], tuple[str, ...]]] = []

    def _fake_load_dq(**_kwargs):
        return dq_reports

    def _fake_load_drift(**_kwargs):
        return drift_reports

    def _fake_filter_by_tags(reports, *, include, exclude):
        return tuple(reports)

    def _fake_filter_by_status(reports, *, include, exclude):
        filter_calls.append((tuple(reports), tuple(include or ()), tuple(exclude or ())))
        if reports is dq_reports:
            return (dq_reports[0],)
        return (drift_reports[0],)

    def _fake_collect(*, data_quality_reports, drift_reports, roles):
        assert data_quality_reports == (dq_reports[0],)
        assert drift_reports == (drift_reports[0],)
        assert roles is None
        return {"risk": (), "compliance": ()}

    monkeypatch.setattr("bot_core.cli.load_recent_data_quality_reports", _fake_load_dq)
    monkeypatch.setattr("bot_core.cli.load_recent_drift_reports", _fake_load_drift)
    monkeypatch.setattr("bot_core.cli.filter_audit_reports_by_tags", _fake_filter_by_tags)
    monkeypatch.setattr("bot_core.cli.filter_audit_reports_by_status", _fake_filter_by_status)
    monkeypatch.setattr("bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect)

    args = _build_args(
        include_report_statuses=["alert"],
        exclude_report_statuses=["ok"],
    )
    exit_code = show_ai_compliance(args)

    assert exit_code == 0
    output = capfd.readouterr().out
    assert "Wymagane statusy raportów: alert" in output
    assert "Wykluczone statusy raportów: ok" in output
    assert len(filter_calls) == 2
    for _reports, include, exclude in filter_calls:
        assert include == ("alert",)
        assert exclude == ("ok",)


def test_show_ai_compliance_filters_sign_off_status(
    monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    dq_reports = (
        {
            "sign_off": {
                "risk": {"status": "Pending"},
                "compliance": {"status": "approved"},
            },
            "category": "dq-keep",
        },
        {
            "sign_off": {
                "risk": {"status": "waived"},
                "compliance": {"status": "approved"},
            },
            "category": "dq-waived",
        },
    )
    drift_reports = (
        {
            "sign_off": {
                "risk": {"status": "investigating"},
            },
            "category": "drift-keep",
        },
        {
            "sign_off": {
                "risk": {"status": "approved"},
            },
            "category": "drift-drop",
        },
    )

    filter_calls: list[
        tuple[
            tuple[dict[str, object], ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...] | None,
        ]
    ] = []

    def _fake_load_dq(**_kwargs):
        return dq_reports

    def _fake_load_drift(**_kwargs):
        return drift_reports

    def _fake_filter_by_tags(reports, *, include, exclude):
        return tuple(reports)

    def _fake_filter_by_status(reports, *, include, exclude, roles):
        filter_calls.append((tuple(reports), tuple(include or ()), tuple(exclude or ()), roles))
        if reports is dq_reports:
            return (dq_reports[0],)
        return (drift_reports[0],)

    def _fake_collect(*, data_quality_reports, drift_reports, roles):
        assert data_quality_reports == (dq_reports[0],)
        assert drift_reports == (drift_reports[0],)
        assert roles == ("risk",)
        return {"risk": (), "compliance": ()}

    monkeypatch.setattr("bot_core.cli.load_recent_data_quality_reports", _fake_load_dq)
    monkeypatch.setattr("bot_core.cli.load_recent_drift_reports", _fake_load_drift)
    monkeypatch.setattr("bot_core.cli.filter_audit_reports_by_tags", _fake_filter_by_tags)
    monkeypatch.setattr(
        "bot_core.cli.filter_audit_reports_by_sign_off_status", _fake_filter_by_status
    )
    monkeypatch.setattr("bot_core.cli.collect_pending_compliance_sign_offs", _fake_collect)

    args = _build_args(
        include_statuses=["pending"],
        exclude_statuses=["approved"],
        roles=["risk"],
    )
    exit_code = show_ai_compliance(args)

    assert exit_code == 0
    output = capfd.readouterr().out
    assert "Wymagane statusy podpisów: pending" in output
    assert "Wykluczone statusy podpisów: approved" in output
    assert len(filter_calls) == 2
    for _reports, include, exclude, roles in filter_calls:
        assert include == ("pending",)
        assert exclude == ("approved",)
        assert roles == ("risk",)
