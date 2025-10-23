from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import live_execution_dry_run
from tests._exchange_adapter_helpers import StubExchangeAdapter


def _stub_factory(credentials, *, environment=None, settings=None):  # noqa: ANN001
    return StubExchangeAdapter(credentials, name=credentials.key_id)


@pytest.mark.integration
def test_live_execution_dry_run_generates_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_execution_dry_run,
        "_resolve_adapter_factories",
        lambda: {"binance_spot": _stub_factory, "kraken_spot": _stub_factory},
    )

    report_path = tmp_path / "dry_run_report.json"
    decision_log = tmp_path / "dry_run_decision_log.jsonl"

    exit_code = live_execution_dry_run.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_live",
            "--report",
            str(report_path),
            "--decision-log",
            str(decision_log),
            "--decision-log-hmac-key",
            "x" * 48,
            "--decision-log-key-id",
            "dry-run-test",
        ]
    )

    assert exit_code == 0

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["environments"], "Raport powinien zawierać środowiska"

    env_report = payload["environments"][0]
    assert env_report["environment"] == "binance_live"
    assert env_report["status"] == "PASS"
    assert env_report["simulation"]["orders_executed"] > 0

    log_lines = decision_log.read_text(encoding="utf-8").splitlines()
    assert log_lines, "Decision log powinien zawierać wpis"
    log_entry = json.loads(log_lines[-1])
    assert log_entry["status"] == "PASS"
    assert log_entry["signature"]["key_id"] == "dry-run-test"
