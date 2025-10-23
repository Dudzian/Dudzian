"""Ensure Stage3 checklist CLI invocations keep working with compatibility aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import generate_mtls_bundle, live_execution_dry_run
from tests._exchange_adapter_helpers import StubExchangeAdapter


def _stub_factory(credentials, *, environment=None, settings=None):  # noqa: ANN001
    return StubExchangeAdapter(credentials, name=credentials.key_id)


def test_stage3_generate_mtls_bundle_cli_accepts_output_alias(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"

    exit_code = generate_mtls_bundle.main(
        [
            "--output",
            str(bundle_dir),
            "--bundle-name",
            "stage3-oem",
        ]
    )

    assert exit_code == 0

    metadata_path = bundle_dir / "bundle.json"
    assert metadata_path.exists()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["bundle"] == "stage3-oem"


@pytest.mark.integration
def test_stage3_live_execution_dry_run_cli_supports_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        live_execution_dry_run,
        "_resolve_adapter_factories",
        lambda: {"binance_spot": _stub_factory, "kraken_spot": _stub_factory},
    )

    report_path = tmp_path / "stage3_report.json"
    decision_log = tmp_path / "stage3_audit.jsonl"

    exit_code = live_execution_dry_run.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_live",
            "--report",
            str(report_path),
            "--audit-json",
            str(decision_log),
            "--decision-log-hmac-key",
            "z" * 48,
            "--decision-log-key-id",
            "stage3-check",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Uruchomiono w trybie dry-run" in captured.out

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["environments"], "Raport powinien zawierać środowiska"

    log_lines = decision_log.read_text(encoding="utf-8").splitlines()
    assert log_lines, "Decision log powinien zawierać wpis"

    log_entry = json.loads(log_lines[-1])
    assert log_entry["status"] == "PASS"
    assert log_entry["signature"]["key_id"] == "stage3-check"
