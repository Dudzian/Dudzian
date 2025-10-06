"""Testy skryptu synchronizującego dziennik JSONL smoke testów."""
from __future__ import annotations

import json
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys_path_added = False

if not sys_path_added:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys_path_added = True

from scripts import sync_paper_smoke_json as sync_script  # noqa: E402


def _write_log(
    path: Path,
    *,
    record_id: str = "J-20250102T030405-abcd",
    timestamp: str = "2025-01-02T03:04:05+00:00",
) -> None:
    payload = {
        "record_id": record_id,
        "timestamp": timestamp,
        "environment": "binance_paper",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_missing_json_log_returns_error(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "missing.jsonl"

    exit_code = sync_script.main(
        [
            "--environment",
            "binance_paper",
            "--json-log",
            str(log_path),
            "--json",
        ]
    )

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert output["error"] == "missing_json_log"


def test_missing_config_returns_error(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path)

    def _raise(*args, **kwargs):  # noqa: ANN001, D401 - pomocniczy stub
        raise FileNotFoundError("config")

    monkeypatch.setattr(sync_script, "load_core_config", _raise)

    exit_code = sync_script.main(
        [
            "--environment",
            "binance_paper",
            "--json-log",
            str(log_path),
            "--json",
        ]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert output["error"] == "missing_config"


def test_successful_sync_uses_latest_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "log.jsonl"
    _write_log(
        log_path,
        record_id="J-20250203T040506-1234",
        timestamp="2025-02-03T04:05:06+00:00",
    )

    config = types.SimpleNamespace(backend="local")
    reporting = types.SimpleNamespace(paper_smoke_json_sync=config)
    monkeypatch.setattr(
        sync_script,
        "load_core_config",
        lambda path: types.SimpleNamespace(reporting=reporting),
    )

    calls: dict[str, object] = {}

    class DummySynchronizer:
        def __init__(self, cfg, *, secret_manager=None) -> None:  # noqa: D401
            assert cfg is config
            self._secret_manager = secret_manager

        def sync(self, json_log_path, *, environment, record_id, timestamp):  # noqa: ANN001
            calls["json_log_path"] = Path(json_log_path)
            calls["environment"] = environment
            calls["record_id"] = record_id
            assert isinstance(timestamp, datetime)
            calls["timestamp"] = timestamp.astimezone(timezone.utc)
            return types.SimpleNamespace(
                backend="local",
                location="file:///audit/copy.jsonl",
                metadata={"acknowledged": "true"},
            )

        @staticmethod
        def resolve_config(reporting_cfg):  # noqa: ANN001 - zachowanie oryginalnej metody
            return getattr(reporting_cfg, "paper_smoke_json_sync", None)

    monkeypatch.setattr(sync_script, "PaperSmokeJsonSynchronizer", DummySynchronizer)

    exit_code = sync_script.main(
        [
            "--environment",
            "binance_paper",
            "--json-log",
            str(log_path),
            "--json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["backend"] == "local"
    assert output["location"] == "file:///audit/copy.jsonl"
    assert output["record_id"] == "J-20250203T040506-1234"
    assert Path(output["json_log_path"]).resolve() == log_path.resolve()

    assert calls["record_id"] == "J-20250203T040506-1234"
    assert calls["environment"] == "binance_paper"
    assert calls["json_log_path"].resolve() == log_path.resolve()
    # timestamp z logu powinien zostać użyty jako punkt odniesienia
    assert calls["timestamp"] == datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


def test_dry_run_skips_synchronizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path)

    config = types.SimpleNamespace(backend="local")
    reporting = types.SimpleNamespace(paper_smoke_json_sync=config)
    monkeypatch.setattr(
        sync_script,
        "load_core_config",
        lambda path: types.SimpleNamespace(reporting=reporting),
    )

    class DummySynchronizer:
        called = False

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
            DummySynchronizer.called = True

        @staticmethod
        def resolve_config(reporting_cfg):  # noqa: ANN001
            return getattr(reporting_cfg, "paper_smoke_json_sync", None)

    monkeypatch.setattr(sync_script, "PaperSmokeJsonSynchronizer", DummySynchronizer)

    exit_code = sync_script.main(
        [
            "--environment",
            "binance_paper",
            "--json-log",
            str(log_path),
            "--json",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "skipped"
    assert output["backend"] == "local"
    assert DummySynchronizer.called is False

