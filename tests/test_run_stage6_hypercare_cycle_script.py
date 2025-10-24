from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts import run_stage6_hypercare_cycle
from scripts._market_intel_paths import resolve_market_intel_path


class _FakeResult:
    def __init__(self, output_path: Path, signature_path: Path, payload: dict[str, object]) -> None:
        self.output_path = output_path
        self.signature_path = signature_path
        self.payload = payload
        self.observability = None
        self.resilience = None
        self.portfolio = None


def test_cli_parses_minimal_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary_path = tmp_path / "summary.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"summary:\n  path: {summary_path.as_posix()}\n",
        encoding="utf-8",
    )

    def fake_cycle_factory(config: run_stage6_hypercare_cycle.Stage6HypercareConfig, **_: object) -> object:
        assert config.output_path == summary_path

        class _FakeCycle:
            def __init__(self, cfg: run_stage6_hypercare_cycle.Stage6HypercareConfig) -> None:
                self._config = cfg

            def run(self) -> _FakeResult:
                payload = {
                    "overall_status": "ok",
                    "components": {
                        "observability": {"status": "skipped"},
                        "resilience": {"status": "skipped"},
                        "portfolio": {"status": "skipped"},
                    },
                    "issues": [],
                    "warnings": [],
                }
                output = self._config.output_path
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload), encoding="utf-8")
                signature_path = output.with_suffix(".sig")
                signature_path.write_text("{}", encoding="utf-8")
                return _FakeResult(output, signature_path, payload)

        return _FakeCycle(config)

    monkeypatch.setattr(run_stage6_hypercare_cycle, "Stage6HypercareCycle", fake_cycle_factory)

    exit_code = run_stage6_hypercare_cycle.run(["--config", str(config_path)])
    assert exit_code == 0
    assert summary_path.exists()


def test_resolve_market_intel_accepts_direct_file(tmp_path: Path) -> None:
    target = tmp_path / "market_intel_stage6_core_20240101T000000Z.json"
    target.write_text("{}", encoding="utf-8")

    resolved = resolve_market_intel_path(
        target,
        str(target),
        environment="binance_paper",
        governor="stage6_core",
    )

    assert resolved == target


def test_resolve_market_intel_prefers_latest_from_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = tmp_path / "market_intel"
    directory.mkdir()
    older = directory / "market_intel_stage6_core_20240101T000000Z.json"
    newer = directory / "market_intel_stage6_core_20240201T000000Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    os.utime(newer, None)

    resolved = resolve_market_intel_path(
        directory,
        str(directory),
        environment="binance_paper",
        governor="stage6_core",
    )

    captured = capsys.readouterr()
    assert "Wybrano raport Market Intel" in captured.err
    assert resolved == newer


def test_resolve_market_intel_expands_timestamp_pattern(tmp_path: Path) -> None:
    directory = tmp_path / "market_intel"
    directory.mkdir()
    candidate = directory / "market_intel_stage6_core_20240301T000000Z.json"
    candidate.write_text("{}", encoding="utf-8")

    resolved = resolve_market_intel_path(
        directory / "market_intel_stage6_core_<timestamp>.json",
        str(directory / "market_intel_stage6_core_<timestamp>.json"),
        environment="binance_paper",
        governor="stage6_core",
    )

    assert resolved == candidate


def test_resolve_market_intel_missing_file_suggests_command(tmp_path: Path) -> None:
    pattern = tmp_path / "market_intel_stage6_core_<timestamp>.json"

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_market_intel_path(
            pattern,
            str(pattern),
            environment="binance_paper",
            governor="stage6_core",
        )

    message = str(excinfo.value)
    assert "build_market_intel_metrics.py" in message
    assert "--output" in message


def test_resolve_market_intel_uses_default_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("var/market_intel/market_intel_stage6_core_20240501T010203Z.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")

    resolved = resolve_market_intel_path(
        None,
        None,
        environment="binance_paper",
        governor="stage6_core",
    )

    assert resolved == target


def test_resolve_market_intel_checks_fallback_directories(tmp_path: Path) -> None:
    primary = tmp_path / "missing"
    fallback = tmp_path / "alt"
    fallback.mkdir()
    candidate = fallback / "market_intel_stage6_core_20240601T010203Z.json"
    candidate.write_text("{}", encoding="utf-8")

    resolved = resolve_market_intel_path(
        primary,
        str(primary),
        environment="binance_paper",
        governor="stage6_core",
        fallback_directories=[fallback],
    )

    assert resolved == candidate
