"""Testy interfejsu `bot_core.cli` dla komendy health-check."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest

from bot_core.cli import CLIUsageError, main
from bot_core.exchanges.core import Mode
from bot_core.exchanges.health import HealthCheckResult, HealthStatus


@dataclass
class _FakeMonitor:
    checks: Sequence

    def run(self) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for check in self.checks:
            check.check()
            results.append(
                HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY,
                    latency=0.01,
                    details={},
                )
            )
        return results


class _RecordingManager:
    instances: list["_RecordingManager"] = []

    def __init__(self, exchange_id: str) -> None:
        self.exchange_id = exchange_id
        self.mode = Mode.SPOT
        self._mode_calls: list[dict[str, object]] = []
        self._credentials: tuple[str | None, str | None] = (None, None)
        self._configured_settings: Mapping[str, object] | None = None
        self.load_markets_calls = 0
        self.fetch_balance_calls = 0
        self.watchdog_config: dict[str, Mapping[str, object]] | None = None
        self._health_checks: Sequence | None = None
        _RecordingManager.instances.append(self)

    # API używane przez CLI
    def set_mode(
        self,
        *,
        paper: bool = False,
        spot: bool = False,
        margin: bool = False,
        futures: bool = False,
        testnet: bool = False,
    ) -> None:
        if paper:
            self.mode = Mode.PAPER
        elif margin:
            self.mode = Mode.MARGIN
        elif futures:
            self.mode = Mode.FUTURES
        else:
            self.mode = Mode.SPOT
        self._mode_calls.append(
            {
                "paper": paper,
                "spot": spot,
                "margin": margin,
                "futures": futures,
                "testnet": testnet,
            }
        )

    def set_credentials(self, api_key: str | None, secret: str | None) -> None:
        self._credentials = (api_key, secret)

    def configure_native_adapter(self, *, settings: Mapping[str, object], mode: Mode | None = None) -> None:
        self._configured_settings = settings

    def configure_watchdog(
        self,
        *,
        retry_policy: Mapping[str, object] | None = None,
        circuit_breaker: Mapping[str, object] | None = None,
    ) -> None:
        self.watchdog_config = {}
        if retry_policy:
            self.watchdog_config["retry_policy"] = retry_policy
        if circuit_breaker:
            self.watchdog_config["circuit_breaker"] = circuit_breaker

    def load_markets(self) -> dict[str, object]:
        self.load_markets_calls += 1
        return {"BTC/USDT": {}}

    def fetch_balance(self) -> dict[str, object]:
        self.fetch_balance_calls += 1
        return {"total": {"USDT": 100.0}}

    def create_health_monitor(self, checks: Iterable) -> _FakeMonitor:
        self._health_checks = tuple(checks)
        return _FakeMonitor(self._health_checks)


def _write_profile(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_health_check_cli_uses_profile_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key = "KEY"
secret = "SECRET"
mode = "margin"
valuation_currency = "USD"
[binance.watchdog.retry_policy]
max_attempts = 4
base_delay = 0.1
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.mode is Mode.MARGIN
    assert manager._credentials == ("KEY", "SECRET")
    assert manager._configured_settings == {"valuation_currency": "USD"}
    assert manager.watchdog_config and manager.watchdog_config["retry_policy"]["max_attempts"] == 4
    assert manager.load_markets_calls == 1
    assert manager.fetch_balance_calls == 1

    captured = capsys.readouterr()
    assert "public_api" in captured.out
    assert "private_api" in captured.out


def test_health_check_cli_skips_private_when_no_credentials(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[kraken]
mode = "spot"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "kraken",
            "--credentials-file",
            str(credentials),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.load_markets_calls == 1
    assert manager.fetch_balance_calls == 0

    captured = capsys.readouterr()
    assert "public_api" in captured.out
    assert "private_api" not in captured.out
    assert "Pomijam test private_api" in captured.err


def test_health_check_cli_returns_nonzero_on_missing_checks(tmp_path: Path) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[zonda]
key = "KEY"
secret = "SECRET"
mode = "spot"
[zonda.health_check]
skip_public = true
skip_private = true
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "zonda",
            "--credentials-file",
            str(credentials),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2


def test_health_check_cli_propagates_cli_usage_error() -> None:
    with pytest.raises(CLIUsageError):
        raise CLIUsageError("Błąd testowy")
