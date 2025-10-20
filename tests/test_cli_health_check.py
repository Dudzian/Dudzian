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
        self._credentials: tuple[str | None, str | None, str | None] = (None, None, None)
        self._configured_settings: Mapping[str, object] | None = None
        self.native_adapter_mode: Mode | None = None
        self.load_markets_calls = 0
        self.fetch_balance_calls = 0
        self.watchdog_config: dict[str, Mapping[str, object]] | None = None
        self._health_checks: Sequence | None = None
        self.paper_variant: str | None = None
        self.paper_balance: tuple[float, str | None] | None = None
        self.paper_fee_rate: float | None = None
        self.paper_simulator_settings: Mapping[str, object] | None = None
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

    def set_credentials(
        self,
        api_key: str | None,
        secret: str | None,
        *,
        passphrase: str | None = None,
    ) -> None:
        self._credentials = (api_key, secret, passphrase)

    def configure_native_adapter(self, *, settings: Mapping[str, object], mode: Mode | None = None) -> None:
        self._configured_settings = settings
        self.native_adapter_mode = mode

    def set_paper_variant(self, variant: str) -> None:
        self.paper_variant = variant

    def set_paper_balance(self, amount: float, asset: str | None = None) -> None:
        self.paper_balance = (float(amount), asset)

    def set_paper_fee_rate(self, fee_rate: float) -> None:
        self.paper_fee_rate = float(fee_rate)

    def configure_paper_simulator(self, **settings: object) -> None:
        self.paper_simulator_settings = dict(settings)

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

    def fetch_ticker(self, symbol: str) -> Mapping[str, object]:
        return {"last": 100.0, "symbol": symbol}

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
passphrase = "PHRASE"
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
    assert manager._credentials == ("KEY", "SECRET", "PHRASE")
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


def test_health_check_cli_prefers_cli_credentials(tmp_path: Path) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key = "PROFILE_KEY"
secret = "PROFILE_SECRET"
passphrase = "PROFILE_PHRASE"
mode = "margin"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--key",
            "CLI_KEY",
            "--secret",
            "CLI_SECRET",
            "--passphrase",
            "CLI_PHRASE",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager._credentials == ("CLI_KEY", "CLI_SECRET", "CLI_PHRASE")


def test_health_check_cli_loads_environment_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
paper_margin:
  description: Margin paper config
  exchange_manager:
    mode: paper
    paper_variant: margin
    paper_initial_cash: 50000
    paper_cash_asset: USDT
    paper_fee_rate: 0.0015
    simulator:
      leverage_limit: 6
      maintenance_margin_ratio: 0.1
    watchdog:
      retry_policy:
        max_attempts: 5
        base_delay: 0.25
  credentials:
    api_key: ${ENV_KEY}
    secret: ${ENV_SECRET}
  health_check:
    public_symbol: ETH/USDT
    skip_private: true
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("ENV_KEY", "LIVEKEY")
    monkeypatch.setenv("ENV_SECRET", "LIVESECRET")

    exit_code = main(
        [
            "health-check",
            "--environment",
            "paper_margin",
            "--environment-config",
            str(env_yaml),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.exchange_id == "binance"
    assert manager.mode is Mode.PAPER
    assert manager.paper_variant == "margin"
    assert manager.paper_balance == (50_000.0, "USDT")
    assert manager.paper_fee_rate == pytest.approx(0.0015)
    assert manager.paper_simulator_settings == {
        "leverage_limit": 6,
        "maintenance_margin_ratio": 0.1,
    }
    assert manager.watchdog_config and manager.watchdog_config["retry_policy"]["max_attempts"] == 5
    assert manager._credentials == ("LIVEKEY", "LIVESECRET", None)
    assert manager.fetch_balance_calls == 0
    assert manager.load_markets_calls == 0
    assert manager._health_checks and manager._health_checks[0].name == "public_api"

    captured = capsys.readouterr()
    assert "Overall status" in captured.out


def test_health_check_cli_environment_requires_environment_name(tmp_path: Path) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text("paper: {}", encoding="utf-8")

    exit_code = main(
        [
            "health-check",
            "--environment-config",
            str(env_yaml),
            "--exchange",
            "binance",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2


def test_health_check_cli_environment_missing_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: kraken
paper:
  exchange_manager:
    mode: paper
  credentials:
    api_key: ${MISSING_KEY}
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "health-check",
            "--environment",
            "paper",
            "--environment-config",
            str(env_yaml),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2


def test_health_check_cli_reads_credentials_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
mode = "margin"
""",
    )

    monkeypatch.setenv("HC_KEY", "ENV_KEY")
    monkeypatch.setenv("HC_SECRET", "ENV_SECRET")
    monkeypatch.setenv("HC_PASSPHRASE", "ENV_PHRASE")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--key-env",
            "HC_KEY",
            "--secret-env",
            "HC_SECRET",
            "--passphrase-env",
            "HC_PASSPHRASE",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager._credentials == ("ENV_KEY", "ENV_SECRET", "ENV_PHRASE")


def test_health_check_cli_requires_present_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--key-env",
            "MISSING_KEY",
            "--skip-private",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2


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
