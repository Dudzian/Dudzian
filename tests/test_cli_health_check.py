"""Testy interfejsu `bot_core.cli` dla komendy health-check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest

from bot_core.cli import CLIUsageError, main
from bot_core.exchanges.core import Mode
from bot_core.exchanges.errors import ExchangeError, ExchangeThrottlingError
from bot_core.exchanges.health import HealthCheckResult, HealthStatus


@dataclass
class _FakeMonitor:
    checks: Sequence

    def run(self) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for check in self.checks:
            try:
                check.check()
            except Exception as exc:  # noqa: BLE001 - symulujemy zachowanie HealthMonitor
                results.append(
                    HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.UNAVAILABLE,
                        latency=0.01,
                        details={"error": str(exc)},
                    )
                )
            else:
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
        self.ticker_requests: list[str] = []
        self.watchdog_config: dict[str, Mapping[str, object]] | None = None
        self._health_checks: Sequence | None = None
        self.paper_variant: str | None = None
        self.paper_balance: tuple[float, str | None] | None = None
        self.paper_fee_rate: float | None = None
        self._paper_initial_cash = 10_000.0
        self._paper_cash_asset: str | None = "USDT"
        self.paper_simulator_settings: dict[str, float] = {}
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
        key_id: str | None,
        secret: str | None,
        *,
        passphrase: str | None = None,
    ) -> None:
        self._credentials = (key_id, secret, passphrase)

    def configure_native_adapter(self, *, settings: Mapping[str, object], mode: Mode | None = None) -> None:
        self._configured_settings = dict(settings)
        self.native_adapter_mode = mode

    def set_paper_variant(self, variant: str) -> None:
        self.paper_variant = variant

    def set_paper_balance(self, amount: float, asset: str | None = None) -> None:
        normalized_asset = asset.upper() if isinstance(asset, str) else asset
        self.paper_balance = (float(amount), normalized_asset)
        self._paper_initial_cash = float(amount)
        if normalized_asset:
            self._paper_cash_asset = normalized_asset

    def set_paper_fee_rate(self, fee_rate: float) -> None:
        self.paper_fee_rate = float(fee_rate)

    def get_paper_variant(self) -> str:
        return self.paper_variant or "spot"

    def get_paper_initial_cash(self) -> float:
        return float(self.paper_balance[0]) if self.paper_balance else float(self._paper_initial_cash)

    def get_paper_cash_asset(self) -> str | None:
        if self.paper_balance and len(self.paper_balance) > 1:
            return self.paper_balance[1]
        return self._paper_cash_asset

    def get_paper_fee_rate(self) -> float:
        return float(self.paper_fee_rate) if self.paper_fee_rate is not None else 0.001

    def configure_paper_simulator(self, **settings: object) -> None:
        for key, value in settings.items():
            self.paper_simulator_settings[key] = float(value)

    def get_paper_simulator_settings(self) -> Mapping[str, float]:
        variant = self.paper_variant or "spot"
        if variant == "spot":
            return {}
        defaults = {
            "leverage_limit": 3.0,
            "maintenance_margin_ratio": 0.15,
            "funding_rate": 0.0,
            "funding_interval_seconds": 0.0,
        }
        if variant == "futures":
            defaults = {
                "leverage_limit": 10.0,
                "maintenance_margin_ratio": 0.05,
                "funding_rate": 0.0001,
                "funding_interval_seconds": 0.0,
            }
        combined = dict(defaults)
        for key, value in self.paper_simulator_settings.items():
            combined[key] = float(value)
        return combined

    def configure_watchdog(
        self,
        *,
        retry_policy: Mapping[str, object] | None = None,
        circuit_breaker: Mapping[str, object] | None = None,
        retry_exceptions: Sequence[type[Exception]] | None = None,
    ) -> None:
        self.watchdog_config = {}
        if retry_policy:
            self.watchdog_config["retry_policy"] = retry_policy
        if circuit_breaker:
            self.watchdog_config["circuit_breaker"] = circuit_breaker
        if retry_exceptions is not None:
            self.watchdog_config["retry_exceptions"] = tuple(retry_exceptions)

    def load_markets(self) -> dict[str, object]:
        self.load_markets_calls += 1
        return {"BTC/USDT": {}}

    def fetch_balance(self) -> dict[str, object]:
        self.fetch_balance_calls += 1
        return {"total": {"USDT": 100.0}}

    def fetch_ticker(self, symbol: str) -> Mapping[str, object]:
        self.ticker_requests.append(symbol)
        return {"last": 100.0, "symbol": symbol}

    def create_health_monitor(self, checks: Iterable) -> _FakeMonitor:
        self._health_checks = tuple(checks)
        return _FakeMonitor(self._health_checks)


def _write_profile(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_environment(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_health_check_cli_uses_profile_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
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
key_id = "PROFILE_KEY_ID"
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
            "--key-id",
            "CLI_KEY_ID",
            "--secret",
            "CLI_SECRET",
            "--passphrase",
            "CLI_PHRASE",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager._credentials == ("CLI_KEY_ID", "CLI_SECRET", "CLI_PHRASE")


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
      retry_exceptions:
        - builtins.TimeoutError
        - bot_core.exchanges.errors.ExchangeThrottlingError
  credentials:
    key_id: ${ENV_KEY_ID}
    secret: ${ENV_SECRET}
  health_check:
    public_symbol: ETH/USDT
    skip_private: true
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("ENV_KEY_ID", "LIVEKEY_ID")
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
        "leverage_limit": pytest.approx(6.0),
        "maintenance_margin_ratio": pytest.approx(0.1),
    }
    assert manager.watchdog_config and manager.watchdog_config["retry_policy"]["max_attempts"] == 5
    retry_excs = manager.watchdog_config.get("retry_exceptions")
    assert retry_excs is not None
    assert set(retry_excs) == {TimeoutError, ExchangeThrottlingError}
    assert manager._credentials == ("LIVEKEY_ID", "LIVESECRET", None)
    assert manager.fetch_balance_calls == 0
    assert manager.load_markets_calls == 0
    assert manager._health_checks and manager._health_checks[0].name == "public_api"

    captured = capsys.readouterr()
    assert "Aktywne środowisko: paper_margin" in captured.out
    assert "paper_variant=margin" in captured.out
    assert "Overall status" in captured.out


def test_health_check_cli_overrides_watchdog_from_cli(
    tmp_path: Path,
) -> None:
    credentials = _write_profile(tmp_path / "desktop.toml", "")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--skip-private",
            "--watchdog-max-attempts",
            "7",
            "--watchdog-base-delay",
            "0.3",
            "--watchdog-max-delay",
            "1.5",
            "--watchdog-jitter-min",
            "0.05",
            "--watchdog-jitter-max",
            "0.25",
            "--watchdog-failure-threshold",
            "4",
            "--watchdog-recovery-timeout",
            "45",
            "--watchdog-half-open-success",
            "3",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.watchdog_config is not None
    retry_cfg = manager.watchdog_config.get("retry_policy")
    assert retry_cfg is not None
    assert retry_cfg["max_attempts"] == 7
    assert retry_cfg["base_delay"] == pytest.approx(0.3)
    assert retry_cfg["max_delay"] == pytest.approx(1.5)
    jitter = retry_cfg.get("jitter")
    assert isinstance(jitter, tuple)
    assert jitter[0] == pytest.approx(0.05)
    assert jitter[1] == pytest.approx(0.25)

    circuit_cfg = manager.watchdog_config.get("circuit_breaker")
    assert circuit_cfg is not None
    assert circuit_cfg["failure_threshold"] == 4
    assert circuit_cfg["recovery_timeout"] == pytest.approx(45.0)
    assert circuit_cfg["half_open_success_threshold"] == 3


def test_health_check_cli_reports_custom_paper_simulator_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(tmp_path / "desktop.toml", "")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--mode",
            "paper",
            "--skip-private",
            "--paper-variant",
            "margin",
            "--paper-simulator-setting",
            "liquidation_buffer=0.07",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.paper_simulator_settings["liquidation_buffer"] == pytest.approx(0.07)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    simulator_cfg = payload["paper"].get("simulator")
    assert simulator_cfg is not None
    assert simulator_cfg["liquidation_buffer"] == pytest.approx(0.07)


def test_health_check_cli_overrides_retry_exceptions_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(tmp_path / "desktop.toml", "")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--skip-private",
            "--output-format",
            "json",
            "--watchdog-retry-exception",
            "builtins.TimeoutError",
            "--watchdog-retry-exception",
            "bot_core.exchanges.errors.ExchangeError",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.watchdog_config is not None
    retry_excs = manager.watchdog_config.get("retry_exceptions")
    assert retry_excs == (TimeoutError, ExchangeError)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    watchdog_cfg = payload.get("watchdog")
    assert watchdog_cfg is not None
    assert watchdog_cfg.get("retry_exceptions") == [
        "builtins.TimeoutError",
        "bot_core.exchanges.errors.ExchangeError",
    ]


def test_health_check_cli_validates_watchdog_jitter_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(tmp_path / "desktop.toml", "")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--skip-private",
            "--watchdog-jitter-min",
            "0.5",
            "--watchdog-jitter-max",
            "0.25",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "watchdog-jitter-max" in captured.err


def test_health_check_cli_overrides_native_adapter_from_cli(tmp_path: Path) -> None:
    _RecordingManager.instances.clear()

    env_yaml = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

live_margin:
  exchange_manager:
    mode: margin
    native_adapter:
      settings:
        margin_mode: cross
""",
    )

    exit_code = main(
        [
            "health-check",
            "--environment",
            "live_margin",
            "--environment-config",
            str(env_yaml),
            "--skip-private",
            "--native-setting",
            "margin_mode=isolated",
            "--native-setting",
            "max_leverage=5",
            "--native-setting",
            "hedge_mode=true",
            "--native-mode",
            "futures",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.native_adapter_mode is Mode.FUTURES
    assert manager._configured_settings == {
        "margin_mode": "isolated",
        "max_leverage": 5,
        "hedge_mode": True,
    }


def test_health_check_cli_validates_native_setting_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
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
            "--native-setting",
            "invalid-setting",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "native-setting" in captured.err


def test_health_check_cli_uses_default_environment_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
  environment: paper
paper:
  description: Domyślne środowisko papierowe
  exchange_manager:
    mode: paper
    paper_variant: futures
    paper_initial_cash: 150000
  credentials:
    key_id: ${DEFAULT_KEY}
    secret: ${DEFAULT_SECRET}
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("DEFAULT_KEY", "DK")
    monkeypatch.setenv("DEFAULT_SECRET", "DS")

    exit_code = main(
        [
            "health-check",
            "--environment-config",
            str(env_yaml),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.exchange_id == "binance"
    assert manager.mode is Mode.PAPER
    assert manager.paper_variant == "futures"
    assert manager.paper_balance == (150_000.0, None)
    assert manager._credentials == ("DK", "DS", None)

    captured = capsys.readouterr()
    assert "Aktywne środowisko: paper" in captured.out
    assert "exchange=binance" in captured.out
    assert "paper_variant=futures" in captured.out


def test_health_check_cli_overrides_public_symbol_via_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
paper:
  exchange_manager:
    mode: paper
  health_check:
    public_symbol: ETH/USDT
    skip_private: true
""",
        encoding="utf-8",
    )

    monkeypatch.delenv("ENV_KEY_ID", raising=False)
    monkeypatch.delenv("ENV_SECRET", raising=False)

    exit_code = main(
        [
            "health-check",
            "--environment",
            "paper",
            "--environment-config",
            str(env_yaml),
            "--public-symbol",
            "BTC/EUR",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.mode is Mode.PAPER
    assert manager.ticker_requests == ["BTC/EUR"]
    assert manager.load_markets_calls == 0


def test_health_check_cli_requires_environment_when_default_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "health-check",
            "--environment-config",
            str(env_yaml),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "defaults.environment" in captured.err


def test_health_check_cli_environment_requires_environment_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    captured = capsys.readouterr()
    assert "--environment" in captured.err


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
    key_id: ${MISSING_KEY}
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


def test_health_check_cli_rejects_archival_aliases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_yaml = tmp_path / "modes.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
paper:
  exchange_manager:
    mode: paper
  credentials:
    api_key: foo
    secret: bar
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
    captured = capsys.readouterr()
    assert "api_key" in captured.err
    assert "--key-id" in captured.err


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

    monkeypatch.setenv("HC_KEY", "ENV_KEY_ID")
    monkeypatch.setenv("HC_SECRET", "ENV_SECRET")
    monkeypatch.setenv("HC_PASSPHRASE", "ENV_PHRASE")

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--key-id-env",
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
    assert manager._credentials == ("ENV_KEY_ID", "ENV_SECRET", "ENV_PHRASE")


def test_health_check_cli_requires_present_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--key-id-env",
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
key_id = "KEY"
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


def test_health_check_cli_outputs_json_payload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["overall_status"] == "healthy"
    assert payload["exchange"] == "binance"
    assert payload["mode"] == Mode.SPOT.value
    assert payload["include_private"] is True
    assert payload["environment_summary"] is None
    assert payload["notes"] == []
    assert payload["native_adapter"] is None
    assert payload["results"] and payload["results"][0]["name"] == "public_api"
    assert payload["paper"]["variant"] == "spot"
    assert payload["paper"]["initial_cash"] == pytest.approx(10_000.0)
    assert payload["paper"]["cash_asset"] == "USDT"
    assert payload["paper"]["fee_rate"] == pytest.approx(0.001)
    assert "simulator" not in payload["paper"]


def test_health_check_cli_outputs_pretty_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--output-format",
            "json-pretty",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("{\n")
    assert "  \"overall_status\": \"healthy\"" in captured.out
    payload = json.loads(captured.out)
    assert payload["overall_status"] == "healthy"
    assert payload["exchange"] == "binance"
    assert payload["native_adapter"] is None


def test_health_check_cli_accepts_generic_simulator_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _RecordingManager.instances.clear()

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--mode",
            "paper",
            "--paper-variant",
            "margin",
            "--paper-simulator-setting",
            "leverage_limit=8.5",
            "--paper-simulator-setting",
            "maintenance_margin_ratio=0.2",
            "--paper-simulator-setting",
            "funding_rate=0.0004",
            "--paper-simulator-setting",
            "funding_interval_seconds=7200",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.paper_simulator_settings == {
        "leverage_limit": pytest.approx(8.5),
        "maintenance_margin_ratio": pytest.approx(0.2),
        "funding_rate": pytest.approx(0.0004),
        "funding_interval_seconds": pytest.approx(7_200.0),
    }

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper"]["simulator"] == {
        "leverage_limit": pytest.approx(8.5),
        "maintenance_margin_ratio": pytest.approx(0.2),
        "funding_rate": pytest.approx(0.0004),
        "funding_interval_seconds": pytest.approx(7_200.0),
    }
    assert payload["native_adapter"] is None


def test_health_check_cli_reports_native_adapter_in_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _RecordingManager.instances.clear()

    env_yaml = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

live_margin:
  exchange_manager:
    mode: margin
    native_adapter:
      settings:
        margin_mode: cross
        max_leverage: 3
""",
    )

    exit_code = main(
        [
            "health-check",
            "--environment",
            "live_margin",
            "--environment-config",
            str(env_yaml),
            "--skip-private",
            "--native-setting",
            "margin_mode=isolated",
            "--native-setting",
            "max_leverage=5",
            "--native-mode",
            "futures",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["native_adapter"] == {
        "mode": "futures",
        "settings": {
            "margin_mode": "isolated",
            "max_leverage": 5,
        },
    }


def test_health_check_cli_overrides_paper_settings_with_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _RecordingManager.instances.clear()

    env_yaml = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

paper_margin:
  exchange_manager:
    mode: paper
    paper_variant: margin
    paper_initial_cash: 50000
    paper_cash_asset: USDT
    paper_fee_rate: 0.002
""",
    )

    exit_code = main(
        [
            "health-check",
            "--environment",
            "paper_margin",
            "--environment-config",
            str(env_yaml),
            "--paper-variant",
            "futures",
            "--paper-initial-cash",
            "75000",
            "--paper-cash-asset",
            "eur",
            "--paper-fee-rate",
            "0.0004",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.paper_variant == "futures"
    assert manager.paper_balance == (75_000.0, "EUR")
    assert manager.paper_fee_rate == pytest.approx(0.0004)

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper"]["variant"] == "futures"
    assert payload["paper"]["initial_cash"] == pytest.approx(75_000.0)
    assert payload["paper"]["cash_asset"] == "EUR"
    assert payload["paper"]["fee_rate"] == pytest.approx(0.0004)
    assert payload["paper"]["simulator"] == {
        "leverage_limit": pytest.approx(10.0),
        "maintenance_margin_ratio": pytest.approx(0.05),
        "funding_rate": pytest.approx(0.0001),
        "funding_interval_seconds": pytest.approx(0.0),
    }
    assert payload["native_adapter"] is None


def test_health_check_cli_overrides_simulator_settings_with_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _RecordingManager.instances.clear()

    env_yaml = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

paper_margin:
  exchange_manager:
    mode: paper
    paper_variant: margin
    simulator:
      leverage_limit: 4.0
      maintenance_margin_ratio: 0.08
""",
    )

    exit_code = main(
        [
            "health-check",
            "--environment",
            "paper_margin",
            "--environment-config",
            str(env_yaml),
            "--paper-leverage-limit",
            "7.5",
            "--paper-maintenance-margin",
            "0.12",
            "--paper-funding-rate",
            "0.0003",
            "--paper-funding-interval",
            "14400",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.paper_simulator_settings == {
        "leverage_limit": pytest.approx(7.5),
        "maintenance_margin_ratio": pytest.approx(0.12),
        "funding_rate": pytest.approx(0.0003),
        "funding_interval_seconds": pytest.approx(14_400.0),
    }

    payload = json.loads(capsys.readouterr().out)
    assert payload["paper"]["simulator"] == {
        "leverage_limit": pytest.approx(7.5),
        "maintenance_margin_ratio": pytest.approx(0.12),
        "funding_rate": pytest.approx(0.0003),
        "funding_interval_seconds": pytest.approx(14_400.0),
    }
    assert payload["native_adapter"] is None


def test_health_check_cli_sets_paper_asset_without_amount() -> None:
    _RecordingManager.instances.clear()

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--mode",
            "paper",
            "--paper-cash-asset",
            "btc",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.paper_balance == (10_000.0, "BTC")


@pytest.mark.parametrize(
    "args, message",
    [
        (["--paper-initial-cash", "0"], "dodatniej"),
        (["--paper-initial-cash", "-1"], "dodatniej"),
        (["--paper-fee-rate", "-0.1"], "nieujemnej"),
        (["--paper-cash-asset", "  "], "niepustej"),
        (["--paper-leverage-limit", "0"], "dodatniej"),
        (["--paper-maintenance-margin", "0"], "dodatniej"),
        (["--paper-funding-interval", "0"], "dodatniej"),
        (["--paper-simulator-setting", "leverage_limit=abc"], "zmiennoprzecinkową"),
    ],
)
def test_health_check_cli_validates_paper_flags(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            *args,
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert message in captured.err


def test_health_check_cli_fails_when_private_asset_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _RecordingManager.instances.clear()

    env_config = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

live:
  exchange_manager:
    mode: margin
  health_check:
    private_asset: usdt
""",
    )

    class _MissingAssetManager(_RecordingManager):
        def fetch_balance(self) -> dict[str, object]:  # type: ignore[override]
            self.fetch_balance_calls += 1
            return {"total": {"BTC": 0.5}}

    exit_code = main(
        [
            "health-check",
            "--environment-config",
            str(env_config),
            "--environment",
            "live",
            "--key-id",
            "KEY",
            "--secret",
            "SECRET",
        ],
        manager_factory=_MissingAssetManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Saldo nie zawiera waluty USDT" in captured.out


def test_health_check_cli_allows_cli_override_of_private_constraints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _RecordingManager.instances.clear()

    env_config = _write_environment(
        tmp_path / "env.yaml",
        """
defaults:
  exchange: binance

profile:
  exchange_manager:
    mode: margin
  health_check:
    private_asset: usdt
    private_min_balance: 7
""",
    )

    class _OverrideManager(_RecordingManager):
        def fetch_balance(self) -> dict[str, object]:  # type: ignore[override]
            self.fetch_balance_calls += 1
            return {"total": {"BTC": 9.5}, "free": {"BTC": 9.5}}

    exit_code = main(
        [
            "health-check",
            "--environment-config",
            str(env_config),
            "--environment",
            "profile",
            "--key-id",
            "KEY",
            "--secret",
            "SECRET",
            "--private-asset",
            "btc",
            "--private-min-balance",
            "8",
            "--output-format",
            "json",
        ],
        manager_factory=_OverrideManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["private_asset"] == "BTC"
    assert payload["private_min_balance"] == 8.0
    assert any(result["name"] == "private_api" for result in payload["results"])


@pytest.mark.parametrize("output_format", ["json", "json-pretty"])
def test_health_check_cli_writes_json_to_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    output_file = tmp_path / "artifacts" / "health.json"

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--output-format",
            output_format,
            "--output-path",
            str(output_file),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload_text = captured.out
    payload = json.loads(payload_text)
    assert output_file.exists()
    saved_contents = output_file.read_text(encoding="utf-8")
    assert json.loads(saved_contents) == payload
    if output_format == "json-pretty":
        assert "\n  \"results\": [" in payload_text
        assert saved_contents.endswith("\n")


def test_health_check_cli_rejects_output_path_in_text_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--output-path",
            str(tmp_path / "out.json"),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "--output-path" in captured.err


def test_health_check_cli_json_includes_skip_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["include_private"] is False
    assert payload["requested_checks"] is None
    assert payload["overall_status"] == "healthy"
    assert any("Pomijam test private_api" in note for note in payload["notes"])


def test_health_check_cli_runs_only_requested_public_check(tmp_path: Path) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "public_api",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.load_markets_calls == 1
    assert manager.fetch_balance_calls == 0


def test_health_check_cli_runs_only_requested_private_check(tmp_path: Path) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "private_api",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    manager = _RecordingManager.instances[-1]
    assert manager.load_markets_calls == 0
    assert manager.fetch_balance_calls == 1


def test_health_check_cli_fails_when_requested_private_without_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "private_api",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Nie można uruchomić testu private_api" in captured.err
    assert "poświadczeń" in captured.err


def test_health_check_cli_json_reports_requested_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
key_id = "KEY"
secret = "SECRET"
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "private_api",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_checks"] == ["private_api"]


def test_health_check_cli_normalizes_requested_check_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "PUBLIC_API",
            "--check",
            " public_api ",
            "--output-format",
            "json",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_checks"] == ["public_api"]
    manager = _RecordingManager.instances[-1]
    assert manager.load_markets_calls == 1


def test_health_check_cli_rejects_unknown_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    credentials = _write_profile(
        tmp_path / "desktop.toml",
        """
[binance]
""",
    )

    exit_code = main(
        [
            "health-check",
            "--exchange",
            "binance",
            "--credentials-file",
            str(credentials),
            "--check",
            "order_routing",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Nieznane testy zdrowia" in captured.err


def test_health_check_cli_lists_available_checks(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "health-check",
            "--list-checks",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Dostępne testy health-check:" in captured.out
    for name in ("public_api", "private_api"):
        assert f"  * {name}" in captured.out


def test_list_environments_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_yaml = tmp_path / "envs.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: kraken
  environment: paper
paper:
  description: Symulator margin dla testów
  exchange_manager:
    mode: paper
    paper_variant: margin
live:
  description: Produkcja margin
  exchange_manager:
    mode: margin
    testnet: false
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "list-environments",
            "--environment-config",
            str(env_yaml),
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Zdefiniowane środowiska" in captured.out
    assert "* live [exchange=kraken, mode=margin, testnet=false] - Produkcja margin" in captured.out
    assert "* paper (default) [exchange=kraken, mode=paper, paper_variant=margin] - Symulator margin dla testów" in captured.out


def test_list_environments_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "list-environments",
            "--environment-config",
            "does/not/exist.yaml",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "nie istnieje" in captured.err


def test_show_environment_prints_merged_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_yaml = tmp_path / "envs.yaml"
    env_yaml.write_text(
        """
defaults:
  exchange: binance
  exchange_manager:
    mode: margin
    testnet: true
    watchdog:
      retry_policy:
        max_attempts: 3
live:
  description: Produkcja margin
  exchange_manager:
    testnet: false
    native_adapter:
      settings:
        margin_mode: cross
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "show-environment",
            "--environment-config",
            str(env_yaml),
            "--environment",
            "live",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Środowisko 'live'" in captured.out
    assert "  exchange: binance" in captured.out
    assert "  description: Produkcja margin" in captured.out
    assert "    mode: margin" in captured.out
    assert "    testnet: false" in captured.out
    assert "    native_adapter:" in captured.out
    assert "      margin_mode: cross" in captured.out
    assert "    watchdog:" in captured.out
    assert "      max_attempts: 3" in captured.out


def test_show_environment_missing_section(tmp_path: Path) -> None:
    env_yaml = tmp_path / "envs.yaml"
    env_yaml.write_text("defaults: {}", encoding="utf-8")

    exit_code = main(
        [
            "show-environment",
            "--environment-config",
            str(env_yaml),
            "--environment",
            "missing",
        ],
        manager_factory=_RecordingManager,
    )

    assert exit_code == 2


def test_health_check_cli_propagates_cli_usage_error() -> None:
    with pytest.raises(CLIUsageError):
        raise CLIUsageError("Błąd testowy")
