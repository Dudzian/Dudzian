from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.run_paper_session as runner
from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel, AlertMessage
from bot_core.config.models import (
    CoreConfig,
    DailyTrendMomentumStrategyConfig,
    EnvironmentConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.data.base import OHLCVResponse
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderResult,
)
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.runtime.bootstrap import BootstrapContext


class _DummySecretStorage:
    def get_secret(self, key: str) -> str | None:  # noqa: D401 - implementacja testowa
        return "{}"

    def set_secret(self, key: str, value: str) -> None:  # noqa: D401 - implementacja testowa
        return None

    def delete_secret(self, key: str) -> None:  # noqa: D401 - implementacja testowa
        return None


class _DummyAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="dummy", environment=Environment.PAPER))

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 100_000.0},
            total_equity=100_000.0,
            available_margin=100_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:  # pragma: no cover - nieużywane w testach
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # pragma: no cover - nieużywane
        return ()

    def place_order(self, request):  # pragma: no cover - bezpieczeństwo testów
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover - bezpieczeństwo
        raise NotImplementedError

    def stream_public_data(self, *, channels: Sequence[str]):  # pragma: no cover - bezpieczeństwo
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]):  # pragma: no cover - bezpieczeństwo
        raise NotImplementedError


class _DummyChannel(AlertChannel):
    def __init__(self) -> None:
        self.name = "dummy"
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return {"status": "ok"}


class _DummyExecution:
    def __init__(self, markets, *, initial_balances=None, **_: object) -> None:
        self.markets = markets
        self.initial_balances = dict(initial_balances or {})
        self.requests: list = []

    def execute(self, request, context):  # noqa: ARG002 - testowy stub
        self.requests.append(request)
        return OrderResult(
            order_id="paper-1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price or 100.0,
            raw_response={"source": "test"},
        )

    def cancel(self, order_id: str, context) -> None:  # noqa: ARG002 - metoda wymagana przez interfejs
        return None

    def flush(self) -> None:
        return None

    def balances(self) -> Mapping[str, float]:
        return {
            "USDT": self.initial_balances.get("USDT", 0.0),
            "BTC": self.initial_balances.get("BTC", 0.0)
            + sum(req.quantity for req in self.requests if req.side.lower() == "buy"),
        }


class _DummyDataSource:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._response = OHLCVResponse(
            columns=("open_time", "open", "high", "low", "close", "volume"),
            rows=rows,
        )

    def fetch_ohlcv(self, request):  # noqa: ARG002 - testowy stub
        return self._response


@pytest.fixture
def trending_rows() -> list[list[float]]:
    rows: list[list[float]] = []
    base = 100.0
    for idx in range(10):
        base += 1.0
        rows.append(
            [
                float(idx * 86_400_000),
                base - 1.0,
                base + 0.5,
                base - 1.5,
                base,
                1_000.0,
            ]
        )
    return rows


def _build_core_config(tmp_path: Path) -> tuple[CoreConfig, EnvironmentConfig]:
    env_cfg = EnvironmentConfig(
        name="binance_paper",
        exchange="binance_spot",
        environment=Environment.PAPER,
        keychain_key="paper_key",
        data_cache_path=str(tmp_path / "cache"),
        risk_profile="balanced",
        alert_channels=[],
        ip_allowlist=[],
        credential_purpose="trading",
        instrument_universe="core_universe",
    )
    risk_cfg = RiskProfileConfig(
        name="balanced",
        max_daily_loss_pct=0.015,
        max_position_pct=0.05,
        target_volatility=0.11,
        max_leverage=3.0,
        stop_loss_atr_multiple=1.5,
        max_open_positions=5,
        hard_drawdown_pct=0.10,
    )
    instrument = InstrumentConfig(
        name="BTC_USDT",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("core",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(
            InstrumentBackfillWindow(interval="1d", lookback_days=365),
        ),
    )
    universe = InstrumentUniverseConfig(
        name="core_universe",
        description="Testowe uniwersum",
        instruments=(instrument,),
    )
    strategy_cfg = DailyTrendMomentumStrategyConfig(
        name="core_daily_trend",
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=5,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=2.0,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    core_config = CoreConfig(
        environments={"binance_paper": env_cfg},
        risk_profiles={"balanced": risk_cfg},
        instrument_universes={"core_universe": universe},
        strategies={"core_daily_trend": strategy_cfg},
        reporting={},
        sms_providers={},
        telegram_channels={},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
    )
    return core_config, env_cfg


def _prepare_context(tmp_path: Path) -> tuple[BootstrapContext, _DummyChannel, ThresholdRiskEngine]:
    core_config, env_cfg = _build_core_config(tmp_path)
    risk_engine = ThresholdRiskEngine()
    audit_log = InMemoryAlertAuditLog()
    channel = _DummyChannel()
    router = DefaultAlertRouter(audit_log=audit_log)
    router.register(channel)
    context = BootstrapContext(
        core_config=core_config,
        environment=env_cfg,
        credentials=ExchangeCredentials(key_id="paper", environment=Environment.PAPER),
        adapter=_DummyAdapter(),
        risk_engine=risk_engine,
        alert_router=router,
        alert_channels={channel.name: channel},
        audit_log=audit_log,
    )
    return context, channel, risk_engine


def test_run_paper_session_executes_single_iteration(monkeypatch, tmp_path, caplog, trending_rows):
    context, channel, _ = _prepare_context(tmp_path)

    monkeypatch.setattr(runner, "create_default_secret_storage", lambda **_: _DummySecretStorage())

    def _fake_bootstrap(environment_name: str, *, config_path: str, secret_manager, adapter_factories=None):  # noqa: ARG001
        assert environment_name == "binance_paper"
        return context

    monkeypatch.setattr(runner, "bootstrap_environment", _fake_bootstrap)

    executed: list[_DummyExecution] = []

    def _execution_factory(*args, **kwargs):
        instance = _DummyExecution(*args, **kwargs)
        executed.append(instance)
        return instance

    monkeypatch.setattr(runner, "PaperTradingExecutionService", _execution_factory)

    data_source = _DummyDataSource(trending_rows)
    monkeypatch.setattr(runner, "_build_data_source", lambda adapter, cache_path: data_source)

    config_path = tmp_path / "core.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    caplog.set_level(logging.INFO, logger="scripts.run_paper_session")

    exit_code = runner.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--strategy",
            "core_daily_trend",
            "--interval",
            "1d",
        ]
    )

    assert exit_code == 0
    assert executed and executed[0].requests, "Powinno zostać złożone co najmniej jedno zlecenie paper trading"
    assert channel.messages, "Alerty powinny zostać zapisane w audycie"
    assert any("Zrealizowano zlecenie" in message for message in caplog.messages)


def test_run_paper_session_errors_when_universe_empty(monkeypatch, tmp_path, caplog):
    context, _, _ = _prepare_context(tmp_path)
    context.core_config.instrument_universes.clear()

    monkeypatch.setattr(runner, "create_default_secret_storage", lambda **_: _DummySecretStorage())
    monkeypatch.setattr(runner, "bootstrap_environment", lambda *_, **__: context)
    monkeypatch.setattr(runner, "_build_data_source", lambda *_, **__: _DummyDataSource([]))

    config_path = tmp_path / "core.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    caplog.set_level(logging.INFO, logger="scripts.run_paper_session")
    exit_code = runner.main([
        "--config",
        str(config_path),
        "--environment",
        "binance_paper",
    ])

    assert exit_code == 1
    assert any("uniwersum" in message for message in caplog.messages)
