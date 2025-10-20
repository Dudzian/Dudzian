"""Test end-to-end papierowej ścieżki sygnału."""
from __future__ import annotations

import math
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, MutableMapping, Sequence

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.alerts import AlertMessage, DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel
from bot_core.data.base import CacheStorage, OHLCVRequest
from bot_core.data.ohlcv.cache import CachedOHLCVSource, PublicAPIDataSource
from bot_core.execution import ExecutionContext, MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
)
from bot_core.runtime.bootstrap import bootstrap_environment
from bot_core.runtime import pipeline as pipeline_module
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.strategies import MarketSnapshot
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy

from tests.test_runtime_bootstrap import _prepare_manager, _write_config


class _InMemoryStorage(CacheStorage):
    """Lekka implementacja magazynu cache do testów."""

    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Sequence[Sequence[float]]]] = {}
        self._metadata: dict[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._store[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        try:
            rows = self._store[key]["rows"]
        except KeyError:
            return None
        if not rows:
            return None
        return float(rows[-1][0])


@dataclass(slots=True)
class _StaticStream:
    """Pusty stream spełniający minimalny kontrakt protokołu."""
    channels: Sequence[str]


class _FakePaperAdapter(ExchangeAdapter):
    """Adapter udostępniający z góry przygotowane dane OHLCV i snapshot konta."""

    name = "fake-paper"

    def __init__(self, candles: Sequence[Sequence[float]]) -> None:
        credentials = ExchangeCredentials(
            key_id="paper-key",
            environment=Environment.PAPER,
            permissions=("read", "trade"),
        )
        super().__init__(credentials)
        self._candles = [tuple(float(value) for value in row) for row in candles]
        self._symbols = ("BTCUSDT",)
        self._configured = False

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401
        self._configured = True

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 100_000.0, "BTC": 0.0},
            total_equity=100_000.0,
            available_margin=100_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return self._symbols

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        del symbol, interval, start, end, limit
        return [tuple(row) for row in self._candles]

    def place_order(self, request: OrderRequest):  # noqa: D401, ANN001
        raise NotImplementedError("Adapter symulacyjny nie obsługuje składania zleceń.")

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # noqa: D401
        raise NotImplementedError("Adapter symulacyjny nie obsługuje anulacji.")

    def stream_public_data(self, *, channels: Sequence[str]) -> _StaticStream:
        return _StaticStream(channels)

    def stream_private_data(self, *, channels: Sequence[str]) -> _StaticStream:
        return _StaticStream(channels)


class _RecordingChannel(AlertChannel):
    name = "recording"

    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:  # noqa: D401
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:
        return {"status": "ok", "delivered": str(len(self.messages))}


def _to_snapshot(row: Sequence[float]) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _position_size(
    *,
    atr: float,
    price: float,
    equity: float,
    risk_pct: float,
    profile: BalancedProfile,
    step_size: float,
) -> float:
    """Wyznacza wielkość pozycji respektując limit ekspozycji profilu."""
    risk_amount = equity * risk_pct
    stop_distance = atr * profile.stop_loss_atr_multiple()
    raw_quantity = max(risk_amount / stop_distance, 0.0)
    max_quantity = profile.max_position_exposure() * equity / price
    constrained = min(raw_quantity, max_quantity)
    if constrained <= 0:
        return 0.0
    if step_size > 0:
        constrained = math.floor(constrained / step_size) * step_size
        constrained = max(constrained, step_size)
    return constrained


def test_paper_pipeline_executes_and_alerts(tmp_path: Path) -> None:
    day_ms = 86_400_000
    base_ts = 1_700_000_000_000
    candles = [
        [base_ts + i * day_ms, 100 + i, 101 + i, 99 + i, 100 + i, 10 + i] for i in range(5)
    ]
    candles.append([base_ts + 5 * day_ms, 105.0, 108.0, 104.0, 107.0, 20.0])

    adapter = _FakePaperAdapter(candles)
    storage = _InMemoryStorage()
    source = CachedOHLCVSource(storage=storage, upstream=PublicAPIDataSource(adapter))
    request = OHLCVRequest(
        symbol="BTCUSDT",
        interval="1d",
        start=base_ts,
        end=base_ts + 5 * day_ms,
    )

    response = source.fetch_ohlcv(request)
    assert len(response.rows) == len(candles)

    adapter._candles = []  # type: ignore[attr-defined]
    cached_response = source.fetch_ohlcv(request)
    assert cached_response.rows == response.rows, "Dane powinny pochodzić z cache"

    snapshots = [_to_snapshot(row) for row in response.rows]
    history, latest = snapshots[:-1], snapshots[-1]

    strategy = DailyTrendMomentumStrategy(
        DailyTrendMomentumSettings(
            fast_ma=3,
            slow_ma=5,
            breakout_lookback=4,
            momentum_window=3,
            atr_window=3,
            atr_multiplier=1.5,
            min_trend_strength=0.0,
            min_momentum=0.0,
        )
    )
    strategy.warm_up(history)
    signals = strategy.on_data(latest)

    assert signals, "Strategia powinna wygenerować sygnał wejścia"
    signal = signals[0]
    assert signal.side == "buy"

    risk_engine = ThresholdRiskEngine()
    profile = BalancedProfile()
    risk_engine.register_profile(profile)

    account = adapter.fetch_account_snapshot()
    atr = float(signal.metadata["atr"])
    price = latest.close
    market = MarketMetadata(
        base_asset="BTC",
        quote_asset="USDT",
        min_quantity=0.001,
        min_notional=10.0,
        step_size=0.001,
    )
    quantity = _position_size(
        atr=atr,
        price=price,
        equity=account.total_equity,
        risk_pct=0.01,
        profile=profile,
        step_size=market.step_size or 0.0,
    )

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=quantity,
        order_type="market",
        price=price,
        stop_price=float(signal.metadata["stop_price"]),
        atr=atr,
    )

    check = risk_engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)
    assert check.allowed, f"Kontrola ryzyka powinna przepuścić zlecenie: {check.reason}"

    markets = {"BTCUSDT": market}
    execution = PaperTradingExecutionService(markets, initial_balances={"USDT": 100_000.0, "BTC": 0.0})
    context = ExecutionContext(portfolio_id="paper-test", risk_profile=profile.name, environment="paper", metadata={})

    result = execution.execute(order, context)
    assert result.status == "filled"
    assert result.avg_price is not None

    risk_engine.on_fill(
        profile_name=profile.name,
        symbol=order.symbol,
        side="long",
        position_value=result.avg_price * result.filled_quantity,
        pnl=0.0,
    )
    assert not risk_engine.should_liquidate(profile_name=profile.name)

    audit_log = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit_log)
    channel = _RecordingChannel()
    router.register(channel)

    alert = AlertMessage(
        category="trade",
        title="Paper trade executed",
        body=f"Kupiono {result.filled_quantity:.4f} BTCUSDT po {result.avg_price:.2f}",
        severity="info",
        context={"profile": profile.name, "strategy": "daily_trend"},
    )
    router.dispatch(alert)

    assert channel.messages == [alert]
    exported = list(audit_log.export())
    assert exported and exported[0]["channel"] == channel.name


def test_bootstrap_environment_provides_paper_execution_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot_core.runtime import bootstrap as bootstrap_module
    from bot_core.security.license import LicenseValidationResult

    dummy_license = LicenseValidationResult(
        status="valid",
        fingerprint="test",
        license_path=Path("license.json"),
        issued_at=None,
        expires_at=None,
        fingerprint_source=None,
        profile=None,
        issuer=None,
        schema=None,
        schema_version=None,
        license_id=None,
        revocation_list_path=None,
        revocation_status=None,
        revocation_reason=None,
        revocation_revoked_at=None,
        revocation_generated_at=None,
        revocation_checked=False,
        revocation_signature_key=None,
        errors=[],
        warnings=[],
        payload=None,
        license_signature_key=None,
        fingerprint_signature_key=None,
    )

    monkeypatch.setattr(
        bootstrap_module,
        "validate_license_from_config",
        lambda _config: dummy_license,
    )

    dummy_router = SimpleNamespace(register=lambda *args, **kwargs: None, audit_log=None)
    monkeypatch.setattr(
        bootstrap_module,
        "build_alert_channels",
        lambda **_kwargs: ({}, dummy_router, SimpleNamespace()),
    )

    config_path = _write_config(tmp_path)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data.setdefault("instrument_universes", {})["paper_test"] = {
        "description": "Test universe for paper execution",
        "instruments": {
            "BTC_USDT": {
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "exchanges": {"binance_spot": "BTCUSDT"},
                "categories": ["paper"],
            }
        },
    }
    config_data["environments"]["binance_paper"]["instrument_universe"] = "paper_test"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper",
        config_path=config_path,
        secret_manager=manager,
    )

    service = context.execution_service
    assert isinstance(service, PaperTradingExecutionService)

    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="market",
        price=30_000.0,
    )
    exec_context = ExecutionContext(
        portfolio_id="autotrader",
        risk_profile=context.risk_profile_name,
        environment=context.environment.environment.value,
        metadata={},
    )

    result = service.execute(order, exec_context)
    assert result.status == "filled"
    assert result.avg_price is not None


def test_select_execution_service_prefers_bootstrap_instance(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }
    service = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 50_000.0},
        ledger_directory=ledger_dir,
    )

    context = SimpleNamespace(execution_service=service)

    resolved = pipeline_module._select_execution_service(
        context,
        markets,
        {
            "initial_balances": {"USDT": 10_000.0},
            "maker_fee": 0.0004,
            "taker_fee": 0.0006,
            "slippage_bps": 5.0,
            "ledger_directory": ledger_dir,
            "ledger_filename_pattern": "ledger-%Y%m%d.jsonl",
            "ledger_retention_days": 730,
            "ledger_fsync": False,
        },
    )

    assert resolved is service


def test_select_execution_service_persists_created_instance(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }

    context = SimpleNamespace(execution_service=None)

    resolved = pipeline_module._select_execution_service(
        context,
        markets,
        {
            "initial_balances": {"USDT": 10_000.0},
            "maker_fee": 0.0004,
            "taker_fee": 0.0006,
            "slippage_bps": 5.0,
            "ledger_directory": ledger_dir,
            "ledger_filename_pattern": "ledger-%Y%m%d.jsonl",
            "ledger_retention_days": 730,
            "ledger_fsync": False,
        },
    )

    assert isinstance(resolved, PaperTradingExecutionService)
    assert context.execution_service is resolved


def test_build_daily_trend_pipeline_reuses_bootstrap_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }
    service = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 50_000.0},
        ledger_directory=ledger_dir,
    )

    environment = SimpleNamespace(
        environment=Environment.PAPER,
        exchange="binance_spot",
        adapter_settings={
            "paper_trading": {
                "quote_assets": ["USDT"],
                "initial_balances": {"USDT": 50_000.0},
            }
        },
        data_cache_path=str(tmp_path / "cache"),
        name="binance_paper",
        default_strategy="daily_trend",
        default_controller="daily_trend",
    )
    bootstrap_ctx = SimpleNamespace(
        core_config=object(),
        environment=environment,
        risk_profile_name="balanced",
        risk_engine=object(),
        adapter=object(),
        execution_service=service,
        tco_reporter=None,
    )

    monkeypatch.setattr(
        pipeline_module,
        "bootstrap_environment",
        lambda *args, **kwargs: bootstrap_ctx,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_strategy",
        lambda *_args, **_kwargs: SimpleNamespace(
            fast_ma=3,
            slow_ma=5,
            breakout_lookback=4,
            momentum_window=3,
            atr_window=3,
            atr_multiplier=1.5,
            min_trend_strength=0.0,
            min_momentum=0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(interval="1d"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_resolve_universe",
        lambda *_args, **_kwargs: SimpleNamespace(
            instruments=[
                SimpleNamespace(
                    exchange_symbols={"binance_spot": "BTCUSDT"},
                    quote_asset="USDT",
                    base_asset="BTC",
                )
            ]
        ),
    )

    cache_stub = SimpleNamespace(
        storage=SimpleNamespace(
            latest_timestamp=lambda _key: None,
            read=lambda _key: {"rows": []},
        ),
        _cache_key=lambda symbol, interval: f"{symbol}:{interval}",
    )
    monkeypatch.setattr(
        pipeline_module,
        "_create_cached_source",
        lambda *_args, **_kwargs: cache_stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "OHLCVBackfillService",
        lambda _source: SimpleNamespace(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_build_account_loader",
        lambda **_kwargs: lambda: SimpleNamespace(),
    )

    class _DummyController:
        def __init__(self, **kwargs):
            self.execution_context = kwargs["execution_context"]
            self.account_loader = kwargs["account_loader"]

    monkeypatch.setattr(pipeline_module, "DailyTrendController", _DummyController)

    call_detected = {"flag": False}

    def _fail_build(*_args, **_kwargs):
        call_detected["flag"] = True
        raise AssertionError("_build_execution_service should not be invoked")

    monkeypatch.setattr(pipeline_module, "_build_execution_service", _fail_build)

    pipeline = pipeline_module.build_daily_trend_pipeline(
        environment_name="binance_paper",
        strategy_name=None,
        controller_name=None,
        config_path=tmp_path / "core.yaml",
        secret_manager=object(),
    )

    assert pipeline.execution_service is service
    assert call_detected["flag"] is False
