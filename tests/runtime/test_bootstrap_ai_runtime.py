from __future__ import annotations

import asyncio
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bot_core.decision.ai_connector import AIManagerDecisionConnector
from bot_core.decision.models import RiskSnapshot
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest
from bot_core.runtime.bootstrap import bootstrap_environment
from bot_core.security import SecretManager, SecretStorage


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


class DummyModel:
    def predict_series(self, df: pd.DataFrame, feature_cols=None) -> pd.Series:
        return pd.Series([0.018] * len(df), index=df.index)

    def predict(self, features) -> float:
        return 0.018


@pytest.fixture
def temp_model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "btc_light.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(DummyModel(), handle)
    return model_path


def _write_config(tmp_path: Path, model_path: Path) -> Path:
    config = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.05,
                "max_position_pct": 0.1,
                "target_volatility": 0.15,
                "max_leverage": 5.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.2,
            }
        },
        "environments": {
            "paper_ai": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "paper_secret",
                "credential_purpose": "trading",
                "data_cache_path": str(tmp_path / "cache"),
                "risk_profile": "balanced",
                "alert_channels": [],
                "ai": {
                    "enabled": True,
                    "threshold_bps": 6.0,
                    "default_strategy": "ai_core",
                    "default_notional": 750.0,
                    "default_action": "enter",
                    "models": [
                        {
                            "symbol": "BTCUSDT",
                            "model_type": "light",
                            "path": str(model_path),
                            "notional": 900.0,
                        }
                    ],
                },
            }
        },
        "alerts": {},
        "permission_profiles": {},
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 15.0,
                "min_net_edge_bps": 1.0,
                "max_daily_loss_pct": 0.05,
                "max_drawdown_pct": 0.2,
                "max_position_ratio": 0.5,
                "max_open_positions": 5,
                "max_latency_ms": 250.0,
            },
            "min_probability": 0.2,
            "require_cost_data": False,
            "penalty_cost_bps": 0.0,
        },
    }
    config_path = tmp_path / "core.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def test_bootstrap_environment_loads_ai_and_executes(tmp_path: Path, temp_model_file: Path) -> None:
    config_path = _write_config(tmp_path, temp_model_file)
    storage = _MemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_exchange_credentials(
        "paper_secret",
        ExchangeCredentials(
            key_id="key",
            secret="secret",
            passphrase="phrase",
            environment=Environment.PAPER,
            permissions=("trade",),
        ),
    )

    context = bootstrap_environment(
        "paper_ai",
        config_path=config_path,
        secret_manager=secret_manager,
    )

    assert context.ai_manager is not None
    assert context.ai_models_loaded == ("BTCUSDT:light",)
    assert context.ai_threshold_bps == pytest.approx(6.0)

    ai_manager = context.ai_manager
    connector = AIManagerDecisionConnector(
        ai_manager=ai_manager,
        strategy="ai_core",
        risk_profile=context.risk_profile_name,
        default_notional=context.environment.ai.default_notional or 750.0,  # type: ignore[union-attr]
        action=context.environment.ai.default_action if context.environment.ai else "enter",  # type: ignore[union-attr]
        threshold_bps=context.ai_threshold_bps,
    )

    df = pd.DataFrame(
        {
            "open": [101.0, 102.0],
            "high": [103.0, 104.0],
            "low": [100.0, 101.0],
            "close": [102.0, 103.0],
            "volume": [12_000.0, 13_000.0],
        },
        index=[datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 1)],
    )

    candidates = asyncio.run(connector.generate_candidates("BTCUSDT", df))
    candidate = candidates[0]
    evaluation = context.decision_orchestrator.evaluate_candidate(
        candidate,
        RiskSnapshot(
            profile=context.risk_profile_name,
            start_of_day_equity=200_000.0,
            daily_realized_pnl=0.0,
            peak_equity=200_000.0,
            last_equity=200_000.0,
            gross_notional=0.0,
            active_positions=0,
            symbols=(),
        ),
    )
    assert evaluation.accepted

    stop_price = 101.0
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=candidate.notional / 103.0,
        order_type="market",
        price=103.0,
        atr=1.2,
        stop_price=stop_price,
        metadata={
            "decision_candidate": candidate.to_mapping(),
            "source": "test",
        },
    )
    account = AccountSnapshot(
        balances={"USDT": 200_000.0},
        total_equity=200_000.0,
        available_margin=200_000.0,
        maintenance_margin=5_000.0,
    )
    risk_result = context.risk_engine.apply_pre_trade_checks(
        request,
        account=account,
        profile_name=context.risk_profile_name,
    )
    assert risk_result.allowed

    execution = PaperTradingExecutionService(
        markets={"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT")},
        initial_balances={"USDT": 200_000.0},
    )
    exec_context = ExecutionContext(
        portfolio_id="paper_ai",
        risk_profile=context.risk_profile_name,
        environment="paper",
        metadata={"source": "test"},
    )
    result = execution.execute(request, exec_context)
    assert result.status == "filled"
    assert result.filled_quantity > 0
