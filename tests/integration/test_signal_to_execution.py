from __future__ import annotations

import json
from pathlib import Path

from bot_core.ai import (
    DecisionModelInference,
    FeatureDataset,
    FeatureVector,
    ModelRepository,
    ModelTrainer,
    generate_model_artifact_bundle,
)
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.models import DecisionCandidate
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.execution.base import ExecutionContext
from bot_core.execution.bridge import ExchangeAdapterExecutionService, decision_to_order_request
from bot_core.execution.paper import MarketMetadata
from bot_core.exchanges.base import ExchangeAdapter, ExchangeCredentials, OrderRequest, OrderResult


class _StubExchangeAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="test"))
        self.orders: list[OrderRequest] = []

    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - konfiguracja nieużywana
        return None

    def fetch_account_snapshot(self):  # pragma: no cover - niepotrzebne w teście
        raise NotImplementedError

    def fetch_symbols(self):  # pragma: no cover - niepotrzebne
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # pragma: no cover - niepotrzebne
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        price = request.price if request.price is not None else 100.0
        return OrderResult(
            order_id=f"test-{len(self.orders)}",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=price,
            raw_response={"fee": 0.0, "fee_asset": "USDT"},
        )

    def cancel_order(self, order_id: str, *, symbol=None) -> None:  # pragma: no cover - brak użycia
        return None

    def stream_public_data(self, *, channels):  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - nieużywane
        raise NotImplementedError


def _synthetic_dataset() -> FeatureDataset:
    base_ts = 1_700_000_000
    vectors = []
    for idx in range(64):
        momentum = 0.5 + idx * 0.01
        volume_ratio = 1.0 + (idx % 5) * 0.02
        vectors.append(
            FeatureVector(
                timestamp=base_ts + idx * 60,
                symbol="BTCUSDT",
                features={"momentum": momentum, "volume_ratio": volume_ratio},
                target_bps=12.0 + momentum * 2.5,
            )
        )
    metadata = {"symbols": ["BTCUSDT"], "window_minutes": 60}
    return FeatureDataset(vectors=tuple(vectors), metadata=metadata)


def test_signal_to_execution_flow(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    trainer = ModelTrainer(validation_split=0.1, test_split=0.1)
    artifact = trainer.train(dataset)

    bundle_dir = tmp_path / "bundle"
    signing_key = b"unit-test-artifact-key"
    bundle = generate_model_artifact_bundle(
        artifact,
        bundle_dir,
        name="btc-trend",
        signing_key=signing_key,
        signing_key_id="ci",
        metadata_overrides={"training_run": "integration-test"},
    )

    artifact_payload = json.loads(bundle.artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["feature_names"], "artefakt powinien zawierać cechy"

    metadata_payload = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["rows"]["train"] == artifact.training_rows
    assert metadata_payload["training_run"] == "integration-test"

    checksums_text = bundle.checksums_path.read_text(encoding="utf-8")
    assert bundle.artifact_path.name in checksums_text

    signature_payload = json.loads(bundle.signature_path.read_text(encoding="utf-8"))
    assert signature_payload["signature"]["algorithm"] == "HMAC-SHA256"
    assert signature_payload["target"] == bundle.artifact_path.name

    repository = ModelRepository(tmp_path / "models")
    repository.save(artifact, "btc-trend.json", version="1.0.0", activate=True)

    inference = DecisionModelInference(repository)
    inference.model_label = "btc-trend"
    inference.load_weights("1.0.0")

    latest_features = dataset.vectors[-1].features
    score = inference.score(latest_features, context={"symbol": "BTCUSDT"})
    assert score.success_probability > 0.0

    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=25.0,
        min_net_edge_bps=-5.0,
        max_daily_loss_pct=0.5,
        max_drawdown_pct=0.5,
        max_position_ratio=150.0,
        max_open_positions=10,
        max_latency_ms=1000.0,
    )
    config = DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.0,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )
    orchestrator = DecisionOrchestrator(config)
    orchestrator.attach_named_inference("btc-trend", inference, set_default=True)

    risk_snapshot = {
        "start_of_day_equity": 1_000_000.0,
        "last_equity": 1_000_500.0,
        "peak_equity": 1_000_500.0,
        "daily_realized_pnl": 0.0,
        "positions": {},
    }

    candidate = DecisionCandidate(
        strategy="trend_follow",
        action="enter",
        risk_profile="core",
        symbol="BTCUSDT",
        notional=10_000.0,
        expected_return_bps=score.expected_return_bps,
        expected_probability=score.success_probability,
        metadata={"features": dict(latest_features)},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, risk_snapshot)
    assert evaluation.accepted, evaluation.reasons
    assert evaluation.model_name == "btc-trend"

    market = MarketMetadata(base_asset="BTC", quote_asset="USDT", step_size=0.001, tick_size=0.1)
    decision_payload = evaluation.to_mapping()
    order_request = decision_to_order_request(decision_payload, price=27000.0, market=market)
    assert order_request.quantity > 0.0

    adapter = _StubExchangeAdapter()
    service = ExchangeAdapterExecutionService(adapter=adapter)
    context = ExecutionContext(
        portfolio_id="core",
        risk_profile="core",
        environment="lab",
        metadata={"strategy": "trend_follow"},
    )
    result = service.execute(order_request, context)

    assert result.status == "filled"
    assert adapter.orders, "zlecenie powinno zostać zarejestrowane"
    recorded_order = adapter.orders[-1]
    assert recorded_order.symbol == "BTCUSDT"
    assert recorded_order.quantity == order_request.quantity
