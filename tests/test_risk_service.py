from datetime import datetime, timezone
from collections.abc import Mapping
import hashlib
import pytest
import threading

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.config.models import ServiceTokenConfig
from bot_core.security.tokens import build_service_token_validator
from bot_core.runtime.risk_service import (
    RiskExposure,
    RiskServer,
    RiskSnapshot,
    RiskSnapshotBuilder,
    RiskSnapshotPublisher,
    RiskSnapshotStore,
)


pytestmark = pytest.mark.requires_trading_stubs


def _snapshot(equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )


def _order(price: float, *, quantity: float = 0.1, atr: float = 200.0) -> OrderRequest:
    stop_price = price - 2.0 * atr
    return OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=quantity,
        order_type="limit",
        price=price,
        stop_price=stop_price,
        atr=atr,
        metadata={"atr": atr, "stop_price": stop_price},
    )


def test_risk_snapshot_builder_generates_exposures() -> None:
    profile = ManualProfile(
        name="paper",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.25,
        daily_loss_limit=0.05,
        max_position_pct=0.6,
        target_volatility=0.1,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(profile)

    account = _snapshot(1_000.0)
    request = _order(20_000.0, quantity=0.02)
    result = engine.apply_pre_trade_checks(
        request,
        account=account,
        profile_name=profile.name,
    )
    assert result.allowed is True

    engine.on_fill(
        profile_name=profile.name,
        symbol="BTCUSDT",
        side="buy",
        position_value=400.0,
        pnl=-50.0,
        timestamp=datetime(2024, 1, 1, 12, 5, 0),
    )

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    )
    snapshot = builder.build(profile.name)
    assert snapshot is not None
    assert snapshot.profile_name == profile.name
    assert snapshot.portfolio_value == pytest.approx(1_000.0)
    assert snapshot.used_leverage == pytest.approx(0.4)
    exposures = {exposure.code: exposure for exposure in snapshot.exposures}
    assert exposures["active_positions"].current == pytest.approx(1.0)
    assert exposures["gross_notional"].current == pytest.approx(400.0)
    assert exposures["largest_position_pct"].current == pytest.approx(0.4)
    assert snapshot.metadata is not None
    assert snapshot.metadata["positions"]

    statistics_meta = snapshot.metadata.get("statistics")
    assert isinstance(statistics_meta, Mapping)
    assert "dailyRealizedPnl" in statistics_meta

    cost_meta = snapshot.metadata.get("cost_breakdown")
    assert isinstance(cost_meta, Mapping)
    assert "averageCostBps" in cost_meta

    stat_exposures = {
        exposure.code: exposure for exposure in snapshot.exposures if exposure.code.startswith("stat:")
    }
    assert "stat:dailyRealizedPnl" in stat_exposures

    cost_exposures = {
        exposure.code: exposure for exposure in snapshot.exposures if exposure.code.startswith("cost:")
    }
    assert "cost:totalCostBps" in cost_exposures


def test_risk_snapshot_builder_includes_recent_decisions(tmp_path) -> None:
    profile = ManualProfile(
        name="audit",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.3,
        daily_loss_limit=0.05,
        max_position_pct=0.5,
        target_volatility=0.15,
        stop_loss_atr_multiple=2.0,
    )

    decision_log = RiskDecisionLog(max_entries=5, jsonl_path=tmp_path / "decisions.jsonl", clock=lambda: datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0), decision_log=decision_log)
    engine.register_profile(profile)

    account = _snapshot(5_000.0)
    allowed_request = _order(25_000.0, quantity=0.05)
    denied_request = _order(25_000.0, quantity=0.05, atr=allowed_request.atr or 200.0)
    denied_request.stop_price = denied_request.price  # wymuś błąd stop loss
    denied_request.metadata = {"atr": denied_request.atr, "stop_price": denied_request.stop_price}

    assert engine.apply_pre_trade_checks(allowed_request, account=account, profile_name=profile.name).allowed
    denied_result = engine.apply_pre_trade_checks(denied_request, account=account, profile_name=profile.name)
    assert denied_result.allowed is False

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
    )

    snapshot = builder.build(profile.name)
    assert snapshot is not None
    metadata = snapshot.metadata
    assert metadata is not None
    recent = metadata.get("recent_decisions")
    assert isinstance(recent, list)
    assert len(recent) == 2
    assert recent[-1]["allowed"] is False


def test_risk_snapshot_builder_attaches_profile_summary() -> None:
    profile = ManualProfile(
        name="manual",
        max_positions=10,
        max_leverage=5.0,
        drawdown_limit=0.4,
        daily_loss_limit=0.1,
        max_position_pct=0.5,
        target_volatility=0.2,
        stop_loss_atr_multiple=2.5,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 5, 4, 10, 30, 0))
    engine.register_profile(profile)

    account = _snapshot(50_000.0)
    engine.apply_pre_trade_checks(
        _order(40_000.0, quantity=0.5),
        account=account,
        profile_name=profile.name,
    )

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 5, 4, 11, 45, 0, tzinfo=timezone.utc),
        profile_summary_resolver=lambda name: {
            "name": name,
            "severity_min": "info",
            "extends_chain": ["balanced"],
        },
    )

    snapshot = builder.build(profile.name)
    assert snapshot is not None
    metadata = snapshot.metadata
    assert metadata is not None
    assert metadata["generated_at"] == snapshot.generated_at.isoformat()
    assert metadata["profile"] == profile.name
    summary = metadata.get("risk_profile_summary")
    assert summary == {
        "name": profile.name,
        "severity_min": "info",
        "extends_chain": ["balanced"],
    }
    assert snapshot.profile_summary() == summary


def test_risk_snapshot_builder_lists_registered_profiles() -> None:
    profile = ManualProfile(
        name="balanced",
        max_positions=4,
        max_leverage=2.5,
        drawdown_limit=0.2,
        daily_loss_limit=0.04,
        max_position_pct=0.4,
        target_volatility=0.12,
        stop_loss_atr_multiple=1.8,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 6, 1, 10, 0, 0))
    engine.register_profile(profile)

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 6, 1, 11, 0, 0, tzinfo=timezone.utc),
    )

    assert profile.name in builder.profile_names()


@pytest.mark.timeout(5)
def test_risk_server_rbac_requires_scope():
    grpc = pytest.importorskip("grpc", reason="Wymaga biblioteki grpcio")
    trading_pb2 = pytest.importorskip(
        "bot_core.generated.trading_pb2",
        reason="Brak wygenerowanych stubów trading_pb2",
    )
    trading_pb2_grpc = pytest.importorskip(
        "bot_core.generated.trading_pb2_grpc",
        reason="Brak wygenerowanych stubów trading_pb2_grpc",
    )

    reader_plain = "risk-reader"
    reader_hash = hashlib.sha256(reader_plain.encode("utf-8")).hexdigest()
    validator = build_service_token_validator(
        [
            ServiceTokenConfig(
                token_id="reader",
                token_hash=f"sha256:{reader_hash}",
                scopes=("risk.read",),
            )
        ],
        default_scope="risk.read",
    )

    server = RiskServer(host="127.0.0.1", port=0, token_validator=validator)
    server.start()
    channel = grpc.insecure_channel(server.address)
    stub = trading_pb2_grpc.RiskServiceStub(channel)

    snapshot = RiskSnapshot(
        profile_name="balanced",
        portfolio_value=10_000.0,
        current_drawdown=0.0,
        daily_loss=0.0,
        used_leverage=0.0,
        exposures=[RiskExposure(code="active_positions", current=0.0)],
        generated_at=datetime.now(timezone.utc),
    )
    server.publish(snapshot)

    try:
        with pytest.raises(grpc.RpcError) as exc:
            stub.GetRiskState(trading_pb2.RiskStateRequest())
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

        metadata = (("authorization", f"Bearer {reader_plain}"),)
        response = stub.GetRiskState(trading_pb2.RiskStateRequest(), metadata=metadata)
        assert response.profile == trading_pb2.RiskProfile.RISK_PROFILE_BALANCED

        stream = stub.StreamRiskState(trading_pb2.RiskStateRequest(), metadata=metadata)
        first = next(stream)
        assert first.profile == trading_pb2.RiskProfile.RISK_PROFILE_BALANCED
    finally:
        server.stop(grace=0)


def test_risk_snapshot_store_appends_and_limits_history() -> None:
    trading_pb2 = pytest.importorskip(
        "bot_core.generated.trading_pb2", reason="Brak wygenerowanych stubów"
    )

    store = RiskSnapshotStore(maxlen=2)
    first = trading_pb2.RiskState()
    first.profile = trading_pb2.RiskProfile.RISK_PROFILE_CONSERVATIVE
    first.portfolio_value = 1_000.0
    store.append(first)

    second = trading_pb2.RiskState()
    second.profile = trading_pb2.RiskProfile.RISK_PROFILE_BALANCED
    second.portfolio_value = 2_000.0
    store.append(second)

    third = trading_pb2.RiskState()
    third.profile = trading_pb2.RiskProfile.RISK_PROFILE_AGGRESSIVE
    third.portfolio_value = 3_000.0
    store.append(third)

    history = store.history()
    assert len(history) == 2
    assert history[0].profile == trading_pb2.RiskProfile.RISK_PROFILE_BALANCED
    assert store.latest().portfolio_value == pytest.approx(3_000.0)


def test_risk_snapshot_store_tracks_metadata() -> None:
    trading_pb2 = pytest.importorskip(
        "bot_core.generated.trading_pb2", reason="Brak wygenerowanych stubów"
    )

    store = RiskSnapshotStore(maxlen=2)
    first = trading_pb2.RiskState()
    first.profile = trading_pb2.RiskProfile.RISK_PROFILE_CONSERVATIVE
    store.append(first, metadata={"profile": "conservative", "sequence": 1})

    second = trading_pb2.RiskState()
    second.profile = trading_pb2.RiskProfile.RISK_PROFILE_BALANCED
    store.append(second, metadata={"profile": "balanced", "sequence": 2})

    metadata_history = store.metadata_history()
    assert metadata_history == [
        {"profile": "conservative", "sequence": 1},
        {"profile": "balanced", "sequence": 2},
    ]

    latest_metadata = store.latest_metadata()
    assert latest_metadata == {"profile": "balanced", "sequence": 2}


def test_risk_snapshot_publisher_publish_once_invokes_sinks() -> None:
    profile = ManualProfile(
        name="paper",
        max_positions=3,
        max_leverage=2.0,
        drawdown_limit=0.15,
        daily_loss_limit=0.03,
        max_position_pct=0.45,
        target_volatility=0.08,
        stop_loss_atr_multiple=1.6,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 2, 1, 9, 0, 0))
    engine.register_profile(profile)

    account = _snapshot(10_000.0)
    engine.apply_pre_trade_checks(
        _order(40_000.0, quantity=0.2),
        account=account,
        profile_name=profile.name,
    )
    engine.on_fill(
        profile_name=profile.name,
        symbol="BTCUSDT",
        side="buy",
        position_value=2_000.0,
        pnl=-150.0,
        timestamp=datetime(2024, 2, 1, 9, 15, 0),
    )

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
    )

    collected: list[RiskSnapshot] = []
    publisher = RiskSnapshotPublisher(builder, sinks=[collected.append])

    snapshots = publisher.publish_once()

    assert snapshots
    assert collected
    assert collected[0].profile_name == profile.name


def test_risk_snapshot_publisher_background_cycle() -> None:
    profile = ManualProfile(
        name="aggressive",
        max_positions=6,
        max_leverage=4.0,
        drawdown_limit=0.3,
        daily_loss_limit=0.08,
        max_position_pct=0.6,
        target_volatility=0.25,
        stop_loss_atr_multiple=2.2,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 3, 10, 8, 0, 0))
    engine.register_profile(profile)

    account = _snapshot(25_000.0)
    engine.apply_pre_trade_checks(
        _order(30_000.0, quantity=0.5),
        account=account,
        profile_name=profile.name,
    )

    builder = RiskSnapshotBuilder(
        engine,
        clock=lambda: datetime(2024, 3, 10, 8, 30, 0, tzinfo=timezone.utc),
    )

    event = threading.Event()
    collected: list[RiskSnapshot] = []

    def _sink(snapshot: RiskSnapshot) -> None:
        collected.append(snapshot)
        event.set()

    publisher = RiskSnapshotPublisher(
        builder,
        sinks=[_sink],
        interval_seconds=0.05,
    )

    try:
        publisher.start()
        assert event.wait(1.0)
    finally:
        publisher.stop()

    assert collected
    assert not publisher.is_running()


def test_risk_snapshot_to_proto_contains_force_liquidation() -> None:
    trading_pb2 = pytest.importorskip(
        "bot_core.generated.trading_pb2", reason="Brak wygenerowanych stubów"
    )

    snapshot = RiskSnapshot(
        profile_name="balanced",
        portfolio_value=1_500.0,
        current_drawdown=0.05,
        daily_loss=0.02,
        used_leverage=1.25,
        exposures=(RiskExposure(code="active_positions", current=1.0, maximum=5.0, threshold=4.0),),
        generated_at=datetime(2024, 1, 1, 14, 30, 0, tzinfo=timezone.utc),
        force_liquidation=True,
        metadata={},
    )

    message = snapshot.to_proto()
    assert message.profile == trading_pb2.RiskProfile.RISK_PROFILE_BALANCED
    codes = {limit.code for limit in message.limits}
    assert "force_liquidation" in codes
    assert message.portfolio_value == pytest.approx(1_500.0)
