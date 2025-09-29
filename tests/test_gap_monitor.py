from datetime import datetime, timedelta, timezone

from bot_core.alerts import AlertMessage
from bot_core.data.ohlcv.backfill import BackfillSummary
from bot_core.data.ohlcv.gap_monitor import DataGapIncidentTracker, GapAlertPolicy


class DummyRouter:
    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def dispatch(self, message: AlertMessage) -> None:
        self.messages.append(message)


def _summary(symbol: str, interval: str, end: int) -> BackfillSummary:
    return BackfillSummary(
        symbol=symbol,
        interval=interval,
        requested_start=end - 600_000,
        requested_end=end,
        fetched_candles=0,
        skipped_candles=0,
    )


def test_gap_tracker_sends_warning_on_threshold_exceeded() -> None:
    router = DummyRouter()
    now_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    metadata = {
        "last_timestamp::BTCUSDT::1h": str(now_ms - 90 * 60_000),
        "row_count::BTCUSDT::1h": "1200",
    }
    policy = GapAlertPolicy(warning_gap_minutes={"1h": 60})
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    tracker.handle_summaries(interval="1h", summaries=[_summary("BTCUSDT", "1h", now_ms)], as_of_ms=now_ms)

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.severity == "warning"
    assert message.context["symbol"] == "BTCUSDT"
    assert message.context["interval"] == "1h"


def test_gap_tracker_opens_incident_after_repeated_warnings() -> None:
    router = DummyRouter()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [base_time + timedelta(minutes=idx) for idx in range(3)]

    def clock() -> datetime:
        return times.pop(0)

    now_ms = int((base_time + timedelta(minutes=30)).timestamp() * 1000)
    metadata = {
        "last_timestamp::ETHUSDT::1h": str(now_ms - 180 * 60_000),
        "row_count::ETHUSDT::1h": "600",
    }
    policy = GapAlertPolicy(
        warning_gap_minutes={"1h": 60},
        incident_threshold_count=3,
        incident_window_minutes=10,
        sms_escalation_minutes=15,
    )
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=clock,
    )

    for _ in range(3):
        tracker.handle_summaries(
            interval="1h",
            summaries=[_summary("ETHUSDT", "1h", now_ms)],
            as_of_ms=now_ms,
        )

    assert len(router.messages) == 3
    assert router.messages[-1].severity == "critical"
    assert "INCIDENT" in router.messages[-1].title


def test_gap_tracker_escalates_sms_and_recovers() -> None:
    router = DummyRouter()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    clock_times = [
        base_time,
        base_time + timedelta(minutes=2),
        base_time + timedelta(minutes=4),
        base_time + timedelta(minutes=20),
        base_time + timedelta(minutes=40),
    ]

    def clock() -> datetime:
        return clock_times.pop(0)

    now_ms = int((base_time + timedelta(minutes=30)).timestamp() * 1000)
    metadata = {
        "last_timestamp::SOLUSDT::15m": str(now_ms - 45 * 60_000),
        "row_count::SOLUSDT::15m": "350",
    }
    policy = GapAlertPolicy(
        warning_gap_minutes={"15m": 10},
        incident_threshold_count=3,
        incident_window_minutes=10,
        sms_escalation_minutes=15,
    )
    tracker = DataGapIncidentTracker(
        router=router,
        metadata_provider=lambda: metadata,
        policy=policy,
        environment_name="test-env",
        exchange="binance_spot",
        clock=clock,
    )

    for _ in range(3):
        tracker.handle_summaries(
            interval="15m",
            summaries=[_summary("SOLUSDT", "15m", now_ms)],
            as_of_ms=now_ms,
        )

    tracker.handle_summaries(
        interval="15m",
        summaries=[_summary("SOLUSDT", "15m", now_ms)],
        as_of_ms=now_ms,
    )

    metadata["last_timestamp::SOLUSDT::15m"] = str(now_ms)
    tracker.handle_summaries(
        interval="15m",
        summaries=[_summary("SOLUSDT", "15m", now_ms)],
        as_of_ms=now_ms,
    )

    assert any("Eskalacja SMS" in msg.title for msg in router.messages)
    assert router.messages[-1].severity == "info"
    assert "Incydent zamknięty" in router.messages[-1].title

