import pytest

from bot_core.data.ohlcv.scheduler import OHLCVRefreshScheduler, _compute_sleep_seconds


class _DummyService:
    def __init__(self):
        self.calls = []

    def synchronize(self, **kwargs):
        self.calls.append(kwargs)
        return []


def test_compute_sleep_seconds_without_jitter():
    assert _compute_sleep_seconds(600, 0) == pytest.approx(600.0)


def test_compute_sleep_seconds_with_jitter(monkeypatch):
    monkeypatch.setattr(
        "bot_core.data.ohlcv.scheduler.random.uniform",
        lambda lower, upper: upper,
    )
    assert _compute_sleep_seconds(300, 30) == pytest.approx(330.0)


def test_add_job_validates_jitter_and_persists_value():
    service = _DummyService()
    scheduler = OHLCVRefreshScheduler(service)

    scheduler.add_job(
        symbols=("BTCUSDT",),
        interval="1h",
        lookback_ms=600_000,
        frequency_seconds=900,
        jitter_seconds=45,
    )

    assert len(scheduler._jobs) == 1
    job = scheduler._jobs[0]
    assert job.jitter_seconds == 45

    with pytest.raises(ValueError):
        scheduler.add_job(
            symbols=("BTCUSDT",),
            interval="1h",
            lookback_ms=600_000,
            frequency_seconds=900,
            jitter_seconds=-1,
        )
