from bot_core.runtime.scheduler_load_test import (
    LoadTestResult,
    LoadTestSettings,
    execute_scheduler_load_test,
)


def test_execute_scheduler_load_test_produces_metrics() -> None:
    settings = LoadTestSettings(iterations=5, schedules=2, signals_per_snapshot=3, simulated_latency_ms=1.0)
    result = execute_scheduler_load_test(settings)

    assert isinstance(result, LoadTestResult)
    assert result.schedules == 2
    assert result.iterations == 5
    assert result.signals_emitted == 5 * 2 * 3
    assert result.avg_latency_ms >= 0.0
    assert result.max_latency_ms >= result.avg_latency_ms
    assert result.resource_status in {"ok", "warning", "error"}
    payload = result.as_dict()
    assert payload["signals_emitted"] == result.signals_emitted
