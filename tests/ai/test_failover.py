from __future__ import annotations

from bot_core.ai.health import ModelHealthMonitor


def test_model_health_monitor_detects_quality_degradation() -> None:
    monitor = ModelHealthMonitor(consecutive_failure_threshold=2)

    first = monitor.record_quality(
        model_name="trend_ai",
        ok=False,
        metrics={"directional_accuracy": 0.42, "mae": 18.0},
        thresholds={"min_directional_accuracy": 0.55, "max_mae": 20.0},
    )
    assert first.degraded is False

    degraded = monitor.record_quality(
        model_name="trend_ai",
        ok=False,
        metrics={"directional_accuracy": 0.4, "mae": 22.5},
        thresholds={"min_directional_accuracy": 0.55, "max_mae": 20.0},
    )

    assert degraded.degraded is True
    assert degraded.reason is not None and degraded.reason.startswith("quality_thresholds_failed")
    assert "trend_ai" in degraded.failing_models
    assert degraded.quality_failures >= 1


def test_model_health_monitor_resolves_after_success() -> None:
    monitor = ModelHealthMonitor(consecutive_failure_threshold=1)
    monitor.record_quality(
        model_name="mean_reversion",
        ok=False,
        metrics={"directional_accuracy": 0.3, "mae": 25.0},
        thresholds={"min_directional_accuracy": 0.6, "max_mae": 15.0},
    )
    assert monitor.is_degraded() is True

    recovered = monitor.record_quality(
        model_name="mean_reversion",
        ok=True,
        metrics={"directional_accuracy": 0.65, "mae": 12.0},
        thresholds={"min_directional_accuracy": 0.6, "max_mae": 15.0},
    )

    assert recovered.degraded is False
    assert monitor.is_degraded() is False


def test_model_health_monitor_backend_failure_flow() -> None:
    monitor = ModelHealthMonitor()
    degraded = monitor.record_backend_failure(
        reason="fallback_ai_models",
        details=("import_error",),
    )
    assert degraded.degraded is True
    assert degraded.backend_degraded is True

    resolved = monitor.resolve_backend_recovery()
    assert resolved.degraded is False
    assert resolved.backend_degraded is False
