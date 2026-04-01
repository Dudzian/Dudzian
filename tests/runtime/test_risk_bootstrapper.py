"""Safety-net tests dla RiskBootstrapper."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from bot_core.runtime.risk_bootstrapper import RiskBootstrapper


class _DummyScheduler:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def configure_signal_limits(self, limits: object) -> None:
        self.calls.append(limits)


def test_bootstrap_context_delegates_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    expected = SimpleNamespace(name="ctx")

    def _fake_bootstrap_environment(environment_name: str, **kwargs: object) -> object:
        captured["environment_name"] = environment_name
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "bot_core.runtime.risk_bootstrapper.bootstrap_environment",
        _fake_bootstrap_environment,
    )

    bootstrapper = RiskBootstrapper()
    secret_manager = object()
    adapter_factories = {"x": object()}
    core_config = object()

    result = bootstrapper.bootstrap_context(
        environment_name="paper",
        config_path="/tmp/config.yaml",
        secret_manager=secret_manager,  # type: ignore[arg-type]
        adapter_factories=adapter_factories,  # type: ignore[arg-type]
        risk_profile_name="balanced",
        core_config=core_config,  # type: ignore[arg-type]
    )

    assert result is expected
    assert captured == {
        "environment_name": "paper",
        "config_path": "/tmp/config.yaml",
        "secret_manager": secret_manager,
        "adapter_factories": adapter_factories,
        "risk_profile_name": "balanced",
        "core_config": core_config,
    }


def test_bootstrap_context_uses_bootstrap_fn_hook() -> None:
    captured: dict[str, object] = {}

    def _custom_bootstrap(environment_name: str, **kwargs: object) -> object:
        captured["environment_name"] = environment_name
        captured.update(kwargs)
        return "custom"

    result = RiskBootstrapper().bootstrap_context(
        environment_name="paper",
        config_path="/tmp/config.yaml",
        secret_manager=object(),  # type: ignore[arg-type]
        bootstrap_fn=_custom_bootstrap,
    )

    assert result == "custom"
    assert captured["environment_name"] == "paper"
    assert captured["config_path"] == "/tmp/config.yaml"

def test_bootstrap_context_logs_and_reraises_without_changing_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _broken_bootstrap_environment(_environment_name: str, **_kwargs: object) -> object:
        raise ValueError("boom")

    monkeypatch.setattr(
        "bot_core.runtime.risk_bootstrapper.bootstrap_environment",
        _broken_bootstrap_environment,
    )

    bootstrapper = RiskBootstrapper()
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError, match="boom"):
        bootstrapper.bootstrap_context(
            environment_name="paper",
            config_path="/tmp/config.yaml",
            secret_manager=object(),  # type: ignore[arg-type]
        )

    assert any(
        "Risk bootstrap failed during bootstrap_environment" in record.getMessage()
        for record in caplog.records
    )


def test_bootstrap_io_guardrails_builds_components_and_configures_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: dict[str, object] = {}

    class _FakeGuardrails:
        def __init__(self, **kwargs: object) -> None:
            events["guardrails_kwargs"] = kwargs

    class _FakeTaskQueue:
        def __init__(self, **kwargs: object) -> None:
            events["task_queue_kwargs"] = kwargs
            self.configured: list[tuple[str, int, int]] = []

        def configure_exchange(self, name: str, *, max_concurrency: int, burst: int) -> None:
            self.configured.append((name, max_concurrency, burst))

    monkeypatch.setattr("bot_core.runtime.risk_bootstrapper.AsyncIOGuardrails", _FakeGuardrails)
    monkeypatch.setattr("bot_core.runtime.risk_bootstrapper.AsyncIOTaskQueue", _FakeTaskQueue)

    runtime_config = SimpleNamespace(
        io_queue=SimpleNamespace(
            log_directory="logs/custom",
            rate_limit_warning_seconds=12.0,
            timeout_warning_seconds=13.0,
            max_concurrency=4,
            burst=7,
            exchanges={
                "binance": SimpleNamespace(max_concurrency=2, burst=3),
                "kraken": SimpleNamespace(max_concurrency=5, burst=6),
            },
        )
    )
    bootstrap_ctx = SimpleNamespace(metrics_ui_alerts_path="logs/ui.jsonl", io_dispatcher=None)

    bootstrapper = RiskBootstrapper()
    dispatcher, guardrails = bootstrapper.bootstrap_io_guardrails(
        runtime_config=runtime_config,  # type: ignore[arg-type]
        bootstrap_ctx=bootstrap_ctx,  # type: ignore[arg-type]
        environment_name="paper",
    )

    assert dispatcher is bootstrap_ctx.io_dispatcher
    assert guardrails is events["task_queue_kwargs"]["event_listener"]
    assert events["guardrails_kwargs"]["environment"] == "paper"
    assert events["task_queue_kwargs"]["default_max_concurrency"] == 4
    assert dispatcher.configured == [("binance", 2, 3), ("kraken", 5, 6)]


def test_bootstrap_io_guardrails_returns_none_and_persists_none_without_io_queue() -> None:
    bootstrap_ctx = SimpleNamespace(io_dispatcher="old")
    bootstrapper = RiskBootstrapper()

    dispatcher, guardrails = bootstrapper.bootstrap_io_guardrails(
        runtime_config=SimpleNamespace(io_queue=None),  # type: ignore[arg-type]
        bootstrap_ctx=bootstrap_ctx,  # type: ignore[arg-type]
        environment_name="paper",
    )

    assert dispatcher is None
    assert guardrails is None
    assert bootstrap_ctx.io_dispatcher is None


def test_bootstrap_io_guardrails_swallows_persist_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="bot_core.runtime.risk_bootstrapper")

    class _ReadOnlyBootstrap:
        def __init__(self) -> None:
            self.metrics_ui_alerts_path = None

        @property
        def io_dispatcher(self) -> None:
            return None

        @io_dispatcher.setter
        def io_dispatcher(self, _value: object) -> None:
            raise RuntimeError("read only")

    class _FakeGuardrails:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _FakeTaskQueue:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def configure_exchange(self, _name: str, *, max_concurrency: int, burst: int) -> None:
            _ = (max_concurrency, burst)

    monkeypatch.setattr("bot_core.runtime.risk_bootstrapper.AsyncIOGuardrails", _FakeGuardrails)
    monkeypatch.setattr("bot_core.runtime.risk_bootstrapper.AsyncIOTaskQueue", _FakeTaskQueue)

    runtime_config = SimpleNamespace(
        io_queue=SimpleNamespace(
            log_directory=None,
            rate_limit_warning_seconds=1.0,
            timeout_warning_seconds=2.0,
            max_concurrency=1,
            burst=1,
            exchanges={},
        )
    )

    bootstrapper = RiskBootstrapper()
    dispatcher, guardrails = bootstrapper.bootstrap_io_guardrails(
        runtime_config=runtime_config,  # type: ignore[arg-type]
        bootstrap_ctx=_ReadOnlyBootstrap(),  # type: ignore[arg-type]
        environment_name="paper",
    )

    assert dispatcher is not None
    assert guardrails is not None
    assert any(
        "Nie udało się zarejestrować io_dispatcher w BootstrapContext" in record.getMessage()
        and record.levelno == logging.DEBUG
        for record in caplog.records
    )


def test_bind_scheduler_limits_configures_when_present() -> None:
    scheduler = _DummyScheduler()

    RiskBootstrapper.bind_scheduler_limits(
        scheduler,  # type: ignore[arg-type]
        signal_limits={"trend": {"balanced": 3}},
    )

    assert scheduler.calls == [{"trend": {"balanced": 3}}]


def test_bind_scheduler_limits_noop_without_limits() -> None:
    scheduler = _DummyScheduler()

    RiskBootstrapper.bind_scheduler_limits(scheduler, signal_limits=None)  # type: ignore[arg-type]

    assert scheduler.calls == []
