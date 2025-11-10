"""Testy CLI skryptu run_multi_strategy_scheduler."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


from scripts import run_multi_strategy_scheduler  # type: ignore  # noqa: E402
from bot_core.runtime.capital_policies import FixedWeightAllocation
from bot_core.security.guards import LicenseCapabilityError
from bot_core.security.license import LicenseValidationError


class _DummyScheduler:
    def __init__(self) -> None:
        self.run_once_called = False
        self.run_forever_called = False
        self.suspend_schedule_calls: list[dict[str, object]] = []
        self.suspend_tag_calls: list[dict[str, object]] = []
        self.resume_schedule_calls: list[str] = []
        self.resume_tag_calls: list[str] = []
        self._active_schedules: set[str] = {"existing"}
        self._active_tags: set[str] = {"sunset"}
        self.configure_signal_limit_calls: list[dict[str, object]] = []
        self._signal_limit_overrides: dict[tuple[str, str], dict[str, object]] = {
            ("trend_following", "balanced"): {"limit": 2, "active": True}
        }
        self._capital_state = {
            "effective": {"alpha": 0.55, "beta": 0.45},
            "raw": {"alpha": 0.6, "beta": 0.4},
            "smoothed": {"alpha": 0.57, "beta": 0.43},
            "profiles": {"balanced": 0.7, "aggressive": 0.3},
            "tags": {"trend": 0.5, "grid": 0.5},
            "tag_members": {"trend": 2, "grid": 1},
        }
        self._capital_diagnostics = {
            "policy_name": "signal_strength",
            "flags": {"drawdown": False},
            "details": {"alpha": {"score": 1.2}},
            "tag_weights": {"trend": 0.5, "grid": 0.5},
            "tag_members": {"trend": 2, "grid": 1},
        }
        self.rebalance_capital_calls = 0
        self.rebalance_kwargs: dict[str, Any] | None = None
        self.replace_policy_calls: list[dict[str, object]] = []
        self.last_policy = None
        self.last_policy_rebalance = True
        self.allocation_interval: float | None = None
        self.describe_calls = 0
        self._schedule_descriptions: Mapping[str, Mapping[str, object]] = {
            "trend_alpha": {
                "strategy_name": "trend_following",
                "risk_profile": "balanced",
                "cadence_seconds": 60.0,
                "max_drift_seconds": 5.0,
                "warmup_bars": 20,
                "base_max_signals": 3,
                "active_max_signals": 2,
                "allocator_weight": 0.5,
                "portfolio_weight": 0.4,
                "tags": ["trend", "spot"],
                "primary_tag": "trend",
                "signal_limit_override": 2,
                "last_run": "2024-01-01T00:00:00+00:00",
            }
        }

    async def run_once(self) -> None:
        self.run_once_called = True

    async def run_forever(self) -> None:
        self.run_forever_called = True

    def stop(self) -> None:  # pragma: no cover - nie wywołujemy w scenariuszach pozytywnych
        pass

    def suspend_schedule(self, name: str, **kwargs: object) -> None:
        self.suspend_schedule_calls.append({"name": name, **kwargs})
        self._active_schedules.add(name)

    def suspend_tag(self, tag: str, **kwargs: object) -> None:
        self.suspend_tag_calls.append({"tag": tag, **kwargs})
        self._active_tags.add(tag)

    def resume_schedule(self, name: str) -> bool:
        self.resume_schedule_calls.append(name)
        existed = name in self._active_schedules
        self._active_schedules.discard(name)
        return existed

    def resume_tag(self, tag: str) -> bool:
        self.resume_tag_calls.append(tag)
        existed = tag in self._active_tags
        self._active_tags.discard(tag)
        return existed

    def configure_signal_limit(
        self,
        strategy: str,
        profile: str,
        limit: int | None,
        *,
        reason: str | None = None,
        until: object | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self.configure_signal_limit_calls.append(
            {
                "strategy": strategy,
                "profile": profile,
                "limit": limit,
                "reason": reason,
                "until": until,
                "duration_seconds": duration_seconds,
            }
        )
        key = (strategy, profile)
        if limit is None:
            self._signal_limit_overrides.pop(key, None)
        else:
            payload: dict[str, object] = {
                "limit": int(limit),
                "active": True,
            }
            if reason:
                payload["reason"] = reason
            if until:
                payload["expires_at"] = until
            if duration_seconds:
                payload["remaining_seconds"] = float(duration_seconds)
            self._signal_limit_overrides[key] = payload

    def suspension_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        schedules = {
            name: {"reason": "manual", "until": "2030-01-01T00:00:00+00:00"}
            for name in sorted(self._active_schedules)
        }
        tags = {
            tag: {"reason": "manual", "until": "bez terminu"}
            for tag in sorted(self._active_tags)
        }
        return {"schedules": schedules, "tags": tags}

    def capital_allocation_state(self) -> Mapping[str, object]:
        return self._capital_state

    def capital_policy_diagnostics(self) -> Mapping[str, object]:
        return self._capital_diagnostics

    def signal_limit_snapshot(self) -> Mapping[str, Mapping[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for (strategy, profile), value in self._signal_limit_overrides.items():
            snapshot.setdefault(strategy, {})[profile] = dict(value)
        return snapshot

    async def rebalance_capital(
        self,
        *,
        timestamp: object | None = None,
        ignore_cooldown: bool = True,
    ) -> None:
        self.rebalance_capital_calls += 1
        self.rebalance_kwargs = {
            "timestamp": timestamp,
            "ignore_cooldown": ignore_cooldown,
        }

    async def replace_capital_policy(
        self,
        policy: object,
        *,
        rebalance: bool = True,
        timestamp: object | None = None,
    ) -> None:
        self.replace_policy_calls.append(
            {"policy": policy, "rebalance": rebalance, "timestamp": timestamp}
        )
        self.last_policy = policy
        self.last_policy_rebalance = rebalance

    def set_allocation_rebalance_seconds(self, value: float | None) -> None:
        self.allocation_interval = None if value is None else float(value)

    def describe_schedules(self) -> Mapping[str, Mapping[str, object]]:
        self.describe_calls += 1
        return self._schedule_descriptions


class _DummySecretStorage:
    def get_secret(self, *_args: object, **_kwargs: object) -> str | None:  # pragma: no cover - proste stuby
        return None

    def put_secret(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - proste stuby
        return None


def _invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: Sequence[str],
) -> tuple[_DummyScheduler, dict[str, Any]]:
    captured: dict[str, Any] = {"async_coro_name": None}
    scheduler = _DummyScheduler()

    def fake_build_scheduler(*, adapter_factories: Mapping[str, object] | None, **kwargs: object) -> _DummyScheduler:
        captured["adapter_factories"] = adapter_factories
        captured["kwargs"] = kwargs
        return scheduler

    def fake_asyncio_run(coro: Any, /) -> None:
        captured["async_coro_name"] = (
            getattr(coro, "cr_code", None).co_name if hasattr(coro, "cr_code") else None
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(run_multi_strategy_scheduler, "_build_scheduler", fake_build_scheduler)
    monkeypatch.setattr(run_multi_strategy_scheduler.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(sys, "argv", ["run_multi_strategy_scheduler.py", *argv])

    run_multi_strategy_scheduler.main()

    return scheduler, captured


def test_main_invokes_run_once_with_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--run-once",
        "--adapter-factory",
        "kucoin_spot=bot_core.exchanges.kucoin:KuCoinSpotAdapter",
        "--adapter-factory",
        "bybit_spot=json:{\"path\": \"bot_core.exchanges.bybit:BybitSpotAdapter\", \"override\": true}",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["async_coro_name"] == "run_once"
    adapter_factories = captured["adapter_factories"]
    assert isinstance(adapter_factories, Mapping)
    assert adapter_factories == {
        "kucoin_spot": "bot_core.exchanges.kucoin:KuCoinSpotAdapter",
        "bybit_spot": {"path": "bot_core.exchanges.bybit:BybitSpotAdapter", "override": True},
    }
    assert scheduler.run_once_called is True
    assert scheduler.run_forever_called is False


def test_main_invokes_run_forever_without_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "okx_paper",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["async_coro_name"] == "run_forever"
    assert captured["adapter_factories"] is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is True


def test_main_supports_remove_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "coinbase_paper",
        "--run-once",
        "--adapter-factory",
        "kraken_spot=!remove",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["adapter_factories"] == {"kraken_spot": {"remove": True}}
    assert captured["async_coro_name"] == "run_once"
    assert scheduler.run_once_called is True
    assert scheduler.run_forever_called is False


def test_management_actions_suspend_and_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--suspend-schedule",
        "alpha",
        "--suspend-tag",
        "beta",
        "--resume-schedule",
        "existing",
        "--resume-tag",
        "sunset",
        "--suspension-reason",
        "maintenance",
        "--suspension-duration",
        "15m",
        "--list-suspensions",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is False
    assert scheduler.resume_schedule_calls == ["existing"]
    assert scheduler.resume_tag_calls == ["sunset"]
    assert scheduler.suspend_schedule_calls == [
        {
            "name": "alpha",
            "reason": "maintenance",
            "until": None,
            "duration_seconds": pytest.approx(900.0),
        }
    ]
    assert scheduler.suspend_tag_calls == [
        {
            "tag": "beta",
            "reason": "maintenance",
            "until": None,
            "duration_seconds": pytest.approx(900.0),
        }
    ]

    out = capsys.readouterr().out
    assert "Wznowiono harmonogram: existing" in out
    assert "Wznowiono tag: sunset" in out
    assert "Zawieszono harmonogram: alpha" in out
    assert "Zawieszenia harmonogramów:" in out
    assert "alpha" in out


def test_export_suspensions_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "suspensions.json"
    argv = [
        "--environment",
        "binance_paper",
        "--export-suspensions",
        str(target),
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is False

    out = capsys.readouterr().out
    assert f"Zapisano zawieszenia do {target}" in out

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schedules"]["existing"]["reason"] == "manual"
    assert payload["tags"]["sunset"]["reason"] == "manual"


def test_management_actions_run_after(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--suspend-schedule",
        "gamma",
        "--run-after-management",
        "--run-once",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["async_coro_name"] == "run_once"
    assert scheduler.run_once_called is True
    assert scheduler.suspend_schedule_calls[0]["name"] == "gamma"


def test_show_capital_state_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--show-capital-state",
        "--show-capital-diagnostics",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is False

    out = capsys.readouterr().out
    assert "Stan alokacji kapitału:" in out
    assert '"alpha": 0.55' in out
    assert "Diagnostyka polityki kapitału:" in out
    assert '"policy_name": "signal_strength"' in out


def test_list_and_export_schedules(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "schedules.json"
    argv = [
        "--environment",
        "binance_paper",
        "--list-schedules",
        "--export-schedules",
        str(target),
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is False

    out = capsys.readouterr().out
    assert "Zarejestrowane harmonogramy:" in out
    assert "trend_alpha" in out
    assert scheduler.describe_calls == 1

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["trend_alpha"]["strategy_name"] == "trend_following"


def test_set_and_list_signal_limits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--set-signal-limit",
        "trend_following:balanced=5",
        "--signal-limit-reason",
        "audit",
        "--signal-limit-duration",
        "180",
        "--list-signal-limits",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.configure_signal_limit_calls == [
        {
            "strategy": "trend_following",
            "profile": "balanced",
            "limit": 5,
            "reason": "audit",
            "until": None,
            "duration_seconds": 180.0,
        }
    ]

    out = capsys.readouterr().out
    assert "Nadpisane limity sygnałów:" in out
    assert "trend_following" in out
    assert "balanced" in out
    assert "5" in out
    assert "powód=audit" in out
    assert "czas=180s" in out


def test_clear_and_export_signal_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "limits.json"
    argv = [
        "--environment",
        "binance_paper",
        "--clear-signal-limit",
        "trend_following:balanced",
        "--export-signal-limits",
        str(target),
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.configure_signal_limit_calls == [
        {
            "strategy": "trend_following",
            "profile": "balanced",
            "limit": None,
            "reason": None,
            "until": None,
            "duration_seconds": None,
        }
    ]

    out = capsys.readouterr().out
    assert f"Zapisano limity sygnałów do {target}" in out

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {}


def test_invalid_signal_limit_spec_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = _DummyScheduler()

    def fake_build_scheduler(*_args: object, **_kwargs: object) -> _DummyScheduler:
        return scheduler

    def fake_asyncio_run(coro: Any, /) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(run_multi_strategy_scheduler, "_build_scheduler", fake_build_scheduler)
    monkeypatch.setattr(run_multi_strategy_scheduler.asyncio, "run", fake_asyncio_run)

    caplog.set_level(logging.ERROR)
    result = run_multi_strategy_scheduler.run(
        [
            "--environment",
            "binance_paper",
            "--set-signal-limit",
            "trend_only",
        ]
    )

    assert result == 1
    assert "Operacja zarządzania schedulerem nie powiodła się" in caplog.text


def test_export_capital_state_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    diag_path = tmp_path / "diag.json"

    argv = [
        "--environment",
        "binance_paper",
        "--export-capital-state",
        str(state_path),
        "--export-capital-diagnostics",
        str(diag_path),
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is False

    out = capsys.readouterr().out
    assert f"Zapisano stan alokacji kapitału do {state_path}" in out
    assert f"Zapisano diagnostykę polityki kapitału do {diag_path}" in out

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    diag_payload = json.loads(diag_path.read_text(encoding="utf-8"))

    assert state_payload["effective"]["alpha"] == pytest.approx(0.55)
    assert diag_payload["policy_name"] == "signal_strength"


def test_rebalance_capital_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--rebalance-capital",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") == "rebalance_capital"
    assert scheduler.rebalance_capital_calls == 1
    assert scheduler.rebalance_kwargs == {
        "timestamp": None,
        "ignore_cooldown": True,
    }

    out = capsys.readouterr().out
    assert "Przeliczono alokację kapitału." in out


def test_set_capital_policy_from_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """
name: fixed_weight
weights:
  alpha: 2
  beta: 1
rebalance_seconds: 42
        """.strip(),
        encoding="utf-8",
    )

    argv = [
        "--environment",
        "binance_paper",
        "--set-capital-policy",
        str(policy_path),
        "--apply-policy-interval",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") == "replace_capital_policy"
    assert isinstance(scheduler.last_policy, FixedWeightAllocation)
    assert scheduler.last_policy_rebalance is True
    assert scheduler.allocation_interval == pytest.approx(42.0)

    out = capsys.readouterr().out
    assert "Zastosowano politykę kapitału" in out
    assert "Ustawiono interwał przeliczeń alokacji" in out


def test_set_capital_policy_without_rebalance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps("risk_parity"), encoding="utf-8")

    argv = [
        "--environment",
        "binance_paper",
        "--set-capital-policy",
        str(policy_path),
        "--skip-policy-rebalance",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") == "replace_capital_policy"
    assert scheduler.last_policy_rebalance is False
    assert scheduler.allocation_interval is None

    out = capsys.readouterr().out
    assert "Załadowano politykę kapitału" in out


def test_set_allocation_interval(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--set-allocation-interval",
        "180",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured.get("async_coro_name") is None
    assert scheduler.allocation_interval == pytest.approx(180.0)

    out = capsys.readouterr().out
    assert "Ustawiono interwał przeliczeń alokacji" in out


def test_build_scheduler_reraises_license_error(monkeypatch: pytest.MonkeyPatch) -> None:
    license_exc = LicenseCapabilityError("Scheduler wymaga modułu Walk Forward.")

    def fake_build_runtime(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bootstrap failed") from license_exc

    monkeypatch.setattr(run_multi_strategy_scheduler, "build_multi_strategy_runtime", fake_build_runtime)
    monkeypatch.setattr(
        run_multi_strategy_scheduler,
        "create_default_secret_storage",
        lambda: _DummySecretStorage(),
    )

    with pytest.raises(LicenseCapabilityError) as exc:
        run_multi_strategy_scheduler._build_scheduler(  # type: ignore[attr-defined]
            config_path=Path("config/core.yaml"),
            environment="binance_paper",
            scheduler_name=None,
            adapter_factories=None,
        )
    assert "Walk Forward" in str(exc.value)


def test_run_returns_error_code_on_license_block(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_build_runtime(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bootstrap failed") from LicenseCapabilityError("Brak modułu multi-strategy")

    monkeypatch.setattr(run_multi_strategy_scheduler, "build_multi_strategy_runtime", fake_build_runtime)
    monkeypatch.setattr(
        run_multi_strategy_scheduler,
        "create_default_secret_storage",
        lambda: _DummySecretStorage(),
    )

    with caplog.at_level(logging.ERROR, logger=run_multi_strategy_scheduler.LOGGER.name):
        exit_code = run_multi_strategy_scheduler.run(["--environment", "binance_paper"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Uruchomienie scheduler-a multi-strategy zablokowane przez licencję" in caplog.text


def test_run_returns_error_code_on_license_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_build_runtime(*_args: object, **_kwargs: object) -> object:
        raise LicenseValidationError("Brak licencji Pro")

    monkeypatch.setattr(run_multi_strategy_scheduler, "build_multi_strategy_runtime", fake_build_runtime)
    monkeypatch.setattr(
        run_multi_strategy_scheduler,
        "create_default_secret_storage",
        lambda: _DummySecretStorage(),
    )

    with caplog.at_level(logging.ERROR, logger=run_multi_strategy_scheduler.LOGGER.name):
        exit_code = run_multi_strategy_scheduler.run(["--environment", "binance_paper"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Walidacja licencji nie powiodła się" in caplog.text
