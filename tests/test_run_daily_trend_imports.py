"""Testy weryfikujące fallback importów w run_daily_trend."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
import textwrap
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ensure_repo_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))


def _import_run_daily_trend(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Importuje moduł CLI po wyczyszczeniu poprzednich wersji."""

    importlib.invalidate_caches()
    sys.modules.pop("scripts.run_daily_trend", None)
    module = importlib.import_module("scripts.run_daily_trend")
    return module


def _reset_run_daily_trend(module: types.ModuleType) -> None:
    sys.modules.pop(module.__name__, None)


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.01
                max_position_pct: 0.03
                target_volatility: 0.07
                max_leverage: 2.0
                stop_loss_atr_multiple: 1.5
                max_open_positions: 3
                hard_drawdown_pct: 0.05
            runtime:
              controllers:
                daily_trend_stub:
                  interval: 1d
                  tick_seconds: 86400
              metrics_service:
                enabled: true
                host: 127.0.0.1
                port: 55100
                history_size: 512
                log_sink: true
                jsonl_path: logs/metrics/telemetry.jsonl
                jsonl_fsync: false
                ui_alerts_jsonl_path: logs/metrics/ui_alerts.jsonl
                tls:
                  enabled: true
                  certificate_path: secrets/metrics/server.crt
                  private_key_path: secrets/metrics/server.key
                  client_ca_path: secrets/metrics/clients.pem
                  require_client_auth: true
            strategies:
              core_daily_trend_stub:
                engine: daily_trend_momentum
                parameters:
                  fast_ma: 25
                  slow_ma: 100
                  breakout_lookback: 55
                  momentum_window: 20
                  atr_window: 14
                  atr_multiplier: 2.0
                  min_trend_strength: 0.005
                  min_momentum: 0.001
            environments:
              binance_paper:
                exchange: binance
                environment: paper
                keychain_key: paper_key
                data_cache_path: var/data
                risk_profile: conservative
                alert_channels: []
                default_strategy: core_daily_trend_stub
                default_controller: daily_trend_stub
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


class _StubAdapter:
    """Lekki adapter wykorzystywany w testowych pipeline'ach."""


class _StubController:
    """Minimalna implementacja kontrolera używana w snapshotach runtime."""

    def __init__(
        self,
        *,
        environment: str,
        risk_profile: str,
        symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
        interval: str = "1d",
        tick_seconds: float = 86400.0,
        position_size: float = 1.0,
    ) -> None:
        self._tick_seconds = float(tick_seconds)
        self.interval = interval
        self.symbols = tuple(symbols)
        self.position_size = position_size
        self.execution_context = SimpleNamespace(
            portfolio_id="paper_portfolio",
            risk_profile=risk_profile,
            environment=environment,
            metadata={"leverage": "1.00x"},
        )

    @property
    def tick_seconds(self) -> float:
        return self._tick_seconds


def _make_pipeline_stub(
    config_path: Path,
    *,
    environment_name: str,
    risk_profile: str | None = None,
) -> SimpleNamespace:
    from bot_core.config.loader import load_core_config
    from bot_core.runtime.metrics_alerts import DEFAULT_UI_ALERTS_JSONL_PATH
    from bot_core.strategies.daily_trend import DailyTrendMomentumStrategy

    config = load_core_config(config_path)
    environment_cfg = config.environments[environment_name]
    effective_profile = risk_profile or environment_cfg.risk_profile

    metrics_cfg = getattr(config, "metrics_service", None)
    metrics_enabled: bool | None = None
    metrics_ui_alert_path: Path | None = None
    metrics_jsonl_path: Path | None = None
    if metrics_cfg is not None:
        metrics_enabled = bool(getattr(metrics_cfg, "enabled", False))
        ui_alert_raw = getattr(metrics_cfg, "ui_alerts_jsonl_path", None)
        if ui_alert_raw:
            metrics_ui_alert_path = Path(ui_alert_raw)
        else:
            metrics_ui_alert_path = DEFAULT_UI_ALERTS_JSONL_PATH
        jsonl_raw = getattr(metrics_cfg, "jsonl_path", None)
        if jsonl_raw:
            metrics_jsonl_path = Path(jsonl_raw)

    bootstrap = SimpleNamespace(
        environment=environment_cfg,
        core_config=config,
        risk_profile_name=effective_profile,
        metrics_server=None,
        decision_journal=None,
        adapter=_StubAdapter(),
        metrics_ui_alerts_path=metrics_ui_alert_path,
        metrics_jsonl_path=metrics_jsonl_path,
        metrics_ui_alert_sink_active=bool(metrics_ui_alert_path),
        metrics_service_enabled=metrics_enabled,
    )

    controller = _StubController(
        environment=getattr(environment_cfg.environment, "value", environment_cfg.environment),
        risk_profile=effective_profile,
    )

    strategy = DailyTrendMomentumStrategy()

    return SimpleNamespace(
        bootstrap=bootstrap,
        strategy=strategy,
        strategy_name="stub_strategy",
        controller=controller,
        controller_name="stub_controller",
        risk_profile_name=effective_profile,
    )


def test_run_daily_trend_prefers_pipeline_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Główny moduł pipeline powinien być użyty, gdy dostępny."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        pipeline_module = importlib.import_module("bot_core.runtime.pipeline")
        realtime_module = importlib.import_module("bot_core.runtime.realtime")

        assert module.build_daily_trend_pipeline is pipeline_module.build_daily_trend_pipeline
        assert module.create_trading_controller is pipeline_module.create_trading_controller
        assert module.DailyTrendRealtimeRunner is realtime_module.DailyTrendRealtimeRunner
    finally:
        _reset_run_daily_trend(module)


def test_run_daily_trend_fallbacks_to_runtime_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Przy braku symboli w pipeline korzystamy z re-eksportu w bot_core.runtime."""

    pipeline_module = importlib.import_module("bot_core.runtime.pipeline")
    runtime_module = importlib.import_module("bot_core.runtime")
    realtime_module = importlib.import_module("bot_core.runtime.realtime")

    monkeypatch.delattr(pipeline_module, "build_daily_trend_pipeline", raising=False)
    monkeypatch.delattr(pipeline_module, "create_trading_controller", raising=False)

    runtime_build = lambda *args, **kwargs: ("runtime", args, kwargs)
    runtime_controller = lambda *args, **kwargs: ("runtime_controller", args, kwargs)
    monkeypatch.setattr(runtime_module, "build_daily_trend_pipeline", runtime_build, raising=False)
    monkeypatch.setattr(runtime_module, "create_trading_controller", runtime_controller, raising=False)

    module = _import_run_daily_trend(monkeypatch)
    try:
        assert module.build_daily_trend_pipeline is runtime_build
        assert module.create_trading_controller is runtime_controller
        assert module.DailyTrendRealtimeRunner is realtime_module.DailyTrendRealtimeRunner
        snapshot = module.get_runtime_module_candidates()
        assert snapshot.pipeline_fallback_used is True
        assert snapshot.realtime_fallback_used is False
    finally:
        _reset_run_daily_trend(module)


def test_runtime_module_snapshot_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot modułów powinien zawierać informacje o domyślnym pochodzeniu."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        snapshot = module.get_runtime_module_candidates()
        assert isinstance(snapshot, module.RuntimeModuleSnapshot)
        assert snapshot.pipeline_origin == module._DEFAULT_PIPELINE_ORIGIN
        assert snapshot.realtime_origin == module._DEFAULT_REALTIME_ORIGIN
        assert snapshot.pipeline_resolved_from == module.build_daily_trend_pipeline.__module__
        assert snapshot.realtime_resolved_from == module.DailyTrendRealtimeRunner.__module__
        assert snapshot.pipeline_modules[0] == "bot_core.runtime.pipeline"
        assert snapshot.realtime_modules[0] == "bot_core.runtime.realtime"
        assert snapshot.pipeline_fallback_used is False
        assert snapshot.realtime_fallback_used is False
    finally:
        _reset_run_daily_trend(module)


def test_run_daily_trend_missing_dependencies_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Czytelny komunikat powinien pojawić się przy braku wymaganych symboli."""

    pipeline_module = importlib.import_module("bot_core.runtime.pipeline")
    runtime_module = importlib.import_module("bot_core.runtime")

    monkeypatch.delattr(pipeline_module, "build_daily_trend_pipeline", raising=False)
    monkeypatch.delattr(pipeline_module, "create_trading_controller", raising=False)
    monkeypatch.delattr(runtime_module, "build_daily_trend_pipeline", raising=False)
    monkeypatch.delattr(runtime_module, "create_trading_controller", raising=False)

    with pytest.raises(ImportError) as excinfo:
        _import_run_daily_trend(monkeypatch)

    message = str(excinfo.value)
    assert "build_daily_trend_pipeline" in message
    assert "create_trading_controller" in message


def test_run_daily_trend_realtime_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Brak klasy w module realtime powinien używać fallbacku z bot_core.runtime."""

    runtime_module = importlib.import_module("bot_core.runtime")
    realtime_module = importlib.import_module("bot_core.runtime.realtime")

    fallback_runner = type("FallbackRunner", (), {})

    monkeypatch.delattr(realtime_module, "DailyTrendRealtimeRunner", raising=False)
    monkeypatch.setattr(runtime_module, "DailyTrendRealtimeRunner", fallback_runner, raising=False)

    module = _import_run_daily_trend(monkeypatch)
    try:
        assert module.DailyTrendRealtimeRunner is fallback_runner
        snapshot = module.get_runtime_module_candidates()
        assert snapshot.pipeline_fallback_used is False
        assert snapshot.realtime_fallback_used is True
    finally:
        _reset_run_daily_trend(module)


def test_run_daily_trend_missing_realtime_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Brak modułu realtime powinien zgłaszać informację o DailyTrendRealtimeRunner."""

    runtime_module = importlib.import_module("bot_core.runtime")
    realtime_module = importlib.import_module("bot_core.runtime.realtime")

    monkeypatch.delattr(realtime_module, "DailyTrendRealtimeRunner", raising=False)
    monkeypatch.delattr(runtime_module, "DailyTrendRealtimeRunner", raising=False)

    with pytest.raises(ImportError) as excinfo:
        _import_run_daily_trend(monkeypatch)

    assert "DailyTrendRealtimeRunner" in str(excinfo.value)


def test_run_daily_trend_allows_runtime_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI umożliwia wskazanie własnych modułów pipeline i realtime."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        pipeline_mod = types.ModuleType("tests.custom_pipeline")
        realtime_mod = types.ModuleType("tests.custom_realtime")

        def _pipeline(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("custom_pipeline", args, dict(kwargs))

        def _controller(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("custom_controller", args, dict(kwargs))

        class CustomRunner:
            pass

        pipeline_mod.build_daily_trend_pipeline = _pipeline  # type: ignore[attr-defined]
        pipeline_mod.create_trading_controller = _controller  # type: ignore[attr-defined]
        realtime_mod.DailyTrendRealtimeRunner = CustomRunner  # type: ignore[attr-defined]

        sys.modules[pipeline_mod.__name__] = pipeline_mod
        sys.modules[realtime_mod.__name__] = realtime_mod

        module._apply_runtime_overrides(
            [pipeline_mod.__name__],
            [realtime_mod.__name__],
            pipeline_origin="test override pipeline",
            realtime_origin="test override realtime",
        )

        assert module.build_daily_trend_pipeline is _pipeline
        assert module.create_trading_controller is _controller
        assert module.DailyTrendRealtimeRunner is CustomRunner
        snapshot = module.get_runtime_module_candidates()

        assert snapshot.pipeline_modules[0] == pipeline_mod.__name__
        assert snapshot.pipeline_modules[-1] == "bot_core.runtime"
        assert snapshot.realtime_modules[0] == realtime_mod.__name__
        assert snapshot.pipeline_origin == "test override pipeline"
        assert snapshot.realtime_origin == "test override realtime"
        assert snapshot.pipeline_resolved_from == pipeline_mod.__name__
        assert snapshot.realtime_resolved_from == realtime_mod.__name__
        assert snapshot.pipeline_fallback_used is False
        assert snapshot.realtime_fallback_used is False

    finally:
        sys.modules.pop("tests.custom_pipeline", None)
        sys.modules.pop("tests.custom_realtime", None)
        _reset_run_daily_trend(module)


def test_print_runtime_modules_short_circuit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flaga CLI powinna wypisać moduły i zakończyć działanie przed bootstrapem."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        exit_code = module.main([
            "--print-runtime-modules",
            "--pipeline-module",
            "bot_core.runtime.pipeline",
            "--realtime-module",
            "bot_core.runtime.realtime",
        ])
        assert exit_code == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["pipeline_modules"][0] == "bot_core.runtime.pipeline"
        assert payload["realtime_modules"][0] == "bot_core.runtime.realtime"
        assert payload["pipeline_resolved_from"] == "bot_core.runtime.pipeline"
        assert payload["realtime_resolved_from"] == "bot_core.runtime.realtime"
        assert payload["pipeline_fallback_used"] is False
        assert payload["realtime_fallback_used"] is False
        assert payload["pipeline"]["resolved_from"] == "bot_core.runtime.pipeline"
        assert payload["realtime"]["resolved_from"] == "bot_core.runtime.realtime"
        assert payload["pipeline"]["fallback_used"] is False
        assert payload["realtime"]["fallback_used"] is False
    finally:
        _reset_run_daily_trend(module)


def test_print_risk_profiles_outputs_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: "Path",
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Flaga --print-risk-profiles powinna wypisać profile wraz z limitami."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                risk_profiles:
                  conservative:
                    max_daily_loss_pct: 0.01
                    max_position_pct: 0.03
                    target_volatility: 0.07
                    max_leverage: 2.0
                    stop_loss_atr_multiple: 1.0
                    max_open_positions: 3
                    hard_drawdown_pct: 0.05
                    data_quality:
                      max_gap_minutes: 1440
                      min_ok_ratio: 0.9
                environments:
                  binance_paper:
                    exchange: binance
                    environment: paper
                    keychain_key: paper_key
                    data_cache_path: var/data
                    risk_profile: conservative
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--print-risk-profiles",
            ]
        )
        assert exit_code == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["environment_profile"] == "conservative"
        conservative = payload["profiles"]["conservative"]
        assert conservative["classification"] == "conservative"
        assert "demo→paper→live" in conservative["deployment_pipeline"]
        assert conservative["limits"]["max_leverage"] == 2.0
        assert conservative["associated_environments"] == ["binance_paper"]
        assert conservative["data_quality"]["max_gap_minutes"] == 1440
        assert payload["available_profiles"] == ["conservative"]
    finally:
        _reset_run_daily_trend(module)


def test_print_risk_profiles_missing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "Path"
) -> None:
    """Brak pliku konfiguracyjnego powinien zwracać kod błędu."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        missing_path = tmp_path / "absent.yaml"
        exit_code = module.main([
            "--config",
            str(missing_path),
            "--print-risk-profiles",
        ])
        assert exit_code == 1
    finally:
        _reset_run_daily_trend(module)


def test_resolve_runtime_symbols_with_generator_reports_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostyka powinna zawierać wszystkie moduły nawet dla generatorów."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        candidate_generator = (name for name in ("tests.missing_pipeline", "tests.missing_runtime"))

        with pytest.raises(ImportError) as excinfo:
            module._resolve_runtime_symbols(
                candidate_generator,
                ("missing_symbol",),
                component_hint="test",
            )

        message = str(excinfo.value)
        assert "tests.missing_pipeline" in message
        assert "tests.missing_runtime" in message
    finally:
        _reset_run_daily_trend(module)


def test_environment_overrides_apply_when_cli_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Zmienne środowiskowe powinny zastępować moduły, gdy brak flag CLI."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        pipeline_mod = types.ModuleType("tests.env_pipeline")
        realtime_mod = types.ModuleType("tests.env_realtime")

        def _pipeline(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("env_pipeline", args, dict(kwargs))

        def _controller(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("env_controller", args, dict(kwargs))

        class EnvRunner:
            pass

        pipeline_mod.build_daily_trend_pipeline = _pipeline  # type: ignore[attr-defined]
        pipeline_mod.create_trading_controller = _controller  # type: ignore[attr-defined]
        realtime_mod.DailyTrendRealtimeRunner = EnvRunner  # type: ignore[attr-defined]

        sys.modules[pipeline_mod.__name__] = pipeline_mod
        sys.modules[realtime_mod.__name__] = realtime_mod

        monkeypatch.setenv("RUN_DAILY_TREND_PIPELINE_MODULES", pipeline_mod.__name__)
        monkeypatch.setenv("RUN_DAILY_TREND_REALTIME_MODULES", realtime_mod.__name__)

        exit_code = module.main(["--print-runtime-modules"])
        assert exit_code == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["pipeline_modules"][0] == pipeline_mod.__name__
        assert payload["realtime_modules"][0] == realtime_mod.__name__
        assert payload["pipeline_resolved_from"] == pipeline_mod.__name__
        assert payload["realtime_resolved_from"] == realtime_mod.__name__
        assert (
            payload["pipeline"]["origin"]
            == "zmienna środowiskowa RUN_DAILY_TREND_PIPELINE_MODULES"
        )
        assert (
            payload["realtime"]["origin"]
            == "zmienna środowiskowa RUN_DAILY_TREND_REALTIME_MODULES"
        )
        assert payload["pipeline"]["resolved_from"] == pipeline_mod.__name__
        assert payload["realtime"]["resolved_from"] == realtime_mod.__name__

        assert module.build_daily_trend_pipeline is _pipeline
        assert module.create_trading_controller is _controller
        assert module.DailyTrendRealtimeRunner is EnvRunner
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        monkeypatch.delenv("RUN_DAILY_TREND_REALTIME_MODULES", raising=False)
        sys.modules.pop("tests.env_pipeline", None)
        sys.modules.pop("tests.env_realtime", None)
        _reset_run_daily_trend(module)


def test_environment_override_blank_value_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pusta wartość zmiennej środowiskowej powinna być zignorowana z ostrzeżeniem."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        env_name = "RUN_DAILY_TREND_PIPELINE_MODULES"
        caplog.set_level(logging.WARNING, logger="scripts.run_daily_trend")
        monkeypatch.setenv(env_name, "   ")

        value = module._modules_from_environment(env_name)
        assert value is None
        assert "nie zawiera żadnych modułów" in caplog.text
        assert env_name in caplog.text
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        _reset_run_daily_trend(module)


def test_environment_override_invalid_tokens_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nieprawidłowe tokeny w zmiennej środowiskowej również powinny być raportowane."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        env_name = "RUN_DAILY_TREND_REALTIME_MODULES"
        caplog.set_level(logging.WARNING, logger="scripts.run_daily_trend")
        monkeypatch.setenv(env_name, " , ;; :: ")

        value = module._modules_from_environment(env_name)
        assert value is None
        assert "nie zawiera poprawnych nazw modułów" in caplog.text
        assert env_name in caplog.text
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_REALTIME_MODULES", raising=False)
        _reset_run_daily_trend(module)


def test_collect_git_metadata_without_git(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Brak binarki git powinien zwrócić None bez wyjątku."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        caplog.set_level(logging.DEBUG, logger="scripts.run_daily_trend")
        monkeypatch.setattr(module.shutil, "which", lambda name: None)

        result = module._collect_git_metadata()

        assert result is None
        assert "Polecenie 'git' nie jest dostępne" in caplog.text
    finally:
        _reset_run_daily_trend(module)


def test_environment_overrides_ignored_when_cli_present(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flagi CLI mają pierwszeństwo przed zmiennymi środowiskowymi."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        cli_pipeline_mod = types.ModuleType("tests.cli_pipeline")
        cli_realtime_mod = types.ModuleType("tests.cli_realtime")
        env_pipeline_mod = types.ModuleType("tests.env_pipeline_override")
        env_realtime_mod = types.ModuleType("tests.env_realtime_override")

        def _cli_pipeline(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("cli_pipeline", args, dict(kwargs))

        def _cli_controller(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("cli_controller", args, dict(kwargs))

        class CliRunner:
            pass

        def _env_pipeline(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("env_pipeline", args, dict(kwargs))

        def _env_controller(*args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
            return ("env_controller", args, dict(kwargs))

        class EnvRunner:
            pass

        cli_pipeline_mod.build_daily_trend_pipeline = _cli_pipeline  # type: ignore[attr-defined]
        cli_pipeline_mod.create_trading_controller = _cli_controller  # type: ignore[attr-defined]
        cli_realtime_mod.DailyTrendRealtimeRunner = CliRunner  # type: ignore[attr-defined]

        env_pipeline_mod.build_daily_trend_pipeline = _env_pipeline  # type: ignore[attr-defined]
        env_pipeline_mod.create_trading_controller = _env_controller  # type: ignore[attr-defined]
        env_realtime_mod.DailyTrendRealtimeRunner = EnvRunner  # type: ignore[attr-defined]

        sys.modules[cli_pipeline_mod.__name__] = cli_pipeline_mod
        sys.modules[cli_realtime_mod.__name__] = cli_realtime_mod
        sys.modules[env_pipeline_mod.__name__] = env_pipeline_mod
        sys.modules[env_realtime_mod.__name__] = env_realtime_mod

        monkeypatch.setenv("RUN_DAILY_TREND_PIPELINE_MODULES", env_pipeline_mod.__name__)
        monkeypatch.setenv("RUN_DAILY_TREND_REALTIME_MODULES", env_realtime_mod.__name__)

        caplog.set_level(logging.INFO, logger="scripts.run_daily_trend")

        exit_code = module.main(
            [
                "--pipeline-module",
                cli_pipeline_mod.__name__,
                "--realtime-module",
                cli_realtime_mod.__name__,
                "--print-runtime-modules",
            ]
        )
        assert exit_code == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["pipeline_modules"][0] == cli_pipeline_mod.__name__
        assert payload["realtime_modules"][0] == cli_realtime_mod.__name__
        assert payload["pipeline_resolved_from"] == cli_pipeline_mod.__name__
        assert payload["realtime_resolved_from"] == cli_realtime_mod.__name__
        assert payload["pipeline"]["origin"] == "flagi CLI (--pipeline-module)"
        assert payload["realtime"]["origin"] == "flagi CLI (--realtime-module)"
        assert payload["pipeline"]["resolved_from"] == cli_pipeline_mod.__name__
        assert payload["realtime"]["resolved_from"] == cli_realtime_mod.__name__

        assert module.build_daily_trend_pipeline is _cli_pipeline
        assert module.create_trading_controller is _cli_controller
        assert module.DailyTrendRealtimeRunner is CliRunner

        log_text = caplog.text
        assert "Pominięto moduły pipeline" in log_text
        assert "Pominięto moduły realtime" in log_text
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        monkeypatch.delenv("RUN_DAILY_TREND_REALTIME_MODULES", raising=False)
        for name in (
            "tests.cli_pipeline",
            "tests.cli_realtime",
            "tests.env_pipeline_override",
            "tests.env_realtime_override",
        ):
            sys.modules.pop(name, None)
        _reset_run_daily_trend(module)


def test_main_logs_runtime_override_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ImportError przy override modułów powinien być raportowany i zakończyć się kodem 2."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        pipeline_module = importlib.import_module("bot_core.runtime.pipeline")
        runtime_module = importlib.import_module("bot_core.runtime")
        realtime_module = importlib.import_module("bot_core.runtime.realtime")

        monkeypatch.delattr(pipeline_module, "build_daily_trend_pipeline", raising=False)
        monkeypatch.delattr(pipeline_module, "create_trading_controller", raising=False)
        monkeypatch.delattr(runtime_module, "build_daily_trend_pipeline", raising=False)
        monkeypatch.delattr(runtime_module, "create_trading_controller", raising=False)

        caplog.set_level(logging.ERROR, logger="scripts.run_daily_trend")
        exit_code = module.main(
            [
                "--pipeline-module",
                pipeline_module.__name__,
                "--realtime-module",
                realtime_module.__name__,
            ]
        )

        assert exit_code == 2
        assert "Nie można załadować modułów runtime" in caplog.text
        assert "flagi CLI (--pipeline-module)" in caplog.text
    finally:
        _reset_run_daily_trend(module)


def test_main_logs_environment_override_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Błąd importu z poziomu zmiennych środowiskowych powinien mieć kontekst."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        pipeline_module = importlib.import_module("bot_core.runtime.pipeline")
        runtime_module = importlib.import_module("bot_core.runtime")

        monkeypatch.delattr(pipeline_module, "build_daily_trend_pipeline", raising=False)
        monkeypatch.delattr(pipeline_module, "create_trading_controller", raising=False)
        monkeypatch.delattr(runtime_module, "build_daily_trend_pipeline", raising=False)
        monkeypatch.delattr(runtime_module, "create_trading_controller", raising=False)

        monkeypatch.setenv("RUN_DAILY_TREND_PIPELINE_MODULES", "tests.missing_pipeline")
        caplog.set_level(logging.ERROR, logger="scripts.run_daily_trend")

        exit_code = module.main(["--print-runtime-modules"])

        assert exit_code == 2
        assert "Nie można załadować modułów runtime" in caplog.text
        assert "zmienna środowiskowa RUN_DAILY_TREND_PIPELINE_MODULES" in caplog.text
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        _reset_run_daily_trend(module)


def test_runtime_plan_jsonl_writes_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Snapshot runtime powinien zapisać wpis JSONL z profilami ryzyka."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        runtime_plan_path = tmp_path / "runtime_plan.jsonl"

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {
                "root": "/repo",
                "commit": "deadbeef",
                "branch": "feature",
                "is_dirty": False,
            },
        )

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
                "--runtime-plan-jsonl",
                str(runtime_plan_path),
            ]
        )
        assert exit_code == 0

        contents = runtime_plan_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(contents) == 1
        payload = json.loads(contents[0])
        assert payload["environment"] == "binance_paper"
        assert payload["runtime_modules"]["pipeline_modules"][0] == "bot_core.runtime.pipeline"
        assert payload["runtime_modules"]["pipeline"]["fallback_used"] is False
        assert payload["runtime_modules"]["realtime"]["fallback_used"] is False
        assert payload["risk_profile"] == "conservative"
        risk_details = payload["risk_profile_details"]
        assert isinstance(risk_details, dict)
        assert risk_details["classification"] == "conservative"
        assert "demo→paper→live" in risk_details["deployment_pipeline"]
        controller_section = payload.get("controller_details")
        assert controller_section and controller_section["name"] == "stub_controller"
        assert controller_section["symbol_count"] == 2
        assert controller_section["interval"] == "1d"
        assert controller_section["tick_seconds"] == pytest.approx(86400.0)
        strategy_section = payload.get("strategy_details")
        assert strategy_section and strategy_section["name"] == "stub_strategy"
        assert strategy_section["class"].endswith("DailyTrendMomentumStrategy")
        assert "settings" in strategy_section
        environment_section = payload.get("environment_details")
        assert environment_section and environment_section["name"] == "binance_paper"
        assert environment_section["exchange"] == "binance"
        pipeline_details = payload.get("pipeline_details")
        assert pipeline_details == {"class": "SimpleNamespace", "module": "types"}
        config_section = payload.get("config_file")
        assert config_section
        assert config_section["path"] == str(config_path)
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert config_section["sha256"] == expected_hash
        assert config_section["size_bytes"] == config_path.stat().st_size
        assert config_section["exists"] is True
        assert "modified_time" in config_section
        assert Path(config_section["absolute_path"]).is_absolute()
        assert config_section["parent_directory"] == str(config_path.parent)
        assert config_section["parent_exists"] is True
        assert config_section["parent_is_dir"] is True
        assert config_section["parent_writable"] in {True, False}
        assert Path(config_section["parent_absolute_path"]).is_absolute()
        metrics_section = payload.get("metrics_service_details")
        assert metrics_section
        assert metrics_section["configured"] is True
        assert metrics_section["enabled"] is True
        assert metrics_section["ui_alert_sink_available"] is True
        assert metrics_section["ui_alerts_source"] == "config"
        expected_jsonl = config_path.parent / "logs/metrics/telemetry.jsonl"
        assert metrics_section["jsonl_path"] == str(expected_jsonl)
        jsonl_file = metrics_section["jsonl_file"]
        assert jsonl_file["path"] == str(expected_jsonl)
        assert Path(jsonl_file["absolute_path"]).is_absolute()
        assert jsonl_file["parent_directory"] == str(expected_jsonl.parent)
        assert jsonl_file["parent_exists"] is False
        assert jsonl_file["parent_is_dir"] is False
        assert jsonl_file["parent_writable"] is False
        assert jsonl_file.get("role") == "jsonl"
        assert isinstance(jsonl_file.get("security_warnings"), list)
        if jsonl_file.get("exists"):
            assert "security_flags" in jsonl_file
        else:
            assert "security_flags" not in jsonl_file
        expected_ui_alerts = config_path.parent / "logs/metrics/ui_alerts.jsonl"
        assert metrics_section["ui_alerts_jsonl_path"] == str(expected_ui_alerts)
        ui_alerts_file = metrics_section["ui_alerts_file"]
        assert ui_alerts_file["path"] == str(expected_ui_alerts)
        assert Path(ui_alerts_file["absolute_path"]).is_absolute()
        assert ui_alerts_file["parent_directory"] == str(expected_ui_alerts.parent)
        assert ui_alerts_file["parent_exists"] is False
        assert ui_alerts_file["parent_is_dir"] is False
        assert ui_alerts_file["parent_writable"] is False
        assert ui_alerts_file.get("role") == "ui_alerts_jsonl"
        assert isinstance(ui_alerts_file.get("security_warnings"), list)
        if ui_alerts_file.get("exists"):
            assert "security_flags" in ui_alerts_file
        else:
            assert "security_flags" not in ui_alerts_file
        assert metrics_section["jsonl_file"]["exists"] is False
        assert Path(metrics_section["jsonl_file"]["absolute_path"]).is_absolute()
        assert metrics_section["tls"]["configured"] is True
        assert metrics_section["tls"]["enabled"] is True
        assert metrics_section["tls"]["require_client_auth"] is True
        expected_cert = config_path.parent / "secrets/metrics/server.crt"
        expected_key = config_path.parent / "secrets/metrics/server.key"
        expected_ca = config_path.parent / "secrets/metrics/clients.pem"
        assert metrics_section["tls"]["certificate"]["path"] == str(expected_cert)
        assert metrics_section["tls"]["private_key"]["path"] == str(expected_key)
        assert metrics_section["tls"]["client_ca"]["path"] == str(expected_ca)
        assert metrics_section["tls"]["certificate"].get("role") == "tls_cert"
        assert metrics_section["tls"]["private_key"].get("role") == "tls_key"
        assert metrics_section["tls"]["client_ca"].get("role") == "tls_client_ca"
        runtime_state = metrics_section.get("runtime_state")
        assert runtime_state is not None
        assert runtime_state["service_enabled"] is True
        assert runtime_state["ui_alert_sink_active"] is True
        assert runtime_state["jsonl_path"] == str(expected_jsonl)
        runtime_jsonl_file = runtime_state["jsonl_file"]
        assert runtime_jsonl_file["path"] == str(expected_jsonl)
        assert runtime_jsonl_file["parent_directory"] == str(expected_jsonl.parent)
        assert runtime_jsonl_file.get("role") == "jsonl"
        assert runtime_state["ui_alerts_jsonl_path"] == str(expected_ui_alerts)
        runtime_ui_alert_file = runtime_state["ui_alerts_file"]
        assert runtime_ui_alert_file["path"] == str(expected_ui_alerts)
        assert runtime_ui_alert_file.get("role") == "ui_alerts_jsonl"
        if "security_warnings" in runtime_state:
            assert isinstance(runtime_state["security_warnings"], list)
        git_section = payload.get("git")
        assert git_section
        assert git_section["commit"] == "deadbeef"
        assert git_section["branch"] == "feature"
    finally:
        _reset_run_daily_trend(module)


def test_fail_on_security_warnings_blocks_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flaga bezpieczeństwa powinna przerwać działanie przy ostrzeżeniach."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {"root": "/repo", "commit": "deadbeef"},
        )

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
                "--fail-on-security-warnings",
            ]
        )

        assert exit_code == 3
        assert "run_daily_trend.runtime_plan" in caplog.text
    finally:
        _reset_run_daily_trend(module)


def test_env_fail_on_security_warnings_blocks_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zmienna środowiskowa powinna aktywować blokadę przy ostrzeżeniach."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {"root": "/repo", "commit": "deadbeef"},
        )

        monkeypatch.setenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", "true")
        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 3
        assert "run_daily_trend.runtime_plan" in caplog.text
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", raising=False)
        _reset_run_daily_trend(module)


def test_print_runtime_plan_outputs_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Flaga inspekcyjna powinna wypisać plan runtime i zakończyć wykonywanie."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {
                "root": "/repo",
                "commit": "cafebabe",
                "branch": "inspect",
                "is_dirty": False,
            },
        )

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--print-runtime-plan",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out.strip())
        assert payload["environment"] == "binance_paper"
        config_section = payload["config_file"]
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert config_section["sha256"] == expected_hash
        assert config_section["exists"] is True
        assert config_section["parent_directory"] == str(config_path.parent)
        assert config_section["parent_exists"] is True
        assert config_section["parent_is_dir"] is True
        assert Path(config_section["parent_absolute_path"]).is_absolute()
        assert payload["runtime_modules"]["pipeline"]["resolved_from"].startswith("bot_core.runtime")
        assert payload["runtime_modules"]["pipeline"]["fallback_used"] is False
        assert payload["runtime_modules"]["realtime"]["fallback_used"] is False
        metrics_section = payload["metrics_service_details"]
        assert metrics_section["configured"] is True
        assert metrics_section["tls"]["configured"] is True
        expected_jsonl = config_path.parent / "logs/metrics/telemetry.jsonl"
        assert metrics_section["jsonl_path"] == str(expected_jsonl)
        jsonl_file = metrics_section["jsonl_file"]
        assert jsonl_file["path"] == str(expected_jsonl)
        assert jsonl_file["parent_directory"] == str(expected_jsonl.parent)
        assert jsonl_file["parent_exists"] is False
        assert jsonl_file["parent_is_dir"] is False
        assert jsonl_file["parent_writable"] is False
        expected_ui_alerts = config_path.parent / "logs/metrics/ui_alerts.jsonl"
        assert metrics_section["ui_alerts_source"] == "config"
        assert metrics_section["ui_alerts_jsonl_path"] == str(expected_ui_alerts)
        ui_alerts_file = metrics_section["ui_alerts_file"]
        assert ui_alerts_file["path"] == str(expected_ui_alerts)
        assert ui_alerts_file["parent_directory"] == str(expected_ui_alerts.parent)
        assert ui_alerts_file["parent_exists"] is False
        assert ui_alerts_file["parent_is_dir"] is False
        assert ui_alerts_file["parent_writable"] is False
        assert metrics_section["ui_alert_sink_available"] is True
        assert metrics_section["tls"]["certificate"]["exists"] is False
        runtime_state = metrics_section["runtime_state"]
        assert runtime_state["service_enabled"] is True
        assert runtime_state["ui_alert_sink_active"] is True
        assert runtime_state["jsonl_path"] == str(expected_jsonl)
        assert runtime_state["ui_alerts_jsonl_path"] == str(expected_ui_alerts)
        if "security_warnings" in runtime_state:
            assert isinstance(runtime_state["security_warnings"], list)
        assert payload["risk_profile"] == "conservative"
        assert payload["controller_details"]["name"] == "stub_controller"
        assert payload["strategy_details"]["name"] == "stub_strategy"
        assert payload["environment_details"]["name"] == "binance_paper"
        git_section = payload.get("git")
        assert git_section and git_section["commit"] == "cafebabe"
        security_section = payload["security"]["fail_on_security_warnings"]
        assert security_section["enabled"] is False
        assert security_section["source"] == "default"
        assert security_section["parameter_source"] == "default"
        assert security_section.get("environment_raw_value") is None
        security_param_sources = payload["security"]["parameter_sources"]
        assert security_param_sources["fail_on_security_warnings"] == "default"
    finally:
        _reset_run_daily_trend(module)


def test_env_fail_on_security_warnings_metadata_in_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Snapshot planu zawiera źródło ustawienia flagi bezpieczeństwa."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {"root": "/repo", "commit": "deadbeef"},
        )

        monkeypatch.setenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", "1")

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--print-runtime-plan",
            ]
        )

        assert exit_code == 3
        captured = capsys.readouterr()
        payload = json.loads(captured.out.strip())
        security_section = payload["security"]["fail_on_security_warnings"]
        assert security_section["enabled"] is True
        assert security_section["source"] == "env:RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS"
        assert security_section["parameter_source"] == "env"
        assert security_section["environment_applied"] is True
        assert security_section["environment_raw_value"] == "1"
        security_param_sources = payload["security"]["parameter_sources"]
        assert security_param_sources["fail_on_security_warnings"] == "env"
        env_overrides = payload["overrides"]["environment"]["fail_on_security_warnings"]
        assert env_overrides["variable"] == "RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS"
        assert env_overrides["applied"] is True
        entries = payload["overrides"]["environment"]["entries"]
        fail_entry = next(
            entry
            for entry in entries
            if entry.get("option") == "fail_on_security_warnings"
        )
        assert fail_entry["applied"] is True
        assert fail_entry["parsed_value"] is True
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", raising=False)
        _reset_run_daily_trend(module)


def test_env_fail_on_security_warnings_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Niepoprawna wartość środowiskowa powinna kończyć działanie kodem 2."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", "maybe")

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 2
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS", raising=False)
        _reset_run_daily_trend(module)


def test_environment_module_override_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan runtime rejestruje szczegóły override'ów modułów z ENV."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)

        fake_cli_module = types.ModuleType("cli_pipeline_module")

        def _cli_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        fake_cli_module.build_daily_trend_pipeline = _cli_pipeline  # type: ignore[attr-defined]
        fake_cli_module.create_trading_controller = lambda *args, **kwargs: object()
        monkeypatch.setitem(sys.modules, "cli_pipeline_module", fake_cli_module)

        fake_env_realtime = types.ModuleType("env_realtime_module")
        fake_env_realtime.DailyTrendRealtimeRunner = type(  # type: ignore[attr-defined]
            "DummyRealtimeRunner",
            (),
            {},
        )
        monkeypatch.setitem(sys.modules, "env_realtime_module", fake_env_realtime)

        monkeypatch.setenv("RUN_DAILY_TREND_PIPELINE_MODULES", "env.pipeline.module")
        monkeypatch.setenv("RUN_DAILY_TREND_REALTIME_MODULES", "env_realtime_module")

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--pipeline-module",
                "cli_pipeline_module",
                "--print-runtime-plan",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out.strip())
        env_section = payload["overrides"]["environment"]
        entries = env_section["entries"]
        pipeline_entry = next(entry for entry in entries if entry["option"] == "pipeline_modules")
        assert pipeline_entry["raw_value"] == "env.pipeline.module"
        assert pipeline_entry["parsed_value"] == ["env.pipeline.module"]
        assert pipeline_entry["applied"] is False
        assert pipeline_entry["reason"] == "cli_override"

        realtime_entry = next(entry for entry in entries if entry["option"] == "realtime_modules")
        assert realtime_entry["parsed_value"] == ["env_realtime_module"]
        assert realtime_entry["applied"] is True
        assert env_section["realtime_modules"] == ["env_realtime_module"]
        assert "pipeline_modules" not in env_section
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        monkeypatch.delenv("RUN_DAILY_TREND_REALTIME_MODULES", raising=False)
        _reset_run_daily_trend(module)


def test_environment_module_invalid_value_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Niepoprawna wartość modułu środowiskowego trafia do wpisu audytowego."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)

        monkeypatch.setenv("RUN_DAILY_TREND_PIPELINE_MODULES", "   ")

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--print-runtime-plan",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out.strip())
        env_section = payload["overrides"]["environment"]
        entries = env_section["entries"]
        pipeline_entry = next(entry for entry in entries if entry["option"] == "pipeline_modules")
        assert pipeline_entry["raw_value"] == "   "
        assert pipeline_entry["applied"] is False
        assert pipeline_entry["reason"] == "empty_value"
    finally:
        monkeypatch.delenv("RUN_DAILY_TREND_PIPELINE_MODULES", raising=False)
        _reset_run_daily_trend(module)

def test_metrics_details_use_default_ui_alert_path_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Jeśli ścieżka UI alerts nie jest skonfigurowana, raportujemy domyślną."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)
        filtered_lines = [
            line
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if "ui_alerts_jsonl_path" not in line
        ]
        config_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--print-runtime-plan",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out.strip())
        metrics_section = payload["metrics_service_details"]
        assert metrics_section["configured"] is True
        assert metrics_section["ui_alerts_source"] == "default"
        expected_relative = str(module.DEFAULT_UI_ALERTS_JSONL_PATH.expanduser())
        expected_absolute = str(
            Path(expected_relative).expanduser().resolve(strict=False)
        )
        assert metrics_section["ui_alerts_jsonl_path"] == expected_relative
        ui_alerts_file = metrics_section["ui_alerts_file"]
        assert ui_alerts_file["path"] == expected_relative
        assert ui_alerts_file["absolute_path"] == expected_absolute
        assert ui_alerts_file["parent_directory"] == str(Path(expected_relative).parent)
        assert Path(ui_alerts_file["parent_absolute_path"]).is_absolute()
        assert isinstance(ui_alerts_file["parent_exists"], bool)
        assert isinstance(ui_alerts_file["parent_is_dir"], bool)
        assert isinstance(ui_alerts_file["parent_writable"], bool)
        assert metrics_section["ui_alert_sink_available"] is True
        runtime_state = metrics_section["runtime_state"]
        assert runtime_state["ui_alert_sink_active"] is True
        assert runtime_state["ui_alerts_jsonl_path"] == expected_relative
        runtime_ui_alert_file = runtime_state["ui_alerts_file"]
        assert runtime_ui_alert_file["absolute_path"] == expected_absolute
        assert runtime_state["jsonl_path"] == metrics_section["jsonl_path"]
        if "security_warnings" in runtime_state:
            assert isinstance(runtime_state["security_warnings"], list)
    finally:
        _reset_run_daily_trend(module)


def test_runtime_plan_jsonl_includes_precheck_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wpis planu runtime zawiera streszczenie paper_precheck wraz z metadanymi audytu."""

    from bot_core.config.loader import load_core_config

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        runtime_plan_path = tmp_path / "runtime_plan.jsonl"

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        precheck_payload = {
            "status": "ok",
            "coverage_status": "ok",
            "risk_status": "ok",
            "coverage_warnings": [],
        }
        precheck_metadata = {
            "path": tmp_path / "audit" / "report.json",
            "sha256": "abc123",
            "created_at": "2024-01-01T00:00:00Z",
            "environment": "binance_paper",
            "status": "ok",
            "size_bytes": 2048,
            "ignored_field": "value",
        }

        def _fake_precheck(**kwargs: object):
            return precheck_payload, 0, precheck_metadata

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(module, "_run_paper_precheck_for_smoke", _fake_precheck)
        monkeypatch.setattr(
            module,
            "_collect_git_metadata",
            lambda base_path=None: {
                "root": "/repo",
                "commit": "feedface",
                "branch": "main",
                "is_dirty": True,
                "dirty_entries": 2,
            },
        )

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--paper-smoke",
                "--paper-smoke-operator",
                "QA Agent",
                "--dry-run",
                "--runtime-plan-jsonl",
                str(runtime_plan_path),
            ]
        )
        assert exit_code == 0

        contents = runtime_plan_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(contents) == 1
        payload = json.loads(contents[0])
        assert payload["paper_precheck"]["status"] == "ok"
        audit_section = payload.get("paper_precheck_audit")
        assert audit_section
        assert audit_section["sha256"] == "abc123"
        assert audit_section["status"] == "ok"
        assert audit_section["environment"] == "binance_paper"
        assert audit_section["size_bytes"] == 2048
        assert audit_section["path"].endswith("report.json")
        assert "ignored_field" not in audit_section
        git_section = payload.get("git")
        assert git_section
        assert git_section["commit"] == "feedface"
        assert git_section["dirty_entries"] == 2
    finally:
        _reset_run_daily_trend(module)


def test_runtime_plan_jsonl_write_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Błąd zapisu planu runtime powinien zakończyć działanie kodem 2."""

    from bot_core.config.loader import load_core_config

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            def load_exchange_credentials(self, *args: object, **kwargs: object) -> object:
                return object()

        def _fake_secret_manager(args: argparse.Namespace) -> DummySecretManager:
            return DummySecretManager()

        def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
            return _make_pipeline_stub(
                kwargs["config_path"],
                environment_name=kwargs["environment_name"],
                risk_profile=kwargs.get("risk_profile_name"),
            )

        monkeypatch.setattr(module, "_create_secret_manager", _fake_secret_manager)
        monkeypatch.setattr(module, "build_daily_trend_pipeline", _fake_pipeline)
        monkeypatch.setattr(module, "_collect_git_metadata", lambda base_path=None: None)

        def _fail_append(path: Path, payload: Mapping[str, object]) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr(module, "_append_runtime_plan_jsonl", _fail_append)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
                "--runtime-plan-jsonl",
                str(tmp_path / "runtime_plan.jsonl"),
            ]
        )

        assert exit_code == 2
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_missing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI powinien zgłosić błąd, gdy środowisko nie istnieje w konfiguracji."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            pass

        monkeypatch.setattr(module, "_create_secret_manager", lambda args: DummySecretManager())

        pipeline_called = False

        def _unexpected_pipeline(**kwargs: object) -> None:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("pipeline should not be constructed when environment is invalid")

        monkeypatch.setattr(module, "build_daily_trend_pipeline", _unexpected_pipeline)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "missing_env",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_called is False
        assert any("Środowisko missing_env" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_unknown_risk_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI powinien sygnalizować brakujący profil ryzyka jeszcze przed bootstrapem pipeline'u."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        class DummySecretManager:
            pass

        monkeypatch.setattr(module, "_create_secret_manager", lambda args: DummySecretManager())

        pipeline_called = False

        def _unexpected_pipeline(**kwargs: object) -> None:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("pipeline should not be constructed when risk profile is invalid")

        monkeypatch.setattr(module, "build_daily_trend_pipeline", _unexpected_pipeline)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--risk-profile",
                "aggressive",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_called is False
        assert any("Profil ryzyka aggressive" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def _prepare_default_secret_manager(module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySecretManager:
        pass

    monkeypatch.setattr(module, "_create_secret_manager", lambda args: DummySecretManager())


def _assert_pipeline_not_called(module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    pipeline_state = SimpleNamespace(called=False)

    def _unexpected_pipeline(**kwargs: object) -> None:
        pipeline_state.called = True
        raise AssertionError("pipeline should not be constructed for invalid configuration")

    monkeypatch.setattr(module, "build_daily_trend_pipeline", _unexpected_pipeline)
    return pipeline_state


def _modify_config(path: Path, *, replace: Mapping[str, str] | None = None, remove: Sequence[str] | None = None) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    replace_map = {needle.strip(): replacement.strip() for needle, replacement in (replace or {}).items()}
    remove_set = {item.strip() for item in (remove or [])}

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped in remove_set:
            continue
        if stripped and stripped in replace_map:
            indent = line[: len(line) - len(line.lstrip())]
            new_value = replace_map[stripped]
            if new_value:
                new_lines.append(f"{indent}{new_value}")
            continue
        new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def test_main_requires_default_strategy_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Środowisko bez domyślnej strategii powinno zakończyć się błędem przed bootstrapem."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)
        _modify_config(
            config_path,
            remove=["                default_strategy: core_daily_trend_stub\n"],
        )

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("domyślnej strategii" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_unknown_environment_default_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Błędne wskazanie strategii w środowisku powinno przerwać wykonywanie."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)
        _modify_config(
            config_path,
            replace={
                "                default_strategy: core_daily_trend_stub\n": "                default_strategy: missing_strategy\n",
            },
        )

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("odwołuje się do strategii" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_unknown_cli_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Podanie nieistniejącej strategii w CLI musi zostać zablokowane."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--strategy",
                "missing_strategy",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("Strategia missing_strategy" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_requires_default_controller_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Brak domyślnego kontrolera powinien kończyć się kodem błędu."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)
        _modify_config(
            config_path,
            remove=["                default_controller: daily_trend_stub\n"],
        )

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("domyślnego kontrolera runtime" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_unknown_environment_default_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Błędny kontroler z konfiguracji środowiska powinien zostać odrzucony."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)
        _modify_config(
            config_path,
            replace={
                "                default_controller: daily_trend_stub\n": "                default_controller: missing_controller\n",
            },
        )

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("kontrolera runtime missing_controller" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)


def test_main_reports_unknown_cli_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nieistniejący kontroler przekazany w CLI musi zostać odrzucony."""

    module = _import_run_daily_trend(monkeypatch)
    try:
        config_path = tmp_path / "core.yaml"
        _write_minimal_config(config_path)

        _prepare_default_secret_manager(module, monkeypatch)
        pipeline_state = _assert_pipeline_not_called(module, monkeypatch)

        caplog.set_level(logging.ERROR)

        exit_code = module.main(
            [
                "--config",
                str(config_path),
                "--environment",
                "binance_paper",
                "--controller",
                "missing_controller",
                "--dry-run",
            ]
        )

        assert exit_code == 2
        assert pipeline_state.called is False
        assert any("Kontroler runtime missing_controller" in record.getMessage() for record in caplog.records)
    finally:
        _reset_run_daily_trend(module)
