from types import SimpleNamespace

from bot_core.config.models import RuntimeExecutionSettings
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.runtime import pipeline as pipeline_module
from bot_core.runtime.execution_bootstrapper import ExecutionBootstrapper


def _paper_settings(tmp_path):
    return {
        "initial_balances": {"USDT": 10_000.0},
        "maker_fee": 0.0004,
        "taker_fee": 0.0006,
        "slippage_bps": 5.0,
        "ledger_directory": tmp_path,
        "ledger_filename_pattern": "ledger-%Y%m%d.jsonl",
        "ledger_retention_days": 730,
        "ledger_fsync": False,
    }


def _markets():
    return {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            step_size=0.001,
        )
    }


def test_execution_bootstrapper_reuses_paper_service(tmp_path) -> None:
    bootstrapper = ExecutionBootstrapper()
    markets = _markets()
    existing = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 5_000.0},
        ledger_directory=tmp_path,
    )
    context = SimpleNamespace(execution_service=existing)

    resolved = bootstrapper.bootstrap_execution_service(
        bootstrap_ctx=context,
        markets=markets,
        paper_settings=_paper_settings(tmp_path),
        runtime_settings=None,
        execution_mode="paper",
        price_resolver=None,
    )

    assert resolved is existing


def test_execution_bootstrapper_builds_new_paper_service_when_context_is_not_reusable(tmp_path) -> None:
    bootstrapper = ExecutionBootstrapper()
    context = SimpleNamespace(execution_service=object())

    resolved = bootstrapper.bootstrap_execution_service(
        bootstrap_ctx=context,
        markets=_markets(),
        paper_settings=_paper_settings(tmp_path),
        runtime_settings=None,
        execution_mode="paper",
        price_resolver=None,
    )

    assert isinstance(resolved, PaperTradingExecutionService)
    assert context.execution_service is resolved


def test_execution_bootstrapper_builds_and_persists_live_service(tmp_path, monkeypatch) -> None:
    bootstrapper = ExecutionBootstrapper()
    built = object()

    def _build_live(*, bootstrap_ctx, environment, runtime_settings):
        assert bootstrap_ctx.environment is environment
        assert isinstance(runtime_settings, RuntimeExecutionSettings)
        return built

    monkeypatch.setattr("bot_core.runtime.execution_bootstrapper.build_live_execution_service", _build_live)

    context = SimpleNamespace(execution_service=None, environment=SimpleNamespace())
    resolved = bootstrapper.bootstrap_execution_service(
        bootstrap_ctx=context,
        markets=_markets(),
        paper_settings=_paper_settings(tmp_path),
        runtime_settings=None,
        execution_mode="live",
        price_resolver=None,
    )

    assert resolved is built
    assert context.execution_service is built


def test_execution_bootstrapper_swallow_persist_errors(tmp_path, monkeypatch, caplog) -> None:
    bootstrapper = ExecutionBootstrapper()

    class ReadOnlyContext:
        def __init__(self):
            self.environment = SimpleNamespace()
            self._execution_service = None

        @property
        def execution_service(self):
            return self._execution_service

        @execution_service.setter
        def execution_service(self, _value):
            raise RuntimeError("read-only")

    built = object()

    def _build_live(*, bootstrap_ctx, environment, runtime_settings):
        return built

    monkeypatch.setattr("bot_core.runtime.execution_bootstrapper.build_live_execution_service", _build_live)

    with caplog.at_level("DEBUG", logger="bot_core.runtime.execution_bootstrapper"):
        resolved = bootstrapper.bootstrap_execution_service(
            bootstrap_ctx=ReadOnlyContext(),
            markets=_markets(),
            paper_settings=_paper_settings(tmp_path),
            runtime_settings=None,
            execution_mode="live",
            price_resolver=None,
        )

    assert resolved is built
    assert any("LiveExecutionRouter" in message for message in caplog.messages)


def test_pipeline_execution_helpers_preserve_contract_by_delegation(tmp_path, monkeypatch) -> None:
    class StubBootstrapper:
        def __init__(self):
            self.paper_args = None
            self.select_args = None

        def build_paper_execution_service(self, markets, paper_settings, *, price_resolver=None):
            self.paper_args = (markets, paper_settings, price_resolver)
            return "paper-service"

        def bootstrap_execution_service(
            self,
            *,
            bootstrap_ctx,
            markets,
            paper_settings,
            runtime_settings,
            execution_mode,
            price_resolver=None,
        ):
            self.select_args = {
                "bootstrap_ctx": bootstrap_ctx,
                "markets": markets,
                "paper_settings": paper_settings,
                "runtime_settings": runtime_settings,
                "execution_mode": execution_mode,
                "price_resolver": price_resolver,
            }
            return "selected-service"

    stub = StubBootstrapper()
    monkeypatch.setattr(pipeline_module, "_EXECUTION_BOOTSTRAPPER", stub)

    markets = _markets()
    paper_settings = _paper_settings(tmp_path)
    price_resolver = lambda _symbol: 1.0
    bootstrap_ctx = SimpleNamespace(execution_service=None)

    built = pipeline_module._build_execution_service(
        markets,
        paper_settings,
        price_resolver=price_resolver,
    )
    selected = pipeline_module._select_execution_service(
        bootstrap_ctx=bootstrap_ctx,
        markets=markets,
        paper_settings=paper_settings,
        runtime_settings=None,
        execution_mode="paper",
        price_resolver=price_resolver,
    )

    assert built == "paper-service"
    assert selected == "selected-service"
    assert stub.paper_args == (markets, paper_settings, price_resolver)
    assert stub.select_args == {
        "bootstrap_ctx": bootstrap_ctx,
        "markets": markets,
        "paper_settings": paper_settings,
        "runtime_settings": None,
        "execution_mode": "paper",
        "price_resolver": price_resolver,
    }
