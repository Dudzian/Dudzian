from __future__ import annotations

from pathlib import Path

from bot_core.optimization import OptimizationScheduler, StrategyOptimizer
from bot_core.config.models import (
    RuntimeOptimizationSettings,
    StrategyDefinitionConfig,
    StrategyOptimizationSearchSpaceConfig,
    StrategyOptimizationTaskConfig,
)
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal
from bot_core.strategies.catalog import (
    StrategyCatalog,
    StrategyDefinition,
    StrategyEngineSpec,
)
from bot_core.runtime.pipeline import _configure_optimization_scheduler


class DummyEngine(StrategyEngine):
    def __init__(self, *, parameters: dict[str, float], metadata: dict[str, object]) -> None:
        self.parameters = parameters
        self.metadata = metadata
        self.history: list[MarketSnapshot] = []

    def warm_up(self, history: tuple[MarketSnapshot, ...] | list[MarketSnapshot]) -> None:
        self.history = list(history)

    def on_data(self, snapshot: MarketSnapshot) -> tuple[StrategySignal, ...]:
        confidence = float(self.parameters.get("alpha", 0.0))
        return (
            StrategySignal(symbol=snapshot.symbol, side="BUY", confidence=confidence),
        )


def _build_catalog() -> tuple[StrategyCatalog, StrategyDefinition]:
    catalog = StrategyCatalog()

    def _factory(*, name: str, parameters: dict[str, float], metadata: dict[str, object]) -> StrategyEngine:
        return DummyEngine(parameters=parameters, metadata=metadata)

    spec = StrategyEngineSpec(
        key="dummy",
        factory=_factory,
        license_tier="basic",
        risk_classes=("a",),
        required_data=("ohlcv",),
    )
    catalog.register(spec)
    definition = StrategyDefinition(
        name="alpha",
        engine="dummy",
        license_tier="basic",
        risk_classes=("a",),
        required_data=("ohlcv",),
        parameters={"alpha": 0.0},
    )
    return catalog, definition


def test_grid_search_returns_best_candidate() -> None:
    catalog, definition = _build_catalog()
    optimizer = StrategyOptimizer(catalog)

    def evaluator(engine: StrategyEngine, params: dict[str, float]):
        score = -abs(params["alpha"] - 2.0)
        return score, {"alpha": params["alpha"]}

    report = optimizer.optimize(
        base_definition=definition,
        algorithm="grid",
        objective="utility",
        goal="maximize",
        search_grid={"alpha": [0.0, 1.0, 2.0, 3.0]},
        search_bounds={},
        max_trials=10,
        evaluator=evaluator,
    )

    assert report.best.parameters["alpha"] == 2.0
    assert len(report.trials) >= 4


def test_bayesian_optimizer_improves_score() -> None:
    catalog, definition = _build_catalog()
    optimizer = StrategyOptimizer(catalog)

    def evaluator(engine: StrategyEngine, params: dict[str, float]):
        score = (params["alpha"] - 1.0) ** 2
        return score, {}

    report = optimizer.optimize(
        base_definition=definition,
        algorithm="bayesian",
        objective="mse",
        goal="minimize",
        search_grid={},
        search_bounds={"alpha": (0.0, 4.0, None)},
        max_trials=15,
        evaluator=evaluator,
        random_seed=42,
    )

    assert abs(report.best.parameters["alpha"] - 1.0) < 0.6


def test_scheduler_runs_task_and_exports_reports(tmp_path: Path) -> None:
    catalog, definition = _build_catalog()
    optimizer = StrategyOptimizer(catalog)

    def evaluator(engine: StrategyEngine, params: dict[str, float]):
        score = -abs(params["alpha"] - 3.0)
        return score, {}

    search_space = StrategyOptimizationSearchSpaceConfig(grid={"alpha": [1.0, 2.0, 3.0]})
    task_config = StrategyOptimizationTaskConfig(
        name="alpha-task",
        strategy="alpha",
        algorithm="grid",
        max_trials=3,
        search_space=search_space,
    )

    scheduler = OptimizationScheduler(optimizer, report_directory=tmp_path)
    scheduler.add_task(
        config=task_config,
        definition=definition,
        evaluator=evaluator,
        default_algorithm="grid",
    )

    reports = scheduler.trigger()
    scheduler.stop()

    assert reports
    assert reports[0].best.parameters["alpha"] == 3.0
    exported = list(tmp_path.glob("*.html"))
    assert exported, "brak wyeksportowanego raportu HTML"


def test_pipeline_configures_optimization_scheduler(tmp_path: Path) -> None:
    catalog, definition = _build_catalog()
    strategy_config = StrategyDefinitionConfig(
        name="alpha",
        engine="dummy",
        parameters={"alpha": 0.0},
        license_tier="basic",
        risk_classes=("a",),
        required_data=(),
    )
    core_config = type("CoreStub", (), {"strategy_definitions": {"alpha": strategy_config}})()
    optimization_cfg = RuntimeOptimizationSettings(
        enabled=True,
        default_algorithm="grid",
        report_directory=tmp_path,
        tasks=(
            StrategyOptimizationTaskConfig(
                name="alpha-task",
                strategy="alpha",
                algorithm="grid",
                max_trials=3,
                search_space=StrategyOptimizationSearchSpaceConfig(grid={"alpha": [0.0, 1.0]}),
            ),
        ),
    )

    class StubFeed:
        def load_history(self, strategy_name: str, bars: int) -> tuple[MarketSnapshot, ...]:
            snapshot = MarketSnapshot(
                symbol="TEST/USDT",
                timestamp=0,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0,
            )
            return tuple(snapshot for _ in range(max(1, bars)))

    class StubRuntime:
        def __init__(self) -> None:
            self.data_feed = StubFeed()
            self.optimization_scheduler = None

    runtime = StubRuntime()
    scheduler = _configure_optimization_scheduler(
        runtime,
        core_config=core_config,
        optimization_cfg=optimization_cfg,
        catalog=catalog,
    )

    assert scheduler is not None
    reports = runtime.optimization_scheduler.trigger()
    runtime.optimization_scheduler.stop()
    assert reports
