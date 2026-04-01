from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot_core.runtime.strategy_bootstrapper import StrategyBootstrapper
from bot_core.strategies.catalog import StrategyDefinition


class _CatalogStub:
    def __init__(self) -> None:
        self._specs: dict[str, object] = {
            "daily_trend_momentum": SimpleNamespace(
                license_tier="pro",
                risk_classes=("trend",),
                required_data=("ohlcv",),
                risk_hooks=("hook_a",),
                default_tags=("default",),
                capability="cap_daily",
            ),
            "mean_reversion": SimpleNamespace(
                license_tier="basic",
                risk_classes=("mean",),
                required_data=("ohlcv",),
                risk_hooks=(),
                default_tags=("legacy",),
                capability=None,
            ),
        }
        self.created: list[StrategyDefinition] = []

    def get(self, engine: str) -> object:
        return self._specs[engine]

    def create(self, definition: StrategyDefinition) -> object:
        self.created.append(definition)
        return {"name": definition.name, "engine": definition.engine}


def _core_config_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        strategy_definitions={
            "main": SimpleNamespace(
                name="main",
                engine="daily_trend_momentum",
                license_tier=None,
                risk_classes=(),
                required_data=(),
                risk_hooks=(),
                parameters={"fast_ma": 5},
                risk_profile="balanced",
                metadata={"source": "config"},
                tags=("custom",),
                capability="cap_override",
            )
        },
        strategies={
            "legacy_daily": SimpleNamespace(
                fast_ma=3,
                slow_ma=8,
                breakout_lookback=12,
                momentum_window=5,
                atr_window=14,
                atr_multiplier=1.2,
                min_trend_strength=0.1,
                min_momentum=0.2,
            )
        },
        mean_reversion_strategies={
            "legacy_mean": SimpleNamespace(
                lookback=20,
                entry_zscore=2.0,
                exit_zscore=0.3,
                max_holding_period=12,
                volatility_cap=0.4,
                min_volume_usd=250_000,
            )
        },
        volatility_target_strategies={},
        cross_exchange_arbitrage_strategies={},
        scalping_strategies={},
        options_income_strategies={},
        statistical_arbitrage_strategies={},
        day_trading_strategies={},
    )


def test_collect_definitions_handles_config_and_legacy_fallbacks() -> None:
    catalog = _CatalogStub()
    bootstrapper = StrategyBootstrapper(catalog=catalog)

    definitions = bootstrapper.collect_definitions(_core_config_fixture())

    assert set(definitions) == {"main", "legacy_daily", "legacy_mean"}
    main = definitions["main"]
    assert main.metadata["source"] == "config"
    assert main.metadata["capability"] == "cap_override"
    assert main.metadata["tags"] == ("default", "custom")
    assert main.tags == ("default", "custom")

    legacy_daily = definitions["legacy_daily"]
    assert legacy_daily.engine == "daily_trend_momentum"
    assert legacy_daily.metadata["capability"] == "cap_daily"
    assert legacy_daily.metadata["tags"] == ("default",)

    legacy_mean = definitions["legacy_mean"]
    assert legacy_mean.engine == "mean_reversion"
    assert legacy_mean.parameters["lookback"] == 20


def test_instantiate_maps_definition_to_catalog_create_and_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _CatalogStub()
    bootstrapper = StrategyBootstrapper(catalog=catalog)
    definitions = bootstrapper.collect_definitions(_core_config_fixture())
    guard_calls: list[tuple[str, str]] = []

    class _Guard:
        def require_strategy(self, capability: str, *, message: str) -> None:
            guard_calls.append((capability, message))

    monkeypatch.setattr(
        "bot_core.runtime.strategy_bootstrapper.get_capability_guard",
        lambda: _Guard(),
    )

    registry = bootstrapper.instantiate(definitions)

    assert set(registry) == {"main", "legacy_daily", "legacy_mean"}
    assert len(catalog.created) == 3
    assert all(engine["name"] in registry for engine in registry.values())
    assert len(guard_calls) == 2
    assert all(call[0] == "cap_daily" for call in guard_calls)


def test_instantiate_preserves_exception_type_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingCatalog(_CatalogStub):
        def create(self, definition: StrategyDefinition) -> object:
            raise ValueError(f"factory exploded for {definition.name}")

    bootstrapper = StrategyBootstrapper(catalog=_FailingCatalog())
    definitions = {
        "broken": StrategyDefinition(
            name="broken",
            engine="daily_trend_momentum",
            license_tier="pro",
            risk_classes=("trend",),
            required_data=("ohlcv",),
            risk_hooks=(),
            parameters={},
            tags=(),
            metadata={},
        )
    }
    monkeypatch.setattr(
        "bot_core.runtime.strategy_bootstrapper.get_capability_guard",
        lambda: None,
    )

    with pytest.raises(ValueError, match="factory exploded for broken"):
        bootstrapper.instantiate(definitions)


def test_validate_schedule_strategies_keeps_contract_message() -> None:
    bootstrapper = StrategyBootstrapper(catalog=_CatalogStub())
    schedules = [SimpleNamespace(strategy="missing")]

    with pytest.raises(KeyError) as exc:
        bootstrapper.validate_schedule_strategies(schedules=schedules, strategies={})

    assert "Strategia missing nie została zarejestrowana w konfiguracji" in str(exc.value)


def test_bootstrap_returns_consistent_definitions_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _CatalogStub()
    bootstrapper = StrategyBootstrapper(catalog=catalog)
    monkeypatch.setattr(
        "bot_core.runtime.strategy_bootstrapper.get_capability_guard",
        lambda: None,
    )

    result = bootstrapper.bootstrap(_core_config_fixture())

    assert set(result.definitions) == {"main", "legacy_daily", "legacy_mean"}
    assert set(result.strategies) == {"main", "legacy_daily", "legacy_mean"}
    for name, strategy in result.strategies.items():
        assert strategy["name"] == name
