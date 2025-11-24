from bot_core.strategies.regime_workflow import StrategyRegimeWorkflow
from bot_core.trading.strategies.plugins import StrategyCatalog


def test_plugin_metadata_exposes_engine_keys() -> None:
    catalog = StrategyCatalog.default()
    for plugin_name in ("scalping", "options_income", "statistical_arbitrage"):
        metadata = catalog.metadata_for(plugin_name)
        assert metadata.get("engine") == plugin_name


def test_regime_workflow_maps_engine_to_plugin() -> None:
    workflow = StrategyRegimeWorkflow()
    for engine in ("scalping", "options_income", "statistical_arbitrage"):
        assert workflow._resolve_plugin_name(engine) == engine
