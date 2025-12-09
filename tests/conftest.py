import pytest

from bot_core.backtest.simulation import SimulationScenario


@pytest.fixture
def latency_one_no_cost_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="latency_one_no_cost",
        latency_bars=1,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def instant_no_cost_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="instant_no_cost",
        latency_bars=0,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def partial_fill_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="partial_fill_half_liquidity",
        latency_bars=0,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=0.5,
    )


@pytest.fixture
def slippage_fee_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="slippage_and_fee",
        latency_bars=0,
        slippage_bps=10.0,
        fee_bps=25.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def heavy_slippage_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="heavy_slippage_no_fee",
        latency_bars=0,
        slippage_bps=100.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )
