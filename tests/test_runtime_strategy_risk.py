from ui.backend.runtime_service import RuntimeService


def test_strategy_and_risk_defaults_and_save(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    service = RuntimeService(default_limit=5, runtime_config_path=runtime_path)

    strategies = service.strategyConfigs
    assert strategies, "powinny istnieć domyślne strategie demo"
    assert any(entry.get("id") == "grid_usdt" for entry in strategies)

    risk = service.riskControls
    assert risk["maxOpenPositions"] > 0

    save_result = service.saveStrategyConfig(
        "scalp_btc",
        {
            "name": "Scalp BTC",
            "mode": "scalping",
            "profile": "aggressive",
            "params": {"exchange": "binance", "symbol": "BTC/USDT", "takeProfitPct": 0.4},
        },
    )
    assert save_result["success"]

    save_risk = service.saveRiskControls(
        {"takeProfitPct": 0.9, "stopLossPct": 1.8, "maxOpenPositions": 3, "killSwitch": True}
    )
    assert save_risk["success"]

    # now reload service reading from files
    reloaded = RuntimeService(default_limit=5, runtime_config_path=runtime_path)
    loaded_strategies = reloaded.strategyConfigs
    assert any(entry.get("id") == "scalp_btc" for entry in loaded_strategies)
    loaded_risk = reloaded.riskControls
    assert loaded_risk["killSwitch"] is True
    assert loaded_risk["maxOpenPositions"] == 3
