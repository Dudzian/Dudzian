from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG, StrategyPresetWizard


def test_preset_wizard_handles_intraday_engines() -> None:
    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    preset = wizard.build_preset(
        "intraday-bundle",
        [
                {
                    "engine": "scalping",
                    "name": "scalping-alpha",
                    "risk_classes": ["custom"],
                    "required_data": ["latency_feed"],
                    "tags": ["custom"],
                    "parameters": {"min_price_change": 0.0006},
                },
                {
                    "engine": "day_trading",
                    "name": "day-momentum",
                    "risk_classes": ["momentum"],
                    "required_data": ["ohlcv"],
                    "tags": ["session_open"],
                    "parameters": {"momentum_window": 3},
                },
        ],
    )

    entries = {entry["name"]: entry for entry in preset["strategies"]}
    scalping_entry = entries["scalping-alpha"]
    assert scalping_entry["capability"] == "scalping"
    assert scalping_entry["license_tier"] == "professional"
    assert scalping_entry["risk_classes"] == ["intraday", "scalping", "custom"]
    assert set(scalping_entry["required_data"]) >= {"ohlcv", "order_book", "latency_feed"}
    assert scalping_entry["metadata"]["capability"] == "scalping"
    assert scalping_entry["metadata"]["risk_classes"] == ("intraday", "scalping", "custom")
    assert scalping_entry["metadata"]["tags"] == ("intraday", "scalping", "custom")

    day_entry = entries["day-momentum"]
    assert day_entry["capability"] == "day_trading"
    assert day_entry["license_tier"] == "standard"
    assert day_entry["risk_classes"] == ["intraday", "momentum"]
    assert day_entry["metadata"]["risk_classes"] == ("intraday", "momentum")
    assert day_entry["metadata"]["tags"] == ("intraday", "momentum", "session_open")

