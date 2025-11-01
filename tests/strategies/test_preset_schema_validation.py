import pytest

from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyCatalog,
    StrategyPresetValidationError,
)


def _build_minimal_preset() -> dict[str, object]:
    return {
        "name": "Test preset",
        "strategies": [
            {
                "name": "grid-entry",
                "engine": "grid_trading",
                "parameters": {"grid_size": 7},
                "risk_classes": ["grid", "grid"],
                "required_data": "ticker",
                "tags": ["swing", "swing"],
            }
        ],
        "metadata": {"id": "test-preset"},
    }


def test_validate_preset_payload_normalizes_sequences() -> None:
    catalog = StrategyCatalog()
    payload = _build_minimal_preset()

    normalized = catalog.validate_preset_payload(payload)

    strategies = normalized["strategies"]
    assert isinstance(strategies, tuple)
    assert strategies[0]["risk_classes"] == ("grid",)
    assert strategies[0]["required_data"] == ("ticker",)
    assert strategies[0]["tags"] == ("swing",)


def test_validate_preset_payload_requires_identifier() -> None:
    catalog = StrategyCatalog()
    payload = _build_minimal_preset()
    payload["metadata"].pop("id")  # type: ignore[index]

    with pytest.raises(StrategyPresetValidationError) as excinfo:
        catalog.validate_preset_payload(payload)

    assert any("metadata.id" in message for message in excinfo.value.errors)


def test_validate_preset_payload_requires_engine_field() -> None:
    catalog = StrategyCatalog()
    payload = _build_minimal_preset()
    payload["strategies"][0].pop("engine")  # type: ignore[index]

    with pytest.raises(StrategyPresetValidationError) as excinfo:
        catalog.validate_preset_payload(payload)

    assert any("strategies.0.engine" in message for message in excinfo.value.errors)


def test_validate_preset_payload_rejects_unknown_engine() -> None:
    payload = _build_minimal_preset()
    payload["strategies"][0]["engine"] = "non_existing"  # type: ignore[index]

    with pytest.raises(StrategyPresetValidationError) as excinfo:
        DEFAULT_STRATEGY_CATALOG.validate_preset_payload(payload)

    assert any("nieznany silnik" in message for message in excinfo.value.errors)
