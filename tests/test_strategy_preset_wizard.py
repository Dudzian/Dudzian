import json
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature
from bot_core.strategies.catalog import StrategyPresetWizard


def test_build_preset_merges_tags() -> None:
    wizard = StrategyPresetWizard()
    preset = wizard.build_preset(
        "demo",
        [
            {
                "engine": "grid_trading",
                "name": "grid-demo",
                "parameters": {"grid_size": 7},
                "tags": ["demo"],
                "risk_profile": "balanced",
                "metadata": {"exchange": "paper"},
            }
        ],
        metadata={"environment": "demo"},
    )

    assert preset["name"] == "demo"
    assert "created_at" in preset
    entry = preset["strategies"][0]
    assert entry["engine"] == "grid_trading"
    assert set(entry["tags"]).issuperset({"grid", "market_making", "demo"})
    assert entry["risk_profile"] == "balanced"
    assert entry["license_tier"] == "professional"
    assert entry["risk_classes"] == ["market_making"]
    assert entry["required_data"] == ["order_book", "ohlcv"]
    assert entry["metadata"]["license_tier"] == "professional"
    assert entry["metadata"]["risk_classes"] == ("market_making",)
    assert entry["metadata"]["required_data"] == ("order_book", "ohlcv")
    assert entry["metadata"]["exchange"] == "paper"
    assert preset["metadata"] == {"environment": "demo"}


def test_build_preset_rejects_mismatched_license() -> None:
    wizard = StrategyPresetWizard()
    with pytest.raises(ValueError):
        wizard.build_preset(
            "invalid",
            [
                {
                    "engine": "grid_trading",
                    "license_tier": "enterprise",
                }
            ],
        )


def test_build_preset_accepts_scalar_metadata() -> None:
    wizard = StrategyPresetWizard()
    preset = wizard.build_preset(
        "scalar-metadata",
        [
            {
                "engine": "volatility_target",
                "risk_classes": "tail_risk",
                "required_data": "stress_metrics",
            }
        ],
    )

    entry = preset["strategies"][0]
    assert entry["risk_classes"] == ["risk_control", "volatility", "tail_risk"]
    assert entry["required_data"] == ["ohlcv", "realized_volatility", "stress_metrics"]
    metadata = entry["metadata"]
    assert metadata["risk_classes"] == ("risk_control", "volatility", "tail_risk")
    assert metadata["required_data"] == (
        "ohlcv",
        "realized_volatility",
        "stress_metrics",
    )


def test_export_signed_preset(tmp_path: Path) -> None:
    wizard = StrategyPresetWizard()
    preset = wizard.build_preset(
        "bundle",
        [
            {
                "engine": "volatility_target",
                "parameters": {"target_volatility": 0.12},
            }
        ],
    )

    key = b"super-secret"
    output = tmp_path / "preset.json"
    wizard.export_signed(preset, signing_key=key, path=output, key_id="wizard-1")

    document = json.loads(output.read_text("utf-8"))
    assert "preset" in document and "signature" in document
    expected_signature = build_hmac_signature(document["preset"], key=key, key_id="wizard-1")
    assert document["signature"] == expected_signature


def test_build_preset_requires_engine() -> None:
    wizard = StrategyPresetWizard()
    with pytest.raises(ValueError):
        wizard.build_preset("broken", [{}])
