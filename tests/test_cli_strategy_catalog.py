from argparse import Namespace
from pathlib import Path

from bot_core.cli import show_strategy_catalog


def _build_args(**overrides):
    defaults = {
        "output_format": "text",
        "engines": [],
        "capabilities": [],
        "tags": [],
        "config": None,
        "scheduler": None,
        "include_parameters": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_show_strategy_catalog_prints_engine_metadata(capfd) -> None:
    args = _build_args()
    assert show_strategy_catalog(args) == 0
    captured = capfd.readouterr().out
    assert "capability=" in captured
    assert "license=" in captured
    assert "risk_classes=[" in captured
    assert "required_data=[" in captured


def test_show_strategy_catalog_prints_definition_metadata(capfd) -> None:
    config_path = Path("config/core.yaml")
    args = _build_args(config=str(config_path))
    assert show_strategy_catalog(args) == 0
    captured = capfd.readouterr().out
    assert "Definicje strategii:" in captured
    assert "license=" in captured
    assert "required_data=" in captured
