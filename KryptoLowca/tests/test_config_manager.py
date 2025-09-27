# tests/test_config_manager.py
# -*- coding: utf-8 -*-
"""Testy walidacji presetów w :mod:`managers.config_manager`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Odporny import ConfigManager: najpierw przestrzeń nazw KryptoLowca, potem lokalnie
try:  # pragma: no cover
    from KryptoLowca.managers.config_manager import ConfigManager  # type: ignore
except Exception:  # pragma: no cover
    from legacy_bridge.managers.config_manager import ConfigManager


@pytest.fixture()
def sample_preset() -> dict:
    return {
        "network": "Testnet",
        "mode": "Spot",
        "timeframe": "5m",
        "fraction": 0.25,
        "ai": {
            "enable": True,
            "seq_len": 256,
            "epochs": 20,
            "batch": 64,
            "retrain_min": 30,
            "train_window": 720,
            "valid_window": 120,
            "ai_threshold_bps": 4.5,
            "train_all": False,
        },
        "risk": {
            "max_daily_loss_pct": 0.02,
            "soft_halt_losses": 2,
            "trade_cooldown_on_error": 60,
            "risk_per_trade": 0.005,
            "portfolio_risk": 0.3,
            "one_trade_per_bar": True,
            "cooldown_s": 15,
            "min_move_pct": 0.01,
        },
        "dca_trailing": {
            "use_trailing": True,
            "atr_period": 14,
            "trail_atr_mult": 2.5,
            "take_atr_mult": 3.5,
            "dca_enabled": True,
            "dca_max_adds": 2,
            "dca_step_atr": 1.5,
        },
        "slippage": {"use_orderbook_vwap": True, "fallback_bps": 6.0},
        "advanced": {
            "rsi_period": 14,
            "ema_fast": 12,
            "ema_slow": 26,
            "atr_period": 14,
            "rsi_buy": 30.0,
            "rsi_sell": 70.0,
        },
        "paper": {"capital": 15_000.0},
        "selected_symbols": ["btc/usdt", "eth/usdt", "btc/usdt"],
    }


@pytest.fixture()
def cfg(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path)


def test_save_and_load_roundtrip(cfg: ConfigManager, sample_preset: dict) -> None:
    path = cfg.save_preset("demo", sample_preset)
    assert path.exists()

    loaded = cfg.load_preset("demo")
    assert loaded["network"] == "Testnet"
    assert loaded["mode"] == "Spot"
    assert loaded["timeframe"] == "5m"
    assert loaded["fraction"] == pytest.approx(0.25)
    # symbole oczyszczone i bez duplikatów
    assert loaded["selected_symbols"] == ["BTC/USDT", "ETH/USDT"]
    # sekcje dodatkowe obecne
    assert loaded["ai"]["epochs"] == 20
    assert loaded["risk"]["risk_per_trade"] == pytest.approx(0.005)
    assert loaded["version"] == ConfigManager.current_version()


def test_invalid_fraction_rejected(cfg: ConfigManager, sample_preset: dict) -> None:
    sample_preset["fraction"] = 1.5
    with pytest.raises(ValueError):
        cfg.save_preset("bad_fraction", sample_preset)


def test_defaults_and_normalisation(cfg: ConfigManager) -> None:
    minimal = {"selected_symbols": [" sol/usdt ", "ada/usdt"], "fraction": 0.0}
    preset = cfg.validate_preset(minimal)

    assert preset.network == "Testnet"
    assert preset.mode == "Spot"
    assert preset.timeframe == "1m"
    assert preset.selected_symbols == ["SOL/USDT", "ADA/USDT"]
    assert preset.fraction == pytest.approx(0.0)
    data = preset.to_dict()
    assert "ai" in data and "risk" in data
    assert preset.version == ConfigManager.current_version()


def test_demo_requirement_blocks_live(cfg: ConfigManager, sample_preset: dict) -> None:
    sample_preset["network"] = "Live"
    with pytest.raises(ValueError):
        cfg.save_preset("live", sample_preset)

    cfg.require_demo_mode(False)
    path = cfg.save_preset("live_allowed", sample_preset)
    assert path.exists()


def test_create_preset_with_audit(cfg: ConfigManager, sample_preset: dict) -> None:
    cfg.save_preset("base", sample_preset)
    audit = cfg.create_preset(
        "custom",
        base="base",
        overrides={"fraction": 0.3, "risk": {"max_daily_loss_pct": 0.25}},
    )
    assert audit["preset"]["network"] == "Testnet"
    assert audit["warnings"], "Zbyt wysoka dzienna strata powinna zostać oznaczona"
    assert "path" in audit and Path(audit["path"]).exists()
    assert audit["version"] == ConfigManager.current_version()
    assert audit["preset"]["version"] == ConfigManager.current_version()


def test_preset_wizard_profiles(cfg: ConfigManager, sample_preset: dict) -> None:
    cfg.save_preset("base", sample_preset)
    wizard = (
        cfg.preset_wizard()
        .from_template("base")
        .with_risk_profile("aggressive")
        .with_symbols(["btc/usdt", "ada/usdt"])
    )
    audit = wizard.build("aggressive_profile")

    assert audit["preset"]["selected_symbols"] == ["BTC/USDT", "ADA/USDT"]
    assert audit["preset"]["fraction"] == pytest.approx(0.35)
    assert audit["is_demo"] is True
    assert audit["preset"]["version"] == ConfigManager.current_version()
