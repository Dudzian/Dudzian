from __future__ import annotations

import json
from datetime import datetime, timezone

from pathlib import Path

import pytest

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.strategies.marketplace import (
    StrategyPreset,
    load_marketplace_index,
    load_marketplace_presets,
    load_preset,
)


def test_load_marketplace_presets_with_metadata(tmp_path: Path) -> None:
    marketplace_dir = tmp_path
    payload = {
        "id": "demo_strategy",
        "name": "Demo Strategy",
        "description": "Testowa strategia do sprawdzenia metadanych.",
        "risk_level": "balanced",
        "recommended_min_balance": 1234.5,
        "timeframe": "1h",
        "exchanges": ["binance"],
        "tags": ["demo"],
        "version": "0.1.0",
        "last_updated": "2024-05-05T10:15:00+00:00",
        "compatibility": {"app": ">=2.7.0"},
        "compliance": {"required_flags": ["compliance_confirmed"]},
        "config": {"strategy": {"preset": "SAFE", "mode": "demo"}},
        "evaluation": {
            "rank": 7,
            "risk_label": "demo",
            "risk_score": 0.15,
            "highlights": ["Test highlight"],
            "backtest": {
                "period": "2024-01-01/2024-03-31",
                "cagr": 0.1,
                "sharpe": 1.0,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
                "trades": 42,
            },
        },
    }
    (marketplace_dir / "demo_strategy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    presets = load_marketplace_presets(base_path=marketplace_dir)
    assert any(p.preset_id == "demo_strategy" for p in presets)

    preset = load_preset("demo_strategy", base_path=marketplace_dir)
    assert isinstance(preset, StrategyPreset)
    assert preset.version == "0.1.0"
    assert preset.compatibility["app"] == ">=2.7.0"
    assert preset.compliance["required_flags"] == ["compliance_confirmed"]
    assert preset.last_updated == datetime(2024, 5, 5, 10, 15, tzinfo=timezone.utc)
    assert preset.evaluation is not None
    assert preset.evaluation.rank == 7
    assert preset.evaluation.backtest is not None
    assert preset.evaluation.backtest.trades == 42

    index = load_marketplace_index(base_path=marketplace_dir)
    assert "demo_strategy" in index
    assert index["demo_strategy"].evaluation is not None


def test_marketplace_risk_summary(tmp_path: Path) -> None:
    marketplace_dir = tmp_path
    first = {
        "id": "preset_a",
        "name": "Preset A",
        "description": "Pierwszy preset",
        "risk_level": "balanced",
        "tags": ["demo"],
        "config": {"strategy": {"preset": "A"}},
        "evaluation": {
            "rank": 1,
            "risk_label": "growth",
            "risk_score": 0.45,
        },
    }
    second = {
        "id": "preset_b",
        "name": "Preset B",
        "description": "Drugi preset",
        "risk_level": "balanced",
        "tags": ["demo"],
        "config": {"strategy": {"preset": "B"}},
        "evaluation": {
            "rank": 4,
            "risk_label": "growth",
            "risk_score": 0.75,
        },
    }
    third = {
        "id": "preset_c",
        "name": "Preset C",
        "description": "Trzeci preset",
        "risk_level": "safe",
        "tags": ["demo"],
        "config": {"strategy": {"preset": "C"}},
    }

    for payload in (first, second, third):
        (marketplace_dir / f"{payload['id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manager = ConfigManager()
    manager.set_marketplace_directory(marketplace_dir)

    summary = manager.get_marketplace_risk_summary()
    assert set(summary.keys()) == {"growth", "safe", "overall"}

    growth = summary["growth"]
    assert growth["count"] == 2
    assert growth["presets_with_score"] == 2
    assert growth["presets_with_rank"] == 2
    assert growth["best_rank"] == 1
    assert growth["worst_rank"] == 4
    assert growth["top_preset"] == "preset_a"
    assert growth["bottom_preset"] == "preset_b"
    assert growth["avg_risk_score"] == pytest.approx((0.45 + 0.75) / 2, rel=1e-6)
    assert growth["min_risk_score"] == pytest.approx(0.45)
    assert growth["max_risk_score"] == pytest.approx(0.75)
    assert growth["avg_rank"] == pytest.approx((1 + 4) / 2)
    assert growth["risk_score_stddev"] == pytest.approx(0.15)
    assert growth["risk_score_p10"] == pytest.approx(0.48)
    assert growth["risk_score_p25"] == pytest.approx(0.525)
    assert growth["risk_score_p75"] == pytest.approx(0.675)
    assert growth["risk_score_p90"] == pytest.approx(0.72)
    assert growth["risk_score_iqr"] == pytest.approx(0.15)
    assert growth["risk_score_cv"] == pytest.approx(0.25)
    assert growth["count_share"] == pytest.approx(2 / 3)
    assert growth["risk_score_median"] == pytest.approx(0.6)
    assert growth["rank_median"] == pytest.approx(2.5)
    assert growth["rank_p10"] == pytest.approx(1.3)
    assert growth["rank_p25"] == pytest.approx(1.75)
    assert growth["rank_p75"] == pytest.approx(3.25)
    assert growth["rank_p90"] == pytest.approx(3.7)
    assert growth["rank_iqr"] == pytest.approx(1.5)
    assert growth["rank_stddev"] == pytest.approx(1.5)
    assert growth["risk_score_variance"] == pytest.approx(0.0225)
    assert growth["risk_score_mad"] == pytest.approx(0.15)
    assert growth["risk_score_range"] == pytest.approx(0.30)
    assert growth["risk_score_skewness"] == pytest.approx(0.0)
    assert growth["risk_score_kurtosis"] == pytest.approx(-2.0)
    assert growth["rank_variance"] == pytest.approx(2.25)
    assert growth["rank_mad"] == pytest.approx(1.5)
    assert growth["rank_range"] == pytest.approx(3.0)
    assert growth["rank_cv"] == pytest.approx(0.6)
    assert growth["rank_skewness"] == pytest.approx(0.0)
    assert growth["rank_kurtosis"] == pytest.approx(-2.0)
    assert growth["risk_score_jarque_bera"] == pytest.approx(1.0 / 3.0)
    assert growth["rank_jarque_bera"] == pytest.approx(1.0 / 3.0)
    assert growth["score_coverage"] == pytest.approx(1.0)
    assert growth["rank_coverage"] == pytest.approx(1.0)
    assert growth["score_rank_count"] == 2
    assert growth["score_rank_covariance"] == pytest.approx(0.225)
    assert growth["score_rank_pearson"] == pytest.approx(1.0)
    assert growth["score_rank_spearman"] == pytest.approx(1.0)
    assert growth["score_rank_regression_slope"] == pytest.approx(10.0)
    assert growth["score_rank_regression_intercept"] == pytest.approx(-3.5)
    assert growth["score_rank_r_squared"] == pytest.approx(1.0)
    assert growth["score_rank_regression_bias"] == pytest.approx(0.0)
    assert growth["score_rank_regression_mae"] == pytest.approx(0.0)
    assert growth["score_rank_regression_mse"] == pytest.approx(0.0)
    assert growth["score_rank_regression_rmse"] == pytest.approx(0.0)
    assert growth["score_rank_regression_residual_variance"] == pytest.approx(0.0)
    assert growth["score_rank_regression_residual_std_error"] == pytest.approx(0.0)

    safe = summary["safe"]
    assert safe["count"] == 1
    assert safe["best_rank"] is None
    assert safe["worst_rank"] is None
    assert safe["avg_risk_score"] is None
    assert safe["presets_with_score"] == 0
    assert safe["min_risk_score"] is None
    assert safe["max_risk_score"] is None
    assert safe["presets_with_rank"] == 0
    assert safe["avg_rank"] is None
    assert safe["risk_score_stddev"] is None
    assert safe["risk_score_p10"] is None
    assert safe["risk_score_p25"] is None
    assert safe["risk_score_p75"] is None
    assert safe["risk_score_p90"] is None
    assert safe["risk_score_iqr"] is None
    assert safe["count_share"] == pytest.approx(1 / 3)
    assert safe["risk_score_median"] is None
    assert safe["rank_median"] is None
    assert safe["rank_p10"] is None
    assert safe["rank_p25"] is None
    assert safe["rank_p75"] is None
    assert safe["rank_p90"] is None
    assert safe["rank_iqr"] is None
    assert safe["rank_stddev"] is None
    assert safe["risk_score_variance"] is None
    assert safe["risk_score_mad"] is None
    assert safe["risk_score_range"] is None
    assert safe["rank_variance"] is None
    assert safe["rank_mad"] is None
    assert safe["rank_range"] is None
    assert safe["risk_score_cv"] is None
    assert safe["rank_cv"] is None
    assert safe["risk_score_skewness"] is None
    assert safe["risk_score_kurtosis"] is None
    assert safe["rank_skewness"] is None
    assert safe["rank_kurtosis"] is None
    assert safe["score_coverage"] == pytest.approx(0.0)
    assert safe["rank_coverage"] == pytest.approx(0.0)
    assert safe["risk_score_jarque_bera"] is None
    assert safe["rank_jarque_bera"] is None
    assert safe["score_rank_count"] == 0
    assert safe["score_rank_covariance"] is None
    assert safe["score_rank_pearson"] is None
    assert safe["score_rank_spearman"] is None
    assert safe["score_rank_regression_slope"] is None
    assert safe["score_rank_regression_intercept"] is None
    assert safe["score_rank_r_squared"] is None
    assert safe["score_rank_regression_bias"] is None
    assert safe["score_rank_regression_mae"] is None
    assert safe["score_rank_regression_mse"] is None
    assert safe["score_rank_regression_rmse"] is None
    assert safe["score_rank_regression_residual_variance"] is None
    assert safe["score_rank_regression_residual_std_error"] is None

    overall = summary["overall"]
    assert overall["count"] == 3
    assert overall["presets_with_score"] == 2
    assert overall["presets_with_rank"] == 2
    assert overall["best_rank"] == 1
    assert overall["worst_rank"] == 4
    assert overall["top_preset"] == "preset_a"
    assert overall["bottom_preset"] == "preset_b"
    assert overall["avg_risk_score"] == pytest.approx((0.45 + 0.75) / 2, rel=1e-6)
    assert overall["avg_rank"] == pytest.approx((1 + 4) / 2)
    assert overall["risk_score_stddev"] == pytest.approx(0.15)
    assert overall["risk_score_variance"] == pytest.approx(0.0225)
    assert overall["risk_score_mad"] == pytest.approx(0.15)
    assert overall["risk_score_range"] == pytest.approx(0.30)
    assert overall["risk_score_p10"] == pytest.approx(0.48)
    assert overall["risk_score_p25"] == pytest.approx(0.525)
    assert overall["risk_score_p75"] == pytest.approx(0.675)
    assert overall["risk_score_p90"] == pytest.approx(0.72)
    assert overall["risk_score_iqr"] == pytest.approx(0.15)
    assert overall["risk_score_cv"] == pytest.approx(0.25)
    assert overall["risk_score_skewness"] == pytest.approx(0.0)
    assert overall["risk_score_kurtosis"] == pytest.approx(-2.0)
    assert overall["risk_score_jarque_bera"] == pytest.approx(1.0 / 3.0)
    assert overall["rank_stddev"] == pytest.approx(1.5)
    assert overall["rank_variance"] == pytest.approx(2.25)
    assert overall["rank_mad"] == pytest.approx(1.5)
    assert overall["rank_range"] == pytest.approx(3.0)
    assert overall["rank_p10"] == pytest.approx(1.3)
    assert overall["rank_p25"] == pytest.approx(1.75)
    assert overall["rank_p75"] == pytest.approx(3.25)
    assert overall["rank_p90"] == pytest.approx(3.7)
    assert overall["rank_iqr"] == pytest.approx(1.5)
    assert overall["rank_cv"] == pytest.approx(0.6)
    assert overall["rank_skewness"] == pytest.approx(0.0)
    assert overall["rank_kurtosis"] == pytest.approx(-2.0)
    assert overall["rank_jarque_bera"] == pytest.approx(1.0 / 3.0)
    assert overall["count_share"] == pytest.approx(1.0)
    assert overall["score_coverage"] == pytest.approx(2 / 3)
    assert overall["rank_coverage"] == pytest.approx(2 / 3)
    assert overall["score_rank_count"] == 2
    assert overall["score_rank_covariance"] == pytest.approx(0.225)
    assert overall["score_rank_pearson"] == pytest.approx(1.0)
    assert overall["score_rank_spearman"] == pytest.approx(1.0)
    assert overall["score_rank_regression_slope"] == pytest.approx(10.0)
    assert overall["score_rank_regression_intercept"] == pytest.approx(-3.5)
    assert overall["score_rank_r_squared"] == pytest.approx(1.0)
    assert overall["score_rank_regression_bias"] == pytest.approx(0.0)
    assert overall["score_rank_regression_mae"] == pytest.approx(0.0)
    assert overall["score_rank_regression_mse"] == pytest.approx(0.0)
    assert overall["score_rank_regression_rmse"] == pytest.approx(0.0)
    assert overall["score_rank_regression_residual_variance"] == pytest.approx(0.0)
    assert overall["score_rank_regression_residual_std_error"] == pytest.approx(0.0)
