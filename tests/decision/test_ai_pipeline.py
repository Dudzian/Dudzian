from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bot_core.ai import (
    DecisionModelInference,
    FeatureEngineer,
    ModelRepository,
    ModelTrainer,
)
from bot_core.data.base import OHLCVResponse


class FakeSource:
    def __init__(self, columns: tuple[str, ...], rows: list[tuple[float, ...]]) -> None:
        self._columns = columns
        self._rows = rows

    def fetch_ohlcv(self, request) -> OHLCVResponse:  # type: ignore[override]
        return OHLCVResponse(columns=self._columns, rows=self._rows)


@pytest.fixture()
def ohlcv_rows() -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    price = 100.0
    for index in range(60):
        open_time = float(1_600_000_000 + index * 60)
        drift = math.sin(index / 5.0) * 0.3
        open_price = price * (1 + drift / 100.0)
        close_price = open_price * (1 + (0.2 - drift) / 100.0)
        high_price = max(open_price, close_price) * 1.005
        low_price = min(open_price, close_price) * 0.995
        volume = 1000.0 + index * 5.0
        rows.append(
            (
                open_time,
                round(open_price, 6),
                round(high_price, 6),
                round(low_price, 6),
                round(close_price, 6),
                round(volume, 4),
            )
        )
        price = close_price
    return rows


def test_training_and_inference(tmp_path: Path, ohlcv_rows: list[tuple[float, ...]]) -> None:
    columns = ("open_time", "open", "high", "low", "close", "volume")
    source = FakeSource(columns, ohlcv_rows)
    engineer = FeatureEngineer(source, feature_window=8, target_horizon=1)
    dataset = engineer.build_dataset(
        symbols=["BTCUSDT"],
        interval="1m",
        start=int(ohlcv_rows[0][0]),
        end=int(ohlcv_rows[-1][0]),
    )
    assert dataset.vectors, "dataset powinien zawierać rekordy do trenowania"
    assert dataset.metadata["row_count"] == len(dataset.vectors)
    assert dataset.metadata["start_timestamp"] <= dataset.metadata["end_timestamp"]
    assert dataset.metadata["symbols"] == ["BTCUSDT"]
    assert "feature_names" in dataset.metadata
    assert "feature_stats" in dataset.metadata
    sample_features = dataset.vectors[0].features
    for key in [
        "ema_fast_gap",
        "ema_slow_gap",
        "rsi",
        "volume_zscore",
        "volatility_trend",
        "atr_ratio",
        "bollinger_position",
        "bollinger_width",
        "macd_line",
        "macd_signal_gap",
        "macd_histogram",
        "ppo_line",
        "ppo_signal",
        "ppo_signal_gap",
        "ppo_histogram",
        "fisher_transform",
        "fisher_signal_gap",
        "schaff_trend_cycle",
        "trix",
        "ultimate_oscillator",
        "ease_of_movement",
        "vortex_positive",
        "vortex_negative",
        "price_rate_of_change",
        "chande_momentum_oscillator",
        "detrended_price_oscillator",
        "aroon_up",
        "aroon_down",
        "aroon_oscillator",
        "balance_of_power",
        "stochastic_rsi",
        "relative_vigor_index",
        "relative_vigor_signal",
        "relative_vigor_signal_gap",
        "elder_ray_bull_power",
        "elder_ray_bear_power",
        "ulcer_index",
        "efficiency_ratio",
        "true_strength_index",
        "true_strength_signal",
        "true_strength_signal_gap",
        "connors_rsi",
        "kama_gap",
        "kama_slope",
        "qstick",
        "dema_gap",
        "tema_gap",
        "frama_gap",
        "frama_slope",
        "stochastic_k",
        "stochastic_d",
        "williams_r",
        "cci",
        "obv_normalized",
        "pvt_normalized",
        "positive_volume_index",
        "negative_volume_index",
        "di_plus",
        "di_minus",
        "adx",
        "mfi",
        "force_index_normalized",
        "accumulation_distribution",
        "chaikin_oscillator",
        "ichimoku_conversion_gap",
        "ichimoku_base_gap",
        "ichimoku_span_a_gap",
        "ichimoku_span_b_gap",
        "ichimoku_cloud_thickness",
        "ichimoku_price_position",
        "donchian_position",
        "donchian_width",
        "chaikin_money_flow",
        "mass_index",
        "heikin_trend",
        "heikin_shadow_ratio",
        "heikin_upper_shadow_ratio",
        "heikin_lower_shadow_ratio",
        "keltner_position",
        "keltner_width",
        "vwap_gap",
        "psar_gap",
        "psar_direction",
        "pivot_gap",
        "pivot_resistance_gap",
        "pivot_support_gap",
        "fractal_high_gap",
        "fractal_low_gap",
        "coppock_curve",
        "choppiness_index",
        "intraday_intensity",
        "intraday_intensity_volume",
        "klinger_oscillator",
        "klinger_signal",
        "klinger_signal_gap",
    ]:
        assert key in sample_features
        assert math.isfinite(sample_features[key])
    assert 0.0 <= sample_features["rsi"] <= 100.0
    assert 0.0 <= sample_features["stochastic_k"] <= 100.0
    assert 0.0 <= sample_features["stochastic_d"] <= 100.0
    assert -110.0 <= sample_features["williams_r"] <= 10.0
    assert sample_features["atr_ratio"] >= 0.0
    assert sample_features["bollinger_width"] >= 0.0
    assert abs(sample_features["obv_normalized"]) < 50.0
    assert abs(sample_features["pvt_normalized"]) < 30.0
    assert -50.0 <= sample_features["ppo_line"] <= 50.0
    assert -50.0 <= sample_features["ppo_signal"] <= 50.0
    assert -50.0 <= sample_features["ppo_signal_gap"] <= 50.0
    assert -50.0 <= sample_features["ppo_histogram"] <= 50.0
    assert -5.0 <= sample_features["positive_volume_index"] <= 5.0
    assert -5.0 <= sample_features["negative_volume_index"] <= 5.0
    assert -5.0 <= sample_features["trix"] <= 5.0
    assert 0.0 <= sample_features["ultimate_oscillator"] <= 100.0
    assert -5.0 <= sample_features["ease_of_movement"] <= 5.0
    assert 0.0 <= sample_features["vortex_positive"] <= 5.0
    assert 0.0 <= sample_features["vortex_negative"] <= 5.0
    assert -5.0 <= sample_features["price_rate_of_change"] <= 5.0
    assert -100.0 <= sample_features["chande_momentum_oscillator"] <= 100.0
    assert -5.0 <= sample_features["detrended_price_oscillator"] <= 5.0
    assert 0.0 <= sample_features["aroon_up"] <= 100.0
    assert 0.0 <= sample_features["aroon_down"] <= 100.0
    assert -100.0 <= sample_features["aroon_oscillator"] <= 100.0
    assert -5.0 <= sample_features["balance_of_power"] <= 5.0
    assert 0.0 <= sample_features["stochastic_rsi"] <= 100.0
    assert -5.0 <= sample_features["relative_vigor_index"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_signal"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_signal_gap"] <= 5.0
    assert -150.0 <= sample_features["true_strength_index"] <= 150.0
    assert -150.0 <= sample_features["true_strength_signal"] <= 150.0
    assert -10.0 <= sample_features["true_strength_signal_gap"] <= 10.0
    assert 0.0 <= sample_features["connors_rsi"] <= 100.0
    assert -5.0 <= sample_features["kama_gap"] <= 5.0
    assert -5.0 <= sample_features["kama_slope"] <= 5.0
    assert -5.0 <= sample_features["qstick"] <= 5.0
    assert -10.0 <= sample_features["fisher_transform"] <= 10.0
    assert -10.0 <= sample_features["fisher_signal_gap"] <= 10.0
    assert 0.0 <= sample_features["schaff_trend_cycle"] <= 100.0
    assert -5.0 <= sample_features["dema_gap"] <= 5.0
    assert -5.0 <= sample_features["tema_gap"] <= 5.0
    assert -5.0 <= sample_features["frama_gap"] <= 5.0
    assert -5.0 <= sample_features["frama_slope"] <= 5.0
    assert -5.0 <= sample_features["elder_ray_bull_power"] <= 5.0
    assert -5.0 <= sample_features["elder_ray_bear_power"] <= 5.0
    assert 0.0 <= sample_features["ulcer_index"] <= 5.0
    assert 0.0 <= sample_features["efficiency_ratio"] <= 1.0
    assert 0.0 <= sample_features["di_plus"] <= 100.0
    assert 0.0 <= sample_features["di_minus"] <= 100.0
    assert 0.0 <= sample_features["adx"] <= 100.0
    assert 0.0 <= sample_features["mfi"] <= 100.0
    assert -5.0 <= sample_features["ichimoku_conversion_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_base_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_span_a_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_span_b_gap"] <= 5.0
    assert 0.0 <= sample_features["ichimoku_cloud_thickness"] <= 5.0
    assert -2.0 <= sample_features["ichimoku_price_position"] <= 3.0
    assert -1.0 <= sample_features["donchian_position"] <= 2.0
    assert 0.0 <= sample_features["donchian_width"] <= 5.0
    assert -1.5 <= sample_features["chaikin_money_flow"] <= 1.5
    assert -5.0 <= sample_features["accumulation_distribution"] <= 5.0
    assert -5.0 <= sample_features["chaikin_oscillator"] <= 5.0
    assert 0.0 <= sample_features["mass_index"] <= 50.0
    assert -5.0 <= sample_features["klinger_oscillator"] <= 5.0
    assert -5.0 <= sample_features["klinger_signal"] <= 5.0
    assert -5.0 <= sample_features["klinger_signal_gap"] <= 5.0
    assert -5.0 <= sample_features["heikin_trend"] <= 5.0
    assert 0.0 <= sample_features["heikin_shadow_ratio"] <= 5.0
    assert 0.0 <= sample_features["heikin_upper_shadow_ratio"] <= 3.0
    assert 0.0 <= sample_features["heikin_lower_shadow_ratio"] <= 3.0
    assert -1.0 <= sample_features["keltner_position"] <= 2.0
    assert 0.0 <= sample_features["keltner_width"] <= 5.0
    assert -5.0 <= sample_features["vwap_gap"] <= 5.0
    assert -5.0 <= sample_features["psar_gap"] <= 5.0
    assert -1.0 <= sample_features["psar_direction"] <= 1.0
    assert -5.0 <= sample_features["pivot_gap"] <= 5.0
    assert -5.0 <= sample_features["pivot_resistance_gap"] <= 5.0
    assert -5.0 <= sample_features["pivot_support_gap"] <= 5.0
    assert -5.0 <= sample_features["fractal_high_gap"] <= 5.0
    assert -5.0 <= sample_features["fractal_low_gap"] <= 5.0
    assert -10.0 <= sample_features["coppock_curve"] <= 10.0
    assert 0.0 <= sample_features["choppiness_index"] <= 100.0
    assert -1.0 <= sample_features["intraday_intensity"] <= 1.0
    assert -5.0 <= sample_features["intraday_intensity_volume"] <= 5.0

    trainer = ModelTrainer(learning_rate=0.2, n_estimators=10, validation_split=0.2)
    artifact = trainer.train(dataset)
    repo = ModelRepository(tmp_path)
    artifact_path = repo.save(artifact, "test_model.json")

    assert "feature_scalers" in artifact.metadata
    scalers = artifact.metadata["feature_scalers"]  # type: ignore[index]
    assert set(scalers.keys()) == set(dataset.feature_names)
    for stats in scalers.values():
        assert "mean" in stats and "stdev" in stats

    assert artifact.metadata["training_rows"] + artifact.metadata["validation_rows"] == len(
        dataset.vectors
    )
    assert artifact.metadata["validation_rows"] > 0
    assert pytest.approx(artifact.metadata["validation_split"], rel=1e-6) == 0.2
    assert "validation_metrics" in artifact.metadata
    assert "validation_mae" in artifact.metrics
    assert "train_mae" in artifact.metrics
    assert artifact.metrics["mae"] == pytest.approx(artifact.metrics["train_mae"])

    inference = DecisionModelInference(repo)
    inference.load_weights(artifact_path)
    sample_vector = dataset.vectors[-1]
    score = inference.score(sample_vector.features)

    assert -500.0 < score.expected_return_bps < 500.0
    assert 0.0 <= score.success_probability <= 1.0

    missing_features_score = inference.score({})
    assert math.isfinite(missing_features_score.expected_return_bps)
