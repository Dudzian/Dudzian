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
        "session_time_sin",
        "session_time_cos",
        "session_time_progress",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "week_of_year_sin",
        "week_of_year_cos",
        "week_of_year_progress",
        "month_sin",
        "month_cos",
        "month_progress",
        "day_of_month_progress",
        "is_month_start",
        "is_month_end",
        "is_quarter_end",
        "price_zscore",
        "price_zscore_change",
        "price_median_gap",
        "price_mad_ratio",
        "price_percentile_low_gap",
        "price_percentile_high_gap",
        "price_percentile_spread",
        "price_percent_rank",
        "volume_percent_rank",
        "volume_median_gap",
        "volume_mad_ratio",
        "volume_percentile_low_gap",
        "volume_percentile_high_gap",
        "volume_percentile_spread",
        "return_skewness",
        "return_kurtosis",
        "return_median_gap_bps",
        "return_mad_ratio",
        "return_percentile_low_gap_bps",
        "return_percentile_high_gap_bps",
        "return_percentile_spread_bps",
        "return_percent_rank",
        "return_positive_share",
        "return_negative_share",
        "return_flat_share",
        "return_sign_balance",
        "return_positive_magnitude_share",
        "return_negative_magnitude_share",
        "return_flat_magnitude_share",
        "return_magnitude_balance",
        "range_percent_rank",
        "atr_percent_rank",
        "ema_fast_gap",
        "ema_slow_gap",
        "rsi",
        "rsi_change",
        "volume_zscore",
        "volume_zscore_change",
        "volume_trend_slope",
        "volume_trend_strength",
        "volume_return_median_gap",
        "volume_return_mad_ratio",
        "volume_return_percentile_low_gap",
        "volume_return_percentile_high_gap",
        "volume_return_percentile_spread",
        "volume_return_percent_rank",
        "volume_return_positive_share",
        "volume_return_negative_share",
        "volume_return_flat_share",
        "volume_return_sign_balance",
        "volume_return_positive_magnitude_share",
        "volume_return_negative_magnitude_share",
        "volume_return_flat_magnitude_share",
        "volume_return_magnitude_balance",
        "return_sharpe_like",
        "return_sortino_like",
        "downside_volatility",
        "upside_volatility",
        "return_volatility_short",
        "return_volatility_long",
        "return_volatility_ratio",
        "volume_mean_short_ratio",
        "volume_volatility_short",
        "volume_volatility_long",
        "volume_volatility_ratio",
        "return_range_correlation",
        "return_range_crosscorr_lag1",
        "return_range_crosscorr_lag3",
        "return_range_crosscorr_lag5",
        "return_atr_correlation",
        "return_atr_crosscorr_lag1",
        "return_atr_crosscorr_lag3",
        "return_atr_crosscorr_lag5",
        "return_volume_correlation",
        "return_volume_crosscorr_lag1",
        "return_volume_crosscorr_lag3",
        "return_volume_crosscorr_lag5",
        "price_volume_correlation",
        "return_autocorr",
        "volume_autocorr",
        "return_entropy",
        "hurst_exponent",
        "fractal_dimension",
        "volatility_trend",
        "atr_ratio",
        "atr_ratio_change",
        "atr_trend_slope",
        "atr_trend_strength",
        "atr_volatility_ratio",
        "max_drawdown_ratio",
        "max_drawdown_duration",
        "drawdown_recovery_ratio",
        "bollinger_position",
        "bollinger_width",
        "bollinger_position_change",
        "bollinger_width_change",
        "macd_line",
        "macd_signal_gap",
        "macd_histogram",
        "macd_histogram_change",
        "ppo_line",
        "ppo_line_change",
        "ppo_signal",
        "ppo_signal_change",
        "ppo_signal_gap",
        "ppo_signal_gap_change",
        "ppo_histogram",
        "ppo_histogram_change",
        "fisher_transform",
        "fisher_transform_change",
        "fisher_signal_gap",
        "fisher_signal_gap_change",
        "fisher_signal_change",
        "schaff_trend_cycle",
        "schaff_trend_cycle_change",
        "trix",
        "ultimate_oscillator",
        "ease_of_movement",
        "vortex_positive",
        "vortex_negative",
        "price_rate_of_change",
        "price_trend_slope",
        "price_trend_strength",
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
        "kama_gap_change",
        "kama_slope_change",
        "qstick",
        "dema_gap",
        "dema_gap_change",
        "tema_gap",
        "tema_gap_change",
        "frama_gap",
        "frama_slope",
        "frama_gap_change",
        "frama_slope_change",
        "hma_gap",
        "hma_slope",
        "hma_gap_change",
        "hma_slope_change",
        "zlema_gap",
        "zlema_gap_change",
        "vwma_gap",
        "vwma_gap_change",
        "t3_gap",
        "t3_slope",
        "t3_gap_change",
        "t3_slope_change",
        "supertrend_gap",
        "supertrend_direction",
        "supertrend_bandwidth",
        "supertrend_direction_change",
        "supertrend_bandwidth_change",
        "supertrend_gap_change",
        "stochastic_k",
        "stochastic_k_change",
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
        "adx_change",
        "mfi",
        "mfi_change",
        "force_index_normalized",
        "accumulation_distribution",
        "chaikin_oscillator",
        "chaikin_oscillator_change",
        "ichimoku_conversion_gap",
        "ichimoku_base_gap",
        "ichimoku_span_a_gap",
        "ichimoku_span_b_gap",
        "ichimoku_cloud_thickness",
        "ichimoku_price_position",
        "ichimoku_price_position_change",
        "donchian_position",
        "donchian_width",
        "donchian_position_change",
        "donchian_width_change",
        "chaikin_money_flow",
        "chaikin_money_flow_change",
        "mass_index",
        "mass_index_change",
        "heikin_trend",
        "heikin_trend_change",
        "heikin_shadow_ratio",
        "heikin_shadow_ratio_change",
        "heikin_upper_shadow_ratio",
        "heikin_lower_shadow_ratio",
        "keltner_position",
        "keltner_width",
        "keltner_position_change",
        "keltner_width_change",
        "vwap_gap",
        "vwap_gap_change",
        "psar_gap",
        "psar_direction",
        "psar_direction_change",
        "pivot_gap",
        "pivot_resistance_gap",
        "pivot_support_gap",
        "fractal_high_gap",
        "fractal_low_gap",
        "coppock_curve",
        "choppiness_index",
        "intraday_intensity",
        "intraday_intensity_change",
        "intraday_intensity_volume",
        "intraday_intensity_volume_change",
        "open_gap",
        "close_location_value",
        "candle_body_ratio",
        "candle_upper_shadow_ratio",
        "candle_lower_shadow_ratio",
        "range_atr_ratio",
        "body_atr_ratio",
        "upper_shadow_atr_ratio",
        "lower_shadow_atr_ratio",
        "price_up_streak_ratio",
        "price_down_streak_ratio",
        "volume_up_streak_ratio",
        "volume_down_streak_ratio",
        "time_since_high_close_ratio",
        "time_since_low_close_ratio",
        "time_since_high_volume_ratio",
        "time_since_low_volume_ratio",
        "time_since_high_return_ratio",
        "time_since_low_return_ratio",
        "volume_flow_imbalance",
        "volume_flow_imbalance_change",
        "volume_flow_ratio",
        "market_facilitation_index",
        "market_facilitation_index_change",
        "klinger_oscillator",
        "klinger_oscillator_change",
        "klinger_signal",
        "klinger_signal_gap",
        "klinger_signal_change",
        "alligator_jaw_gap",
        "alligator_jaw_gap_change",
        "alligator_teeth_gap",
        "alligator_teeth_gap_change",
        "alligator_lips_gap",
        "alligator_lips_gap_change",
        "gator_oscillator_upper",
        "gator_oscillator_upper_change",
        "gator_oscillator_lower",
        "gator_oscillator_lower_change",
    ]:
        assert key in sample_features
        assert math.isfinite(sample_features[key])
    assert 0.0 <= sample_features["rsi"] <= 100.0
    assert -100.0 <= sample_features["rsi_change"] <= 100.0
    assert 0.0 <= sample_features["stochastic_k"] <= 100.0
    assert -100.0 <= sample_features["stochastic_k_change"] <= 100.0
    assert 0.0 <= sample_features["stochastic_d"] <= 100.0
    assert -110.0 <= sample_features["williams_r"] <= 10.0
    assert -1.0 <= sample_features["return_autocorr"] <= 1.0
    assert -1.0 <= sample_features["volume_autocorr"] <= 1.0
    assert 0.0 <= sample_features["return_volatility_short"] <= 5.0
    assert 0.0 <= sample_features["return_volatility_long"] <= 5.0
    assert 0.0 <= sample_features["return_volatility_ratio"] <= 10.0
    assert -5.0 <= sample_features["return_volatility_diff"] <= 5.0
    assert "parkinson_volatility" in sample_features
    assert "garman_klass_volatility" in sample_features
    assert "rogers_satchell_volatility" in sample_features
    assert "yang_zhang_volatility" in sample_features
    assert 0.0 <= sample_features["parkinson_volatility"] <= 10.0
    assert 0.0 <= sample_features["garman_klass_volatility"] <= 10.0
    assert 0.0 <= sample_features["rogers_satchell_volatility"] <= 10.0
    assert 0.0 <= sample_features["yang_zhang_volatility"] <= 10.0
    assert 0.0 <= sample_features["volume_mean_short_ratio"] <= 20.0
    assert 0.0 <= sample_features["volume_volatility_short"] <= 10.0
    assert 0.0 <= sample_features["volume_volatility_long"] <= 10.0
    assert 0.0 <= sample_features["volume_volatility_ratio"] <= 10.0
    assert -10.0 <= sample_features["volume_zscore_change"] <= 10.0
    assert 0.0 <= sample_features["volume_spike_ratio"] <= 1.0
    assert -1.0 <= sample_features["return_range_correlation"] <= 1.0
    assert -1.0 <= sample_features["return_range_crosscorr_lag1"] <= 1.0
    assert -1.0 <= sample_features["return_range_crosscorr_lag3"] <= 1.0
    assert -1.0 <= sample_features["return_range_crosscorr_lag5"] <= 1.0
    assert -1.0 <= sample_features["return_atr_correlation"] <= 1.0
    assert -1.0 <= sample_features["return_atr_crosscorr_lag1"] <= 1.0
    assert -1.0 <= sample_features["return_atr_crosscorr_lag3"] <= 1.0
    assert -1.0 <= sample_features["return_atr_crosscorr_lag5"] <= 1.0
    assert -1.0 <= sample_features["return_volume_correlation"] <= 1.0
    assert -1.0 <= sample_features["return_volume_crosscorr_lag1"] <= 1.0
    assert -1.0 <= sample_features["return_volume_crosscorr_lag3"] <= 1.0
    assert -1.0 <= sample_features["return_volume_crosscorr_lag5"] <= 1.0
    assert -1.0 <= sample_features["price_volume_correlation"] <= 1.0
    assert 0.0 <= sample_features["return_entropy"] <= 1.0
    assert -1.0 <= sample_features["session_time_sin"] <= 1.0
    assert -1.0 <= sample_features["session_time_cos"] <= 1.0
    assert 0.0 <= sample_features["session_time_progress"] <= 1.0
    assert -1.0 <= sample_features["day_of_week_sin"] <= 1.0
    assert -1.0 <= sample_features["day_of_week_cos"] <= 1.0
    assert 0.0 <= sample_features["is_weekend"] <= 1.0
    assert -1.0 <= sample_features["week_of_year_sin"] <= 1.0
    assert -1.0 <= sample_features["week_of_year_cos"] <= 1.0
    assert 0.0 <= sample_features["week_of_year_progress"] <= 1.0
    assert -1.0 <= sample_features["month_sin"] <= 1.0
    assert -1.0 <= sample_features["month_cos"] <= 1.0
    assert 0.0 <= sample_features["month_progress"] <= 1.0
    assert 0.0 <= sample_features["day_of_month_progress"] <= 1.0
    assert 0.0 <= sample_features["is_month_start"] <= 1.0
    assert 0.0 <= sample_features["is_month_end"] <= 1.0
    assert 0.0 <= sample_features["is_quarter_end"] <= 1.0
    assert 0.0 <= sample_features["hurst_exponent"] <= 1.0
    assert 0.0 <= sample_features["fractal_dimension"] <= 2.0
    assert sample_features["atr_ratio"] >= 0.0
    assert -5.0 <= sample_features["atr_ratio_change"] <= 5.0
    assert -5.0 <= sample_features["price_zscore"] <= 5.0
    assert -10.0 <= sample_features["price_zscore_change"] <= 10.0
    assert -5.0 <= sample_features["price_median_gap"] <= 5.0
    assert -20.0 <= sample_features["price_mad_ratio"] <= 20.0
    assert -5.0 <= sample_features["price_percentile_low_gap"] <= 5.0
    assert -5.0 <= sample_features["price_percentile_high_gap"] <= 5.0
    assert 0.0 <= sample_features["price_percentile_spread"] <= 5.0
    assert 0.0 <= sample_features["price_percent_rank"] <= 100.0
    assert -50.0 <= sample_features["return_median_gap_bps"] <= 50.0
    assert -20.0 <= sample_features["return_mad_ratio"] <= 20.0
    assert -200.0 <= sample_features["return_percentile_low_gap_bps"] <= 200.0
    assert -200.0 <= sample_features["return_percentile_high_gap_bps"] <= 200.0
    assert 0.0 <= sample_features["return_percentile_spread_bps"] <= 400.0
    assert 0.0 <= sample_features["return_percent_rank"] <= 100.0
    assert 0.0 <= sample_features["return_positive_share"] <= 1.0
    assert 0.0 <= sample_features["return_negative_share"] <= 1.0
    assert 0.0 <= sample_features["return_flat_share"] <= 1.0
    assert -1.0 <= sample_features["return_sign_balance"] <= 1.0
    assert 0.0 <= sample_features["return_positive_magnitude_share"] <= 1.0
    assert 0.0 <= sample_features["return_negative_magnitude_share"] <= 1.0
    assert 0.0 <= sample_features["return_flat_magnitude_share"] <= 1.0
    assert -1.0 <= sample_features["return_magnitude_balance"] <= 1.0
    assert 0.0 <= sample_features["volume_percent_rank"] <= 100.0
    assert -5.0 <= sample_features["volume_median_gap"] <= 5.0
    assert -20.0 <= sample_features["volume_mad_ratio"] <= 20.0
    assert -5.0 <= sample_features["volume_return_median_gap"] <= 5.0
    assert -20.0 <= sample_features["volume_return_mad_ratio"] <= 20.0
    assert -5.0 <= sample_features["volume_return_percentile_low_gap"] <= 5.0
    assert -5.0 <= sample_features["volume_return_percentile_high_gap"] <= 5.0
    assert 0.0 <= sample_features["volume_return_percentile_spread"] <= 5.0
    assert 0.0 <= sample_features["volume_return_percent_rank"] <= 100.0
    assert 0.0 <= sample_features["volume_return_positive_share"] <= 1.0
    assert 0.0 <= sample_features["volume_return_negative_share"] <= 1.0
    assert 0.0 <= sample_features["volume_return_flat_share"] <= 1.0
    assert -1.0 <= sample_features["volume_return_sign_balance"] <= 1.0
    assert (
        0.0 <= sample_features["volume_return_positive_magnitude_share"] <= 1.0
    )
    assert (
        0.0 <= sample_features["volume_return_negative_magnitude_share"] <= 1.0
    )
    assert 0.0 <= sample_features["volume_return_flat_magnitude_share"] <= 1.0
    assert -1.0 <= sample_features["volume_return_magnitude_balance"] <= 1.0
    assert -5.0 <= sample_features["volume_percentile_low_gap"] <= 5.0
    assert -5.0 <= sample_features["volume_percentile_high_gap"] <= 5.0
    assert 0.0 <= sample_features["volume_percentile_spread"] <= 5.0
    assert 0.0 <= sample_features["range_percent_rank"] <= 100.0
    assert 0.0 <= sample_features["range_mean_ratio"] <= 10.0
    assert 0.0 <= sample_features["atr_percent_rank"] <= 100.0
    assert -10.0 <= sample_features["return_skewness"] <= 10.0
    assert -10.0 <= sample_features["return_kurtosis"] <= 10.0
    assert sample_features["bollinger_width"] >= 0.0
    assert -3.0 <= sample_features["bollinger_position_change"] <= 3.0
    assert -5.0 <= sample_features["bollinger_width_change"] <= 5.0
    assert -5.0 <= sample_features["atr_trend_slope"] <= 5.0
    assert -1.0 <= sample_features["atr_trend_strength"] <= 1.0
    assert 0.0 <= sample_features["atr_volatility_ratio"] <= 5.0
    assert 0.0 <= sample_features["max_drawdown_ratio"] <= 1.0
    assert 0.0 <= sample_features["max_drawdown_duration"] <= 1.0
    assert -1.0 <= sample_features["drawdown_recovery_ratio"] <= 2.0
    assert abs(sample_features["obv_normalized"]) < 50.0
    assert abs(sample_features["pvt_normalized"]) < 30.0
    assert -5.0 <= sample_features["volume_trend_slope"] <= 5.0
    assert -5.0 <= sample_features["volume_trend_acceleration"] <= 5.0
    assert -1.0 <= sample_features["volume_trend_strength"] <= 1.0
    assert -50.0 <= sample_features["ppo_line"] <= 50.0
    assert -100.0 <= sample_features["ppo_line_change"] <= 100.0
    assert -50.0 <= sample_features["ppo_signal"] <= 50.0
    assert -100.0 <= sample_features["ppo_signal_change"] <= 100.0
    assert -50.0 <= sample_features["ppo_signal_gap"] <= 50.0
    assert -100.0 <= sample_features["ppo_signal_gap_change"] <= 100.0
    assert -50.0 <= sample_features["ppo_histogram"] <= 50.0
    assert -100.0 <= sample_features["ppo_histogram_change"] <= 100.0
    assert -10.0 <= sample_features["macd_histogram_change"] <= 10.0
    assert -5.0 <= sample_features["positive_volume_index"] <= 5.0
    assert -5.0 <= sample_features["negative_volume_index"] <= 5.0
    assert -5.0 <= sample_features["trix"] <= 5.0
    assert -5.0 <= sample_features["trix_change"] <= 5.0
    assert 0.0 <= sample_features["ultimate_oscillator"] <= 100.0
    assert -100.0 <= sample_features["ultimate_oscillator_change"] <= 100.0
    assert -5.0 <= sample_features["ease_of_movement"] <= 5.0
    assert -5.0 <= sample_features["ease_of_movement_change"] <= 5.0
    assert 0.0 <= sample_features["vortex_positive"] <= 5.0
    assert 0.0 <= sample_features["vortex_negative"] <= 5.0
    assert -5.0 <= sample_features["vortex_positive_change"] <= 5.0
    assert -5.0 <= sample_features["vortex_negative_change"] <= 5.0
    assert -5.0 <= sample_features["price_rate_of_change"] <= 5.0
    assert -5.0 <= sample_features["price_rate_of_change_change"] <= 5.0
    assert -5.0 <= sample_features["price_trend_slope"] <= 5.0
    assert -5.0 <= sample_features["price_trend_acceleration"] <= 5.0
    assert -1.0 <= sample_features["price_trend_strength"] <= 1.0
    assert -100.0 <= sample_features["chande_momentum_oscillator"] <= 100.0
    assert -200.0 <= sample_features["chande_momentum_change"] <= 200.0
    assert -5.0 <= sample_features["detrended_price_oscillator"] <= 5.0
    assert -5.0 <= sample_features["detrended_price_oscillator_change"] <= 5.0
    assert 0.0 <= sample_features["aroon_up"] <= 100.0
    assert 0.0 <= sample_features["aroon_down"] <= 100.0
    assert -100.0 <= sample_features["aroon_oscillator"] <= 100.0
    assert -5.0 <= sample_features["balance_of_power"] <= 5.0
    assert 0.0 <= sample_features["stochastic_rsi"] <= 100.0
    assert -100.0 <= sample_features["stochastic_rsi_change"] <= 100.0
    assert -5.0 <= sample_features["relative_vigor_index"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_signal"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_signal_gap"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_index_change"] <= 5.0
    assert -5.0 <= sample_features["relative_vigor_signal_change"] <= 5.0
    assert -150.0 <= sample_features["true_strength_index"] <= 150.0
    assert -150.0 <= sample_features["true_strength_signal"] <= 150.0
    assert -10.0 <= sample_features["true_strength_signal_gap"] <= 10.0
    assert -150.0 <= sample_features["true_strength_index_change"] <= 150.0
    assert -150.0 <= sample_features["true_strength_signal_change"] <= 150.0
    assert 0.0 <= sample_features["connors_rsi"] <= 100.0
    assert -100.0 <= sample_features["connors_rsi_change"] <= 100.0
    assert -5.0 <= sample_features["kama_gap"] <= 5.0
    assert -5.0 <= sample_features["kama_slope"] <= 5.0
    assert -5.0 <= sample_features["kama_gap_change"] <= 5.0
    assert -5.0 <= sample_features["kama_slope_change"] <= 5.0
    assert -5.0 <= sample_features["qstick"] <= 5.0
    assert -5.0 <= sample_features["qstick_change"] <= 5.0
    assert -10.0 <= sample_features["fisher_transform"] <= 10.0
    assert -10.0 <= sample_features["fisher_transform_change"] <= 10.0
    assert -10.0 <= sample_features["fisher_signal_gap"] <= 10.0
    assert -10.0 <= sample_features["fisher_signal_gap_change"] <= 10.0
    assert -10.0 <= sample_features["fisher_signal_change"] <= 10.0
    assert 0.0 <= sample_features["schaff_trend_cycle"] <= 100.0
    assert -100.0 <= sample_features["schaff_trend_cycle_change"] <= 100.0
    assert -5.0 <= sample_features["dema_gap"] <= 5.0
    assert -5.0 <= sample_features["dema_gap_change"] <= 5.0
    assert -5.0 <= sample_features["tema_gap"] <= 5.0
    assert -5.0 <= sample_features["tema_gap_change"] <= 5.0
    assert -5.0 <= sample_features["frama_gap"] <= 5.0
    assert -5.0 <= sample_features["frama_slope"] <= 5.0
    assert -5.0 <= sample_features["frama_gap_change"] <= 5.0
    assert -5.0 <= sample_features["frama_slope_change"] <= 5.0
    assert -5.0 <= sample_features["hma_gap"] <= 5.0
    assert -5.0 <= sample_features["hma_slope"] <= 5.0
    assert -5.0 <= sample_features["hma_gap_change"] <= 5.0
    assert -5.0 <= sample_features["hma_slope_change"] <= 5.0
    assert -5.0 <= sample_features["zlema_gap"] <= 5.0
    assert -5.0 <= sample_features["zlema_gap_change"] <= 5.0
    assert -5.0 <= sample_features["vwma_gap"] <= 5.0
    assert -5.0 <= sample_features["vwma_gap_change"] <= 5.0
    assert -5.0 <= sample_features["t3_gap"] <= 5.0
    assert -5.0 <= sample_features["t3_slope"] <= 5.0
    assert -5.0 <= sample_features["t3_gap_change"] <= 5.0
    assert -5.0 <= sample_features["t3_slope_change"] <= 5.0
    assert -5.0 <= sample_features["supertrend_gap"] <= 5.0
    assert -1.0 <= sample_features["supertrend_direction"] <= 1.0
    assert 0.0 <= sample_features["supertrend_bandwidth"] <= 5.0
    assert -2.0 <= sample_features["supertrend_direction_change"] <= 2.0
    assert -5.0 <= sample_features["supertrend_bandwidth_change"] <= 5.0
    assert -5.0 <= sample_features["supertrend_gap_change"] <= 5.0
    assert -5.0 <= sample_features["elder_ray_bull_power"] <= 5.0
    assert -5.0 <= sample_features["elder_ray_bear_power"] <= 5.0
    assert 0.0 <= sample_features["ulcer_index"] <= 5.0
    assert 0.0 <= sample_features["efficiency_ratio"] <= 1.0
    assert -5.0 <= sample_features["alligator_jaw_gap"] <= 5.0
    assert -5.0 <= sample_features["alligator_jaw_gap_change"] <= 5.0
    assert -5.0 <= sample_features["alligator_teeth_gap"] <= 5.0
    assert -5.0 <= sample_features["alligator_teeth_gap_change"] <= 5.0
    assert -5.0 <= sample_features["alligator_lips_gap"] <= 5.0
    assert -5.0 <= sample_features["alligator_lips_gap_change"] <= 5.0
    assert 0.0 <= sample_features["gator_oscillator_upper"] <= 5.0
    assert -5.0 <= sample_features["gator_oscillator_upper_change"] <= 5.0
    assert 0.0 <= sample_features["gator_oscillator_lower"] <= 5.0
    assert -5.0 <= sample_features["gator_oscillator_lower_change"] <= 5.0
    assert -5.0 <= sample_features["market_facilitation_index"] <= 5.0
    assert 0.0 <= sample_features["di_plus"] <= 100.0
    assert 0.0 <= sample_features["di_minus"] <= 100.0
    assert 0.0 <= sample_features["adx"] <= 100.0
    assert 0.0 <= sample_features["mfi"] <= 100.0
    assert -100.0 <= sample_features["mfi_change"] <= 100.0
    assert -100.0 <= sample_features["adx_change"] <= 100.0
    assert -5.0 <= sample_features["ichimoku_conversion_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_base_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_span_a_gap"] <= 5.0
    assert -5.0 <= sample_features["ichimoku_span_b_gap"] <= 5.0
    assert 0.0 <= sample_features["ichimoku_cloud_thickness"] <= 5.0
    assert -2.0 <= sample_features["ichimoku_price_position"] <= 3.0
    assert -3.0 <= sample_features["ichimoku_price_position_change"] <= 3.0
    assert -1.0 <= sample_features["donchian_position"] <= 2.0
    assert 0.0 <= sample_features["donchian_width"] <= 5.0
    assert -3.0 <= sample_features["donchian_position_change"] <= 3.0
    assert -5.0 <= sample_features["donchian_width_change"] <= 5.0
    assert -1.5 <= sample_features["chaikin_money_flow"] <= 1.5
    assert -2.0 <= sample_features["chaikin_money_flow_change"] <= 2.0
    assert -5.0 <= sample_features["accumulation_distribution"] <= 5.0
    assert -5.0 <= sample_features["chaikin_oscillator"] <= 5.0
    assert -10.0 <= sample_features["chaikin_oscillator_change"] <= 10.0
    assert 0.0 <= sample_features["mass_index"] <= 50.0
    assert -10.0 <= sample_features["mass_index_change"] <= 10.0
    assert -5.0 <= sample_features["klinger_oscillator"] <= 5.0
    assert -5.0 <= sample_features["klinger_oscillator_change"] <= 5.0
    assert -5.0 <= sample_features["klinger_signal"] <= 5.0
    assert -5.0 <= sample_features["klinger_signal_gap"] <= 5.0
    assert -5.0 <= sample_features["klinger_signal_change"] <= 5.0
    assert -5.0 <= sample_features["heikin_trend"] <= 5.0
    assert -5.0 <= sample_features["heikin_trend_change"] <= 5.0
    assert 0.0 <= sample_features["heikin_shadow_ratio"] <= 5.0
    assert -5.0 <= sample_features["heikin_shadow_ratio_change"] <= 5.0
    assert 0.0 <= sample_features["heikin_upper_shadow_ratio"] <= 3.0
    assert 0.0 <= sample_features["heikin_lower_shadow_ratio"] <= 3.0
    assert -1.0 <= sample_features["keltner_position"] <= 2.0
    assert 0.0 <= sample_features["keltner_width"] <= 5.0
    assert -3.0 <= sample_features["keltner_position_change"] <= 3.0
    assert -5.0 <= sample_features["keltner_width_change"] <= 5.0
    assert -5.0 <= sample_features["vwap_gap"] <= 5.0
    assert -5.0 <= sample_features["vwap_gap_change"] <= 5.0
    assert -5.0 <= sample_features["psar_gap"] <= 5.0
    assert -1.0 <= sample_features["psar_direction"] <= 1.0
    assert -2.0 <= sample_features["psar_direction_change"] <= 2.0
    assert -5.0 <= sample_features["pivot_gap"] <= 5.0
    assert -5.0 <= sample_features["pivot_resistance_gap"] <= 5.0
    assert -5.0 <= sample_features["pivot_support_gap"] <= 5.0
    assert -5.0 <= sample_features["fractal_high_gap"] <= 5.0
    assert -5.0 <= sample_features["fractal_low_gap"] <= 5.0
    assert -10.0 <= sample_features["coppock_curve"] <= 10.0
    assert 0.0 <= sample_features["choppiness_index"] <= 100.0
    assert -1.0 <= sample_features["intraday_intensity"] <= 1.0
    assert -2.0 <= sample_features["intraday_intensity_change"] <= 2.0
    assert -5.0 <= sample_features["intraday_intensity_volume"] <= 5.0
    assert -5.0 <= sample_features["intraday_intensity_volume_change"] <= 5.0
    assert -5.0 <= sample_features["open_gap"] <= 5.0
    assert -1.0 <= sample_features["close_location_value"] <= 1.0
    assert -1.0 <= sample_features["candle_body_ratio"] <= 1.0
    assert 0.0 <= sample_features["candle_upper_shadow_ratio"] <= 1.0
    assert 0.0 <= sample_features["candle_lower_shadow_ratio"] <= 1.0
    assert 0.0 <= sample_features["range_atr_ratio"] <= 10.0
    assert 0.0 <= sample_features["body_atr_ratio"] <= 10.0
    assert 0.0 <= sample_features["upper_shadow_atr_ratio"] <= 10.0
    assert 0.0 <= sample_features["lower_shadow_atr_ratio"] <= 10.0
    assert 0.0 <= sample_features["price_up_streak_ratio"] <= 1.0
    assert 0.0 <= sample_features["price_down_streak_ratio"] <= 1.0
    assert 0.0 <= sample_features["volume_up_streak_ratio"] <= 1.0
    assert 0.0 <= sample_features["volume_down_streak_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_high_close_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_low_close_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_high_volume_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_low_volume_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_high_return_ratio"] <= 1.0
    assert 0.0 <= sample_features["time_since_low_return_ratio"] <= 1.0
    assert -1.0 <= sample_features["volume_flow_imbalance"] <= 1.0
    assert -2.0 <= sample_features["volume_flow_imbalance_change"] <= 2.0
    assert 0.0 <= sample_features["volume_flow_ratio"] <= 20.0
    assert -5.0 <= sample_features["market_facilitation_index_change"] <= 5.0

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
