# test_data_preprocessor.py
import numpy as np
import pandas as pd
import pytest

from data_preprocessor import DataPreprocessor, PreprocessorConfig, TradingParameters

def _make_ohlcv(n=200, start=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    price = pd.Series(np.cumsum(np.random.randn(n)) * 0.5 + start, index=idx).abs() + 10.0
    high = price * (1 + np.random.rand(n) * 0.01)
    low = price * (1 - np.random.rand(n) * 0.01)
    open_ = price.shift(1).fillna(price.iloc[0])
    vol = np.random.rand(n) * 1000 + 100
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": price, "volume": vol}, index=idx)

def test_validation_and_indicators():
    df = _make_ohlcv(300)
    prep = DataPreprocessor(PreprocessorConfig(sequence_length=40, cache_indicators=True))
    enriched = prep.add_indicators(df, TradingParameters())
    for col in ["bb_mid","bb_up","bb_dn","rsi","atr","macd","macd_signal","macd_hist","stoch_k","stoch_d","ema_fast","ema_slow","sma_trend"]:
        assert col in enriched.columns
    for col in ["rsi_n","stoch_k_n","stoch_d_n","bb_bwidth"]:
        assert col in enriched.columns

def test_cache_signature():
    df = _make_ohlcv(260)
    prep = DataPreprocessor(PreprocessorConfig(cache_indicators=True))
    p = TradingParameters()
    e1 = prep.add_indicators(df, p)
    e2 = prep.add_indicators(df, p)  # cache hit
    pd.testing.assert_frame_equal(e1, e2)

def test_process_many_dict_and_export():
    df1 = _make_ohlcv(220)
    df2 = _make_ohlcv(240)
    prep = DataPreprocessor()
    out = prep.process_many({"BTCUSDT": df1, "ETHUSDT": df2})
    assert set(out.keys()) == {"BTCUSDT","ETHUSDT"}
    csv_txt = DataPreprocessor.export_df(out["BTCUSDT"], fmt="csv")
    json_txt = DataPreprocessor.export_df(out["BTCUSDT"], fmt="json")
    assert "close" in csv_txt and json_txt.startswith("[")

def test_validation_failures():
    df = _make_ohlcv(50)
    df = df.drop(columns=["high"])
    prep = DataPreprocessor()
    with pytest.raises(ValueError):
        prep.add_indicators(df)
