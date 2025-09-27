# -*- coding: utf-8 -*-
"""DataPreprocessor
=====================
Moduł do przygotowania danych OHLCV pod strategie/AI:
- walidacja wejścia,
- opcjonalne resampling,
- wypełnianie braków,
- obliczanie podstawowych wskaźników (EMA/RSI/MACD/ATR, zwroty),
- (opcjonalnie) skalowanie robust,
- wsparcie batch: `process_many`,
- eksport: `export_df` (csv/json).

Zależności: pandas, numpy (tylko standardowe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple
import numpy as np
import pandas as pd

# Opcjonalnie do typów (nie wymagane do działania)
try:
    from KryptoLowca.strategies import TradingParameters  # type: ignore
except Exception:  # pragma: no cover
    try:
        from strategies import TradingParameters  # type: ignore
    except Exception:
        TradingParameters = object  # type: ignore

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

@dataclass
class PreprocessorConfig:
    resample: Optional[str] = None           # np. '1h', '15min', '1D'
    fillna: Literal["ffill", "bfill", "interpolate", "none"] = "ffill"
    dropna: bool = True
    compute_indicators: bool = True
    indicators: List[str] = field(default_factory=lambda: ["ema20", "ema50", "rsi14", "macd", "atr14", "returns"])
    winsorize_z: Optional[float] = 6.0       # winsoryzacja zwrotów; None = wyłącz
    scale: bool = False                      # robust scaling
    scaler_center_median: bool = True
    scaler_iqr_eps: float = 1e-9

class DataPreprocessor:
    def __init__(self, cfg: Optional[PreprocessorConfig] = None):
        self.cfg = cfg or PreprocessorConfig()
        # Bufory skalera per-kolumna
        self._median_: Dict[str, float] = {}
        self._iqr_: Dict[str, float] = {}

    # ---------- API wysokiego poziomu ----------
    def process(self, df: pd.DataFrame, cfg: Optional[PreprocessorConfig] = None) -> pd.DataFrame:
        """Przetwórz pojedynczy DataFrame OHLCV -> zwraca DataFrame z cechami.
        Wymaga kolumn: open, high, low, close, volume; indeks: datetime-like.
        """
        cfg = cfg or self.cfg
        df = self._validate_df(df.copy())
        if cfg.resample:
            df = self._resample_ohlcv(df, cfg.resample)
        df = self._fillna(df, method=cfg.fillna)
        if cfg.compute_indicators:
            df = self._compute_indicators(df, cfg.indicators)
        if cfg.winsorize_z:
            df = self._winsorize_returns(df, z=cfg.winsorize_z)
        if cfg.dropna:
            df = df.dropna()
        if cfg.scale:
            df = self._robust_scale(df)
        return df

    def process_many(self, data: Dict[str, pd.DataFrame], cfg: Optional[PreprocessorConfig] = None) -> Dict[str, pd.DataFrame]:
        """Przetwórz wiele symboli naraz.

        Example: `out = DataPreprocessor().process_many({"BTCUSDT": df1, "ETHUSDT": df2})`
        """
        cfg = cfg or self.cfg
        out: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            out[sym] = self.process(df, cfg=cfg)
        return out

    @staticmethod
    def export_df(df: pd.DataFrame, fmt: Literal["csv", "json"] = "csv") -> str:
        if fmt == "csv":
            return df.to_csv(index=True)
        elif fmt == "json":
            # indeks na kolumnę, ISO daty
            return df.reset_index().to_json(orient="records", date_format="iso")
        else:
            raise ValueError("fmt must be 'csv' or 'json'")

    # ---------- Walidacje ----------
    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Kolumny
        for c in REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")
        # Indeks czasowy
        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            # spróbuj skonwertować
            try:
                df.index = pd.to_datetime(df.index, utc=False)
            except Exception as e:
                raise ValueError("Index must be datetime-like") from e
        # Monotoniczność
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        # Typy
        for c in REQUIRED_COLS:
            if not np.issubdtype(df[c].dtype, np.number):
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    raise ValueError(f"Column {c} must be numeric")
        return df

    # ---------- Transformacje ----------
    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        return df.resample(rule).agg(agg)

    def _fillna(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        if method == "ffill":
            return df.ffill()
        elif method == "bfill":
            return df.bfill()
        elif method == "interpolate":
            return df.interpolate(limit_direction="both")
        elif method == "none":
            return df
        else:
            raise ValueError("Unsupported fillna method")

    def _compute_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        close = df["close"]
        high, low = df["high"], df["low"]
        # EMA
        if any(ind.startswith("ema") for ind in indicators):
            for p in [5, 10, 20, 50, 100, 200]:
                key = f"ema{p}"
                if key in indicators or key in ("ema20","ema50","ema200"):
                    df[key] = close.ewm(span=p, adjust=False).mean()
        # RSI
        if any(ind.startswith("rsi") for ind in indicators):
            period = 14
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            rs = (avg_gain / (avg_loss.replace(0, np.nan)))
            df[f"rsi{period}"] = 100 - (100 / (1 + rs))
        # MACD
        if "macd" in indicators:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = macd - signal
        # ATR
        if any(ind.startswith("atr") for ind in indicators):
            period = 14
            prev_close = close.shift(1)
            tr = pd.concat([(high - low).abs(),
                            (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
            df[f"atr{period}"] = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        # Zwroty
        if "returns" in indicators:
            df["return"] = close.pct_change().fillna(0.0)
            df["log_return"] = np.log(close).diff().fillna(0.0)
        return df

    def _winsorize_returns(self, df: pd.DataFrame, z: float = 6.0) -> pd.DataFrame:
        if "return" not in df.columns:
            return df
        r = df["return"]
        mu = r.mean()
        sd = r.std(ddof=0) or 1e-9
        lower = mu - z * sd
        upper = mu + z * sd
        df["return"] = r.clip(lower=lower, upper=upper)
        return df

    def _robust_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Skaluje tylko kolumny numeryczne (bez indeksu)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for c in num_cols:
            x = df[c].to_numpy(copy=False)
            med = float(np.nanmedian(x))
            q1 = float(np.nanpercentile(x, 25))
            q3 = float(np.nanpercentile(x, 75))
            iqr = max(q3 - q1, self.cfg.scaler_iqr_eps)
            self._median_[c] = med
            self._iqr_[c] = iqr
            if self.cfg.scaler_center_median:
                df[c] = (x - med) / iqr
            else:
                df[c] = x / iqr
        return df
