"""Feature engineering dla modeli decyzyjnych bazujących na danych OHLCV."""

from __future__ import annotations

import calendar
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.data.base import OHLCVRequest, OHLCVResponse


@dataclass(slots=True)
class FeatureVector:
    """Pojedynczy wektor cech z przypisanym targetem (future return w bps)."""

    timestamp: float
    symbol: str
    features: Mapping[str, float]
    target_bps: float


@dataclass(slots=True)
class FeatureDataset:
    """Znormalizowany zbiór danych gotowy do trenowania modelu."""

    vectors: Sequence[FeatureVector]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.vectors, key=lambda item: (item.timestamp, item.symbol)))
        object.__setattr__(self, "vectors", ordered)

        if isinstance(self.metadata, MutableMapping):
            normalized: MutableMapping[str, object] = dict(self.metadata)
        else:
            normalized = {str(key): value for key, value in self.metadata.items()}

        normalized["row_count"] = len(ordered)
        if ordered:
            normalized["start_timestamp"] = ordered[0].timestamp
            normalized["end_timestamp"] = ordered[-1].timestamp

        feature_names = tuple(sorted({name for vector in ordered for name in vector.features}))
        normalized["feature_names"] = list(feature_names)

        feature_stats: MutableMapping[str, MutableMapping[str, float]] = {}
        if ordered and feature_names:
            for name in feature_names:
                values = [float(vector.features.get(name, 0.0)) for vector in ordered]
                mean_value = sum(values) / len(values)
                stdev_value = pstdev(values) if len(values) > 1 else 0.0
                feature_stats[str(name)] = {
                    "mean": float(mean_value),
                    "stdev": float(stdev_value),
                    "min": float(min(values) if values else 0.0),
                    "max": float(max(values) if values else 0.0),
                }
        normalized["feature_stats"] = dict(feature_stats)

        object.__setattr__(self, "metadata", normalized)

    @property
    def features(self) -> list[Mapping[str, float]]:
        return [vector.features for vector in self.vectors]

    @property
    def targets(self) -> list[float]:
        return [vector.target_bps for vector in self.vectors]

    @property
    def timestamps(self) -> list[float]:
        return [vector.timestamp for vector in self.vectors]

    @property
    def target_scale(self) -> float:
        """Zwraca skalę targetu (odchylenie standardowe) dla konwersji na prawdopodobieństwo."""

        targets = self.targets
        if not targets:
            return 1.0
        dispersion = pstdev(targets)
        if math.isfinite(dispersion) and dispersion > 0:
            return dispersion
        return max(abs(mean(targets)) or 1.0, 1.0)

    @property
    def feature_names(self) -> Sequence[str]:
        raw = self.metadata.get("feature_names") if isinstance(self.metadata, Mapping) else None
        if isinstance(raw, Sequence):
            return tuple(str(name) for name in raw)
        return tuple(sorted({name for vector in self.vectors for name in vector.features}))

    @property
    def feature_stats(self) -> Mapping[str, Mapping[str, float]]:
        raw = self.metadata.get("feature_stats") if isinstance(self.metadata, Mapping) else None
        if isinstance(raw, Mapping):
            return {str(name): dict(stats) for name, stats in raw.items() if isinstance(stats, Mapping)}
        summary: MutableMapping[str, Mapping[str, float]] = {}
        for name in self.feature_names:
            values = [float(vector.features.get(name, 0.0)) for vector in self.vectors]
            summary[name] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "stdev": pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        return summary

    def to_learning_arrays(self) -> tuple[list[list[float]], list[float], list[str]]:
        """Zwraca macierz cech, listę targetów i uporządkowane nazwy cech."""

        feature_names = list(self.feature_names)
        matrix = [
            [float(vector.features.get(name, 0.0)) for name in feature_names]
            for vector in self.vectors
        ]
        return matrix, self.targets, feature_names

    def subset(self, indices: Sequence[int]) -> "FeatureDataset":
        """Zwraca nowy zbiór danych zawierający wybrane wektory."""

        if not indices:
            return FeatureDataset(vectors=(), metadata=dict(self.metadata))

        unique = sorted({idx for idx in (int(i) for i in indices) if 0 <= idx < len(self.vectors)})
        selected = [self.vectors[idx] for idx in unique]
        metadata = dict(self.metadata)
        subset = FeatureDataset(vectors=tuple(selected), metadata=metadata)
        subset.metadata["target_scale"] = subset.target_scale
        return subset


class FeatureEngineer:
    """Buduje cechy i targety dla modeli AI Decision Engine."""

    def __init__(
        self,
        data_source: "CachedOHLCVSource",
        *,
        feature_window: int = 10,
        target_horizon: int = 1,
    ) -> None:
        if feature_window <= 1:
            raise ValueError("feature_window musi być większe od 1")
        if target_horizon <= 0:
            raise ValueError("target_horizon musi być dodatni")
        self._source = data_source
        self._window = feature_window
        self._horizon = target_horizon

    def build_dataset(
        self,
        *,
        symbols: Sequence[str],
        interval: str,
        start: int,
        end: int,
    ) -> FeatureDataset:
        """Buduje zbiór danych na podstawie historii OHLCV."""

        vectors: list[FeatureVector] = []
        for symbol in symbols:
            request = OHLCVRequest(symbol=symbol, interval=interval, start=start, end=end)
            response = self._source.fetch_ohlcv(request)
            vectors.extend(self._build_symbol_vectors(symbol, response))

        metadata: MutableMapping[str, object] = {
            "symbols": sorted({str(symbol) for symbol in symbols}),
            "interval": interval,
            "start": start,
            "end": end,
            "feature_window": self._window,
            "target_horizon": self._horizon,
        }
        dataset = FeatureDataset(vectors=tuple(vectors), metadata=metadata)
        dataset.metadata["target_scale"] = dataset.target_scale
        dataset.metadata["row_count"] = len(dataset.vectors)
        return dataset

    # ------------------------------------------------------------------ helpers --
    def _build_symbol_vectors(self, symbol: str, response: OHLCVResponse) -> Iterable[FeatureVector]:
        columns = list(response.columns)
        column_map = {name: idx for idx, name in enumerate(columns)}
        required = ["open_time", "open", "high", "low", "close", "volume"]
        for item in required:
            if item not in column_map:
                raise KeyError(f"Kolumna {item} nie występuje w odpowiedzi OHLCV")

        rows = list(response.rows)
        if len(rows) <= self._window + self._horizon:
            return []

        closes = [float(row[column_map["close"]]) for row in rows]
        volumes = [float(row[column_map["volume"]]) for row in rows]
        highs = [float(row[column_map["high"]]) for row in rows]
        lows = [float(row[column_map["low"]]) for row in rows]
        opens = [float(row[column_map["open"]]) for row in rows]
        timestamps = [float(row[column_map["open_time"]]) for row in rows]

        vectors: list[FeatureVector] = []
        stochastic_history: list[float] = []
        obv_value = 0.0
        pvt_value = 0.0
        fisher_trigger = 0.0
        fisher_value = 0.0
        fisher_signal = 0.0
        frama_value = closes[self._window - 1]
        schaff_ema_fast = closes[self._window - 1]
        schaff_ema_slow = closes[self._window - 1]
        schaff_macd_history: list[float] = []
        schaff_stoch_ema = 50.0
        schaff_raw_value = 50.0
        pvi_value = 1_000.0
        nvi_value = 1_000.0
        pvi_history: list[float] = []
        nvi_history: list[float] = []
        coppock_history: list[float] = []
        adl_value = 0.0
        adl_history: list[float] = []
        range_history: list[float] = []
        atr_history: list[float] = []
        mass_range_history: list[float] = []
        mass_single_history: list[float] = []
        mass_ratio_history: list[float] = []
        klinger_volume_history: list[float] = []
        klinger_oscillator_history: list[float] = []
        klinger_trend = 1.0
        median_initial = float((highs[0] + lows[0]) / 2.0)
        alligator_jaw = median_initial
        alligator_teeth = median_initial
        alligator_lips = median_initial
        supertrend_period = max(3, min(self._window, 10))
        supertrend_multiplier = 3.0
        supertrend_value = float(closes[self._window - 1])
        supertrend_direction = 1.0
        supertrend_upper = float(closes[self._window - 1])
        supertrend_lower = float(closes[self._window - 1])
        hma_period = max(5, self._window)
        previous_hma = self._hma(closes[: self._window], hma_period)
        t3_period = max(5, self._window)
        previous_t3 = self._t3(closes[: self._window], t3_period)
        heikin_open_prev = float(
            (opens[self._window - 1] + closes[self._window - 1]) / 2.0
        )
        heikin_close_prev = float(
            (opens[self._window - 1]
            + highs[self._window - 1]
            + lows[self._window - 1]
            + closes[self._window - 1])
            / 4.0
        )
        prev_typical_price = float(
            (highs[0] + lows[0] + closes[0]) / 3.0
        )
        for warm_index in range(self._window):
            high_value = highs[warm_index]
            low_value = lows[warm_index]
            close_value = closes[warm_index]
            volume_value = volumes[warm_index]
            typical_price = (high_value + low_value + close_value) / 3.0
            multiplier = self._money_flow_multiplier(high_value, low_value, close_value)
            money_flow_volume = multiplier * volume_value
            adl_value += money_flow_volume
            adl_history.append(adl_value)
            range_history.append(max(0.0, high_value - low_value))
            if warm_index > 0:
                warm_slice = slice(max(0, warm_index - self._window + 1), warm_index)
                atr_history.append(
                    self._average_true_range(
                        highs[warm_slice],
                        lows[warm_slice],
                        closes[warm_slice],
                        closes[warm_index - 1],
                        high_value,
                        low_value,
                    )
                )
            else:
                atr_history.append(max(0.0, high_value - low_value))
            mass_range_history.append(max(0.0, high_value - low_value))
            mass_single = self._ema(mass_range_history, 9)
            mass_single_history.append(mass_single)
            mass_double = self._ema(mass_single_history, 9)
            ratio = mass_single / mass_double if mass_double > 0 else 0.0
            mass_ratio_history.append(ratio)
            trend = klinger_trend
            if typical_price > prev_typical_price:
                trend = 1.0
            elif typical_price < prev_typical_price:
                trend = -1.0
            klinger_trend = trend
            volume_force = money_flow_volume * trend
            klinger_volume_history.append(volume_force)
            klinger_raw = self._ema(klinger_volume_history, 34) - self._ema(
                klinger_volume_history, 55
            )
            klinger_oscillator_history.append(klinger_raw)
            prev_typical_price = typical_price
            median_price = (high_value + low_value) / 2.0
            alligator_jaw = self._smma_next(alligator_jaw, median_price, 13)
            alligator_teeth = self._smma_next(alligator_teeth, median_price, 8)
            alligator_lips = self._smma_next(alligator_lips, median_price, 5)
            if warm_index > 0:
                (
                    supertrend_value,
                    supertrend_direction,
                    supertrend_upper,
                    supertrend_lower,
                ) = self._supertrend_step(
                    highs,
                    lows,
                    closes,
                    warm_index,
                    supertrend_value,
                    supertrend_direction,
                    supertrend_upper,
                    supertrend_lower,
                    supertrend_period,
                    supertrend_multiplier,
                )
            previous_hma = self._hma(
                closes[: warm_index + 1],
                hma_period,
            )
            previous_t3 = self._t3(
                closes[: warm_index + 1],
                t3_period,
            )

        for index in range(self._window, len(rows) - self._horizon):
            lookback_slice = slice(index - self._window, index)
            inclusive_slice = slice(index - self._window, index + 1)
            prev_close = closes[index - 1]
            current_close = closes[index]
            next_close = closes[index + self._horizon]
            if prev_close <= 0:
                continue
            if current_close <= 0:
                continue

            current_timestamp = timestamps[index]
            current_dt = datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
            day_seconds = (
                current_dt.hour * 3600
                + current_dt.minute * 60
                + current_dt.second
            )
            session_progress = day_seconds / 86_400.0
            session_progress = max(0.0, min(1.0, session_progress))
            session_angle = 2.0 * math.pi * session_progress
            session_time_sin = math.sin(session_angle)
            session_time_cos = math.cos(session_angle)

            day_of_week = current_dt.weekday()
            day_angle = 2.0 * math.pi * (day_of_week / 7.0)
            day_of_week_sin = math.sin(day_angle)
            day_of_week_cos = math.cos(day_angle)
            is_weekend = 1.0 if day_of_week >= 5 else 0.0

            _, iso_week, _ = current_dt.isocalendar()
            week_progress = (iso_week - 1) / 52.0 if iso_week else 0.0
            week_progress = max(0.0, min(1.0, week_progress))
            week_angle = 2.0 * math.pi * week_progress
            week_of_year_sin = math.sin(week_angle)
            week_of_year_cos = math.cos(week_angle)

            month = current_dt.month
            month_angle = 2.0 * math.pi * ((month - 1) / 12.0)
            month_sin = math.sin(month_angle)
            month_cos = math.cos(month_angle)
            month_progress = (month - 1) / 11.0 if month > 1 else 0.0
            month_progress = max(0.0, min(1.0, month_progress))

            days_in_month = calendar.monthrange(current_dt.year, month)[1]
            day_of_month = current_dt.day
            denominator = float(max(days_in_month - 1, 1))
            day_of_month_progress = (day_of_month - 1) / denominator
            day_of_month_progress = max(0.0, min(1.0, day_of_month_progress))
            is_month_start = 1.0 if day_of_month == 1 else 0.0
            is_month_end = 1.0 if day_of_month == days_in_month else 0.0
            is_quarter_end = 1.0 if month in {3, 6, 9, 12} and is_month_end else 0.0

            lookback_closes = closes[lookback_slice]
            lookback_closes_with_current = closes[inclusive_slice]
            returns = self._compute_returns(closes, lookback_slice)
            volatility = pstdev(returns) if len(returns) > 1 else 0.0
            mean_return = mean(returns) if returns else 0.0
            volume_window = volumes[lookback_slice]
            avg_volume = mean(volume_window) if volume_window else 0.0
            current_volume = volumes[index]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            prev_volume = volumes[index - 1]
            if prev_close > 0:
                close_change = (current_close - prev_close) / prev_close
                if current_volume > prev_volume:
                    pvi_value += pvi_value * close_change
                elif current_volume < prev_volume:
                    nvi_value += nvi_value * close_change
            pvi_history.append(pvi_value)
            nvi_history.append(nvi_value)
            pvi_gap = self._normalized_index_gap(pvi_history)
            nvi_gap = self._normalized_index_gap(nvi_history)
            price_range = (max(highs[lookback_slice]) - min(lows[lookback_slice])) / prev_close
            open_close_diff = (opens[index] - current_close) / current_close
            highest_high = max(highs[lookback_slice]) if volume_window else highs[index]
            lowest_low = min(lows[lookback_slice]) if volume_window else lows[index]
            range_span = highest_high - lowest_low

            target_return = ((next_close - current_close) / current_close) * 10_000.0
            ema_fast = self._ema(lookback_closes, max(2, self._window // 2))
            ema_slow = self._ema(lookback_closes, self._window)
            ema_fast_gap = (current_close - ema_fast) / ema_fast if ema_fast > 0 else 0.0
            ema_slow_gap = (current_close - ema_slow) / ema_slow if ema_slow > 0 else 0.0
            rsi = self._compute_rsi(lookback_closes)
            volume_zscore = 0.0
            if volume_window:
                vol_stdev = pstdev(volume_window) if len(volume_window) > 1 else 0.0
                if vol_stdev > 0:
                    volume_zscore = (volumes[index] - avg_volume) / vol_stdev
            vol_trend = self._volatility_trend(returns)
            atr = self._average_true_range(
                highs[lookback_slice],
                lows[lookback_slice],
                closes[lookback_slice],
                prev_close,
                highs[index],
                lows[index],
            )
            atr_history.append(atr)
            bollinger_position, bollinger_width = self._bollinger_metrics(
                lookback_closes_with_current, current_close
            )
            macd_line, macd_signal, macd_hist = self._macd_components(lookback_closes_with_current)
            ppo_line, ppo_signal, ppo_hist = self._ppo_components(closes[: index + 1])
            ppo_signal_gap = self._clamp(ppo_line - ppo_signal, -50.0, 50.0)

            fisher_trigger, fisher_value, fisher_signal = self._fisher_transform(
                closes,
                index,
                fisher_trigger,
                fisher_value,
                fisher_signal,
            )
            fisher_signal_gap = self._clamp(fisher_value - fisher_signal, -10.0, 10.0)

            (
                schaff_ema_fast,
                schaff_ema_slow,
                schaff_stoch_ema,
                schaff_raw_value,
                schaff_cycle,
            ) = self._schaff_trend_cycle(
                current_close,
                schaff_macd_history,
                schaff_ema_fast,
                schaff_ema_slow,
                schaff_stoch_ema,
                schaff_raw_value,
            )

            frama_value, frama_gap, frama_slope = self._frama_metrics(
                highs,
                lows,
                closes,
                index,
                frama_value,
            )

            (
                supertrend_value,
                supertrend_direction,
                supertrend_upper,
                supertrend_lower,
            ) = self._supertrend_step(
                highs,
                lows,
                closes,
                index,
                supertrend_value,
                supertrend_direction,
                supertrend_upper,
                supertrend_lower,
                supertrend_period,
                supertrend_multiplier,
            )

            hma_value = self._hma(closes[: index + 1], hma_period)
            zlema_value = self._zlema(closes[: index + 1], hma_period)
            t3_value = self._t3(closes[: index + 1], t3_period)
            hma_gap = 0.0
            hma_slope = 0.0
            zlema_gap = 0.0
            t3_gap = 0.0
            t3_slope = 0.0
            if current_close > 0:
                hma_gap = (current_close - hma_value) / current_close
                zlema_gap = (current_close - zlema_value) / current_close
                hma_slope = (hma_value - previous_hma) / current_close
                if t3_value != 0.0:
                    t3_gap = (current_close - t3_value) / current_close
                if math.isfinite(previous_t3):
                    t3_slope = (t3_value - previous_t3) / current_close
            hma_gap = self._clamp(hma_gap, -5.0, 5.0)
            hma_slope = self._clamp(hma_slope, -5.0, 5.0)
            zlema_gap = self._clamp(zlema_gap, -5.0, 5.0)
            t3_gap = self._clamp(t3_gap, -5.0, 5.0)
            t3_slope = self._clamp(t3_slope, -5.0, 5.0)
            previous_hma = hma_value
            previous_t3 = t3_value

            supertrend_gap = 0.0
            supertrend_bandwidth = 0.0
            if current_close > 0:
                supertrend_gap = (current_close - supertrend_value) / current_close
                supertrend_bandwidth = (supertrend_upper - supertrend_lower) / current_close
            supertrend_gap = self._clamp(supertrend_gap, -5.0, 5.0)
            supertrend_bandwidth = self._clamp(supertrend_bandwidth, 0.0, 5.0)
            supertrend_direction_clamped = self._clamp(supertrend_direction, -1.0, 1.0)

            choppiness_index = self._choppiness_index(highs, lows, closes, index)
            roc_short_period = min(11, index)
            roc_long_period = min(14, index)
            coppock_input = 0.0
            if roc_short_period > 0 and roc_long_period > 0:
                roc_short = self._rate_of_change(closes, index, roc_short_period)
                roc_long = self._rate_of_change(closes, index, roc_long_period)
                coppock_input = roc_short + roc_long
            coppock_history.append(coppock_input)
            coppock_curve = self._coppock_curve(coppock_history)
            intraday_intensity = self._intraday_intensity(highs[index], lows[index], current_close)
            intraday_intensity_volume = self._clamp(
                intraday_intensity * volume_ratio,
                -5.0,
                5.0,
            )

            market_facilitation_index = 0.0
            if current_volume > 0:
                market_facilitation_index = self._clamp(
                    (highs[index] - lows[index]) / current_volume,
                    -5.0,
                    5.0,
                )

            money_flow_multiplier = self._money_flow_multiplier(
                highs[index], lows[index], current_close
            )
            money_flow_volume = money_flow_multiplier * current_volume
            adl_value += money_flow_volume
            adl_history.append(adl_value)
            accumulation_distribution = self._normalized_index_gap(adl_history)
            chaikin_short = self._ema(adl_history, 3)
            chaikin_long = self._ema(adl_history, 10)
            chaikin_scale = max(abs(chaikin_long), 1.0)
            chaikin_oscillator = self._clamp(
                (chaikin_short - chaikin_long) / chaikin_scale,
                -5.0,
                5.0,
            )

            mass_range_history.append(max(0.0, highs[index] - lows[index]))
            mass_single = self._ema(mass_range_history, 9)
            mass_single_history.append(mass_single)
            mass_double = self._ema(mass_single_history, 9)
            mass_ratio = mass_single / mass_double if mass_double > 0 else 0.0
            mass_ratio_history.append(mass_ratio)
            mass_index_value = self._mass_index(mass_ratio_history)

            typical_price = (highs[index] + lows[index] + current_close) / 3.0
            trend = klinger_trend
            if typical_price > prev_typical_price:
                trend = 1.0
            elif typical_price < prev_typical_price:
                trend = -1.0
            klinger_trend = trend
            volume_force = money_flow_volume * trend
            klinger_volume_history.append(volume_force)
            klinger_short = self._ema(klinger_volume_history, 34)
            klinger_long = self._ema(klinger_volume_history, 55)
            klinger_raw = klinger_short - klinger_long
            klinger_oscillator_history.append(klinger_raw)
            klinger_signal_raw = self._ema(klinger_oscillator_history, 13)
            klinger_scale = max(abs(klinger_long), 1.0)
            klinger_oscillator = self._clamp(klinger_raw / klinger_scale, -5.0, 5.0)
            klinger_signal = self._clamp(klinger_signal_raw / klinger_scale, -5.0, 5.0)
            klinger_signal_gap = self._clamp(
                klinger_oscillator - klinger_signal,
                -5.0,
                5.0,
            )
            prev_typical_price = typical_price
            median_price = (highs[index] + lows[index]) / 2.0
            alligator_jaw = self._smma_next(alligator_jaw, median_price, 13)
            alligator_teeth = self._smma_next(alligator_teeth, median_price, 8)
            alligator_lips = self._smma_next(alligator_lips, median_price, 5)
            alligator_jaw_gap = 0.0
            alligator_teeth_gap = 0.0
            alligator_lips_gap = 0.0
            gator_upper = 0.0
            gator_lower = 0.0
            if current_close > 0:
                alligator_jaw_gap = self._clamp(
                    (current_close - alligator_jaw) / current_close,
                    -5.0,
                    5.0,
                )
                alligator_teeth_gap = self._clamp(
                    (current_close - alligator_teeth) / current_close,
                    -5.0,
                    5.0,
                )
                alligator_lips_gap = self._clamp(
                    (current_close - alligator_lips) / current_close,
                    -5.0,
                    5.0,
                )
                gator_upper = self._clamp(
                    abs(alligator_jaw - alligator_teeth) / current_close,
                    0.0,
                    5.0,
                )
                gator_lower = self._clamp(
                    abs(alligator_teeth - alligator_lips) / current_close,
                    0.0,
                    5.0,
                )

            dema_value = self._dema(lookback_closes_with_current, self._window)
            tema_value = self._tema(lookback_closes_with_current, self._window)
            dema_gap = 0.0
            tema_gap = 0.0
            if current_close > 0:
                dema_gap = (current_close - dema_value) / current_close
                tema_gap = (current_close - tema_value) / current_close
            dema_gap = self._clamp(dema_gap, -5.0, 5.0)
            tema_gap = self._clamp(tema_gap, -5.0, 5.0)

            trix = self._trix(lookback_closes_with_current)
            ultimate_oscillator = self._ultimate_oscillator(highs, lows, closes, index)
            ease_of_movement = self._ease_of_movement(highs, lows, volumes, index)
            vortex_positive, vortex_negative = self._vortex_indicator(highs, lows, closes, index)
            price_rate_of_change = self._rate_of_change(closes, index, min(self._window, index))
            chande_momentum = self._chande_momentum_oscillator(closes, index)
            detrended_price_oscillator = self._detrended_price_oscillator(closes, index)
            aroon_up, aroon_down = self._aroon(highs, lows, index)
            balance_of_power = self._balance_of_power(
                opens[index], highs[index], lows[index], current_close
            )
            stochastic_rsi_value = self._stochastic_rsi(closes, index, rsi)
            relative_vigor_index, relative_vigor_signal = self._relative_vigor_index(
                opens, highs, lows, closes, index
            )
            elder_bull_power, elder_bear_power = self._elder_ray(
                lookback_closes_with_current, highs[index], lows[index], current_close
            )
            ulcer_index = self._ulcer_index(lookback_closes_with_current)
            efficiency_ratio = self._efficiency_ratio(closes, index)
            true_strength_index, true_strength_signal = self._true_strength_index(closes, index)
            connors_rsi = self._connors_rsi(closes, index)
            kama_gap, kama_slope = self._kama_metrics(closes, index)
            qstick = self._qstick(opens, closes, index)

            ichimoku_conversion = self._rolling_midpoint(highs, lows, index, 9)
            ichimoku_base = self._rolling_midpoint(highs, lows, index, 26)
            ichimoku_span_b = self._rolling_midpoint(highs, lows, index, 52)
            ichimoku_span_a = (ichimoku_conversion + ichimoku_base) / 2.0
            cloud_top = max(ichimoku_span_a, ichimoku_span_b)
            cloud_bottom = min(ichimoku_span_a, ichimoku_span_b)
            ichimoku_conversion_gap = 0.0
            ichimoku_base_gap = 0.0
            ichimoku_span_a_gap = 0.0
            ichimoku_span_b_gap = 0.0
            ichimoku_cloud_thickness = 0.0
            ichimoku_price_position = 0.0
            if current_close > 0:
                ichimoku_conversion_gap = (current_close - ichimoku_conversion) / current_close
                ichimoku_base_gap = (current_close - ichimoku_base) / current_close
                ichimoku_span_a_gap = (current_close - ichimoku_span_a) / current_close
                ichimoku_span_b_gap = (current_close - ichimoku_span_b) / current_close
                cloud_span = cloud_top - cloud_bottom
                if cloud_top > 0:
                    ichimoku_cloud_thickness = cloud_span / cloud_top
                if cloud_span != 0:
                    ichimoku_price_position = (current_close - cloud_bottom) / cloud_span

            ichimoku_conversion_gap = self._clamp(ichimoku_conversion_gap, -5.0, 5.0)
            ichimoku_base_gap = self._clamp(ichimoku_base_gap, -5.0, 5.0)
            ichimoku_span_a_gap = self._clamp(ichimoku_span_a_gap, -5.0, 5.0)
            ichimoku_span_b_gap = self._clamp(ichimoku_span_b_gap, -5.0, 5.0)
            ichimoku_cloud_thickness = self._clamp(ichimoku_cloud_thickness, 0.0, 5.0)
            ichimoku_price_position = self._clamp(ichimoku_price_position, -2.0, 3.0)

            donchian_high, donchian_low = self._donchian_channel(highs, lows, index, self._window * 2)
            donchian_position = 0.0
            donchian_width = 0.0
            if donchian_high > donchian_low:
                donchian_position = (current_close - donchian_low) / (donchian_high - donchian_low)
                donchian_width = (donchian_high - donchian_low) / current_close if current_close > 0 else 0.0
            donchian_position = self._clamp(donchian_position, -1.0, 2.0)
            donchian_width = self._clamp(donchian_width, 0.0, 5.0)

            chaikin_money_flow = self._chaikin_money_flow(highs, lows, closes, volumes, index)

            heikin_close = (opens[index] + highs[index] + lows[index] + current_close) / 4.0
            heikin_open = (heikin_open_prev + heikin_close_prev) / 2.0
            heikin_high = max(highs[index], heikin_open, heikin_close)
            heikin_low = min(lows[index], heikin_open, heikin_close)
            heikin_range = heikin_high - heikin_low
            heikin_body = heikin_close - heikin_open
            heikin_trend = 0.0
            heikin_shadow_ratio = 0.0
            heikin_upper_shadow = max(0.0, heikin_high - max(heikin_close, heikin_open))
            heikin_lower_shadow = max(0.0, min(heikin_close, heikin_open) - heikin_low)
            if heikin_open != 0.0:
                heikin_trend = heikin_body / heikin_open
            if heikin_range > 0 and current_close > 0:
                heikin_shadow_ratio = heikin_range / current_close
            heikin_trend = self._clamp(heikin_trend, -5.0, 5.0)
            heikin_shadow_ratio = self._clamp(heikin_shadow_ratio, 0.0, 5.0)
            heikin_upper_shadow = self._clamp(
                heikin_upper_shadow / current_close if current_close > 0 else 0.0, 0.0, 3.0
            )
            heikin_lower_shadow = self._clamp(
                heikin_lower_shadow / current_close if current_close > 0 else 0.0, 0.0, 3.0
            )

            heikin_open_prev = heikin_open
            heikin_close_prev = heikin_close

            keltner_basis = self._ema(lookback_closes_with_current, self._window)
            keltner_multiplier = 1.5
            keltner_upper = keltner_basis + keltner_multiplier * atr
            keltner_lower = keltner_basis - keltner_multiplier * atr
            keltner_position = 0.0
            keltner_width = 0.0
            if keltner_upper > keltner_lower and current_close > 0:
                channel_span = keltner_upper - keltner_lower
                keltner_position = (current_close - keltner_lower) / channel_span
                keltner_width = channel_span / current_close
            keltner_position = self._clamp(keltner_position, -1.0, 2.0)
            keltner_width = self._clamp(keltner_width, 0.0, 5.0)

            vwap_gap = 0.0
            vwap_volume_window = volumes[inclusive_slice]
            if vwap_volume_window:
                vwap_numerator = 0.0
                vwap_denominator = 0.0
                for pos in range(inclusive_slice.start, inclusive_slice.stop):
                    vwap_numerator += closes[pos] * volumes[pos]
                    vwap_denominator += volumes[pos]
                if vwap_denominator > 0 and current_close > 0:
                    vwap = vwap_numerator / vwap_denominator
                    vwap_gap = (current_close - vwap) / current_close
            vwap_gap = self._clamp(vwap_gap, -5.0, 5.0)

            psar_value, psar_direction = self._parabolic_sar(highs, lows, closes, index)
            psar_gap = 0.0
            if current_close > 0:
                psar_gap = (current_close - psar_value) / current_close
            psar_gap = self._clamp(psar_gap, -5.0, 5.0)
            psar_direction = self._clamp(psar_direction, -1.0, 1.0)

            pivot_high = max(highs[lookback_slice]) if lookback_slice.stop > lookback_slice.start else highs[index - 1]
            pivot_low = min(lows[lookback_slice]) if lookback_slice.stop > lookback_slice.start else lows[index - 1]
            pivot_close = closes[index - 1]
            pivot_point = (pivot_high + pivot_low + pivot_close) / 3.0
            pivot_resistance = (2.0 * pivot_point) - pivot_low
            pivot_support = (2.0 * pivot_point) - pivot_high
            pivot_gap = 0.0
            pivot_resistance_gap = 0.0
            pivot_support_gap = 0.0
            if current_close > 0:
                pivot_gap = (current_close - pivot_point) / current_close
                pivot_resistance_gap = (current_close - pivot_resistance) / current_close
                pivot_support_gap = (current_close - pivot_support) / current_close
            pivot_gap = self._clamp(pivot_gap, -5.0, 5.0)
            pivot_resistance_gap = self._clamp(pivot_resistance_gap, -5.0, 5.0)
            pivot_support_gap = self._clamp(pivot_support_gap, -5.0, 5.0)

            fractal_high = self._recent_fractal(highs, index, is_high=True)
            fractal_low = self._recent_fractal(lows, index, is_high=False)
            fractal_high_gap = 0.0
            fractal_low_gap = 0.0
            if fractal_high is not None and current_close > 0:
                fractal_high_gap = (current_close - fractal_high) / current_close
            if fractal_low is not None and current_close > 0:
                fractal_low_gap = (current_close - fractal_low) / current_close
            fractal_high_gap = self._clamp(fractal_high_gap, -5.0, 5.0)
            fractal_low_gap = self._clamp(fractal_low_gap, -5.0, 5.0)

            directional_start = max(1, index - self._window + 1)
            dm_plus_sum = 0.0
            dm_minus_sum = 0.0
            tr_sum = 0.0
            dx_values: list[float] = []
            money_flow_positive = 0.0
            money_flow_negative = 0.0
            for pos in range(directional_start, index + 1):
                prev_pos = pos - 1
                high_value = highs[pos]
                low_value = lows[pos]
                prev_high = highs[prev_pos]
                prev_low = lows[prev_pos]
                prev_close_value = closes[prev_pos]
                tr = max(
                    high_value - low_value,
                    abs(high_value - prev_close_value),
                    abs(low_value - prev_close_value),
                )
                tr_sum += tr
                up_move = high_value - prev_high
                down_move = prev_low - low_value
                dm_plus = up_move if up_move > down_move and up_move > 0 else 0.0
                dm_minus = down_move if down_move > up_move and down_move > 0 else 0.0
                dm_plus_sum += dm_plus
                dm_minus_sum += dm_minus
                if tr > 0:
                    di_plus_component = (dm_plus / tr) * 100.0
                    di_minus_component = (dm_minus / tr) * 100.0
                    denominator = di_plus_component + di_minus_component
                    if denominator > 0:
                        dx = abs(di_plus_component - di_minus_component) / denominator * 100.0
                        dx_values.append(dx)

                typical_price_pos = (high_value + low_value + closes[pos]) / 3.0
                prev_typical_price = (prev_high + prev_low + closes[prev_pos]) / 3.0
                raw_money_flow = typical_price_pos * volumes[pos]
                if typical_price_pos > prev_typical_price:
                    money_flow_positive += raw_money_flow
                elif typical_price_pos < prev_typical_price:
                    money_flow_negative += raw_money_flow

            di_plus = 0.0
            di_minus = 0.0
            adx = 0.0
            if tr_sum > 0:
                di_plus = 100.0 * (dm_plus_sum / tr_sum)
                di_minus = 100.0 * (dm_minus_sum / tr_sum)
            if dx_values:
                adx = sum(dx_values) / len(dx_values)

            mfi = 50.0
            if money_flow_negative > 0 and money_flow_positive >= 0:
                ratio = money_flow_positive / money_flow_negative
                mfi = 100.0 - (100.0 / (1.0 + ratio))
            elif money_flow_positive > 0 and money_flow_negative == 0:
                mfi = 100.0
            mfi = max(0.0, min(100.0, mfi))

            force_index = (current_close - prev_close) * volumes[index]
            force_index_normalized = 0.0
            if avg_volume > 0 and prev_close > 0:
                force_index_normalized = force_index / (avg_volume * prev_close)

            stochastic_k = 50.0
            if range_span > 0:
                stochastic_k = ((current_close - lowest_low) / range_span) * 100.0
            stochastic_k = max(0.0, min(100.0, stochastic_k))
            stochastic_history.append(stochastic_k)
            recent_k = stochastic_history[-3:]
            stochastic_d = mean(recent_k) if recent_k else stochastic_k

            williams_r = 0.0
            if range_span > 0:
                williams_r = -100.0 * (highest_high - current_close) / range_span

            typical_price = (highs[index] + lows[index] + current_close) / 3.0
            lookback_typical = [
                (highs[pos] + lows[pos] + closes[pos]) / 3.0 for pos in range(index - self._window, index)
            ]
            cci = 0.0
            if lookback_typical:
                tp_mean = mean(lookback_typical)
                mean_dev = mean(abs(tp - tp_mean) for tp in lookback_typical)
                if mean_dev > 0:
                    cci = (typical_price - tp_mean) / (0.015 * mean_dev)

            if current_close > prev_close:
                obv_value += volumes[index]
            elif current_close < prev_close:
                obv_value -= volumes[index]
            obv_normalized = 0.0
            if avg_volume > 0 and self._window > 0:
                obv_normalized = obv_value / (avg_volume * self._window)

            if prev_close > 0:
                pvt_value += ((current_close - prev_close) / prev_close) * volumes[index]
            pvt_normalized = 0.0
            if avg_volume > 0 and self._window > 0:
                pvt_normalized = pvt_value / (avg_volume * self._window)
            pvt_normalized = self._clamp(pvt_normalized, -25.0, 25.0)

            price_zscore = 0.0
            if lookback_closes:
                mean_price = mean(lookback_closes)
                stdev_price = pstdev(lookback_closes) if len(lookback_closes) > 1 else 0.0
                if stdev_price > 0.0:
                    price_zscore = (current_close - mean_price) / stdev_price
            price_zscore = self._clamp(price_zscore, -5.0, 5.0)

            vwma_gap = 0.0
            lookback_volumes_inclusive = volumes[inclusive_slice]
            vwma_value = self._vwma(lookback_closes_with_current, lookback_volumes_inclusive)
            if current_close > 0.0:
                vwma_gap = (current_close - vwma_value) / current_close
            vwma_gap = self._clamp(vwma_gap, -5.0, 5.0)

            return_skewness = self._skewness(returns)
            return_kurtosis = self._kurtosis(returns)

            price_percent_rank = self._percent_rank(
                lookback_closes_with_current, current_close
            )
            volume_percent_rank = self._percent_rank(
                lookback_volumes_inclusive, current_volume
            )

            range_value = max(0.0, highs[index] - lows[index])
            range_history.append(range_value)
            range_percent_rank = self._percent_rank(range_history, range_value)
            atr_percent_rank = self._percent_rank(atr_history, atr)

            price_lr_slope, _, price_lr_rvalue = self._linear_regression(
                lookback_closes_with_current
            )
            price_trend_slope = 0.0
            if current_close > 0:
                price_trend_slope = price_lr_slope / current_close
            price_trend_slope = self._clamp(price_trend_slope, -5.0, 5.0)
            price_trend_strength = max(-1.0, min(1.0, price_lr_rvalue))

            volume_lr_slope, _, volume_lr_rvalue = self._linear_regression(
                volumes[inclusive_slice]
            )
            volume_scale = avg_volume if avg_volume > 0 else 1.0
            volume_trend_slope = self._clamp(volume_lr_slope / volume_scale, -5.0, 5.0)
            volume_trend_strength = max(-1.0, min(1.0, volume_lr_rvalue))

            downside_returns = [ret for ret in returns if ret < 0.0]
            upside_returns = [ret for ret in returns if ret > 0.0]
            downside_volatility = 0.0
            if downside_returns:
                downside_volatility = math.sqrt(
                    sum(ret * ret for ret in downside_returns) / len(downside_returns)
                )
            upside_volatility = 0.0
            if upside_returns:
                upside_volatility = math.sqrt(
                    sum(ret * ret for ret in upside_returns) / len(upside_returns)
                )
            sharpe_like = 0.0
            if volatility > 0.0:
                sharpe_like = mean_return / volatility
            sortino_like = 0.0
            if downside_volatility > 0.0:
                sortino_like = mean_return / downside_volatility
            sharpe_like = self._clamp(sharpe_like, -50.0, 50.0)
            sortino_like = self._clamp(sortino_like, -50.0, 50.0)
            downside_volatility = self._clamp(downside_volatility, 0.0, 5.0)
            upside_volatility = self._clamp(upside_volatility, 0.0, 5.0)

            return_autocorr = self._autocorrelation(returns, 1)
            volume_autocorr = self._autocorrelation(volume_window, 1)
            return_entropy = self._return_entropy(returns)
            hurst_exponent = self._hurst_exponent(returns)
            fractal_dimension = self._clamp(2.0 - hurst_exponent, 0.0, 2.0)

            features: MutableMapping[str, float] = {
                "ret_mean": float(mean_return),
                "ret_volatility": float(volatility),
                "volume_ratio": float(volume_ratio),
                "price_range": float(price_range),
                "open_close_diff": float(open_close_diff),
                "momentum": float((current_close - prev_close) / prev_close * 10_000.0),
                "session_time_sin": float(session_time_sin),
                "session_time_cos": float(session_time_cos),
                "session_time_progress": float(session_progress),
                "day_of_week_sin": float(day_of_week_sin),
                "day_of_week_cos": float(day_of_week_cos),
                "is_weekend": float(is_weekend),
                "week_of_year_sin": float(week_of_year_sin),
                "week_of_year_cos": float(week_of_year_cos),
                "week_of_year_progress": float(week_progress),
                "month_sin": float(month_sin),
                "month_cos": float(month_cos),
                "month_progress": float(month_progress),
                "day_of_month_progress": float(day_of_month_progress),
                "is_month_start": float(is_month_start),
                "is_month_end": float(is_month_end),
                "is_quarter_end": float(is_quarter_end),
                "price_zscore": float(price_zscore),
                "price_percent_rank": float(price_percent_rank),
                "volume_percent_rank": float(volume_percent_rank),
                "return_skewness": float(return_skewness),
                "return_kurtosis": float(return_kurtosis),
                "range_percent_rank": float(range_percent_rank),
                "atr_percent_rank": float(atr_percent_rank),
                "ema_fast_gap": float(ema_fast_gap),
                "ema_slow_gap": float(ema_slow_gap),
                "rsi": float(max(0.0, min(100.0, rsi))),
                "volume_zscore": float(volume_zscore),
                "volume_trend_slope": float(volume_trend_slope),
                "volume_trend_strength": float(volume_trend_strength),
                "return_sharpe_like": float(sharpe_like),
                "return_sortino_like": float(sortino_like),
                "downside_volatility": float(downside_volatility),
                "upside_volatility": float(upside_volatility),
                "return_autocorr": float(return_autocorr),
                "volume_autocorr": float(volume_autocorr),
                "return_entropy": float(return_entropy),
                "hurst_exponent": float(hurst_exponent),
                "fractal_dimension": float(fractal_dimension),
                "volatility_trend": float(vol_trend),
                "atr_ratio": float(atr / current_close if current_close > 0 else 0.0),
                "bollinger_position": float(bollinger_position),
                "bollinger_width": float(bollinger_width),
                "macd_line": float(macd_line),
                "macd_signal_gap": float(macd_line - macd_signal),
                "macd_histogram": float(macd_hist),
                "ppo_line": float(ppo_line),
                "ppo_signal": float(ppo_signal),
                "ppo_signal_gap": float(ppo_signal_gap),
                "ppo_histogram": float(ppo_hist),
                "fisher_transform": float(fisher_value),
                "fisher_signal_gap": float(fisher_signal_gap),
                "schaff_trend_cycle": float(schaff_cycle),
                "trix": float(trix),
                "ultimate_oscillator": float(ultimate_oscillator),
                "ease_of_movement": float(ease_of_movement),
                "vortex_positive": float(vortex_positive),
                "vortex_negative": float(vortex_negative),
                "price_rate_of_change": float(price_rate_of_change),
                "price_trend_slope": float(price_trend_slope),
                "price_trend_strength": float(price_trend_strength),
                "chande_momentum_oscillator": float(chande_momentum),
                "detrended_price_oscillator": float(detrended_price_oscillator),
                "aroon_up": float(aroon_up),
                "aroon_down": float(aroon_down),
                "aroon_oscillator": float(aroon_up - aroon_down),
                "balance_of_power": float(balance_of_power),
                "stochastic_rsi": float(stochastic_rsi_value),
                "relative_vigor_index": float(relative_vigor_index),
                "relative_vigor_signal": float(relative_vigor_signal),
                "relative_vigor_signal_gap": float(
                    self._clamp(relative_vigor_index - relative_vigor_signal, -5.0, 5.0)
                ),
                "elder_ray_bull_power": float(elder_bull_power),
                "elder_ray_bear_power": float(elder_bear_power),
                "ulcer_index": float(ulcer_index),
                "efficiency_ratio": float(efficiency_ratio),
                "true_strength_index": float(true_strength_index),
                "true_strength_signal": float(true_strength_signal),
                "true_strength_signal_gap": float(
                    self._clamp(true_strength_index - true_strength_signal, -10.0, 10.0)
                ),
                "connors_rsi": float(connors_rsi),
                "kama_gap": float(kama_gap),
                "kama_slope": float(kama_slope),
                "qstick": float(qstick),
                "dema_gap": float(dema_gap),
                "tema_gap": float(tema_gap),
                "frama_gap": float(frama_gap),
                "frama_slope": float(frama_slope),
                "hma_gap": float(hma_gap),
                "hma_slope": float(hma_slope),
                "zlema_gap": float(zlema_gap),
                "vwma_gap": float(vwma_gap),
                "t3_gap": float(t3_gap),
                "t3_slope": float(t3_slope),
                "supertrend_gap": float(supertrend_gap),
                "supertrend_direction": float(supertrend_direction_clamped),
                "supertrend_bandwidth": float(supertrend_bandwidth),
                "stochastic_k": float(stochastic_k),
                "stochastic_d": float(stochastic_d),
                "williams_r": float(williams_r),
                "cci": float(cci),
                "obv_normalized": float(obv_normalized),
                "pvt_normalized": float(pvt_normalized),
                "positive_volume_index": float(pvi_gap),
                "negative_volume_index": float(nvi_gap),
                "di_plus": float(max(0.0, min(100.0, di_plus))),
                "di_minus": float(max(0.0, min(100.0, di_minus))),
                "adx": float(max(0.0, min(100.0, adx))),
                "mfi": float(mfi),
                "force_index_normalized": float(force_index_normalized),
                "accumulation_distribution": float(accumulation_distribution),
                "chaikin_oscillator": float(chaikin_oscillator),
                "ichimoku_conversion_gap": float(ichimoku_conversion_gap),
                "ichimoku_base_gap": float(ichimoku_base_gap),
                "ichimoku_span_a_gap": float(ichimoku_span_a_gap),
                "ichimoku_span_b_gap": float(ichimoku_span_b_gap),
                "ichimoku_cloud_thickness": float(ichimoku_cloud_thickness),
                "ichimoku_price_position": float(ichimoku_price_position),
                "donchian_position": float(donchian_position),
                "donchian_width": float(donchian_width),
                "chaikin_money_flow": float(chaikin_money_flow),
                "mass_index": float(mass_index_value),
                "heikin_trend": float(heikin_trend),
                "heikin_shadow_ratio": float(heikin_shadow_ratio),
                "heikin_upper_shadow_ratio": float(heikin_upper_shadow),
                "heikin_lower_shadow_ratio": float(heikin_lower_shadow),
                "keltner_position": float(keltner_position),
                "keltner_width": float(keltner_width),
                "vwap_gap": float(vwap_gap),
                "psar_gap": float(psar_gap),
                "psar_direction": float(psar_direction),
                "pivot_gap": float(pivot_gap),
                "pivot_resistance_gap": float(pivot_resistance_gap),
                "pivot_support_gap": float(pivot_support_gap),
                "fractal_high_gap": float(fractal_high_gap),
                "fractal_low_gap": float(fractal_low_gap),
                "coppock_curve": float(coppock_curve),
                "choppiness_index": float(choppiness_index),
                "intraday_intensity": float(intraday_intensity),
                "intraday_intensity_volume": float(intraday_intensity_volume),
                "market_facilitation_index": float(market_facilitation_index),
                "klinger_oscillator": float(klinger_oscillator),
                "klinger_signal": float(klinger_signal),
                "klinger_signal_gap": float(klinger_signal_gap),
                "alligator_jaw_gap": float(alligator_jaw_gap),
                "alligator_teeth_gap": float(alligator_teeth_gap),
                "alligator_lips_gap": float(alligator_lips_gap),
                "gator_oscillator_upper": float(gator_upper),
                "gator_oscillator_lower": float(gator_lower),
            }

            vectors.append(
                FeatureVector(
                    timestamp=timestamps[index],
                    symbol=symbol,
                    features=features,
                    target_bps=float(target_return),
                )
            )

        return vectors

    def _compute_returns(
        self, closes: Sequence[float], lookback_slice: slice
    ) -> list[float]:
        returns: list[float] = []
        start, stop, step = lookback_slice.start, lookback_slice.stop, lookback_slice.step or 1
        if start is None or stop is None:
            return returns
        for idx in range(start + step, stop, step):
            prev = closes[idx - step]
            current = closes[idx]
            if prev <= 0:
                continue
            returns.append((current - prev) / prev)
        return returns

    def _ema(self, values: Sequence[float], span: int) -> float:
        if not values:
            return 0.0
        span = max(1, int(span))
        alpha = 2.0 / (span + 1.0)
        ema = float(values[0])
        for value in values[1:]:
            ema = alpha * float(value) + (1.0 - alpha) * ema
        return ema

    def _compute_rsi(self, closes: Sequence[float]) -> float:
        if len(closes) < 2:
            return 50.0
        gains: list[float] = []
        losses: list[float] = []
        for prev, current in zip(closes[:-1], closes[1:]):
            delta = float(current) - float(prev)
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        if not gains and not losses:
            return 50.0
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _volatility_trend(self, returns: Sequence[float]) -> float:
        if len(returns) < 4:
            return 0.0
        midpoint = len(returns) // 2
        first_half = returns[:midpoint]
        second_half = returns[midpoint:]
        if len(first_half) < 2 or len(second_half) < 2:
            return 0.0
        first_vol = pstdev(first_half)
        second_vol = pstdev(second_half)
        if first_vol == 0.0:
            return float(second_vol)
        return (second_vol - first_vol) / first_vol

    def _average_true_range(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        prev_close: float,
        current_high: float,
        current_low: float,
    ) -> float:
        true_ranges: list[float] = []
        previous = float(prev_close)
        for high, low, close in zip(highs, lows, closes):
            high_value = float(high)
            low_value = float(low)
            close_value = float(close)
            true_ranges.append(
                max(
                    high_value - low_value,
                    abs(high_value - previous),
                    abs(low_value - previous),
                )
            )
            previous = close_value

        true_ranges.append(
            max(
                float(current_high) - float(current_low),
                abs(float(current_high) - previous),
                abs(float(current_low) - previous),
            )
        )

        if not true_ranges:
            return 0.0
        return sum(true_ranges) / len(true_ranges)

    def _bollinger_metrics(
        self, closes: Sequence[float], current_close: float
    ) -> tuple[float, float]:
        if len(closes) < 2:
            return 0.0, 0.0
        mean_price = mean(closes)
        stdev_price = pstdev(closes) if len(closes) > 1 else 0.0
        if stdev_price == 0.0:
            return 0.0, 0.0
        upper = mean_price + 2 * stdev_price
        lower = mean_price - 2 * stdev_price
        width = (upper - lower) / mean_price if mean_price != 0.0 else 0.0
        position = (current_close - mean_price) / (upper - lower) if (upper - lower) != 0 else 0.0
        return position, width

    def _macd_components(self, closes: Sequence[float]) -> tuple[float, float, float]:
        if len(closes) < 3:
            return 0.0, 0.0, 0.0
        fast_span = max(2, min(len(closes) // 2, 12))
        slow_span = max(fast_span + 1, min(len(closes), 26))
        alpha_fast = 2.0 / (fast_span + 1.0)
        alpha_slow = 2.0 / (slow_span + 1.0)
        fast_ema = float(closes[0])
        slow_ema = float(closes[0])
        macd_series: list[float] = []
        for price in closes:
            fast_ema = alpha_fast * float(price) + (1.0 - alpha_fast) * fast_ema
            slow_ema = alpha_slow * float(price) + (1.0 - alpha_slow) * slow_ema
            macd_series.append(fast_ema - slow_ema)

        macd_line = macd_series[-1]
        signal_span = max(2, min(len(macd_series), 9))
        alpha_signal = 2.0 / (signal_span + 1.0)
        signal = macd_series[0]
        for value in macd_series[1:]:
            signal = alpha_signal * value + (1.0 - alpha_signal) * signal
        histogram = macd_line - signal
        return macd_line, signal, histogram

    def _ppo_components(self, closes: Sequence[float]) -> tuple[float, float, float]:
        if len(closes) < 3:
            return 0.0, 0.0, 0.0
        fast_span = max(2, min(len(closes) // 2, 12))
        slow_span = max(fast_span + 1, min(len(closes), 26))
        alpha_fast = 2.0 / (fast_span + 1.0)
        alpha_slow = 2.0 / (slow_span + 1.0)
        fast_ema = float(closes[0])
        slow_ema = float(closes[0])
        ppo_series: list[float] = []
        for price in closes:
            fast_ema = alpha_fast * float(price) + (1.0 - alpha_fast) * fast_ema
            slow_ema = alpha_slow * float(price) + (1.0 - alpha_slow) * slow_ema
            if slow_ema == 0.0:
                ppo_series.append(0.0)
            else:
                ppo_series.append(((fast_ema - slow_ema) / slow_ema) * 100.0)
        ppo_line = ppo_series[-1]
        signal_span = max(2, min(len(ppo_series), 9))
        alpha_signal = 2.0 / (signal_span + 1.0)
        signal = ppo_series[0]
        for value in ppo_series[1:]:
            signal = alpha_signal * value + (1.0 - alpha_signal) * signal
        histogram = ppo_line - signal
        return (
            self._clamp(ppo_line, -50.0, 50.0),
            self._clamp(signal, -50.0, 50.0),
            self._clamp(histogram, -50.0, 50.0),
        )

    def _trix(self, closes: Sequence[float]) -> float:
        if len(closes) < 4:
            return 0.0
        period = min(max(self._window, 5), len(closes))
        ema1 = self._ema_series(closes, period)
        ema2 = self._ema_series(ema1, period)
        ema3 = self._ema_series(ema2, period)
        if len(ema3) < 2:
            return 0.0
        prev = ema3[-2]
        current = ema3[-1]
        if prev == 0.0:
            return 0.0
        value = (current - prev) / abs(prev)
        return self._clamp(value, -5.0, 5.0)

    def _ema_series(self, values: Sequence[float], span: int) -> list[float]:
        if not values:
            return []
        span = max(1, int(span))
        alpha = 2.0 / (span + 1.0)
        ema_values: list[float] = []
        ema = float(values[0])
        ema_values.append(ema)
        for value in values[1:]:
            ema = alpha * float(value) + (1.0 - alpha) * ema
            ema_values.append(ema)
        return ema_values

    def _ultimate_oscillator(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
    ) -> float:
        if index <= 0:
            return 50.0
        periods = (7, 14, 28)
        start = max(1, index - periods[-1] + 1)
        bp_values: list[float] = []
        tr_values: list[float] = []
        for pos in range(start, index + 1):
            prev_close = float(closes[pos - 1])
            high = float(highs[pos])
            low = float(lows[pos])
            close_value = float(closes[pos])
            bp = close_value - min(low, prev_close)
            tr = max(high, prev_close) - min(low, prev_close)
            bp_values.append(bp)
            tr_values.append(tr if tr > 0 else 0.0)
        weights = (4.0, 2.0, 1.0)
        accumulator = 0.0
        weight_sum = 0.0
        for period, weight in zip(periods, weights):
            if len(bp_values) >= period:
                bp_sum = sum(bp_values[-period:])
                tr_sum = sum(tr_values[-period:])
                if tr_sum > 0:
                    accumulator += weight * (bp_sum / tr_sum)
                    weight_sum += weight
        if weight_sum == 0.0:
            return 50.0
        oscillator = (accumulator / weight_sum) * 100.0
        return max(0.0, min(100.0, oscillator))

    def _ease_of_movement(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        volumes: Sequence[float],
        index: int,
    ) -> float:
        if index <= 0:
            return 0.0
        period = min(max(self._window, 5), 20)
        start = max(1, index - period + 1)
        values: list[float] = []
        for pos in range(start, index + 1):
            high = float(highs[pos])
            low = float(lows[pos])
            prev_high = float(highs[pos - 1])
            prev_low = float(lows[pos - 1])
            volume = float(volumes[pos])
            range_span = high - low
            if range_span == 0.0 or volume == 0.0:
                continue
            mid_move = ((high + low) / 2.0) - ((prev_high + prev_low) / 2.0)
            box_ratio = volume / range_span
            if box_ratio == 0.0:
                continue
            values.append(mid_move / box_ratio)
        if not values:
            return 0.0
        average = sum(values) / len(values)
        return self._clamp(average, -5.0, 5.0)

    def _stochastic_rsi(
        self, closes: Sequence[float], index: int, current_rsi: float
    ) -> float:
        if index <= 0:
            return 50.0
        period = min(max(self._window, 5), 14)
        start = max(1, index - period + 1)
        rsi_values: list[float] = []
        for pos in range(start, index + 1):
            window_start = max(0, pos - period + 1)
            window = closes[window_start : pos + 1]
            if len(window) < 2:
                continue
            rsi_values.append(self._compute_rsi(window))
        if not rsi_values:
            return 50.0
        min_rsi = min(rsi_values)
        max_rsi = max(rsi_values)
        if max_rsi == min_rsi:
            return 50.0
        normalized = (current_rsi - min_rsi) / (max_rsi - min_rsi)
        return self._clamp(normalized * 100.0, 0.0, 100.0)

    def _relative_vigor_index(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
    ) -> tuple[float, float]:
        if index < 3:
            return 0.0, 0.0
        period = min(max(self._window, 4), 14)
        start = max(3, index - period + 1)
        rvi_values: list[float] = []
        for pos in range(start, index + 1):
            if pos - 3 < 0:
                continue
            numerator = (
                (closes[pos] - opens[pos])
                + 2.0 * (closes[pos - 1] - opens[pos - 1])
                + 2.0 * (closes[pos - 2] - opens[pos - 2])
                + (closes[pos - 3] - opens[pos - 3])
            ) / 6.0
            denominator = (
                (highs[pos] - lows[pos])
                + 2.0 * (highs[pos - 1] - lows[pos - 1])
                + 2.0 * (highs[pos - 2] - lows[pos - 2])
                + (highs[pos - 3] - lows[pos - 3])
            ) / 6.0
            if denominator == 0.0:
                rvi_values.append(0.0)
            else:
                rvi_values.append(numerator / denominator)
        if not rvi_values:
            return 0.0, 0.0
        current_value = rvi_values[-1]
        if len(rvi_values) >= 4:
            signal = (
                rvi_values[-1]
                + 2.0 * rvi_values[-2]
                + 2.0 * rvi_values[-3]
                + rvi_values[-4]
            ) / 6.0
        else:
            signal = sum(rvi_values) / len(rvi_values)
        return self._clamp(current_value, -5.0, 5.0), self._clamp(signal, -5.0, 5.0)

    def _elder_ray(
        self,
        closes_with_current: Sequence[float],
        high_price: float,
        low_price: float,
        current_close: float,
    ) -> tuple[float, float]:
        if not closes_with_current:
            return 0.0, 0.0
        period = min(max(self._window, 5), len(closes_with_current))
        ema_value = self._ema(closes_with_current, period)
        if current_close == 0.0:
            return 0.0, 0.0
        bull_power = (float(high_price) - ema_value) / current_close
        bear_power = (float(low_price) - ema_value) / current_close
        return self._clamp(bull_power, -5.0, 5.0), self._clamp(bear_power, -5.0, 5.0)

    def _ulcer_index(self, closes: Sequence[float]) -> float:
        if len(closes) < 2:
            return 0.0
        peak = float(closes[0])
        squared_drawdown = 0.0
        count = 0
        for value in closes:
            price = float(value)
            if price > peak:
                peak = price
            if peak <= 0.0:
                continue
            drawdown = max(0.0, (peak - price) / peak)
            squared_drawdown += drawdown * drawdown
            count += 1
        if count == 0:
            return 0.0
        ulcer = math.sqrt(squared_drawdown / count)
        return self._clamp(ulcer, 0.0, 5.0)

    def _efficiency_ratio(self, closes: Sequence[float], index: int) -> float:
        period = min(max(self._window, 5), 20)
        if index - period < 0:
            return 0.0
        start = index - period + 1
        change = abs(float(closes[index]) - float(closes[start]))
        volatility = 0.0
        for pos in range(start + 1, index + 1):
            volatility += abs(float(closes[pos]) - float(closes[pos - 1]))
        if volatility == 0.0:
            return 0.0
        ratio = change / volatility
        return self._clamp(ratio, 0.0, 1.0)

    def _rate_of_change(self, closes: Sequence[float], index: int, period: int) -> float:
        if period <= 0 or index - period < 0:
            return 0.0
        previous = float(closes[index - period])
        current = float(closes[index])
        if previous == 0.0:
            return 0.0
        change = (current - previous) / previous
        return self._clamp(change, -5.0, 5.0)

    def _chande_momentum_oscillator(self, closes: Sequence[float], index: int) -> float:
        if index <= 0:
            return 0.0
        period = min(max(self._window, 5), 20)
        start = max(1, index - period + 1)
        gains = 0.0
        losses = 0.0
        for pos in range(start, index + 1):
            delta = float(closes[pos]) - float(closes[pos - 1])
            if delta > 0:
                gains += delta
            elif delta < 0:
                losses += abs(delta)
        denominator = gains + losses
        if denominator == 0.0:
            return 0.0
        oscillator = 100.0 * (gains - losses) / denominator
        return self._clamp(oscillator, -100.0, 100.0)

    def _detrended_price_oscillator(self, closes: Sequence[float], index: int) -> float:
        period = min(max(self._window, 5), 20)
        offset = period // 2 + 1
        reference_index = index - offset
        if reference_index <= 0 or reference_index >= len(closes):
            return 0.0
        start = reference_index - period + 1
        if start < 0:
            return 0.0
        window = closes[start : reference_index + 1]
        if not window:
            return 0.0
        sma = mean(window)
        if sma == 0.0:
            return 0.0
        value = (float(closes[reference_index]) - sma) / abs(sma)
        return self._clamp(value, -5.0, 5.0)

    def _aroon(self, highs: Sequence[float], lows: Sequence[float], index: int) -> tuple[float, float]:
        period = min(max(self._window, 5), 25)
        start = max(0, index - period + 1)
        window_highs = highs[start : index + 1]
        window_lows = lows[start : index + 1]
        if not window_highs or not window_lows:
            return 0.0, 0.0
        length = len(window_highs)
        if length <= 1:
            return 0.0, 0.0
        highest = max(window_highs)
        lowest = min(window_lows)
        last_high_index = 0
        last_low_index = 0
        for pos, value in enumerate(window_highs):
            if value >= highest:
                last_high_index = pos
        for pos, value in enumerate(window_lows):
            if value <= lowest:
                last_low_index = pos
        periods_since_high = length - 1 - last_high_index
        periods_since_low = length - 1 - last_low_index
        if length <= 0:
            return 0.0, 0.0
        aroon_up = 100.0 * (length - periods_since_high) / length
        aroon_down = 100.0 * (length - periods_since_low) / length
        return (
            self._clamp(aroon_up, 0.0, 100.0),
            self._clamp(aroon_down, 0.0, 100.0),
        )

    def _balance_of_power(
        self, open_price: float, high_price: float, low_price: float, close_price: float
    ) -> float:
        high_value = float(high_price)
        low_value = float(low_price)
        if high_value == low_value:
            return 0.0
        bop = (float(close_price) - float(open_price)) / (high_value - low_value)
        return self._clamp(bop, -5.0, 5.0)

    def _vortex_indicator(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
    ) -> tuple[float, float]:
        if index <= 0:
            return 0.0, 0.0
        period = min(max(self._window, 5), 21)
        start = max(1, index - period + 1)
        tr_sum = 0.0
        vm_plus = 0.0
        vm_minus = 0.0
        for pos in range(start, index + 1):
            high = float(highs[pos])
            low = float(lows[pos])
            prev_high = float(highs[pos - 1])
            prev_low = float(lows[pos - 1])
            prev_close = float(closes[pos - 1])
            tr_sum += max(high - low, abs(high - prev_close), abs(low - prev_close))
            vm_plus += abs(high - prev_low)
            vm_minus += abs(low - prev_high)
        if tr_sum == 0.0:
            return 0.0, 0.0
        positive = vm_plus / tr_sum
        negative = vm_minus / tr_sum
        return self._clamp(positive, 0.0, 5.0), self._clamp(negative, 0.0, 5.0)

    def _rolling_midpoint(
        self, highs: Sequence[float], lows: Sequence[float], index: int, period: int
    ) -> float:
        if index < 0 or period <= 1:
            return (float(highs[index]) + float(lows[index])) / 2.0
        start = max(0, index - period + 1)
        window_highs = highs[start : index + 1]
        window_lows = lows[start : index + 1]
        if not window_highs or not window_lows:
            return (float(highs[index]) + float(lows[index])) / 2.0
        return (max(window_highs) + min(window_lows)) / 2.0

    def _donchian_channel(
        self, highs: Sequence[float], lows: Sequence[float], index: int, period: int
    ) -> tuple[float, float]:
        if period <= 1:
            value = (float(highs[index]) + float(lows[index])) / 2.0
            return value, value
        start = max(0, index - period + 1)
        window_highs = highs[start : index + 1]
        window_lows = lows[start : index + 1]
        return max(window_highs) if window_highs else highs[index], min(window_lows) if window_lows else lows[index]

    def _chaikin_money_flow(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        index: int,
    ) -> float:
        period = min(max(self._window, 5), 20)
        start = max(0, index - period + 1)
        mf_volume = 0.0
        volume_sum = 0.0
        for pos in range(start, index + 1):
            high = float(highs[pos])
            low = float(lows[pos])
            close_value = float(closes[pos])
            volume_value = float(volumes[pos])
            if high == low:
                multiplier = 0.0
            else:
                multiplier = ((close_value - low) - (high - close_value)) / (high - low)
            mf_volume += multiplier * volume_value
            volume_sum += volume_value
        if volume_sum == 0.0:
            return 0.0
        cmf = mf_volume / volume_sum
        if not math.isfinite(cmf):
            return 0.0
        return max(-1.5, min(1.5, cmf))

    def _parabolic_sar(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
    ) -> tuple[float, float]:
        if index <= 0 or index >= len(highs):
            price = float(closes[index]) if index < len(closes) else float(closes[-1])
            return price, 0.0

        window_span = max(self._window * 3, 6)
        start = max(0, index - window_span)
        if start == index:
            price = float(closes[index])
            return price, 0.0

        step = 0.02
        max_step = 0.2
        uptrend = True
        if start + 1 <= index:
            uptrend = float(closes[start + 1]) >= float(closes[start])
        sar = float(lows[start]) if uptrend else float(highs[start])
        extreme = float(highs[start]) if uptrend else float(lows[start])

        for pos in range(start + 1, index + 1):
            prev_sar = sar
            if uptrend:
                sar = prev_sar + step * (extreme - prev_sar)
                sar = min(sar, float(lows[pos - 1]), float(lows[pos]))
                if float(highs[pos]) > extreme:
                    extreme = float(highs[pos])
                    step = min(step + 0.02, max_step)
                if float(lows[pos]) < sar:
                    uptrend = False
                    sar = extreme
                    extreme = float(lows[pos])
                    step = 0.02
            else:
                sar = prev_sar + step * (extreme - prev_sar)
                sar = max(sar, float(highs[pos - 1]), float(highs[pos]))
                if float(lows[pos]) < extreme:
                    extreme = float(lows[pos])
                    step = min(step + 0.02, max_step)
                if float(highs[pos]) > sar:
                    uptrend = True
                    sar = extreme
                    extreme = float(highs[pos])
                    step = 0.02

        direction = 1.0 if uptrend else -1.0
        return float(sar), direction

    def _recent_fractal(
        self,
        series: Sequence[float],
        index: int,
        *,
        is_high: bool,
    ) -> float | None:
        if index < 4:
            return None
        lookback = max(self._window, 5)
        start = max(2, index - lookback)
        end = index - 2
        if end < start:
            return None

        last_value: float | None = None
        for pos in range(start, end + 1):
            if pos + 2 >= len(series):
                break
            center = float(series[pos])
            neighbors = (
                float(series[pos - 2]),
                float(series[pos - 1]),
                float(series[pos + 1]),
                float(series[pos + 2]),
            )
            if is_high:
                if all(center >= neighbor for neighbor in neighbors):
                    last_value = center
            else:
                if all(center <= neighbor for neighbor in neighbors):
                    last_value = center
        return last_value

    def _true_strength_index(
        self, closes: Sequence[float], index: int
    ) -> tuple[float, float]:
        if index < 2:
            return 0.0, 0.0
        short_period = max(2, min(13, index))
        long_period = max(short_period + 1, min(25, index + 1))
        lookback = max(long_period * 3, short_period * 3)
        start = max(1, index - lookback + 1)
        alpha_short = 2.0 / (short_period + 1.0)
        alpha_long = 2.0 / (long_period + 1.0)
        ema1_num = 0.0
        ema1_den = 0.0
        ema2_num = 0.0
        ema2_den = 0.0
        tsi_history: list[float] = []
        first_iteration = True
        for pos in range(start, index + 1):
            momentum = float(closes[pos]) - float(closes[pos - 1])
            abs_momentum = abs(momentum)
            if first_iteration:
                ema1_num = momentum
                ema1_den = abs_momentum
                ema2_num = ema1_num
                ema2_den = ema1_den
                first_iteration = False
            else:
                ema1_num = alpha_short * momentum + (1.0 - alpha_short) * ema1_num
                ema1_den = alpha_short * abs_momentum + (1.0 - alpha_short) * ema1_den
                ema2_num = alpha_long * ema1_num + (1.0 - alpha_long) * ema2_num
                ema2_den = alpha_long * ema1_den + (1.0 - alpha_long) * ema2_den
            tsi_value = 0.0
            if ema2_den != 0.0:
                tsi_value = 100.0 * ema2_num / ema2_den
            tsi_history.append(tsi_value)
        if not tsi_history:
            return 0.0, 0.0
        tsi_current = self._clamp(tsi_history[-1], -150.0, 150.0)
        signal_period = max(2, min(7, len(tsi_history)))
        alpha_signal = 2.0 / (signal_period + 1.0)
        signal = tsi_history[0]
        for value in tsi_history[1:]:
            signal = alpha_signal * value + (1.0 - alpha_signal) * signal
        signal = self._clamp(signal, -150.0, 150.0)
        return tsi_current, signal

    def _connors_rsi(self, closes: Sequence[float], index: int) -> float:
        if index < 2:
            return 50.0
        lookback = max(3, min(self._window * 2, 20))
        price_period = min(3, index)
        price_slice_start = max(0, index - price_period)
        price_slice = closes[price_slice_start : index + 1]
        price_rsi = 50.0
        if len(price_slice) >= 2:
            price_rsi = self._compute_rsi(price_slice)

        streaks: list[int] = []
        streak_value = 0
        start = max(0, index - lookback + 1)
        for pos in range(start + 1, index + 1):
            delta = float(closes[pos]) - float(closes[pos - 1])
            if delta > 0:
                streak_value = streak_value + 1 if streak_value >= 0 else 1
            elif delta < 0:
                streak_value = streak_value - 1 if streak_value <= 0 else -1
            else:
                streak_value = 0
            streaks.append(streak_value)
        streak_rsi = 50.0
        if streaks:
            streak_series = [0.0] + [float(value) for value in streaks]
            streak_period = min(2, len(streak_series) - 1)
            if streak_period >= 1:
                streak_slice = streak_series[-(streak_period + 1) :]
                if len(streak_slice) >= 2:
                    streak_rsi = self._compute_rsi(streak_slice)

        returns: list[float] = []
        for pos in range(start + 1, index + 1):
            prev_close = float(closes[pos - 1])
            current_close = float(closes[pos])
            if prev_close != 0.0:
                returns.append((current_close - prev_close) / prev_close)
            else:
                returns.append(0.0)
        percent_rank = 50.0
        if returns:
            last_return = returns[-1]
            less_count = sum(1 for value in returns if value < last_return)
            equal_count = sum(1 for value in returns if value == last_return)
            percent_rank = 100.0 * (less_count + 0.5 * equal_count) / len(returns)

        connors = (price_rsi + streak_rsi + percent_rank) / 3.0
        return self._clamp(connors, 0.0, 100.0)

    def _kama_metrics(self, closes: Sequence[float], index: int) -> tuple[float, float]:
        if index < 2:
            return 0.0, 0.0
        base_period = max(10, self._window)
        period = min(base_period, index)
        start = max(0, index - period)
        kama = float(closes[start])
        prev_kama = kama
        fast_sc = 2.0 / (2 + 1.0)
        slow_sc = 2.0 / (30 + 1.0)
        for pos in range(start + 1, index + 1):
            baseline_index = max(0, pos - period)
            change = abs(float(closes[pos]) - float(closes[baseline_index]))
            volatility = 0.0
            for step in range(max(1, pos - period + 1), pos + 1):
                volatility += abs(float(closes[step]) - float(closes[step - 1]))
            efficiency_ratio = 0.0
            if volatility > 0.0:
                efficiency_ratio = change / volatility
            smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
            price = float(closes[pos])
            prev_kama = kama
            kama = kama + smoothing_constant * (price - kama)
        current_close = float(closes[index])
        kama_gap = 0.0
        if current_close != 0.0:
            kama_gap = (current_close - kama) / current_close
        kama_slope = 0.0
        if current_close != 0.0 and prev_kama != 0.0:
            kama_slope = (kama - prev_kama) / current_close
        return self._clamp(kama_gap, -5.0, 5.0), self._clamp(kama_slope, -5.0, 5.0)

    def _qstick(
        self, opens: Sequence[float], closes: Sequence[float], index: int
    ) -> float:
        period = min(max(3, self._window // 2), index + 1)
        start = max(0, index - period + 1)
        differences = [
            float(closes[pos]) - float(opens[pos]) for pos in range(start, index + 1)
        ]
        if not differences:
            return 0.0
        average_difference = sum(differences) / len(differences)
        current_close = float(closes[index])
        if current_close == 0.0:
            return 0.0
        normalized = average_difference / current_close
        return self._clamp(normalized, -5.0, 5.0)

    def _ema_next(self, previous: float, value: float, span: int) -> float:
        span = max(1, int(span))
        alpha = 2.0 / (span + 1.0)
        return alpha * float(value) + (1.0 - alpha) * float(previous)

    def _ema_series(self, values: Sequence[float], span: int) -> list[float]:
        if not values:
            return []
        span = max(1, int(span))
        alpha = 2.0 / (span + 1.0)
        ema_values: list[float] = [float(values[0])]
        ema = ema_values[0]
        for value in values[1:]:
            ema = alpha * float(value) + (1.0 - alpha) * ema
            ema_values.append(ema)
        return ema_values

    def _dema(self, values: Sequence[float], span: int) -> float:
        if not values:
            return 0.0
        ema_first_series = self._ema_series(values, span)
        if not ema_first_series:
            return 0.0
        ema_first = ema_first_series[-1]
        ema_second_series = self._ema_series(ema_first_series, span)
        ema_second = ema_second_series[-1] if ema_second_series else ema_first
        return 2.0 * ema_first - ema_second

    def _tema(self, values: Sequence[float], span: int) -> float:
        if not values:
            return 0.0
        ema_first_series = self._ema_series(values, span)
        if not ema_first_series:
            return 0.0
        ema_first = ema_first_series[-1]
        ema_second_series = self._ema_series(ema_first_series, span)
        if not ema_second_series:
            return ema_first
        ema_second = ema_second_series[-1]
        ema_third_series = self._ema_series(ema_second_series, span)
        ema_third = ema_third_series[-1] if ema_third_series else ema_second
        return 3.0 * ema_first - 3.0 * ema_second + ema_third

    def _choppiness_index(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
    ) -> float:
        if index <= 1:
            return 50.0
        period = min(max(self._window, 5), 30)
        start = max(1, index - period + 1)
        length = index - start + 1
        if length < 2:
            return 50.0
        tr_sum = 0.0
        highest_high = float("-inf")
        lowest_low = float("inf")
        for pos in range(start, index + 1):
            high_value = float(highs[pos])
            low_value = float(lows[pos])
            prev_close = float(closes[pos - 1])
            tr_sum += max(
                high_value - low_value,
                abs(high_value - prev_close),
                abs(low_value - prev_close),
            )
            if high_value > highest_high:
                highest_high = high_value
            if low_value < lowest_low:
                lowest_low = low_value
        range_span = highest_high - lowest_low
        if range_span <= 0.0 or tr_sum <= 0.0:
            return 50.0
        denominator = math.log10(length)
        if denominator <= 0.0:
            return 50.0
        choppiness = 100.0 * math.log10(tr_sum / range_span) / denominator
        return self._clamp(choppiness, 0.0, 100.0)

    def _intraday_intensity(self, high: float, low: float, close: float) -> float:
        high_value = float(high)
        low_value = float(low)
        close_value = float(close)
        range_span = high_value - low_value
        if range_span == 0.0:
            return 0.0
        intensity = ((close_value - low_value) - (high_value - close_value)) / range_span
        return self._clamp(intensity, -1.0, 1.0)

    def _money_flow_multiplier(self, high: float, low: float, close: float) -> float:
        high_value = float(high)
        low_value = float(low)
        close_value = float(close)
        range_span = high_value - low_value
        if range_span == 0.0:
            return 0.0
        multiplier = ((close_value - low_value) - (high_value - close_value)) / range_span
        return self._clamp(multiplier, -1.0, 1.0)

    def _mass_index(self, ratios: Sequence[float]) -> float:
        if not ratios:
            return 0.0
        window = [float(value) for value in ratios[-25:]]
        total = sum(window)
        return self._clamp(total, 0.0, 50.0)

    def _normalized_index_gap(self, history: Sequence[float]) -> float:
        if not history:
            return 0.0
        period = min(len(history), max(self._window, 10))
        if period <= 0:
            return 0.0
        window = [float(value) for value in history[-period:]]
        if not window:
            return 0.0
        average = sum(window) / len(window)
        if average == 0.0:
            return 0.0
        current = window[-1]
        return self._clamp((current - average) / average, -5.0, 5.0)

    def _weighted_moving_average(self, values: Sequence[float], period: int) -> float:
        if period <= 0:
            return 0.0
        window = [float(value) for value in values[-period:]]
        if not window:
            return 0.0
        denominator = len(window) * (len(window) + 1) / 2.0
        if denominator == 0.0:
            return 0.0
        weighted_sum = 0.0
        for weight, value in enumerate(window, start=1):
            weighted_sum += weight * value
        return weighted_sum / denominator

    @staticmethod
    def _smma_next(previous: float, current: float, period: int) -> float:
        period = max(1, period)
        prev_value = float(previous)
        current_value = float(current)
        return prev_value + (current_value - prev_value) / period

    def _coppock_curve(self, history: Sequence[float]) -> float:
        if not history:
            return 0.0
        period = min(len(history), max(3, min(self._window, 10)))
        wma = self._weighted_moving_average(history, period)
        return self._clamp(wma, -10.0, 10.0)

    def _fisher_transform(
        self,
        closes: Sequence[float],
        index: int,
        prev_trigger: float,
        prev_fisher: float,
        prev_signal: float,
        *,
        period: int | None = None,
    ) -> tuple[float, float, float]:
        window = period or max(10, self._window)
        if index < window:
            return prev_trigger, self._clamp(prev_fisher, -10.0, 10.0), self._clamp(prev_signal, -10.0, 10.0)
        start = index - window + 1
        window_values = closes[start : index + 1]
        highest = max(window_values)
        lowest = min(window_values)
        if highest == lowest:
            trigger = prev_trigger
        else:
            normalized = (float(closes[index]) - lowest) / (highest - lowest)
            trigger = 0.66 * (normalized - 0.5) + 0.67 * prev_trigger
        trigger = self._clamp(trigger, -0.999, 0.999)
        fisher = 0.5 * math.log((1.0 + trigger) / (1.0 - trigger))
        fisher = self._clamp(fisher, -10.0, 10.0)
        signal = 0.5 * (fisher + prev_fisher)
        return trigger, fisher, self._clamp(signal, -10.0, 10.0)

    def _schaff_trend_cycle(
        self,
        current_close: float,
        macd_history: list[float],
        ema_fast_prev: float,
        ema_slow_prev: float,
        stoch_prev: float,
        stc_prev: float,
        *,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
    ) -> tuple[float, float, float, float, float]:
        ema_fast = self._ema_next(ema_fast_prev, current_close, fast_period)
        ema_slow = self._ema_next(ema_slow_prev, current_close, slow_period)
        macd_value = ema_fast - ema_slow
        macd_history.append(macd_value)
        history_limit = max(slow_period * 2, cycle_period * 3)
        if len(macd_history) > history_limit:
            del macd_history[: len(macd_history) - history_limit]
        stoch_macd = self._stochastic_from_history(macd_history, cycle_period)
        stoch_ema = self._ema_next(stoch_prev, stoch_macd, cycle_period)
        stc_raw = self._ema_next(stc_prev, stoch_ema, cycle_period)
        stc_value = self._clamp(stc_raw, 0.0, 100.0)
        return ema_fast, ema_slow, stoch_ema, stc_raw, stc_value

    def _stochastic_from_history(
        self, values: Sequence[float], period: int
    ) -> float:
        if not values:
            return 50.0
        period = max(1, int(period))
        window = values[-period:]
        highest = max(window)
        lowest = min(window)
        if highest == lowest:
            return 50.0
        latest = values[-1]
        normalized = (latest - lowest) / (highest - lowest)
        return self._clamp(normalized * 100.0, 0.0, 100.0)

    def _frama_metrics(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
        prev_frama: float,
        *,
        window: int | None = None,
    ) -> tuple[float, float, float]:
        period = window or max(16, self._window)
        period = max(2, min(period, index + 1))
        half = period // 2
        if period < 2 or index < period:
            current_price = float(closes[index])
            gap = 0.0 if current_price == 0.0 else (current_price - current_price) / current_price
            return current_price, gap, 0.0
        start = index - period + 1
        first_slice = slice(start, start + half)
        second_slice = slice(start + half, start + period)
        first_high = max(highs[first_slice]) if first_slice.stop <= len(highs) else max(highs[start:index + 1])
        first_low = min(lows[first_slice]) if first_slice.stop <= len(lows) else min(lows[start:index + 1])
        second_high = max(highs[second_slice]) if second_slice.stop <= len(highs) else max(highs[start:index + 1])
        second_low = min(lows[second_slice]) if second_slice.stop <= len(lows) else min(lows[start:index + 1])
        full_high = max(highs[start : index + 1])
        full_low = min(lows[start : index + 1])
        n1 = float(first_high) - float(first_low)
        n2 = float(second_high) - float(second_low)
        n3 = float(full_high) - float(full_low)
        if n1 <= 0.0 or n2 <= 0.0 or n3 <= 0.0:
            dimension = 1.0
        else:
            dimension = (math.log(n1 + n2) - math.log(n3)) / math.log(2.0)
        alpha = math.exp(-4.6 * (dimension - 1.0)) if dimension != 1.0 else 1.0
        alpha = max(0.01, min(alpha, 1.0))
        previous = prev_frama if prev_frama is not None else float(closes[index - 1])
        current_price = float(closes[index])
        frama = alpha * current_price + (1.0 - alpha) * previous
        slope = frama - previous
        gap = 0.0
        slope_ratio = 0.0
        if current_price != 0.0:
            gap = (current_price - frama) / current_price
            slope_ratio = slope / current_price
        return frama, self._clamp(gap, -5.0, 5.0), self._clamp(slope_ratio, -5.0, 5.0)

    def _supertrend_step(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        index: int,
        prev_value: float,
        prev_direction: float,
        prev_upper: float,
        prev_lower: float,
        period: int,
        multiplier: float,
    ) -> tuple[float, float, float, float]:
        if index <= 0:
            price = float(closes[index])
            return price, 1.0, price, price
        start = max(0, index - period)
        atr_slice = slice(start, index)
        prev_close = float(closes[index - 1])
        atr = self._average_true_range(
            highs[atr_slice],
            lows[atr_slice],
            closes[atr_slice],
            prev_close,
            highs[index],
            lows[index],
        )
        mid_price = (float(highs[index]) + float(lows[index])) / 2.0
        basic_upper = mid_price + multiplier * atr
        basic_lower = mid_price - multiplier * atr
        final_upper = basic_upper
        final_lower = basic_lower
        direction = 1.0 if prev_direction >= 0 else -1.0
        if index > 1:
            if basic_upper < prev_upper or prev_direction < 0 and prev_close > prev_upper:
                final_upper = basic_upper
            else:
                final_upper = prev_upper
            if basic_lower > prev_lower or prev_direction > 0 and prev_close < prev_lower:
                final_lower = basic_lower
            else:
                final_lower = prev_lower
        value = prev_value
        if direction > 0:
            if mid_price < final_lower:
                direction = -1.0
                value = final_upper
            else:
                value = final_lower
        else:
            if mid_price > final_upper:
                direction = 1.0
                value = final_lower
            else:
                value = final_upper
        return value, direction, final_upper, final_lower

    def _wma_series(self, values: Sequence[float], period: int) -> list[float]:
        if not values or period <= 0:
            return []
        period = min(int(period), len(values))
        weights = list(range(1, period + 1))
        denominator = sum(weights)
        series: list[float] = []
        for idx in range(period - 1, len(values)):
            window = values[idx - period + 1 : idx + 1]
            weighted_sum = 0.0
            for weight, value in zip(weights, window, strict=False):
                weighted_sum += weight * float(value)
            series.append(weighted_sum / denominator if denominator else float(window[-1]))
        return series

    def _wma(self, values: Sequence[float], period: int) -> float:
        series = self._wma_series(values, period)
        if not series:
            return float(values[-1]) if values else 0.0
        return series[-1]

    def _hma(self, values: Sequence[float], period: int) -> float:
        if not values:
            return 0.0
        period = max(2, int(period))
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(math.sqrt(period)))
        wma_half_series = self._wma_series(values, half_period)
        wma_full_series = self._wma_series(values, period)
        if not wma_half_series or not wma_full_series:
            return float(values[-1])
        length = min(len(wma_half_series), len(wma_full_series))
        derived: list[float] = []
        for offset in range(-length, 0):
            derived.append(2.0 * wma_half_series[offset] - wma_full_series[offset])
        return self._wma(derived, min(sqrt_period, len(derived)))

    def _zlema(self, values: Sequence[float], period: int) -> float:
        if not values:
            return 0.0
        period = max(1, int(period))
        lag = max(0, (period - 1) // 2)
        adjusted: list[float] = []
        for idx, value in enumerate(values):
            price = float(value)
            lag_index = idx - lag
            lag_price = float(values[lag_index]) if lag_index >= 0 else price
            adjusted_value = price + (price - lag_price)
            adjusted.append(adjusted_value)
        return self._ema(adjusted, period)

    def _vwma(self, closes: Sequence[float], volumes: Sequence[float]) -> float:
        if not closes or not volumes:
            return 0.0
        length = min(len(closes), len(volumes))
        weighted_sum = 0.0
        volume_sum = 0.0
        for price, volume in zip(closes[-length:], volumes[-length:], strict=False):
            vol = float(volume)
            volume_sum += vol
            weighted_sum += float(price) * vol
        if volume_sum == 0.0:
            return float(closes[-1])
        return weighted_sum / volume_sum

    def _t3(
        self,
        closes: Sequence[float],
        period: int,
        volume_factor: float = 0.7,
    ) -> float:
        if not closes:
            return 0.0
        period = max(1, int(period))
        vf = max(0.0, min(float(volume_factor), 1.0))
        ema1 = self._ema_series(closes, period)
        if not ema1:
            return float(closes[-1])
        ema2 = self._ema_series(ema1, period)
        ema3 = self._ema_series(ema2, period)
        ema4 = self._ema_series(ema3, period)
        ema5 = self._ema_series(ema4, period)
        ema6 = self._ema_series(ema5, period)
        ema3_last = ema3[-1] if ema3 else float(closes[-1])
        ema4_last = ema4[-1] if ema4 else ema3_last
        ema5_last = ema5[-1] if ema5 else ema4_last
        ema6_last = ema6[-1] if ema6 else ema5_last
        c1 = -vf**3
        c2 = 3.0 * vf**2 + 3.0 * vf**3
        c3 = -3.0 * vf - 6.0 * vf**2 - 3.0 * vf**3
        c4 = 1.0 + 3.0 * vf + 3.0 * vf**2 + vf**3
        return c1 * ema6_last + c2 * ema5_last + c3 * ema4_last + c4 * ema3_last

    def _skewness(self, values: Sequence[float]) -> float:
        if len(values) < 3:
            return 0.0
        mean_value = mean(values)
        stdev_value = pstdev(values)
        if stdev_value == 0.0:
            return 0.0
        moment = sum((float(value) - mean_value) ** 3 for value in values) / len(values)
        skew = moment / (stdev_value**3)
        return self._clamp(skew, -10.0, 10.0)

    def _kurtosis(self, values: Sequence[float]) -> float:
        if len(values) < 4:
            return 0.0
        mean_value = mean(values)
        stdev_value = pstdev(values)
        if stdev_value == 0.0:
            return 0.0
        moment = sum((float(value) - mean_value) ** 4 for value in values) / len(values)
        kurtosis = moment / (stdev_value**4)
        excess = kurtosis - 3.0
        return self._clamp(excess, -10.0, 10.0)

    def _percent_rank(self, values: Sequence[float], target: float) -> float:
        if not values:
            return 50.0
        count = sum(1 for value in values if float(value) <= float(target))
        percent = (count / len(values)) * 100.0
        return self._clamp(percent, 0.0, 100.0)

    def _autocorrelation(self, values: Sequence[float], lag: int = 1) -> float:
        if lag <= 0:
            return 0.0
        if len(values) <= lag:
            return 0.0
        float_values = [float(value) for value in values]
        mean_value = mean(float_values)
        numerator = 0.0
        denominator = 0.0
        for idx, value in enumerate(float_values):
            diff = value - mean_value
            denominator += diff * diff
            if idx >= lag:
                numerator += diff * (float_values[idx - lag] - mean_value)
        if denominator == 0.0:
            return 0.0
        correlation = numerator / denominator
        return self._clamp(correlation, -1.0, 1.0)

    def _return_entropy(self, returns: Sequence[float]) -> float:
        if not returns:
            return 0.0
        positive = sum(1 for value in returns if value > 0.0)
        negative = sum(1 for value in returns if value < 0.0)
        flat = len(returns) - positive - negative
        total = len(returns)
        entropy = 0.0
        for count in (positive, negative, flat):
            if count <= 0:
                continue
            probability = count / total
            entropy -= probability * math.log(probability)
        max_entropy = math.log(3)
        if max_entropy == 0.0:
            return 0.0
        normalized = entropy / max_entropy
        return self._clamp(normalized, 0.0, 1.0)

    def _hurst_exponent(self, returns: Sequence[float]) -> float:
        if len(returns) < 8:
            return 0.5
        float_returns = [float(value) for value in returns]
        max_lag = min(len(float_returns) // 2, 20)
        if max_lag < 2:
            return 0.5
        log_lags: list[float] = []
        log_rs: list[float] = []
        for lag in range(2, max_lag + 1):
            segment_length = lag
            segment_count = len(float_returns) // segment_length
            if segment_count <= 0:
                continue
            rs_values: list[float] = []
            for segment_index in range(segment_count):
                start = segment_index * segment_length
                end = start + segment_length
                segment = float_returns[start:end]
                mean_segment = mean(segment)
                deviations = [value - mean_segment for value in segment]
                cumulative: list[float] = []
                total = 0.0
                for deviation in deviations:
                    total += deviation
                    cumulative.append(total)
                if not cumulative:
                    continue
                span = max(cumulative) - min(cumulative)
                stdev = pstdev(segment) if len(segment) > 1 else 0.0
                if stdev > 0.0:
                    rs_values.append(span / stdev)
            if not rs_values:
                continue
            average_rs = sum(rs_values) / len(rs_values)
            if average_rs <= 0.0:
                continue
            log_lags.append(math.log(segment_length))
            log_rs.append(math.log(average_rs))
        if len(log_lags) < 2 or len(log_rs) < 2:
            return 0.5
        mean_log_lag = sum(log_lags) / len(log_lags)
        mean_log_rs = sum(log_rs) / len(log_rs)
        numerator = 0.0
        denominator = 0.0
        for lag_value, rs_value in zip(log_lags, log_rs, strict=False):
            diff_lag = lag_value - mean_log_lag
            numerator += diff_lag * (rs_value - mean_log_rs)
            denominator += diff_lag * diff_lag
        if denominator == 0.0:
            return 0.5
        hurst = numerator / denominator
        return self._clamp(hurst, 0.0, 1.0)

    def _linear_regression(self, values: Sequence[float]) -> tuple[float, float, float]:
        if len(values) < 2:
            last = float(values[-1]) if values else 0.0
            return 0.0, last, 0.0
        n = len(values)
        mean_x = (n - 1) / 2.0
        mean_y = mean(values)
        numerator = 0.0
        denominator = 0.0
        variance_y = 0.0
        for idx, raw_value in enumerate(values):
            x = float(idx)
            y = float(raw_value)
            diff_x = x - mean_x
            diff_y = y - mean_y
            numerator += diff_x * diff_y
            denominator += diff_x * diff_x
            variance_y += diff_y * diff_y
        slope = numerator / denominator if denominator > 0 else 0.0
        intercept = mean_y - slope * mean_x
        rvalue = 0.0
        if denominator > 0 and variance_y > 0:
            rvalue = numerator / math.sqrt(denominator * variance_y)
            rvalue = max(-1.0, min(1.0, rvalue))
        return slope, intercept, rvalue

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))


# Import na końcu, aby uniknąć cyklicznych zależności z modułem data.ohlcv.cache
from bot_core.data.ohlcv.cache import CachedOHLCVSource  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["FeatureDataset", "FeatureEngineer", "FeatureVector"]
