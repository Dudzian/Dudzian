# -*- coding: utf-8 -*-
"""Asynchroniczny menedżer modeli AI zgodny z testami jednostkowymi.

Moduł zapewnia interfejs wysokiego poziomu wykorzystywany przez stare
skrypty i testy (``tests/test_ai_manager.py``). Został zaprojektowany tak,
aby współpracował z nową architekturą, ale jednocześnie zachował kontrakt
API znany z pierwszych iteracji projektu.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:  # pragma: no cover - środowisko testowe może nie mieć joblib
    import joblib
except Exception:  # pragma: no cover - fallback na pickle
    joblib = None  # type: ignore
    import pickle

    def _joblib_dump(obj: Any, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(obj, fh)

    def _joblib_load(path: Path) -> Any:
        with path.open("rb") as fh:
            return pickle.load(fh)
else:

    def _joblib_dump(obj: Any, path: Path) -> None:
        joblib.dump(obj, path)

    def _joblib_load(path: Path) -> Any:
        return joblib.load(path)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_AI_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - w testach zastępujemy _AIModels atrapą
    from ai_models import AIModels as _DefaultAIModels  # type: ignore
except Exception as exc:  # pragma: no cover - brak zależności na CI
    _AI_IMPORT_ERROR = exc

    class _DefaultAIModels:
        """Minimalny model fallback używany w testach bez zależności ML."""

        def __init__(self, input_size: int, seq_len: int, model_type: str = "rf") -> None:
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self._coef = np.zeros(input_size, dtype=float)

        async def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int,
            batch_size: int,
            progress_callback: Optional[Callable[..., None]] = None,
            model_out: Optional[str] = None,
            verbose: bool = False,
        ) -> None:
            if len(X) == 0:
                return
            self._coef = np.nan_to_num(X.mean(axis=1).mean(axis=0))
            if model_out:
                _joblib_dump(self, Path(model_out))

        def predict(self, X: np.ndarray) -> np.ndarray:
            if X.ndim == 3:
                features = X.reshape(X.shape[0], -1)
            else:
                features = X
            coef = self._coef
            if features.shape[1] != coef.size:
                coef = np.resize(coef, features.shape[1])
            return features @ coef

        async def predict_series(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
            arr = df[feature_cols].to_numpy(dtype=float)
            if arr.size == 0:
                return pd.Series(np.zeros(len(df)), index=df.index)
            coef = np.resize(self._coef, arr.shape[1])
            preds = arr @ coef
            return pd.Series(preds, index=df.index)

        @staticmethod
        def load_model(path: str) -> "_DefaultAIModels":
            return _joblib_load(Path(path))

# --- Import funkcji windowize z różnych możliwych miejsc, z bezpiecznym fallbackiem ---
_default_windowize: Callable[..., Tuple[np.ndarray, np.ndarray]]
try:  # najpierw wariant namespacowany
    from KryptoLowca.data_preprocessor import windowize as _default_windowize  # type: ignore
except Exception:
    try:  # następnie wariant lokalny
        from data_preprocessor import windowize as _default_windowize  # type: ignore
    except Exception:  # fallback minimalny – działa w środowisku testowym bez zależności
        def _default_windowize(df: pd.DataFrame, feature_cols: List[str], seq_len: int, target_col: str):
            """Minimalna implementacja okienkująca dane dla modeli sekwencyjnych."""
            if seq_len <= 0 or len(df) <= seq_len:
                raise ValueError("Za mało danych do przygotowania sekwencji.")
            X: List[np.ndarray] = []
            y: List[float] = []
            values = df[feature_cols].to_numpy(dtype=float)
            target = df[target_col].to_numpy(dtype=float)
            for idx in range(seq_len, len(df)):
                X.append(values[idx - seq_len : idx])
                prev = target[idx - 1] if target[idx - 1] != 0 else 1e-12
                y.append((target[idx] / prev) - 1.0)
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

# Zmienne modułowe podmieniane w testach jednostkowych
_AIModels = _DefaultAIModels
_windowize = _default_windowize


@dataclass(slots=True)
class TrainResult:
    """Podsumowanie treningu pojedynczego modelu."""

    model_type: str
    hit_rate: float
    model_path: Optional[str]


@dataclass(slots=True)
class ModelEvaluation:
    """Szczegółowa ocena modelu wykorzystywana przy selekcji strategii."""

    model_type: str
    hit_rate: float
    pnl: float
    sharpe: float
    cv_scores: List[float] = field(default_factory=list)
    model_path: Optional[str] = None

    def composite_score(self) -> float:
        """Łączna ocena – średnia trafności ważona Sharpe'em."""

        sharpe_bonus = max(self.sharpe, 0.0)
        return float(self.hit_rate * (1.0 + sharpe_bonus))


@dataclass(slots=True)
class StrategySelectionResult:
    """Wynik wyboru najlepszego modelu dla strategii."""

    symbol: str
    best_model: str
    evaluations: List[ModelEvaluation]
    decided_at: datetime
    drift_report: Optional["DriftReport"] = None
    predictions: Optional[pd.Series] = None


@dataclass(slots=True)
class DriftReport:
    """Raport detekcji dryfu danych wejściowych."""

    feature_drift: float
    volatility_shift: float
    triggered: bool
    threshold: float


@dataclass(slots=True)
class TrainingSchedule:
    """Reprezentuje zaplanowane zadanie treningowe."""

    symbol: str
    interval_seconds: float
    task: asyncio.Task
    model_types: Tuple[str, ...]
    seq_len: int


class AIManager:
    """Wysokopoziomowy kontroler treningu i predykcji modeli AI.

    Interfejs jest asynchroniczny, aby łatwo współpracował z GUI oraz
    umożliwiał blokadę podczas długich operacji treningowych. Równolegle
    dba o higienę danych oraz ograniczenie sygnałów do rozsądnego zakresu.
    """

    def __init__(self, *, ai_threshold_bps: float = 5.0, model_dir: str | Path = "models") -> None:
        self.ai_threshold_bps = float(ai_threshold_bps)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.models: Dict[str, Any] = {}
        self._schedules: Dict[str, TrainingSchedule] = {}
        self._recent_signals: Dict[str, deque[float]] = {}

    # -------------------------- API pomocnicze --------------------------
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        sym = (symbol or "").strip().lower()
        if not sym:
            raise ValueError("Symbol nie może być pusty.")
        return sym

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, feature_cols: Iterable[str]) -> None:
        if df is None or df.empty:
            raise ValueError("DataFrame z danymi jest pusty.")
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Brak kolumn wymaganych przez model: {missing}")
        if df.isna().any().any():
            raise ValueError("Dane wejściowe zawierają braki (NaN).")

    def _model_key(self, symbol: str, model_type: str) -> str:
        return f"{self._normalize_symbol(symbol)}:{model_type.lower()}"

    def _sanitize_predictions(self, series: pd.Series) -> pd.Series:
        """Ogranicz sygnały do zakresu [-1, 1] i usuń wartości odstające."""
        sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sanitized = sanitized.clip(lower=-1.0, upper=1.0)
        return sanitized

    def _track_signal(self, symbol: str, signals: pd.Series) -> None:
        """Zachowaj ostatnie predykcje do monitorowania dryfu."""

        key = self._normalize_symbol(symbol)
        buffer = self._recent_signals.setdefault(key, deque(maxlen=500))
        buffer.extend(float(v) for v in signals.values if np.isfinite(v))

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        if std == 0.0:
            return 0.0
        return mean / std

    async def _train_single_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seq_len: int,
        epochs: int,
        batch_size: int,
        model_path: Path,
    ) -> Any:
        model = _AIModels(input_size=X.shape[-1], seq_len=int(seq_len), model_type=model_name)

        async def _run_train() -> None:
            train_kwargs = dict(
                X=X,
                y=y,
                epochs=int(max(1, epochs)),
                batch_size=int(max(1, batch_size)),
                progress_callback=(lambda *_: None),
                model_out=str(model_path),
                verbose=False,
            )
            result = model.train(**train_kwargs)
            if inspect.isawaitable(result):
                await result

        await _run_train()
        return model

    def detect_drift(
        self,
        baseline: pd.DataFrame,
        recent: pd.DataFrame,
        feature_cols: Optional[Iterable[str]] = None,
        *,
        threshold: float = 0.35,
    ) -> DriftReport:
        """Porównaj zmienność cech i zgłoś dryf."""

        feats = list(feature_cols or ["open", "high", "low", "close", "volume"])
        self._validate_dataframe(baseline, feats)
        self._validate_dataframe(recent, feats)

        baseline_pct = baseline[feats].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        recent_pct = recent[feats].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        if baseline_pct.empty or recent_pct.empty:
            return DriftReport(0.0, 0.0, False, threshold)

        baseline_std = baseline_pct.std().replace(0.0, np.nan)
        recent_std = recent_pct.std()
        volatility_shift = float(
            ((recent_std - baseline_std).abs() / (baseline_std.abs() + 1e-9))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .max()
        )

        baseline_mean = baseline_pct.mean()
        recent_mean = recent_pct.mean()
        feature_drift = float(
            ((recent_mean - baseline_mean).abs() / (baseline_std.abs() + 1e-9))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .max()
        )

        triggered = volatility_shift > threshold or feature_drift > threshold
        return DriftReport(feature_drift=feature_drift, volatility_shift=volatility_shift, triggered=triggered, threshold=threshold)

    async def rank_models(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_types: Iterable[str],
        *,
        seq_len: int = 64,
        folds: int = 3,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> List[ModelEvaluation]:
        """Przeprowadź walidację krzyżową modeli i zwróć ranking."""

        feature_cols = ["open", "high", "low", "close", "volume"]
        self._validate_dataframe(df, feature_cols)
        symbol_key = self._normalize_symbol(symbol)
        folds = max(2, int(folds))

        async with self._lock:
            X, y = _windowize(df, feature_cols, int(seq_len), "close")
            if X is None or y is None or len(X) <= folds:
                raise ValueError("Za mało danych do walidacji modeli AI.")

            total = len(X)
            fold_size = max(1, total // folds)
            evaluations: List[ModelEvaluation] = []

            for model_type in model_types:
                model_name = str(model_type).lower()
                cv_scores: List[float] = []
                pnl_scores: List[float] = []

                for fold_idx in range(folds):
                    start = fold_idx * fold_size
                    end = total if fold_idx == folds - 1 else min(total, start + fold_size)
                    if start >= end:
                        continue

                    X_val = X[start:end]
                    y_val = y[start:end]
                    X_train = np.concatenate((X[:start], X[end:]), axis=0) if start > 0 or end < total else X
                    y_train = np.concatenate((y[:start], y[end:]), axis=0) if start > 0 or end < total else y

                    if len(X_train) == 0 or len(X_val) == 0:
                        continue

                    model_path = self.model_dir / f"{symbol_key}:{model_name}.fold{fold_idx}.joblib"
                    model = await self._train_single_model(
                        model_name,
                        X_train,
                        y_train,
                        seq_len=seq_len,
                        epochs=epochs,
                        batch_size=batch_size,
                        model_path=model_path,
                    )

                    pred = model.predict(X_val)
                    if inspect.isawaitable(pred):
                        pred = await pred
                    preds = np.asarray(pred, dtype=float).flatten()
                    if preds.shape[0] != y_val.shape[0]:
                        preds = np.resize(preds, y_val.shape[0])

                    guesses = np.sign(preds)
                    target = np.sign(np.asarray(y_val, dtype=float))
                    cv_scores.append(float((guesses == target).mean()))

                    pnl_vector = preds * np.asarray(y_val, dtype=float)
                    pnl_scores.append(float(np.sum(pnl_vector)))

                hit_rate = float(np.mean(cv_scores)) if cv_scores else 0.0
                pnl = float(np.mean(pnl_scores)) if pnl_scores else 0.0
                sharpe = self._sharpe_ratio(np.asarray(pnl_scores, dtype=float))
                evaluations.append(
                    ModelEvaluation(
                        model_type=model_name,
                        hit_rate=hit_rate,
                        pnl=pnl,
                        sharpe=sharpe,
                        cv_scores=cv_scores,
                        model_path=str(self.model_dir / f"{symbol_key}:{model_name}.joblib"),
                    )
                )

            evaluations.sort(key=lambda ev: ev.composite_score(), reverse=True)
            return evaluations

    async def select_best_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_types: Iterable[str],
        *,
        seq_len: int = 64,
        folds: int = 3,
    ) -> StrategySelectionResult:
        """Wybierz najlepszy model w oparciu o ranking."""

        evaluations = await self.rank_models(
            symbol,
            df,
            model_types,
            seq_len=seq_len,
            folds=folds,
        )
        if not evaluations:
            raise ValueError("Nie udało się ocenić żadnego modelu.")

        best = evaluations[0]
        decided_at = datetime.now(timezone.utc)
        return StrategySelectionResult(
            symbol=self._normalize_symbol(symbol),
            best_model=best.model_type,
            evaluations=evaluations,
            decided_at=decided_at,
        )

    def cancel_schedule(self, symbol: str) -> None:
        key = self._normalize_symbol(symbol)
        schedule = self._schedules.pop(key, None)
        if schedule and not schedule.task.done():
            schedule.task.cancel()

    def active_schedules(self) -> Dict[str, TrainingSchedule]:
        return dict(self._schedules)

    def _schedule_runner(
        self,
        symbol: str,
        df_provider: Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]],
        model_types: Iterable[str],
        interval_seconds: float,
        seq_len: int,
        epochs: int,
        batch_size: int,
    ) -> Callable[[], Awaitable[None]]:
        async def _runner() -> None:
            while True:
                try:
                    df_candidate = df_provider()
                    if inspect.isawaitable(df_candidate):
                        df_candidate = await df_candidate
                    await self.train_all_models(
                        symbol,
                        df_candidate,
                        model_types,
                        seq_len=seq_len,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # pragma: no cover - błędy runtime
                    logger.error("Błąd harmonogramu treningowego %s: %s", symbol, exc)
                    await asyncio.sleep(min(60.0, interval_seconds))

        return _runner

    def schedule_periodic_training(
        self,
        symbol: str,
        df_provider: Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]],
        model_types: Iterable[str],
        *,
        interval_seconds: float = 3600.0,
        seq_len: int = 64,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> TrainingSchedule:
        """Utwórz cykliczny trening modeli AI."""

        model_types_tuple = tuple(model_types)
        runner = self._schedule_runner(
            symbol,
            df_provider,
            model_types_tuple,
            interval_seconds,
            seq_len,
            epochs,
            batch_size,
        )
        task = asyncio.create_task(runner())
        schedule = TrainingSchedule(
            symbol=self._normalize_symbol(symbol),
            interval_seconds=float(interval_seconds),
            task=task,
            model_types=model_types_tuple,
            seq_len=int(seq_len),
        )
        self._schedules[schedule.symbol] = schedule
        return schedule

    async def run_pipeline(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_types: Iterable[str],
        *,
        seq_len: int = 64,
        folds: int = 3,
        baseline: Optional[pd.DataFrame] = None,
    ) -> StrategySelectionResult:
        """Zautomatyzowany pipeline: selekcja modelu → trening → predykcja."""

        selection = await self.select_best_model(
            symbol,
            df,
            model_types,
            seq_len=seq_len,
            folds=folds,
        )
        await self.train_all_models(
            symbol,
            df,
            [selection.best_model],
            seq_len=seq_len,
        )
        predictions = await self.predict_series(
            symbol,
            df,
            model_types=[selection.best_model],
        )
        selection.predictions = predictions
        if baseline is not None:
            selection.drift_report = self.detect_drift(baseline, df)
        self._track_signal(symbol, predictions)
        return selection

    # ----------------------------- Trening -----------------------------
    async def train_all_models(
        self,
        symbol: str,
        df: pd.DataFrame,
        model_types: Iterable[str],
        *,
        seq_len: int = 64,
        epochs: int = 10,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[str, float, Optional[float]], None]] = None,
    ) -> Dict[str, TrainResult]:
        """Wytrenuj sekwencyjnie modele i zwróć metryki trafności."""

        feature_cols = ["open", "high", "low", "close", "volume"]
        self._validate_dataframe(df, feature_cols)
        symbol_key = self._normalize_symbol(symbol)

        async with self._lock:
            X, y = _windowize(df, feature_cols, int(seq_len), "close")
            if X is None or y is None or len(X) == 0:
                raise ValueError("Za mało danych do treningu modeli AI.")

            results: Dict[str, TrainResult] = {}
            for model_type in model_types:
                model_name = str(model_type).lower()
                model_path = self.model_dir / f"{symbol_key}:{model_name}.joblib"
                model = await self._train_single_model(
                    model_name,
                    X,
                    y,
                    seq_len=seq_len,
                    epochs=epochs,
                    batch_size=batch_size,
                    model_path=model_path,
                )

                preds = None
                try:
                    pred = model.predict(X)
                    if inspect.isawaitable(pred):
                        pred = await pred
                    preds = np.asarray(pred, dtype=float).flatten()
                except Exception:
                    logger.debug("Model %s nie udostępnia metody predict – pomijam ocenę.", model_name)

                hit_rate = 0.0
                if preds is not None and len(preds) == len(y):
                    target = np.sign(np.asarray(y, dtype=float))
                    guesses = np.sign(preds)
                    hit_rate = float((guesses == target).mean())

                key = self._model_key(symbol, model_name)
                self.models[key] = model
                results[model_name] = TrainResult(model_type=model_name, hit_rate=hit_rate, model_path=str(model_path))

                if progress_callback:
                    try:
                        progress_callback(model_name, 1.0, hit_rate)
                    except Exception:  # pragma: no cover - callback użytkownika
                        logger.debug("Callback post-treningowy zgłosił wyjątek.")

            return results

    # ---------------------------- Predykcje ----------------------------
    async def predict_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        model_types: Optional[Iterable[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.Series:
        """Wygeneruj prognozę zwróconą jako seria Pandas."""

        feats = feature_cols or ["open", "high", "low", "close", "volume"]
        self._validate_dataframe(df, feats)
        symbol_key = self._normalize_symbol(symbol)
        candidates = list(model_types or [mt.split(":")[-1] for mt in self.models.keys() if mt.startswith(symbol_key)])
        if not candidates:
            raise ValueError("Brak wytrenowanych modeli dla podanego symbolu.")

        async with self._lock:
            model: Any = None
            chosen_type: Optional[str] = None
            for candidate in candidates:
                key = self._model_key(symbol, candidate)
                model = self.models.get(key)
                if model is not None:
                    chosen_type = candidate
                    break
                path = self.model_dir / f"{symbol_key}:{candidate.lower()}.joblib"
                if path.exists():
                    try:
                        model = _joblib_load(path)
                        self.models[key] = model
                        chosen_type = candidate
                        break
                    except Exception as exc:
                        logger.error("Nie można wczytać modelu %s: %s", path, exc)
            if model is None:
                # automatyczne doraźne przetrenowanie prostego modelu fallback
                candidate = candidates[0]
                try:
                    X_tmp, y_tmp = _windowize(
                        df, feats, min(len(df) // 2, max(2, int(self.ai_threshold_bps and 10))), "close"
                    )
                except Exception as exc:
                    logger.error("Fallback windowize failed: %s", exc)
                    X_tmp, y_tmp = None, None
                if X_tmp is not None and y_tmp is not None and len(X_tmp) > 0:
                    model = _AIModels(input_size=X_tmp.shape[-1], seq_len=X_tmp.shape[1], model_type=candidate)
                    result = model.train(
                        X_tmp,
                        y_tmp,
                        epochs=1,
                        batch_size=max(1, min(32, len(X_tmp))),
                        progress_callback=None,
                        model_out=None,
                        verbose=False,
                    )
                    if inspect.isawaitable(result):
                        await result
                    key = self._model_key(symbol, candidate)
                    self.models[key] = model
                    chosen_type = candidate
                else:
                    raise ValueError("Nie znaleziono żadnego modelu spełniającego kryteria.")

            preds = model.predict_series(df, feature_cols=feats)
            if inspect.isawaitable(preds):
                preds = await preds
            if not isinstance(preds, pd.Series):
                preds = pd.Series(np.asarray(preds, dtype=float), index=df.index)
            sanitized = self._sanitize_predictions(preds)
            self._track_signal(symbol, sanitized)
            logger.debug("Zwracam predykcje modelu %s dla %s", chosen_type, symbol_key)
            return sanitized

    # --------------------------- Import modeli --------------------------
    async def import_model(self, symbol: str, model_type: str, path: str | Path) -> None:
        """Załaduj model zapisany na dysku i zarejestruj go w menedżerze."""

        symbol_key = self._normalize_symbol(symbol)
        key = self._model_key(symbol, model_type)
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        def _load() -> Any:
            return _joblib_load(model_path)

        model = await asyncio.to_thread(_load)
        async with self._lock:
            self.models[key] = model
            target = self.model_dir / f"{symbol_key}:{model_type.lower()}.joblib"
            if target != model_path:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(_joblib_dump, model, target)

        logger.info("Zaimportowano model %s z pliku %s", key, model_path)


__all__ = [
    "AIManager",
    "TrainResult",
    "ModelEvaluation",
    "StrategySelectionResult",
    "DriftReport",
    "TrainingSchedule",
    "_AIModels",
    "_windowize",
]
