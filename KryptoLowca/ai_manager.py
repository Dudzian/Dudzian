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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

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


try:  # pragma: no cover - domyślna implementacja
    from data_preprocessor import windowize as _default_windowize  # type: ignore
except Exception:  # pragma: no cover - brak modułu w środowisku testowym

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
                try:
                    model = _AIModels(input_size=X.shape[-1], seq_len=int(seq_len), model_type=model_name)
                except Exception as exc:
                    logger.error("Nie można utworzyć modelu %s: %s", model_name, exc)
                    raise

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
            model = None
            chosen_type = None
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
                    X_tmp, y_tmp = _windowize(df, feats, min(len(df) // 2, max(2, self.ai_threshold_bps and 10)), "close")
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


__all__ = ["AIManager", "TrainResult", "_AIModels", "_windowize"]
