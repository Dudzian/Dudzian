# managers/ai_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# --- Defensywny import AIModels z ai_models.py ---
_AI_IMPORT_ERR: Optional[BaseException] = None
AIModels = None  # type: ignore[assignment]

try:
    # preferowany wariant – tak jak w Twoim repo
    from ai_models import AIModels as _AIModels  # type: ignore[import-not-found]
    AIModels = _AIModels  # type: ignore[assignment]
except Exception as e1:
    _AI_IMPORT_ERR = e1
    try:
        # spróbuj załadować moduł i poszukać alternatywnych nazw
        import importlib
        _aim = importlib.import_module("ai_models")
        for candidate in ("AIModels", "AIModel", "Models", "EnsembleModels"):
            if hasattr(_aim, candidate):
                AIModels = getattr(_aim, candidate)  # type: ignore[assignment]
                logger.info(f"AI Manager: using '{candidate}' from ai_models as AIModels.")
                break
        if AIModels is None:
            raise ImportError("No AIModels-like class found in ai_models.py")
    except Exception as e2:
        _AI_IMPORT_ERR = e2
        # Utwórz shim z tą samą sygnaturą, który podniesie czytelny wyjątek dopiero przy użyciu
        class _AIModelsShim:  # zgodny interfejs
            def __init__(self, *a, **kw):
                raise RuntimeError(
                    "AI models are unavailable: expected class 'AIModels' in ai_models.py. "
                    f"Import failures: first={repr(e1)}, second={repr(e2)}. "
                    "Upewnij się, że plik ai_models.py zawiera klasę AIModels z metodami "
                    "train(...) i predict_series(...)."
                )
            def train(self, *a, **kw):  # pragma: no cover
                raise RuntimeError("AIModels shim – training not available.")
            def predict_series(self, *a, **kw):  # pragma: no cover
                raise RuntimeError("AIModels shim – prediction not available.")
        AIModels = _AIModelsShim  # type: ignore[assignment]
        logger.error("AI Manager: failed to import AIModels. Using shim; GUI will still start, "
                     "but training/prediction will raise a clear error.")

# ========================= AIManager =========================
class AIManager:
    """
    Trening modeli, predykcje, cykliczny retrain + ranking po hit-rate.
    Thread-safe: blokada RLock żeby GUI mogło bezpiecznie sprawdzać stan.
    """
    def __init__(self):
        # Przechowujemy wiele modeli (np. do ensemblera); klucz = nazwa typu modelu
        self.models: Dict[str, Any] = {}
        self.training_lock = threading.RLock()
        self.current_model: Optional[Any] = None
        self._training_flag = threading.Event()

    # --------- pomocnicze: przygotowanie X/y ---------
    def _build_xy(self, df: pd.DataFrame, feats: List[str], seq_len: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if df is None or df.empty or any(c not in df.columns for c in feats + ["close"]):
            return None, None
        arr = df[feats].to_numpy(dtype=float)
        cl = df["close"].to_numpy(dtype=float)
        if len(arr) <= seq_len + 1:
            return None, None
        X, y = [], []
        for i in range(seq_len, len(arr)):
            X.append(arr[i - seq_len:i])
            prev = cl[i - 1] if cl[i - 1] != 0 else 1e-12
            y.append((cl[i] / prev) - 1.0)
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    # --------- predykcja dla aktualnego modelu ---------
    def predict_series(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Optional[pd.Series]:
        feats = feature_cols or ["open", "high", "low", "close", "volume"]
        with self.training_lock:
            if self.current_model is None:
                logger.debug("AIManager.predict_series: no current model.")
                return None
            try:
                return self.current_model.predict_series(df, feature_cols=feats)
            except Exception as e:
                logger.error(f"AIManager.predict_series error: {e}")
                return None

    # --------- metryka: trafność kierunku ---------
    def _eval_hit_rate(self, model: Any, df: pd.DataFrame, window: int) -> float:
        try:
            K = int(max(10, min(window, len(df) - 2)))
            if K <= 0:
                return -1.0
            df_eval = df.tail(K + 1).copy()
            preds = model.predict_series(df_eval, feature_cols=["open", "high", "low", "close", "volume"])
            if preds is None or len(preds) < 1:
                return -1.0
            df_eval["ret"] = df_eval["close"].pct_change().fillna(0.0)
            preds = preds.reindex(df_eval.index).fillna(0.0)
            real_sign = np.sign(df_eval["ret"].values[1:])
            pred_sign = np.sign(preds.values[:-1])
            ok = (real_sign == pred_sign).astype(float)
            return float(ok.mean())
        except Exception as e:
            logger.error(f"_eval_hit_rate error: {e}")
            return -1.0

    # --------- główna procedura treningu wielu modeli ---------
    def train_all_models(
        self,
        df: pd.DataFrame,
        seq_len: int,
        epochs: int,
        batch_size: int,
        model_types: List[str],
        progress_callback: Callable[[str, float, Optional[float]], None],
    ) -> Optional[str]:
        """
        Trenuje sekwencyjnie i wybiera najlepszy model po hit-rate.
        Zwraca nazwę najlepszego modelu lub None.
        """
        if self._training_flag.is_set():
            logger.info("AI training already running — skipping.")
            return None

        self._training_flag.set()
        try:
            feats = ["open", "high", "low", "close", "volume"]
            X, y = self._build_xy(df, feats, seq_len)
            if X is None or y is None:
                logger.warning("AI training skipped: insufficient data.")
                return None

            best_model: Optional[Any] = None
            best_type: Optional[str] = None
            best_score: float = -1.0

            for mtype in model_types:
                try:
                    # konstrukcja modelu
                    ai = AIModels(input_size=len(feats), seq_len=seq_len, model_type=mtype)

                    # raportowanie postępu (0..1)
                    def cb(epoch, total_epochs, train_loss, val_loss):
                        try:
                            p = float(epoch) / float(total_epochs) if total_epochs else 0.0
                        except Exception:
                            p = 0.0
                        try:
                            progress_callback(mtype, p, None)
                        except Exception:
                            pass

                    # trening
                    ai.train(
                        X,
                        y,
                        epochs=int(max(1, epochs)),
                        batch_size=int(max(1, batch_size)),
                        progress_callback=cb,
                        model_out=None,
                        verbose=False,
                    )

                    # metryka trafności
                    score = self._eval_hit_rate(ai, df, max(50, seq_len))
                    try:
                        progress_callback(mtype, 1.0, score)
                    except Exception:
                        pass

                    if score > best_score:
                        best_model, best_type, best_score = ai, mtype, score

                except Exception as e:
                    # dla każdego modelu log i przechodzimy dalej
                    logger.error(f"AI training error ({mtype}): {e}")
                    try:
                        progress_callback(mtype, 1.0, -1.0)
                    except Exception:
                        pass
                    continue

            # ustaw najlepszy i dodaj do puli
            with self.training_lock:
                self.current_model = best_model
                if best_type and best_model is not None:
                    self.models[best_type] = best_model

            if best_type is None:
                logger.warning("AI training finished; no valid model.")
            else:
                logger.info(f"AI training finished. Best={best_type}.")

            return best_type
        finally:
            self._training_flag.clear()

    # Ręczne ustawienie modelu (np. po wczytaniu z dysku)
    def set_model(self, model: Any):
        with self.training_lock:
            self.current_model = model
