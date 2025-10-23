# -*- coding: utf-8 -*-
"""Asynchroniczny menedżer modeli AI zgodny z testami jednostkowymi.

Moduł zapewnia interfejs wysokiego poziomu wykorzystywany przez stare
skrypty i testy (``tests/test_ai_manager.py``). Został zaprojektowany tak,
aby współpracował z nową architekturą, ale jednocześnie zachował kontrakt
API znany z pierwszych iteracji projektu.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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

from ._license import ensure_ai_signals_enabled
from .feature_engineering import FeatureDataset
from .inference import DecisionModelInference, ModelRepository
from .models import ModelArtifact, ModelScore
from .regime import (
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSummary,
)
from .scheduler import (
    RetrainingScheduler,
    ScheduledTrainingJob,
    TrainingScheduler,
    WalkForwardResult,
    WalkForwardValidator,
)
from .training import ModelTrainer

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
_history_logger = logger.getChild("history")


def _utcnow() -> datetime:
    """Zwróć bieżący czas w strefie UTC."""

    return datetime.now(timezone.utc)


def _select_metric_block(metrics: Mapping[str, object]) -> Mapping[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    if metrics and any(isinstance(value, Mapping) for value in metrics.values()):
        summary = metrics.get("summary")
        if isinstance(summary, Mapping):
            return {
                str(key): float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float))
            }
        for key in ("test", "validation", "train"):
            candidate = metrics.get(key)
            if isinstance(candidate, Mapping) and candidate:
                return {
                    str(metric_name): float(metric_value)
                    for metric_name, metric_value in candidate.items()
                    if isinstance(metric_value, (int, float))
                }
        for value in metrics.values():
            if isinstance(value, Mapping):
                return {
                    str(metric_name): float(metric_value)
                    for metric_name, metric_value in value.items()
                    if isinstance(metric_value, (int, float))
                }
        return {}
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }

LoggerLike = Union[logging.Logger, logging.LoggerAdapter]


def _ensure_logger(logger_like: Optional[LoggerLike]) -> LoggerLike:
    if logger_like is None:
        return _history_logger
    if not isinstance(logger_like, (logging.Logger, logging.LoggerAdapter)):
        raise TypeError("logger musi być instancją logging.Logger lub logging.LoggerAdapter")
    return logger_like


def _emit_history_log(
    message: str,
    *,
    level: int,
    logger_like: Optional[LoggerLike] = None,
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    resolved_logger = _ensure_logger(logger_like)
    if not isinstance(level, int):
        raise TypeError("level musi być liczbą całkowitą")
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra musi być mapowaniem jeśli jest podane")
    log_kwargs: Dict[str, Any] = {}
    if extra:
        log_kwargs["extra"] = dict(extra)
    log_kwargs["stacklevel"] = stacklevel
    resolved_logger.log(level, message, **log_kwargs)


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_mean(values: Iterable[object]) -> Optional[float]:
    collected: List[float] = []
    for value in values:
        number = _safe_float(value)
        if number is not None:
            collected.append(number)
    if not collected:
        return None
    return sum(collected) / len(collected)

_AI_IMPORT_ERROR: Optional[BaseException] = None
_FALLBACK_ACTIVE = False


def _bundle_import_errors(primary: BaseException, secondary: BaseException) -> BaseException:
    """Połącz dwa wyjątki importu w jeden obiekt z zachowaniem kontekstu."""

    try:
        return ExceptionGroup("AI backend import failed", [primary, secondary])  # type: ignore[name-defined]
    except NameError:  # pragma: no cover - Python < 3.11
        secondary.__cause__ = primary  # type: ignore[attr-defined]
        return secondary


def _flatten_exception_messages(error: BaseException) -> str:
    """Zbuduj zwięzły opis z łańcucha wyjątków importu."""

    pieces: List[str] = []
    stack: List[BaseException] = [error]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        identifier = id(current)
        if identifier in seen:
            continue
        seen.add(identifier)

        message = str(current).strip()
        if not message:
            message = current.__class__.__name__
        else:
            message = f"{current.__class__.__name__}: {message}"
        pieces.append(message)

        nested = getattr(current, "exceptions", None)
        if isinstance(nested, Iterable):
            for child in nested:
                if isinstance(child, BaseException):
                    stack.append(child)

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)

        context = getattr(current, "__context__", None)
        suppressed = getattr(current, "__suppress_context__", False)
        if not suppressed and isinstance(context, BaseException):
            stack.append(context)

    unique_messages = []
    seen_text: set[str] = set()
    for piece in pieces:
        if piece not in seen_text:
            unique_messages.append(piece)
            seen_text.add(piece)

    return " | ".join(unique_messages)


def _iter_exception_chain(root: BaseException) -> Iterable[BaseException]:
    """Breadth-first traversal over nested, caused and contextual exceptions."""

    queue: deque[BaseException] = deque([root])
    seen: set[int] = set()
    while queue:
        current = queue.popleft()
        identifier = id(current)
        if identifier in seen:
            continue
        seen.add(identifier)
        yield current

        nested = getattr(current, "exceptions", None)
        if isinstance(nested, Iterable):
            for child in nested:
                if isinstance(child, BaseException):
                    queue.append(child)

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            queue.append(cause)

        context = getattr(current, "__context__", None)
        if not getattr(current, "__suppress_context__", False) and isinstance(context, BaseException):
            queue.append(context)


def _collect_exception_messages(error: BaseException) -> Tuple[str, ...]:
    """Collect formatted messages from an exception chain preserving order."""

    messages: list[str] = []
    seen_text: set[str] = set()
    for exc in _iter_exception_chain(error):
        message = str(exc).strip()
        if message:
            formatted = f"{exc.__class__.__name__}: {message}"
        else:
            formatted = exc.__class__.__name__
        if formatted not in seen_text:
            messages.append(formatted)
            seen_text.add(formatted)
    return tuple(messages)


def _collect_exception_types(error: BaseException) -> Tuple[str, ...]:
    """Collect fully-qualified exception type names from a chain."""

    types: list[str] = []
    seen: set[str] = set()
    for exc in _iter_exception_chain(error):
        name = f"{exc.__class__.__module__}.{exc.__class__.__qualname__}"
        if name not in seen:
            types.append(name)
            seen.add(name)
    return tuple(types)


def _build_exception_diagnostics(error: BaseException) -> Tuple[ExceptionDiagnostics, ...]:
    """Create structured diagnostics for each exception in the chain."""

    diagnostics: list[ExceptionDiagnostics] = []
    seen: set[int] = set()
    for exc in _iter_exception_chain(error):
        identifier = id(exc)
        if identifier in seen:
            continue
        seen.add(identifier)
        diagnostics.append(
            ExceptionDiagnostics(
                module=getattr(exc.__class__, "__module__", ""),
                qualname=getattr(exc.__class__, "__qualname__", exc.__class__.__name__),
                message=str(exc).strip(),
            )
        )
    return tuple(diagnostics)


def _is_fallback_degradation(reason: Optional[str]) -> bool:
    """Sprawdź, czy wskazany powód oznacza degradację przez fallback modeli."""

    if reason is None:
        return True
    prefix = reason.split(":", 1)[0].strip()
    return prefix in {"fallback_ai_models", "backend_validation_failed"}


def _build_fallback_ai_models() -> type:
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

    return _DefaultAIModels


try:  # pragma: no cover - w testach zastępujemy _AIModels atrapą
    from ai_models import AIModels as _DefaultAIModels  # type: ignore
except Exception as exc:  # pragma: no cover - brak zależności na CI
    try:
        _kryptolowca_ai_models = importlib.import_module("KryptoLowca.ai_models")
    except Exception as namespace_exc:
        _AI_IMPORT_ERROR = _bundle_import_errors(exc, namespace_exc)
        _FALLBACK_ACTIVE = True
        _DefaultAIModels = _build_fallback_ai_models()
    else:
        try:
            _DefaultAIModels = getattr(_kryptolowca_ai_models, "AIModels")
        except AttributeError as attr_exc:
            _AI_IMPORT_ERROR = _bundle_import_errors(exc, attr_exc)
            _FALLBACK_ACTIVE = True
            _DefaultAIModels = _build_fallback_ai_models()

# --- Import funkcji windowize z różnych możliwych miejsc, z bezpiecznym fallbackiem ---
_default_windowize: Callable[..., Tuple[np.ndarray, np.ndarray]]
try:  # najpierw wariant namespacowany
    from bot_core.data.preprocessing import windowize as _default_windowize  # type: ignore
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

try:  # pragma: no cover - DecisionOrchestrator nie zawsze dostępny w minimalnych buildach
    from bot_core.decision.orchestrator import DecisionOrchestrator as _DecisionOrchestrator
except Exception:  # pragma: no cover - komponent optionalny
    _DecisionOrchestrator = None  # type: ignore[misc]


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


@dataclass(frozen=True, slots=True)
class EnsembleDefinition:
    """Opis zespołu modeli łączonego w trakcie generowania predykcji."""

    name: str
    components: Tuple[str, ...]
    aggregation: str = "mean"
    weights: Optional[Tuple[float, ...]] = None

    def require_weights(self) -> Tuple[float, ...]:
        if self.weights is None:
            raise ValueError("Definicja zespołu nie zawiera wag.")
        return self.weights


@dataclass(slots=True)
class EnsembleRegistrySnapshot:
    """Migawka zarejestrowanych zespołów modeli."""

    ensembles: Dict[str, EnsembleDefinition] = field(default_factory=dict)

    def total_ensembles(self) -> int:
        return len(self.ensembles)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.ensembles))

    def get(self, name: str) -> Optional[EnsembleDefinition]:
        return self.ensembles.get(name)


@dataclass(slots=True)
class EnsembleRegistryDiff:
    """Różnica pomiędzy dwiema migawkami rejestru zespołów modeli."""

    added: Dict[str, EnsembleDefinition] = field(default_factory=dict)
    removed: Dict[str, EnsembleDefinition] = field(default_factory=dict)
    changed: Dict[str, Tuple[EnsembleDefinition, EnsembleDefinition]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def added_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.added))

    def removed_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.removed))

    def changed_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.changed))

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.added) | set(self.removed) | set(self.changed)))


@dataclass(frozen=True, slots=True)
class ExceptionDiagnostics:
    """Struktura opisująca pojedynczy wyjątek z łańcucha degradacji."""

    module: str
    qualname: str
    message: str

    @property
    def type_name(self) -> str:
        return f"{self.module}.{self.qualname}" if self.module else self.qualname

    @property
    def formatted(self) -> str:
        text = self.message.strip()
        if text:
            return f"{self.type_name}: {text}"
        return self.type_name

    def as_dict(self) -> Dict[str, str]:
        data: Dict[str, str] = {
            "module": self.module,
            "qualname": self.qualname,
            "type": self.type_name,
            "formatted": self.formatted,
        }
        if self.message:
            data["message"] = self.message
        return data


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Migawka stanu backendu modeli AI."""

    degraded: bool
    reason: str | None
    details: Tuple[str, ...] = ()
    exception_types: Tuple[str, ...] = ()
    exception_diagnostics: Tuple[ExceptionDiagnostics, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "degraded": self.degraded,
            "reason": self.reason,
            "details": list(self.details),
            "exception_types": list(self.exception_types),
            "exception_diagnostics": [diag.as_dict() for diag in self.exception_diagnostics],
        }


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


@dataclass(slots=True)
class PipelineSchedule:
    """Zaplanowany pipeline selekcji modeli i generowania sygnałów."""

    symbol: str
    interval_seconds: float
    task: asyncio.Task
    model_types: Tuple[str, ...]
    seq_len: int
    folds: int


@dataclass(slots=True)
class PipelineExecutionRecord:
    """Zapis pojedynczego uruchomienia pipeline'u selekcji modeli."""

    symbol: str
    decided_at: datetime
    best_model: str
    evaluations: Tuple[ModelEvaluation, ...]
    drift_report: Optional[DriftReport] = None
    prediction_count: int = 0
    prediction_mean: Optional[float] = None
    prediction_std: Optional[float] = None
    prediction_min: Optional[float] = None
    prediction_max: Optional[float] = None


@dataclass(slots=True)
class PipelineHistorySnapshot:
    """Migawka historii pipeline'u dla jednego lub wielu symboli."""

    records: Dict[str, Tuple[PipelineExecutionRecord, ...]] = field(default_factory=dict)

    def total_symbols(self) -> int:
        return len(self.records)

    def total_records(self) -> int:
        return sum(len(entries) for entries in self.records.values())

    def symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.records))

    def for_symbol(self, symbol: str) -> Tuple[PipelineExecutionRecord, ...]:
        return self.records.get(symbol, ())


@dataclass(slots=True)
class PipelineHistoryDiff:
    """Różnice pomiędzy dwiema migawkami historii pipeline'u."""

    added: Dict[str, Tuple[PipelineExecutionRecord, ...]] = field(default_factory=dict)
    removed: Dict[str, Tuple[PipelineExecutionRecord, ...]] = field(default_factory=dict)
    changed: Dict[str, Tuple[Tuple[PipelineExecutionRecord, ...], Tuple[PipelineExecutionRecord, ...]]] = field(
        default_factory=dict
    )

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def added_symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.added))

    def removed_symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.removed))

    def changed_symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.changed))

    def symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.added) | set(self.removed) | set(self.changed)))

    def total_added_records(self) -> int:
        return sum(len(records) for records in self.added.values())

    def total_removed_records(self) -> int:
        return sum(len(records) for records in self.removed.values())

    def total_changed_records(self) -> int:
        return sum(len(after) for _, after in self.changed.values())


PathInput = Union[str, Path, PathLike[str]]


class AIManager:
    """Wysokopoziomowy kontroler treningu i predykcji modeli AI.

    Interfejs jest asynchroniczny, aby łatwo współpracował z GUI oraz
    umożliwiał blokadę podczas długich operacji treningowych. Równolegle
    dba o higienę danych oraz ograniczenie sygnałów do rozsądnego zakresu.
    """

    def __init__(self, *, ai_threshold_bps: float = 5.0, model_dir: str | Path = "models") -> None:
        ensure_ai_signals_enabled("zarządzania modelami AI")
        self.ai_threshold_bps = float(ai_threshold_bps)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.models: Dict[str, Any] = {}
        self._schedules: Dict[str, TrainingSchedule] = {}
        self._pipeline_schedules: Dict[str, PipelineSchedule] = {}
        self._recent_signals: Dict[str, deque[float]] = {}
        self._active_models: Dict[str, str] = {}
        self._pipeline_history: Dict[str, deque[PipelineExecutionRecord]] = {}
        self._pipeline_history_limit = 100
        self._model_init_kwargs: Dict[str, Any] = {}
        self._ensembles: Dict[str, EnsembleDefinition] = {}
        self._regime_classifier = MarketRegimeClassifier()
        self._latest_regimes: Dict[str, MarketRegimeAssessment] = {}
        self._regime_histories: Dict[str, RegimeHistory] = {}
        self._degraded = False
        self._degradation_reason: str | None = None
        self._degradation_details: Tuple[str, ...] = ()
        self._degradation_exceptions: Tuple[BaseException, ...] = ()
        self._degradation_exception_types: Tuple[str, ...] = ()
        self._degradation_exception_diagnostics: Tuple[ExceptionDiagnostics, ...] = ()
        self._decision_model_repository: ModelRepository | None = None
        self._model_repository: ModelRepository | None = None
        self._repository_models: Dict[str, DecisionModelInference] = {}
        self._repository_paths: Dict[str, Path] = {}
        self._decision_inferences: Dict[str, DecisionModelInference] = {}
        self._decision_default_name: str | None = None
        self._decision_orchestrator: Any | None = None
        self._training_scheduler: TrainingScheduler | None = None
        self._scheduled_training_jobs: Dict[str, ScheduledTrainingJob] = {}
        self._job_artifact_paths: Dict[str, Path] = {}
        if _AI_IMPORT_ERROR is not None:
            self._degraded = True
            messages = _collect_exception_messages(_AI_IMPORT_ERROR)
            if messages:
                self._degradation_details = messages
                summary = _flatten_exception_messages(_AI_IMPORT_ERROR)
                self._degradation_reason = (
                    f"fallback_ai_models: {summary}" if summary else "fallback_ai_models"
                )
            else:
                self._degradation_reason = "fallback_ai_models"
            self._degradation_exceptions = (_AI_IMPORT_ERROR,)
            self._degradation_exception_types = _collect_exception_types(_AI_IMPORT_ERROR)
            self._degradation_exception_diagnostics = _build_exception_diagnostics(
                _AI_IMPORT_ERROR
            )
        elif _FALLBACK_ACTIVE:
            self._degraded = True
            self._degradation_reason = "fallback_ai_models"
        try:
            init_signature = inspect.signature(_AIModels.__init__)  # type: ignore[attr-defined]
        except (TypeError, ValueError, AttributeError):
            init_signature = None
        if init_signature is not None and "model_dir" in init_signature.parameters:
            self._model_init_kwargs["model_dir"] = self.model_dir

    # -------------------------- API pomocnicze --------------------------
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        sym = (symbol or "").strip().lower()
        if not sym:
            raise ValueError("Symbol nie może być pusty.")
        return sym

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        variant = (model_type or "").strip()
        if not variant:
            raise ValueError("Typ modelu nie może być pusty.")
        if ":" in variant:
            variant = variant.split(":")[-1]
        return variant.lower()

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
        return f"{self._normalize_symbol(symbol)}:{self._normalize_model_type(model_type)}"

    # --------------------------- Market regimes ---------------------------
    def assess_market_regime(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        *,
        price_col: str = "close",
    ) -> MarketRegimeAssessment:
        """Classify the current market regime for ``symbol``."""

        if not isinstance(market_data, pd.DataFrame):
            raise TypeError("market_data must be a pandas DataFrame")

        normalized_symbol = self._normalize_symbol(symbol)
        sanitized = market_data.dropna(subset=[price_col])
        if sanitized.empty:
            raise ValueError("market_data must contain at least one complete row")
        if price_col not in sanitized.columns:
            raise ValueError(f"Column {price_col!r} missing from market data")

        assessment = self._regime_classifier.assess(
            sanitized,
            price_col=price_col,
            symbol=normalized_symbol,
        )
        self._latest_regimes[normalized_symbol] = assessment
        history = self._regime_histories.setdefault(
            normalized_symbol,
            RegimeHistory(
                thresholds_loader=self._regime_classifier.thresholds_loader,
            ),
        )
        history.reload_thresholds(
            thresholds=self._regime_classifier.thresholds_snapshot()
        )
        history.update(assessment)
        _emit_history_log(
            f"Regime[{normalized_symbol}] => {assessment.regime.value} (risk={assessment.risk_score:.2f}, confidence={assessment.confidence:.2f})",
            level=logging.DEBUG,
        )
        return assessment

    def get_last_regime_assessment(self, symbol: str) -> Optional[MarketRegimeAssessment]:
        """Return a defensive copy of cached regime classification if available."""

        assessment = self._latest_regimes.get(self._normalize_symbol(symbol))
        if assessment is None:
            return None
        return deepcopy(assessment)

    def get_regime_summary(self, symbol: str) -> Optional[RegimeSummary]:
        """Zwróć wygładzoną historię reżimów dla danego symbolu."""

        history = self._regime_histories.get(self._normalize_symbol(symbol))
        if history is None:
            return None
        summary = history.summarise()
        if summary is None:
            return None
        return deepcopy(summary)

    def get_regime_thresholds(self, symbol: str | None = None) -> Mapping[str, Any]:
        """Expose defensive copy of thresholds either per symbol or classifier defaults."""

        if symbol is not None:
            history = self._regime_histories.get(self._normalize_symbol(symbol))
            if history is not None:
                return history.thresholds_snapshot()
        return self._regime_classifier.thresholds_snapshot()

    # --------------------------- Modele aktywne ---------------------------
    def set_active_model(self, symbol: str, model_type: str | None) -> None:
        """Ustaw aktywny model dla symbolu lub usuń go, podając ``None``."""

        key = self._normalize_symbol(symbol)
        if model_type is None or not str(model_type).strip():
            self._active_models.pop(key, None)
            return
        self._active_models[key] = self._normalize_model_type(str(model_type))

    def clear_active_model(self, symbol: str) -> None:
        """Usuń informację o aktywnym modelu dla danego symbolu."""

        self._active_models.pop(self._normalize_symbol(symbol), None)

    def get_active_model(self, symbol: str) -> Optional[str]:
        """Zwróć aktualnie aktywny model lub ``None`` jeśli nie ustawiono."""

        return self._active_models.get(self._normalize_symbol(symbol))

    def list_active_models(self) -> Dict[str, str]:
        """Zwróć kopię mapowania aktywnych modeli."""

        return dict(self._active_models)

    @property
    def is_degraded(self) -> bool:
        """Czy menedżer działa w trybie degradacji (tylko fallback)?"""

        return self._degraded

    @property
    def degradation_reason(self) -> str | None:
        """Zwraca powód degradacji (jeśli aktywna)."""

        return self._degradation_reason

    @property
    def degradation_details(self) -> Tuple[str, ...]:
        """Zwraca szczegóły degradacji backendu."""

        return self._degradation_details

    @property
    def degradation_exceptions(self) -> Tuple[BaseException, ...]:
        """Zwraca krotkę wyjątków opisujących degradację (jeśli dostępne)."""

        return self._degradation_exceptions

    @property
    def degradation_exception_types(self) -> Tuple[str, ...]:
        """Zwraca nazwy klas wyjątków związanych z degradacją."""

        return self._degradation_exception_types

    @property
    def degradation_exception_diagnostics(self) -> Tuple[ExceptionDiagnostics, ...]:
        """Zwraca szczegółowe dane wyjątków użyte do diagnostyki backendu."""

        return self._degradation_exception_diagnostics

    def backend_status(self) -> BackendStatus:
        """Zwraca migawkę stanu backendu modeli AI."""

        return BackendStatus(
            degraded=self._degraded,
            reason=self._degradation_reason,
            details=self._degradation_details,
            exception_types=self._degradation_exception_types,
            exception_diagnostics=self._degradation_exception_diagnostics,
        )

    def require_real_models(self) -> None:
        """Zgłasza wyjątek gdy dostępne są tylko fallbackowe modele AI."""

        if self._degraded or (not self._decision_inferences and not self._repository_models):
            reason = self._degradation_reason or "AI backend in degraded mode"
            detail_text = ""
            if self._degradation_details:
                joined = " | ".join(self._degradation_details)
                detail_text = f" (details: {joined})"
            raise RuntimeError(
                "Real models are required for live trading; current backend is degraded: "
                + str(reason)
                + detail_text
            )

    def _activate_degradation(
        self,
        reason: str,
        *,
        details: Tuple[str, ...] = (),
        exceptions: Tuple[BaseException, ...] = (),
        exception_types: Tuple[str, ...] = (),
        exception_diagnostics: Tuple[ExceptionDiagnostics, ...] = (),
        since: datetime | None = None,
    ) -> None:
        """Ustaw stan degradacji i zarejestruj zdarzenie w historii."""

        activation_time = since or _utcnow()
        if self._degraded:
            self._finalize_degradation_event(ended_at=activation_time)
        self._degraded = True
        self._degradation_reason = reason
        self._degradation_details = details
        self._degradation_exceptions = exceptions
        self._degradation_exception_types = exception_types
        self._degradation_exception_diagnostics = exception_diagnostics
        self._degradation_since = activation_time
        event = DegradationEvent(
            reason=reason,
            details=details,
            exception_types=exception_types,
            exception_diagnostics=exception_diagnostics,
            started_at=activation_time,
        )
        self._degradation_history.append(event)

    def _finalize_degradation_event(self, *, ended_at: datetime | None = None) -> None:
        if not self._degradation_history:
            return
        last = self._degradation_history[-1]
        if last.ended_at is not None:
            return
        self._degradation_history[-1] = last.resolve(ended_at=ended_at)

    def _resolve_degradation(self) -> None:
        if not self._degraded:
            return
        self._degraded = False
        self._degradation_reason = None
        self._degradation_details = ()
        self._degradation_exceptions = ()
        self._degradation_exception_types = ()
        self._degradation_exception_diagnostics = ()
        self._degradation_since = None
        self._finalize_degradation_event()

    # -------------------------- Harmonogram treningów --------------------------
    def _ensure_training_scheduler(self) -> TrainingScheduler:
        if self._training_scheduler is None:
            self._training_scheduler = TrainingScheduler()
        return self._training_scheduler

    def list_training_jobs(self) -> Tuple[str, ...]:
        """Zwróć posortowaną listę nazw zarejestrowanych zadań treningowych."""

        return tuple(sorted(self._scheduled_training_jobs))

    def get_training_job(self, name: str) -> ScheduledTrainingJob | None:
        """Zwróć zadanie treningowe o podanej nazwie (bez modyfikacji)."""

        key = name.strip().lower()
        return self._scheduled_training_jobs.get(key)

    def unregister_training_job(self, name: str) -> None:
        """Usuń zadanie treningowe z harmonogramu."""

        key = name.strip().lower()
        if self._training_scheduler is not None:
            self._training_scheduler.unregister(name)
        self._scheduled_training_jobs.pop(key, None)
        self._job_artifact_paths.pop(key, None)

    def register_training_job(
        self,
        name: str,
        schedule: RetrainingScheduler,
        dataset_provider: Callable[[], FeatureDataset],
        *,
        trainer_factory: Callable[[], ModelTrainer] | None = None,
        validator_factory: Callable[[FeatureDataset], WalkForwardValidator] | None = None,
        artifact_metadata: Mapping[str, object] | None = None,
        repository_base: PathInput | None = None,
        repository_filename: str | None = None,
        attach_to_decision: bool = False,
        decision_name: str | None = None,
        decision_repository_root: PathInput | None = None,
        set_default_decision: bool = False,
        symbol: str | None = None,
        model_type: str | None = None,
    ) -> ScheduledTrainingJob:
        """Zarejestruj zadanie treningowe i opcjonalnie podłącz inference do DecisionOrchestratora."""

        key = name.strip().lower()
        if not key:
            raise ValueError("Nazwa zadania treningowego nie może być pusta")
        if key in self._scheduled_training_jobs:
            raise ValueError(f"Zadanie treningowe {name!r} jest już zarejestrowane")

        scheduler = self._ensure_training_scheduler()
        model_factory = trainer_factory or (lambda: ModelTrainer())

        repository: ModelRepository
        if attach_to_decision:
            root = decision_repository_root or repository_base
            if root is None:
                root = self.model_dir / "decision_engine"
            repository = self.configure_decision_repository(root)
        else:
            root = repository_base or (self.model_dir / "repository")
            repository = self.configure_model_repository(root)

        last_metadata: Dict[str, object] = {}
        last_symbols: List[str] = []

        def _dataset_provider() -> FeatureDataset:
            dataset = dataset_provider()
            metadata = dict(dataset.metadata)
            last_metadata.clear()
            last_metadata.update(metadata)
            last_symbols.clear()
            last_symbols.extend(sorted({vector.symbol for vector in dataset.vectors}))
            return dataset

        def _serialize_validation(result: WalkForwardResult | None) -> Mapping[str, object] | None:
            if result is None:
                return None
            return {
                "average_mae": float(result.average_mae),
                "average_directional_accuracy": float(result.average_directional_accuracy),
                "windows": [dict(window) for window in result.windows],
            }

        normalized_model_type = (
            self._normalize_model_type(model_type)
            if model_type is not None
            else self._normalize_model_type(name)
        )

        def _resolve_symbol() -> str | None:
            if symbol is not None:
                return self._normalize_symbol(symbol)
            if last_symbols:
                return self._normalize_symbol(last_symbols[0])
            symbols_meta = last_metadata.get("symbols")
            if isinstance(symbols_meta, Sequence) and symbols_meta:
                first = symbols_meta[0]
                if isinstance(first, str):
                    return self._normalize_symbol(first)
            return None

        def _on_completed(artifact: ModelArtifact, validation: WalkForwardResult | None) -> None:
            inferred_symbol = _resolve_symbol()
            timestamp = artifact.trained_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = repository_filename or f"{(inferred_symbol or 'model')}_{normalized_model_type}_{timestamp}.json"

            metadata: Dict[str, object] = dict(artifact.metadata)
            metadata.setdefault("job_name", name)
            metadata.setdefault("model_type", normalized_model_type)
            if inferred_symbol is not None:
                metadata.setdefault("symbol", inferred_symbol)
            if last_symbols:
                metadata.setdefault("symbols", list(last_symbols))
            metadata.setdefault(
                "schedule_interval_seconds",
                float(schedule.interval.total_seconds()),
            )
            validation_payload = _serialize_validation(validation)
            if validation_payload is not None:
                metadata["walk_forward"] = validation_payload
            if artifact_metadata:
                metadata.update(dict(artifact_metadata))

            enriched = ModelArtifact(
                feature_names=artifact.feature_names,
                model_state=artifact.model_state,
                trained_at=artifact.trained_at,
                metrics=artifact.metrics,
                metadata=metadata,
                target_scale=artifact.target_scale,
                training_rows=artifact.training_rows,
                validation_rows=artifact.validation_rows,
                test_rows=artifact.test_rows,
                feature_scalers=artifact.feature_scalers,
                decision_journal_entry_id=artifact.decision_journal_entry_id,
                backend=artifact.backend,
            )
            destination = repository.save(enriched, filename)

            inference = DecisionModelInference(repository)
            inference.load_weights(destination)
            self._job_artifact_paths[key] = destination

            if inferred_symbol is not None:
                repository_key = self._model_key(inferred_symbol, normalized_model_type)
                self._repository_models[repository_key] = inference
                self._repository_paths[repository_key] = destination
            self._mark_backend_ready(inference)

            performance_name = normalized_model_type
            if attach_to_decision:
                decision_label = decision_name or normalized_model_type
                performance_name = decision_label
                try:
                    self.load_decision_artifact(
                        decision_label,
                        destination,
                        repository_root=repository.base_path,
                        set_default=set_default_decision,
                    )
                except Exception:
                    logger.exception(
                        "Nie udało się zarejestrować artefaktu Decision Engine %s",
                        decision_label,
                    )

            self._update_decision_orchestrator_performance(
                performance_name,
                artifact=enriched,
                metadata=metadata,
                validation=validation,
                strategy=normalized_model_type,
            )

        job = ScheduledTrainingJob(
            name=name,
            scheduler=schedule,
            trainer_factory=model_factory,
            dataset_provider=_dataset_provider,
            validator_factory=validator_factory,
            on_completed=_on_completed,
        )
        scheduler.register(job)
        self._scheduled_training_jobs[key] = job
        return job

    def run_due_training_jobs(
        self, now: datetime | None = None
    ) -> Tuple[tuple[ScheduledTrainingJob, ModelArtifact, Path | None], ...]:
        """Uruchom wszystkie zaplanowane zadania wymagające odświeżenia."""

        if self._training_scheduler is None:
            return ()
        results: List[tuple[ScheduledTrainingJob, ModelArtifact, Path | None]] = []
        for job, artifact in self._training_scheduler.run_due_jobs(now):
            path = self._job_artifact_paths.get(job.name.strip().lower())
            results.append((job, artifact, path))
        return tuple(results)

    def get_last_trained_artifact_path(self, name: str) -> Path | None:
        """Zwróć ścieżkę do ostatniego artefaktu zapisanego przez zadanie."""

        return self._job_artifact_paths.get(name.strip().lower())

    # ------------------------ Decision Engine repo ------------------------
    def configure_decision_repository(self, base_path: PathInput) -> ModelRepository:
        """Skonfiguruj repozytorium artefaktów Decision Engine."""

        repository_path = Path(base_path).expanduser().resolve()
        repository_path.mkdir(parents=True, exist_ok=True)
        repository = ModelRepository(repository_path)
        self._decision_model_repository = repository
        return repository

    def configure_model_repository(self, base_path: PathInput) -> ModelRepository:
        """Configure repository used for live trading models."""

        repository_path = Path(base_path).expanduser().resolve()
        repository_path.mkdir(parents=True, exist_ok=True)
        repository = ModelRepository(repository_path)
        self._model_repository = repository
        return repository

    def attach_decision_orchestrator(
        self,
        orchestrator: Any,
        *,
        default_model: str | None = None,
    ) -> None:
        """Podłącz istniejący DecisionOrchestrator do menedżera."""

        if _DecisionOrchestrator is not None and not isinstance(orchestrator, _DecisionOrchestrator):
            raise TypeError("orchestrator must be DecisionOrchestrator instance")
        self._decision_orchestrator = orchestrator
        if default_model:
            self._decision_default_name = self._normalize_model_type(default_model)
        for index, (name, inference) in enumerate(self._decision_inferences.items()):
            set_default = self._decision_default_name == name or (
                self._decision_default_name is None and index == 0
            )
            try:
                orchestrator.attach_named_inference(name, inference, set_default=set_default)
            except Exception:  # pragma: no cover - zabezpieczenie przed zmianami API
                logger.exception("Failed to attach inference %s to DecisionOrchestrator", name)

    def detach_decision_orchestrator(self) -> None:
        """Odłącz aktualny DecisionOrchestrator."""

        if self._decision_orchestrator is None:
            return
        orchestrator = self._decision_orchestrator
        for name in list(self._decision_inferences):
            try:
                orchestrator.detach_named_inference(name)
            except Exception:  # pragma: no cover - API defensywne
                logger.debug("Could not detach inference %s", name, exc_info=True)
        self._decision_orchestrator = None

    def _decision_repository(self) -> ModelRepository:
        if self._decision_model_repository is None:
            repository_path = self.model_dir / "decision_engine"
            repository_path.mkdir(parents=True, exist_ok=True)
            self._decision_model_repository = ModelRepository(repository_path)
        return self._decision_model_repository

    def load_decision_artifact(
        self,
        name: str,
        artifact: PathInput,
        *,
        repository_root: PathInput | None = None,
        set_default: bool = False,
    ) -> DecisionModelInference:
        """Załaduj artefakt Decision Engine i zarejestruj inference."""

        normalized_name = self._normalize_model_type(name)
        if repository_root is not None:
            repository = self.configure_decision_repository(repository_root)
        else:
            repository = self._decision_model_repository or self._decision_repository()

        inference = DecisionModelInference(repository)
        artifact_path = Path(artifact)
        if artifact_path.is_absolute():
            if not artifact_path.exists():
                raise FileNotFoundError(artifact_path)
            load_target: Path | str = artifact_path
        else:
            load_target = artifact_path

        inference.load_weights(load_target)
        self._decision_inferences[normalized_name] = inference
        if set_default or self._decision_default_name is None:
            self._decision_default_name = normalized_name
        quality_ok, quality_details = self._evaluate_decision_model_quality(inference, normalized_name)
        if self._decision_orchestrator is not None:
            try:
                self._decision_orchestrator.attach_named_inference(
                    normalized_name,
                    inference,
                    set_default=set_default or self._decision_default_name == normalized_name,
                )
            except Exception:  # pragma: no cover - defensywnie logujemy
                logger.exception(
                    "Nie udało się podłączyć inference %s do DecisionOrchestratora", normalized_name
                )
        self._mark_backend_ready(inference)
        if not quality_ok:
            message = f"Decision model {normalized_name} failed quality thresholds"
            details = tuple(quality_details) if quality_details else (message,)
            error = RuntimeError(message)
            self._degraded = True
            self._degradation_reason = message
            self._degradation_details = details
            self._degradation_exceptions = (error,)
            self._degradation_exception_types = (
                f"{error.__class__.__module__}.{error.__class__.__qualname__}",
            )
            self._degradation_exception_diagnostics = _build_exception_diagnostics(error)
        return inference

    def ingest_model_repository(
        self,
        base_path: PathInput | None = None,
        *,
        pattern: str = "*.json",
    ) -> int:
        """Load all model artifacts matching ``pattern`` into the manager.

        The repository must contain artefacts produced by the Decision Engine
        pipeline.  Each artefact should include ``symbol`` and ``model_type``
        metadata so that it can be addressed at runtime.
        """

        if base_path is not None:
            repository = self.configure_model_repository(base_path)
        elif self._model_repository is None:
            repository_root = self.model_dir / "repository"
            repository = self.configure_model_repository(repository_root)
        else:
            repository = self._model_repository

        loaded = 0
        for artifact_path in sorted(repository.base_path.glob(pattern)):
            try:
                inference = DecisionModelInference(repository)
                inference.load_weights(artifact_path)
            except Exception:
                logger.warning("Failed to load model artifact from %s", artifact_path, exc_info=True)
                continue

            artifact = getattr(inference, "_artifact", None)
            metadata = dict(getattr(artifact, "metadata", {})) if artifact else {}
            symbol = metadata.get("symbol")
            model_type = metadata.get("model_type") or metadata.get("name") or artifact_path.stem
            if symbol is None:
                logger.debug("Skipping artifact %s without symbol metadata", artifact_path)
                continue
            key = self._model_key(str(symbol), str(model_type))
            self._repository_models[key] = inference
            self._repository_paths[key] = artifact_path if isinstance(artifact_path, Path) else Path(artifact_path)
            self._mark_backend_ready(inference)
            loaded += 1
        return loaded

    def _resolve_decision_inference(self, model_name: str | None = None) -> DecisionModelInference:
        if not self._decision_inferences:
            raise RuntimeError("Decision model inference repository is empty")
        if model_name is None:
            if self._decision_default_name is None:
                model = next(iter(self._decision_inferences.values()))
                return model
            model_name = self._decision_default_name
        normalized = self._normalize_model_type(model_name)
        try:
            return self._decision_inferences[normalized]
        except KeyError as exc:
            raise KeyError(f"Decision model {normalized!r} nie jest załadowany") from exc

    def score_decision_features(
        self,
        features: Mapping[str, float],
        *,
        model_name: str | None = None,
    ) -> ModelScore:
        """Zwróć wynik inference Decision Engine dla podanych cech."""

        inference = self._resolve_decision_inference(model_name)
        return inference.score(features)

    def build_decision_engine_payload(
        self,
        *,
        strategy: str,
        action: str,
        risk_profile: str,
        symbol: str,
        notional: float,
        features: Mapping[str, float],
        model_name: str | None = None,
    ) -> Mapping[str, object]:
        """Przygotuj metadane zgodne z oczekiwaniami RiskEngine."""

        score = self.score_decision_features(features, model_name=model_name)
        candidate_payload: Dict[str, object] = {
            "strategy": strategy,
            "action": action,
            "risk_profile": risk_profile,
            "symbol": symbol,
            "notional": float(notional),
            "expected_return_bps": float(score.expected_return_bps),
            "expected_probability": float(score.success_probability),
            "metadata": {
                "decision_engine": {
                    "features": dict(features),
                }
            },
        }
        threshold = float(getattr(self, "ai_threshold_bps", 0.0))
        ai_payload = {
            "prediction_bps": float(score.expected_return_bps),
            "probability": float(score.success_probability),
            "model": model_name or self._decision_default_name or "default",
            "threshold_bps": threshold,
        }
        return {
            "candidate": candidate_payload,
            "ai": ai_payload,
        }

    # --------------------------- Zespoły modeli ---------------------------
    def register_ensemble(
        self,
        name: str,
        components: Iterable[str],
        *,
        aggregation: str = "mean",
        weights: Optional[Iterable[float]] = None,
        override: bool = False,
    ) -> EnsembleDefinition:
        """Zarejestruj zespół modeli dostępny przy generowaniu predykcji."""

        normalized_name = self._normalize_model_type(name)
        comp_list = [self._normalize_model_type(component) for component in components if str(component).strip()]
        if not comp_list:
            raise ValueError("Zespół modeli wymaga co najmniej jednego komponentu.")
        if len(set(comp_list)) != len(comp_list):
            raise ValueError("Lista komponentów zespołu nie może zawierać duplikatów.")
        agg = aggregation.strip().lower()
        allowed = {"mean", "median", "max", "min", "weighted"}
        if agg not in allowed:
            raise ValueError(f"Nieznany typ agregacji zespołu: {aggregation!r}.")
        weights_tuple: Optional[Tuple[float, ...]] = None
        if weights is not None:
            weights_tuple = tuple(float(value) for value in weights)
            if len(weights_tuple) != len(comp_list):
                raise ValueError("Liczba wag musi odpowiadać liczbie komponentów.")
            if agg != "weighted":
                raise ValueError("Wagi mogą być podane tylko dla agregacji 'weighted'.")
            if not any(weight != 0.0 for weight in weights_tuple):
                raise ValueError("Co najmniej jedna waga musi być niezerowa.")
        elif agg == "weighted":
            raise ValueError("Agregacja 'weighted' wymaga listy wag.")

        if not override and normalized_name in self._ensembles:
            raise ValueError(f"Zespół {normalized_name!r} jest już zarejestrowany.")

        definition = EnsembleDefinition(
            name=normalized_name,
            components=tuple(comp_list),
            aggregation=agg,
            weights=weights_tuple,
        )
        self._ensembles[normalized_name] = definition
        return definition

    def unregister_ensemble(self, name: str, *, missing_ok: bool = False) -> None:
        """Usuń definicję zespołu modeli."""

        normalized_name = self._normalize_model_type(name)
        if normalized_name not in self._ensembles:
            if missing_ok:
                return
            raise KeyError(f"Zespół {normalized_name!r} nie jest zarejestrowany.")
        del self._ensembles[normalized_name]

    def get_ensemble(self, name: str) -> Optional[EnsembleDefinition]:
        """Pobierz definicję zespołu modeli, jeśli istnieje."""

        return self._ensembles.get(self._normalize_model_type(name))

    def list_ensembles(self) -> Dict[str, EnsembleDefinition]:
        """Zwróć kopię zarejestrowanych zespołów modeli."""

        return dict(self._ensembles)

    def snapshot_ensembles(self) -> EnsembleRegistrySnapshot:
        """Wykonaj migawkę aktualnego rejestru zespołów modeli."""

        return EnsembleRegistrySnapshot(dict(self._ensembles))

    def restore_ensembles(
        self,
        snapshot: EnsembleRegistrySnapshot,
        *,
        clear_missing: bool = True,
    ) -> None:
        """Przywróć rejestr zespołów z dostarczonej migawki."""

        if not isinstance(snapshot, EnsembleRegistrySnapshot):
            raise TypeError("Oczekiwano instancji EnsembleRegistrySnapshot")

        desired = {self._normalize_model_type(name): definition for name, definition in snapshot.ensembles.items()}

        if clear_missing:
            for name in list(self._ensembles.keys()):
                if name not in desired:
                    del self._ensembles[name]

        for name, definition in desired.items():
            self.register_ensemble(
                name,
                definition.components,
                aggregation=definition.aggregation,
                weights=definition.weights,
                override=True,
            )

    def _sanitize_predictions(self, series: pd.Series) -> pd.Series:
        """Ogranicz sygnały do zakresu [-1, 1] i usuń wartości odstające."""
        sanitized = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sanitized = sanitized.clip(lower=-1.0, upper=1.0)
        return sanitized

    def _aggregate_ensemble_predictions(
        self,
        series_list: Iterable[pd.Series],
        definition: EnsembleDefinition,
    ) -> pd.Series:
        frame = pd.concat(list(series_list), axis=1)
        if frame.empty:
            raise ValueError("Brak predykcji komponentów do agregacji.")
        agg = definition.aggregation
        if agg == "mean":
            combined = frame.mean(axis=1)
        elif agg == "median":
            combined = frame.median(axis=1)
        elif agg == "max":
            combined = frame.max(axis=1)
        elif agg == "min":
            combined = frame.min(axis=1)
        elif agg == "weighted":
            weights = np.asarray(definition.require_weights(), dtype=float)
            total = float(weights.sum())
            if total == 0.0:
                raise ValueError("Suma wag zespołu nie może być zerowa.")
            normalized = weights / total
            combined = frame.to_numpy(dtype=float) @ normalized
            combined = pd.Series(combined, index=frame.index)
        else:  # pragma: no cover - zabezpieczenie przed przyszłymi wartościami
            raise ValueError(f"Nieobsługiwana agregacja zespołu: {agg}")
        return combined if isinstance(combined, pd.Series) else pd.Series(combined, index=frame.index)

    def _predict_series_from_inference(
        self,
        inference: DecisionModelInference,
        df: pd.DataFrame,
        feature_cols: Iterable[str],
    ) -> pd.Series:
        rows: list[float] = []
        for _, row in df.iterrows():
            features = {name: float(row[name]) for name in feature_cols if name in row}
            score = inference.score(features)
            rows.append(float(score.expected_return_bps))
        series = pd.Series(rows, index=df.index)
        return self._sanitize_predictions(series)

    async def _predict_model_series(
        self,
        symbol_key: str,
        model_type: str,
        df: pd.DataFrame,
        feats: List[str],
        cache: Dict[str, pd.Series],
        visited: Optional[set[str]] = None,
    ) -> pd.Series:
        normalized_type = self._normalize_model_type(model_type)
        if normalized_type in cache:
            return cache[normalized_type]
        if visited is None:
            visited = set()
        if normalized_type in visited:
            raise ValueError(f"Wykryto cykliczną definicję zespołu dla {normalized_type!r}.")
        visited.add(normalized_type)
        try:
            ensemble = self._ensembles.get(normalized_type)
            if ensemble is not None:
                component_series = [
                    await self._predict_model_series(symbol_key, component, df, feats, cache, visited)
                    for component in ensemble.components
                ]
                combined = self._aggregate_ensemble_predictions(component_series, ensemble)
                cache[normalized_type] = combined
                return combined

            key = self._model_key(symbol_key, normalized_type)
            repository_model = self._repository_models.get(key)
            if repository_model is not None:
                preds = self._predict_series_from_inference(repository_model, df, feats)
                cache[normalized_type] = preds
                return preds
            model = self.models.get(key)
            if model is None:
                path = self.model_dir / f"{symbol_key}:{normalized_type}.joblib"
                if path.exists():
                    try:
                        model = self._load_model_from_disk(path)
                        self.models[key] = model
                    except Exception as exc:
                        logger.error("Nie można wczytać modelu %s: %s", path, exc)
                        model = None
            if model is not None and not self._supports_series_prediction(model):
                logger.warning(
                    "Model %s nie udostępnia predict_series – użyję fallbacku kompatybilności",
                    key,
                )
            if model is None:
                candidate = normalized_type
                try:
                    X_tmp, y_tmp = _windowize(
                        df,
                        feats,
                        min(len(df) // 2, max(2, int(self.ai_threshold_bps and 10))),
                        "close",
                    )
                except Exception as exc:
                    logger.error("Fallback windowize failed: %s", exc)
                    X_tmp, y_tmp = None, None
                if X_tmp is None or y_tmp is None or len(X_tmp) == 0:
                    raise ValueError("Nie znaleziono żadnego modelu spełniającego kryteria.")
                ctor_kwargs = dict(self._model_init_kwargs)
                model = _AIModels(
                    input_size=X_tmp.shape[-1],
                    seq_len=X_tmp.shape[1],
                    model_type=candidate,
                    **ctor_kwargs,
                )
                await self._invoke_model_method(
                    model,
                    "train",
                    prefer_thread=True,
                    X=X_tmp,
                    y=y_tmp,
                    epochs=1,
                    batch_size=max(1, min(32, len(X_tmp))),
                    progress_callback=None,
                    model_out=None,
                    verbose=False,
                )
                self.models[key] = model

            preds = await self._invoke_model_method(
                model,
                "predict_series",
                df,
                feats,
                prefer_thread=True,
            )
            if not isinstance(preds, pd.Series):
                preds = pd.Series(np.asarray(preds, dtype=float), index=df.index)
            cache[normalized_type] = preds
            return preds
        finally:
            visited.remove(normalized_type)

    def _supports_series_prediction(self, model: Any) -> bool:
        return any(
            callable(getattr(model, attr, None))
            for attr in ("predict_series", "predict", "batch_predict", "predict_batch")
        )

    async def _invoke_model_method(
        self,
        model: Any,
        method_name: str,
        *args: Any,
        prefer_thread: bool = False,
        **kwargs: Any,
    ) -> Any:
        method = getattr(model, method_name, None)
        if method is None and method_name == "predict_series":
            return await self._invoke_predict_series_fallback(
                model, *args, prefer_thread=prefer_thread, **kwargs
            )
        if method is None:
            raise AttributeError(f"Model {model!r} nie ma metody {method_name!r}")
        return await self._call_model_callable(method, *args, prefer_thread=prefer_thread, **kwargs)

    async def _call_model_callable(
        self,
        method: Callable[..., Any],
        *args: Any,
        prefer_thread: bool = False,
        **kwargs: Any,
    ) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        if prefer_thread:
            return await asyncio.to_thread(method, *args, **kwargs)
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _invoke_predict_series_fallback(
        self,
        model: Any,
        *args: Any,
        prefer_thread: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not args and "df" not in kwargs:
            raise AttributeError(f"Model {model!r} nie ma metody 'predict_series'")

        df = kwargs.get("df") if "df" in kwargs else args[0]
        feature_cols = None
        if len(args) >= 2:
            feature_cols = args[1]
        elif "feature_cols" in kwargs:
            feature_cols = kwargs["feature_cols"]

        if not isinstance(df, pd.DataFrame):
            raise AttributeError(
                f"Model {model!r} nie ma metody 'predict_series' i nie można zastosować fallbacku"
            )

        candidate_methods: list[tuple[str, Callable[..., Any]]] = []
        for name in ("predict", "batch_predict", "predict_batch"):
            candidate = getattr(model, name, None)
            if callable(candidate):
                candidate_methods.append((name, candidate))

        if not candidate_methods:
            logger.warning(
                "Model %r nie udostępnia predict/predict_series – użyję stałej predykcji",
                model,
            )
            baseline_bps = _safe_float(self.ai_threshold_bps)
            fallback_value = 0.01 if baseline_bps is None else max(0.01, baseline_bps / 10000.0)
            return pd.Series(np.full(len(df), fallback_value, dtype=float), index=df.index)

        feature_frame: pd.DataFrame
        try:
            feature_frame = df.loc[:, list(feature_cols)] if feature_cols else df
        except Exception:
            feature_frame = df

        feature_array = feature_frame.to_numpy(dtype=float, copy=False)

        call_variants: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []
        if feature_cols is not None:
            call_variants.append(((df, feature_cols), {}))
        call_variants.append(((df,), {}))
        if feature_frame is not df:
            call_variants.append(((feature_frame,), {}))
        call_variants.append(((feature_array,), {}))

        suppressed_errors: list[str] = []

        for method_name, method in candidate_methods:
            for call_args, call_kwargs in call_variants:
                try:
                    result = await self._call_model_callable(
                        method, *call_args, prefer_thread=prefer_thread, **call_kwargs
                    )
                except TypeError as exc:
                    message = str(exc)
                    lowered = message.lower()
                    if "argument" in lowered and any(
                        token in lowered for token in ("positional", "keyword", "unexpected")
                    ):
                        suppressed_errors.append(f"{method_name}{call_args}: {message}")
                        continue
                    raise

                if isinstance(result, pd.Series):
                    if result.empty:
                        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
                    return result.reindex(df.index, fill_value=float(result.iloc[0]))

                array_result = np.asarray(result, dtype=float)
                if array_result.ndim == 0:
                    array_result = np.repeat(float(array_result), len(df))
                if array_result.shape[0] != len(df):
                    raise ValueError(
                        f"Metoda {method_name} zwróciła {array_result.shape[0]} wyników przy {len(df)} wierszach"
                    )
                return pd.Series(array_result, index=df.index)

        if suppressed_errors:
            detail = "; ".join(suppressed_errors)
        else:
            detail = "brak kompatybilnych podpisów metod"
        raise AttributeError(
            f"Model {model!r} nie obsługuje fallbacku predict_series ({detail})."
        )

    def _load_model_from_disk(self, path: Path) -> Any:
        loader = getattr(_AIModels, "load_model", None)
        if callable(loader):
            try:
                model = loader(path)
                self._mark_backend_ready(model)
                return model
            except Exception:
                logger.debug("Nowy loader modeli nie powiódł się dla %s", path, exc_info=True)
        model = _joblib_load(path)
        self._mark_backend_ready(model)
        return model

    def _mark_backend_ready(self, model: Any) -> None:
        if not self._degraded:
            return
        if isinstance(model, DecisionModelInference) and getattr(model, "is_ready", False):
            if _is_fallback_degradation(self._degradation_reason):
                self._degraded = False
                self._degradation_reason = None
                self._degradation_details = ()
                self._degradation_exceptions = ()
                self._degradation_exception_types = ()
                self._degradation_exception_diagnostics = ()
            return
        if _AI_IMPORT_ERROR is not None:
            return
        if getattr(model, "feature_names", None):
            if _is_fallback_degradation(self._degradation_reason):
                self._degraded = False
                self._degradation_reason = None
                self._degradation_details = ()
                self._degradation_exceptions = ()
                self._degradation_exception_types = ()
                self._degradation_exception_diagnostics = ()

    def _evaluate_decision_model_quality(
        self, inference: DecisionModelInference, name: str
    ) -> Tuple[bool, Tuple[str, ...]]:
        artifact = getattr(inference, "_artifact", None)
        if artifact is None:
            return True, ()
        metadata = getattr(artifact, "metadata", {}) or {}
        thresholds = metadata.get("quality_thresholds", {}) if isinstance(metadata, Mapping) else {}
        min_directional = float(thresholds.get("min_directional_accuracy", 0.5))
        max_mae = float(thresholds.get("max_mae", 20.0))
        metrics_payload = getattr(artifact, "metrics", {}) or {}
        summary_metrics = _select_metric_block(metrics_payload)
        directional = float(summary_metrics.get("directional_accuracy", 0.0))
        mae = float(summary_metrics.get("mae", 0.0))
        if directional < min_directional or mae > max_mae:
            logger.warning(
                "Decision model %s failed quality thresholds (directional=%.3f, mae=%.3f, min_directional=%.3f, max_mae=%.3f)",
                name,
                directional,
                mae,
                min_directional,
                max_mae,
            )
            details: List[str] = [
                f"Decision model {name} failed quality thresholds",
            ]
            if directional < min_directional:
                details.append(
                    f"directional_accuracy={directional:.3f} < min {min_directional:.3f}"
                )
            if mae > max_mae:
                details.append(f"mae={mae:.3f} > max {max_mae:.3f}")
            return False, tuple(details)
        return True, ()

    def _compose_performance_metrics(
        self,
        base_metrics: Mapping[str, object],
        metadata: Mapping[str, object],
        validation: WalkForwardResult | None,
    ) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        structured = _select_metric_block(base_metrics)
        for key, value in structured.items():
            number = _safe_float(value)
            if number is None:
                continue
            summary[str(key)] = number

        cross_validation = metadata.get("cross_validation")
        if isinstance(cross_validation, Mapping):
            mae_values = cross_validation.get("mae", ())
            mae_mean = _safe_mean(mae_values if isinstance(mae_values, Iterable) else ())
            if mae_mean is not None:
                summary.setdefault("cv_mae_mean", mae_mean)
            directional_values = cross_validation.get("directional_accuracy", ())
            directional_mean = _safe_mean(
                directional_values if isinstance(directional_values, Iterable) else ()
            )
            if directional_mean is not None:
                summary.setdefault("cv_directional_accuracy_mean", directional_mean)
            folds = _safe_float(cross_validation.get("folds"))
            if folds is not None:
                summary.setdefault("cv_folds", folds)

        if validation is not None:
            mae = _safe_float(validation.average_mae)
            if mae is not None:
                summary.setdefault("walk_forward_mae", mae)
            directional = _safe_float(validation.average_directional_accuracy)
            if directional is not None:
                summary.setdefault("walk_forward_directional_accuracy", directional)

        walk_forward_meta = metadata.get("walk_forward")
        if isinstance(walk_forward_meta, Mapping):
            wf_mae = _safe_float(walk_forward_meta.get("average_mae"))
            if wf_mae is not None:
                summary.setdefault("walk_forward_mae", wf_mae)
            wf_directional = _safe_float(
                walk_forward_meta.get("average_directional_accuracy")
            )
            if wf_directional is not None:
                summary.setdefault("walk_forward_directional_accuracy", wf_directional)

        return summary

    def _update_decision_orchestrator_performance(
        self,
        name: str,
        *,
        artifact: ModelArtifact,
        metadata: Mapping[str, object],
        validation: WalkForwardResult | None,
        strategy: str,
    ) -> None:
        orchestrator = self._decision_orchestrator
        if orchestrator is None:
            return
        metrics = self._compose_performance_metrics(artifact.metrics, metadata, validation)
        risk_profile = metadata.get("risk_profile")
        profile_value = str(risk_profile) if isinstance(risk_profile, str) else None
        try:
            orchestrator.update_model_performance(
                name,
                metrics,
                strategy=strategy,
                risk_profile=profile_value,
                timestamp=artifact.trained_at,
            )
        except Exception:  # pragma: no cover - zapewniamy odporność na zmiany API
            logger.exception(
                "Nie udało się zaktualizować metryk modelu %s w DecisionOrchestrator", name
            )

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
        ctor_kwargs = dict(self._model_init_kwargs)
        model = _AIModels(
            input_size=X.shape[-1],
            seq_len=int(seq_len),
            model_type=model_name,
            **ctor_kwargs,
        )

        train_kwargs = dict(
            X=X,
            y=y,
            epochs=int(max(1, epochs)),
            batch_size=int(max(1, batch_size)),
            progress_callback=(lambda *_: None),
            model_out=str(model_path),
            verbose=False,
        )
        await self._invoke_model_method(model, "train", prefer_thread=True, **train_kwargs)
        return model

    def _safe_pct_change(self, frame: pd.DataFrame, feats: Iterable[str]) -> pd.DataFrame:
        """Oblicz zmiany procentowe z obsługą awaryjnego trybu dla ramek legacy."""

        columns = list(feats)
        try:
            return frame[columns].pct_change()
        except (TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError):
            numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
            shifted = numeric.shift(1)
            return (numeric - shifted) / shifted

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

        baseline_pct = (
            self._safe_pct_change(baseline, feats).replace([np.inf, -np.inf], np.nan).dropna()
        )
        recent_pct = (
            self._safe_pct_change(recent, feats).replace([np.inf, -np.inf], np.nan).dropna()
        )

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
                key = self._model_key(symbol_key, model_name)
                repository_inference = self._repository_models.get(key)
                if repository_inference is not None:
                    artifact = getattr(repository_inference, "_artifact", None)
                    metrics = dict(getattr(artifact, "metrics", {})) if artifact else {}
                    metadata = dict(getattr(artifact, "metadata", {})) if artifact else {}
                    cv_raw = metadata.get("cross_validation")
                    cv_meta = cv_raw if isinstance(cv_raw, Mapping) else {}
                    cv_scores = [
                        float(value)
                        for value in cv_meta.get("directional_accuracy", [])
                        if isinstance(value, (int, float))
                    ]
                    pnl = float(metrics.get("expected_pnl", 0.0))
                    hit_rate = (
                        float(np.mean(cv_scores))
                        if cv_scores
                        else float(metrics.get("directional_accuracy", 0.0))
                    )
                    sharpe = self._sharpe_ratio(np.asarray(cv_scores, dtype=float)) if cv_scores else 0.0
                    model_path = self._repository_paths.get(key)
                    evaluations.append(
                        ModelEvaluation(
                            model_type=model_name,
                            hit_rate=hit_rate,
                            pnl=pnl,
                            sharpe=sharpe,
                            cv_scores=cv_scores,
                            model_path=str(model_path) if model_path is not None else None,
                        )
                    )
                    continue

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

                    pred = await self._invoke_model_method(
                        model,
                        "predict",
                        X_val,
                        prefer_thread=True,
                    )
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

    def cancel_pipeline_schedule(self, symbol: str) -> None:
        key = self._normalize_symbol(symbol)
        schedule = self._pipeline_schedules.pop(key, None)
        if schedule and not schedule.task.done():
            schedule.task.cancel()

    def active_pipeline_schedules(self) -> Dict[str, PipelineSchedule]:
        return dict(self._pipeline_schedules)

    def _get_history_buffer(self, symbol: str) -> deque[PipelineExecutionRecord]:
        key = self._normalize_symbol(symbol)
        history = self._pipeline_history.get(key)
        if history is None or history.maxlen != self._pipeline_history_limit:
            history = deque(history or (), maxlen=self._pipeline_history_limit)
            self._pipeline_history[key] = history
        return history

    def _record_pipeline_selection(self, selection: StrategySelectionResult) -> PipelineExecutionRecord:
        symbol = self._normalize_symbol(selection.symbol)
        predictions = selection.predictions
        prediction_count = 0
        prediction_mean: Optional[float] = None
        prediction_std: Optional[float] = None
        prediction_min: Optional[float] = None
        prediction_max: Optional[float] = None

        if predictions is not None:
            if not isinstance(predictions, pd.Series):
                predictions = pd.Series(np.asarray(predictions, dtype=float))
            values = predictions.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            prediction_count = int(finite.size)
            if prediction_count:
                prediction_mean = float(np.mean(finite))
                prediction_std = float(np.std(finite))
                prediction_min = float(np.min(finite))
                prediction_max = float(np.max(finite))

        record = PipelineExecutionRecord(
            symbol=symbol,
            decided_at=selection.decided_at,
            best_model=selection.best_model,
            evaluations=tuple(selection.evaluations),
            drift_report=selection.drift_report,
            prediction_count=prediction_count,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            prediction_min=prediction_min,
            prediction_max=prediction_max,
        )

        history = self._get_history_buffer(symbol)
        history.append(record)
        return record

    def get_pipeline_history_limit(self) -> int:
        return self._pipeline_history_limit

    def set_pipeline_history_limit(self, limit: int) -> None:
        if not isinstance(limit, int):
            raise TypeError("limit musi być liczbą całkowitą")
        if limit <= 0:
            raise ValueError("limit historii musi być dodatni")
        if limit == self._pipeline_history_limit:
            return
        self._pipeline_history_limit = int(limit)
        for symbol, history in list(self._pipeline_history.items()):
            new_history = deque(history, maxlen=self._pipeline_history_limit)
            if not new_history:
                if symbol in self._pipeline_history:
                    self._pipeline_history[symbol] = new_history
                continue
            self._pipeline_history[symbol] = new_history

    def get_pipeline_history(self, symbol: str, limit: Optional[int] = None) -> List[PipelineExecutionRecord]:
        if limit is not None and limit <= 0:
            return []
        history = self._pipeline_history.get(self._normalize_symbol(symbol))
        if not history:
            return []
        records = list(history)
        if limit is None or limit >= len(records):
            return records
        return records[-int(limit) :]

    def last_pipeline_selection(self, symbol: str) -> Optional[PipelineExecutionRecord]:
        history = self._pipeline_history.get(self._normalize_symbol(symbol))
        if not history:
            return None
        try:
            return history[-1]
        except IndexError:
            return None

    def clear_pipeline_history(self, symbol: str) -> None:
        self._pipeline_history.pop(self._normalize_symbol(symbol), None)

    def pipeline_history(self) -> Dict[str, List[PipelineExecutionRecord]]:
        return {symbol: list(records) for symbol, records in self._pipeline_history.items()}

    def snapshot_pipeline_history(
        self,
        symbol: Optional[str] = None,
        *,
        limit: Optional[int] = None,
    ) -> PipelineHistorySnapshot:
        if limit is not None and limit <= 0:
            return PipelineHistorySnapshot()
        if symbol is not None:
            key = self._normalize_symbol(symbol)
            history = self._pipeline_history.get(key)
            if not history:
                return PipelineHistorySnapshot()
            records = list(history)
            if limit is not None and len(records) > limit:
                records = records[-int(limit) :]
            return PipelineHistorySnapshot({key: tuple(records)})

        snapshot: Dict[str, Tuple[PipelineExecutionRecord, ...]] = {}
        for key, history in self._pipeline_history.items():
            if not history:
                continue
            records = list(history)
            if limit is not None and len(records) > limit:
                records = records[-int(limit) :]
            snapshot[key] = tuple(records)
        return PipelineHistorySnapshot(snapshot)

    def restore_pipeline_history(
        self,
        snapshot: PipelineHistorySnapshot,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(snapshot, PipelineHistorySnapshot):
            raise TypeError("snapshot musi być instancją PipelineHistorySnapshot")
        if replace:
            self._pipeline_history.clear()
        for symbol, records in snapshot.records.items():
            key = self._normalize_symbol(symbol)
            buffer = deque(records, maxlen=self._pipeline_history_limit)
            if replace:
                self._pipeline_history[key] = buffer
            else:
                target = self._get_history_buffer(key)
                target.extend(records)

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

    def _pipeline_schedule_runner(
        self,
        symbol: str,
        df_provider: Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]],
        model_types: Iterable[str],
        interval_seconds: float,
        seq_len: int,
        folds: int,
        baseline_provider: Optional[Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]]] = None,
        on_result: Optional[Callable[[StrategySelectionResult], Union[None, Awaitable[None]]]] = None,
    ) -> Callable[[], Awaitable[None]]:
        async def _runner() -> None:
            while True:
                try:
                    df_candidate = df_provider()
                    if inspect.isawaitable(df_candidate):
                        df_candidate = await df_candidate
                    baseline_df = None
                    if baseline_provider is not None:
                        baseline_candidate = baseline_provider()
                        if inspect.isawaitable(baseline_candidate):
                            baseline_candidate = await baseline_candidate
                        baseline_df = baseline_candidate
                    selection = await self.run_pipeline(
                        symbol,
                        df_candidate,
                        model_types,
                        seq_len=seq_len,
                        folds=folds,
                        baseline=baseline_df,
                    )
                    if on_result is not None:
                        try:
                            maybe_awaitable = on_result(selection)
                            if inspect.isawaitable(maybe_awaitable):
                                await maybe_awaitable
                        except Exception:  # pragma: no cover - callback użytkownika
                            logger.debug("Callback pipeline'u zgłosił wyjątek.")
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # pragma: no cover - błędy runtime
                    logger.error("Błąd harmonogramu pipeline'u %s: %s", symbol, exc)
                    await asyncio.sleep(min(60.0, interval_seconds))

        return _runner

    def schedule_pipeline(
        self,
        symbol: str,
        df_provider: Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]],
        model_types: Iterable[str],
        *,
        interval_seconds: float = 3600.0,
        seq_len: int = 64,
        folds: int = 3,
        baseline_provider: Optional[Callable[[], Union[pd.DataFrame, Awaitable[pd.DataFrame]]]] = None,
        on_result: Optional[Callable[[StrategySelectionResult], Union[None, Awaitable[None]]]] = None,
    ) -> PipelineSchedule:
        """Zaplanuj cykliczne uruchamianie pipeline'u selekcji modeli."""

        model_types_tuple = tuple(model_types)
        runner = self._pipeline_schedule_runner(
            symbol,
            df_provider,
            model_types_tuple,
            interval_seconds,
            seq_len,
            folds,
            baseline_provider,
            on_result,
        )
        task = asyncio.create_task(runner())
        schedule = PipelineSchedule(
            symbol=self._normalize_symbol(symbol),
            interval_seconds=float(interval_seconds),
            task=task,
            model_types=model_types_tuple,
            seq_len=int(seq_len),
            folds=int(folds),
        )
        self._pipeline_schedules[schedule.symbol] = schedule
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
        self.set_active_model(symbol, selection.best_model)
        if baseline is not None:
            selection.drift_report = self.detect_drift(baseline, df)
        self._track_signal(symbol, predictions)
        self._record_pipeline_selection(selection)
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
                    pred = await self._invoke_model_method(
                        model,
                        "predict",
                        X,
                        prefer_thread=True,
                    )
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
        if model_types is None:
            active_model = self._active_models.get(symbol_key)
            if active_model:
                candidates = [active_model]
            else:
                candidates = [
                    self._normalize_model_type(mt)
                    for mt in self.models.keys()
                    if mt.startswith(symbol_key)
                ]
        else:
            candidates = [self._normalize_model_type(mt) for mt in model_types]
        if not candidates:
            raise ValueError("Brak wytrenowanych modeli dla podanego symbolu.")

        async with self._lock:
            cache: Dict[str, pd.Series] = {}
            last_error: Optional[BaseException] = None
            for candidate in candidates:
                try:
                    raw_predictions = await self._predict_model_series(
                        symbol_key,
                        candidate,
                        df,
                        feats,
                        cache,
                        set(),
                    )
                except Exception as exc:
                    last_error = exc
                    logger.debug(
                        "Nie udało się uzyskać predykcji modelu %s dla %s: %s",
                        candidate,
                        symbol_key,
                        exc,
                        exc_info=True,
                    )
                    continue
                sanitized = self._sanitize_predictions(raw_predictions)
                cache[self._normalize_model_type(candidate)] = sanitized
                self._track_signal(symbol, sanitized)
                logger.debug("Zwracam predykcje modelu %s dla %s", candidate, symbol_key)
                return sanitized
            if last_error is not None:
                raise last_error
            raise ValueError("Nie znaleziono żadnego modelu spełniającego kryteria.")

    # --------------------------- Import modeli --------------------------
    async def import_model(self, symbol: str, model_type: str, path: str | Path) -> None:
        """Załaduj model zapisany na dysku i zarejestruj go w menedżerze."""

        symbol_key = self._normalize_symbol(symbol)
        key = self._model_key(symbol, model_type)
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        def _load() -> Any:
            return self._load_model_from_disk(model_path)

        model = await asyncio.to_thread(_load)
        async with self._lock:
            self.models[key] = model
            target = self.model_dir / f"{symbol_key}:{model_type.lower()}.joblib"
            if target != model_path:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(_joblib_dump, model, target)

        logger.info("Zaimportowano model %s z pliku %s", key, model_path)


def _serialize_ensemble_definition(definition: EnsembleDefinition) -> Dict[str, Any]:
    return {
        "name": definition.name,
        "components": list(definition.components),
        "aggregation": definition.aggregation,
        "weights": None if definition.weights is None else list(definition.weights),
    }


def _deserialize_ensemble_definition(name: str, payload: Mapping[str, Any]) -> EnsembleDefinition:
    if not isinstance(payload, Mapping):
        raise TypeError("Definicja zespołu musi być mapowaniem")

    key_name = str(name or "").strip()
    if not key_name:
        raise ValueError("Nazwa zespołu nie może być pusta")

    provided_name_raw = payload.get("name", key_name)
    provided_name = str(provided_name_raw or "").strip()
    if provided_name:
        normalized_provided = AIManager._normalize_model_type(provided_name)
        normalized_key = AIManager._normalize_model_type(key_name)
        if normalized_provided != normalized_key:
            raise ValueError("Nazwa w definicji nie zgadza się z kluczem migawki")
        normalized_name = normalized_provided
    else:
        normalized_name = AIManager._normalize_model_type(key_name)

    components_raw = payload.get("components")
    if not isinstance(components_raw, Iterable):
        raise TypeError("Pole components musi być iterowalne")
    components: list[str] = []
    for component in components_raw:
        text = str(component or "").strip()
        if not text:
            continue
        components.append(AIManager._normalize_model_type(text))
    if not components:
        raise ValueError("Lista komponentów zespołu nie może być pusta")
    if len(set(components)) != len(components):
        raise ValueError("Lista komponentów nie może zawierać duplikatów")

    aggregation_raw = payload.get("aggregation", "mean")
    agg = str(aggregation_raw or "").strip().lower() or "mean"
    allowed = {"mean", "median", "max", "min", "weighted"}
    if agg not in allowed:
        raise ValueError(f"Nieznany typ agregacji zespołu: {aggregation_raw!r}")

    weights_raw = payload.get("weights", None)
    weights_tuple: Optional[Tuple[float, ...]] = None
    if weights_raw is not None:
        if not isinstance(weights_raw, Iterable):
            raise TypeError("Pole weights musi być iterowalne")
        weights_tuple = tuple(float(value) for value in weights_raw)
        if len(weights_tuple) != len(components):
            raise ValueError("Liczba wag musi odpowiadać liczbie komponentów")
        if agg != "weighted":
            raise ValueError("Wagi dozwolone są tylko przy agregacji 'weighted'")
        if not any(weight != 0.0 for weight in weights_tuple):
            raise ValueError("Co najmniej jedna waga zespołu musi być niezerowa")
    elif agg == "weighted":
        raise ValueError("Agregacja 'weighted' wymaga podania wag")

    return EnsembleDefinition(
        name=normalized_name,
        components=tuple(components),
        aggregation=agg,
        weights=weights_tuple,
    )


def ensemble_registry_snapshot_to_dict(snapshot: EnsembleRegistrySnapshot) -> Dict[str, Dict[str, Any]]:
    if not isinstance(snapshot, EnsembleRegistrySnapshot):
        raise TypeError("Oczekiwano instancji EnsembleRegistrySnapshot")
    return {name: _serialize_ensemble_definition(definition) for name, definition in snapshot.ensembles.items()}


def ensemble_registry_snapshot_from_dict(payload: Mapping[str, Any]) -> EnsembleRegistrySnapshot:
    if not isinstance(payload, Mapping):
        raise TypeError("Dane wejściowe muszą być mapowaniem")
    ensembles: Dict[str, EnsembleDefinition] = {}
    for name, definition_payload in payload.items():
        if definition_payload is None:
            continue
        if not isinstance(name, str):
            raise TypeError("Klucz migawki musi być tekstem")
        normalized_name = AIManager._normalize_model_type(name)
        definition = _deserialize_ensemble_definition(normalized_name, definition_payload)
        ensembles[normalized_name] = definition
    return EnsembleRegistrySnapshot(ensembles)


def ensemble_registry_snapshot_to_json(snapshot: EnsembleRegistrySnapshot, *, indent: Optional[int] = None) -> str:
    data = ensemble_registry_snapshot_to_dict(snapshot)
    return json.dumps(data, indent=indent, sort_keys=True)


def ensemble_registry_snapshot_from_json(payload: Union[str, bytes, bytearray]) -> EnsembleRegistrySnapshot:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("Payload JSON musi być tekstem lub bajtami")
    data = json.loads(payload)
    return ensemble_registry_snapshot_from_dict(data)


def ensemble_registry_snapshot_to_file(
    snapshot: EnsembleRegistrySnapshot,
    path: PathInput,
    *,
    indent: Optional[int] = 2,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ensemble_registry_snapshot_to_json(snapshot, indent=indent)
    target.write_text(payload, encoding="utf-8")


def ensemble_registry_snapshot_from_file(path: PathInput) -> EnsembleRegistrySnapshot:
    source = Path(path)
    data = source.read_text(encoding="utf-8")
    return ensemble_registry_snapshot_from_json(data)


def diff_ensemble_snapshots(
    before: EnsembleRegistrySnapshot,
    after: EnsembleRegistrySnapshot,
) -> EnsembleRegistryDiff:
    if not isinstance(before, EnsembleRegistrySnapshot):
        raise TypeError("before musi być instancją EnsembleRegistrySnapshot")
    if not isinstance(after, EnsembleRegistrySnapshot):
        raise TypeError("after musi być instancją EnsembleRegistrySnapshot")

    diff = EnsembleRegistryDiff()

    before_defs = {name: definition for name, definition in before.ensembles.items()}
    after_defs = {name: definition for name, definition in after.ensembles.items()}

    for name in before_defs.keys() - after_defs.keys():
        diff.removed[name] = before_defs[name]

    for name in after_defs.keys() - before_defs.keys():
        diff.added[name] = after_defs[name]

    for name in before_defs.keys() & after_defs.keys():
        previous = before_defs[name]
        current = after_defs[name]
        if previous != current:
            diff.changed[name] = (previous, current)

    return diff


def ensemble_registry_diff_to_dict(diff: EnsembleRegistryDiff) -> Dict[str, Any]:
    if not isinstance(diff, EnsembleRegistryDiff):
        raise TypeError("Oczekiwano instancji EnsembleRegistryDiff")

    result: Dict[str, Any] = {
        "added": {},
        "removed": {},
        "changed": {},
    }

    for name, definition in diff.added.items():
        result["added"][name] = _serialize_ensemble_definition(definition)

    for name, definition in diff.removed.items():
        result["removed"][name] = _serialize_ensemble_definition(definition)

    for name, (before_def, after_def) in diff.changed.items():
        result["changed"][name] = {
            "before": _serialize_ensemble_definition(before_def),
            "after": _serialize_ensemble_definition(after_def),
        }

    return result


def ensemble_registry_diff_from_dict(payload: Mapping[str, Any]) -> EnsembleRegistryDiff:
    if not isinstance(payload, Mapping):
        raise TypeError("Dane wejściowe muszą być mapowaniem diffu")

    diff = EnsembleRegistryDiff()

    added_raw = payload.get("added", {})
    if not isinstance(added_raw, Mapping):
        raise TypeError("Sekcja 'added' musi być mapowaniem")
    for name, definition_payload in added_raw.items():
        if definition_payload is None:
            continue
        if not isinstance(name, str):
            raise TypeError("Nazwa zespołu musi być tekstem")
        normalized = AIManager._normalize_model_type(name)
        diff.added[normalized] = _deserialize_ensemble_definition(normalized, definition_payload)

    removed_raw = payload.get("removed", {})
    if not isinstance(removed_raw, Mapping):
        raise TypeError("Sekcja 'removed' musi być mapowaniem")
    for name, definition_payload in removed_raw.items():
        if definition_payload is None:
            continue
        if not isinstance(name, str):
            raise TypeError("Nazwa zespołu musi być tekstem")
        normalized = AIManager._normalize_model_type(name)
        diff.removed[normalized] = _deserialize_ensemble_definition(normalized, definition_payload)

    changed_raw = payload.get("changed", {})
    if not isinstance(changed_raw, Mapping):
        raise TypeError("Sekcja 'changed' musi być mapowaniem")
    for name, definition_payload in changed_raw.items():
        if definition_payload is None:
            continue
        if not isinstance(name, str):
            raise TypeError("Nazwa zespołu musi być tekstem")
        if not isinstance(definition_payload, Mapping):
            raise TypeError("Dane zmienionego zespołu muszą być mapowaniem")
        before_payload = definition_payload.get("before")
        after_payload = definition_payload.get("after")
        normalized = AIManager._normalize_model_type(name)
        before_definition = _deserialize_ensemble_definition(normalized, before_payload)
        after_definition = _deserialize_ensemble_definition(normalized, after_payload)
        diff.changed[normalized] = (before_definition, after_definition)

    return diff


def ensemble_registry_diff_to_json(diff: EnsembleRegistryDiff, *, indent: Optional[int] = None) -> str:
    data = ensemble_registry_diff_to_dict(diff)
    return json.dumps(data, indent=indent, sort_keys=True)


def ensemble_registry_diff_from_json(payload: Union[str, bytes, bytearray]) -> EnsembleRegistryDiff:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("Payload JSON musi być tekstem lub bajtami")
    data = json.loads(payload)
    return ensemble_registry_diff_from_dict(data)


def ensemble_registry_diff_to_file(
    diff: EnsembleRegistryDiff,
    path: PathInput,
    *,
    indent: Optional[int] = 2,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ensemble_registry_diff_to_json(diff, indent=indent)
    target.write_text(payload, encoding="utf-8")


def ensemble_registry_diff_from_file(path: PathInput) -> EnsembleRegistryDiff:
    source = Path(path)
    data = source.read_text(encoding="utf-8")
    return ensemble_registry_diff_from_json(data)


def _serialize_model_evaluation(evaluation: ModelEvaluation) -> Dict[str, Any]:
    return {
        "model_type": evaluation.model_type,
        "hit_rate": float(evaluation.hit_rate),
        "pnl": float(evaluation.pnl),
        "sharpe": float(evaluation.sharpe),
        "cv_scores": [float(score) for score in evaluation.cv_scores],
        "model_path": evaluation.model_path,
    }


def _deserialize_model_evaluation(payload: Mapping[str, Any]) -> ModelEvaluation:
    if not isinstance(payload, Mapping):
        raise TypeError("Oczekiwano mapowania do rekonstrukcji ModelEvaluation")
    model_type = str(payload.get("model_type", "")).strip()
    if not model_type:
        raise ValueError("Brak pola model_type w ModelEvaluation")
    cv_scores_raw = payload.get("cv_scores", [])
    if not isinstance(cv_scores_raw, Iterable):
        raise TypeError("Pole cv_scores musi być iterowalne")
    cv_scores = [float(score) for score in cv_scores_raw]
    return ModelEvaluation(
        model_type=model_type,
        hit_rate=float(payload.get("hit_rate", 0.0)),
        pnl=float(payload.get("pnl", 0.0)),
        sharpe=float(payload.get("sharpe", 0.0)),
        cv_scores=cv_scores,
        model_path=payload.get("model_path"),
    )


def _serialize_drift_report(report: Optional[DriftReport]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    return {
        "feature_drift": float(report.feature_drift),
        "volatility_shift": float(report.volatility_shift),
        "triggered": bool(report.triggered),
        "threshold": float(report.threshold),
    }


def _deserialize_drift_report(payload: Optional[Mapping[str, Any]]) -> Optional[DriftReport]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("Raport dryfu musi być mapowaniem lub None")
    return DriftReport(
        feature_drift=float(payload.get("feature_drift", 0.0)),
        volatility_shift=float(payload.get("volatility_shift", 0.0)),
        triggered=bool(payload.get("triggered", False)),
        threshold=float(payload.get("threshold", 0.0)),
    )


def _serialize_pipeline_execution_record(record: PipelineExecutionRecord) -> Dict[str, Any]:
    decided_at = record.decided_at
    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=timezone.utc)
    else:
        decided_at = decided_at.astimezone(timezone.utc)
    return {
        "symbol": record.symbol,
        "decided_at": decided_at.isoformat(),
        "best_model": record.best_model,
        "evaluations": [_serialize_model_evaluation(ev) for ev in record.evaluations],
        "drift_report": _serialize_drift_report(record.drift_report),
        "prediction_count": int(record.prediction_count),
        "prediction_mean": record.prediction_mean,
        "prediction_std": record.prediction_std,
        "prediction_min": record.prediction_min,
        "prediction_max": record.prediction_max,
    }


def _deserialize_pipeline_execution_record(payload: Mapping[str, Any]) -> PipelineExecutionRecord:
    if not isinstance(payload, Mapping):
        raise TypeError("Rekord historii pipeline'u musi być mapowaniem")
    symbol = str(payload.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("Brak pola symbol w rekordzie historii pipeline'u")
    decided_at_raw = payload.get("decided_at")
    if not isinstance(decided_at_raw, str) or not decided_at_raw.strip():
        raise ValueError("Pole decided_at musi być tekstem w formacie ISO")
    decided_at = datetime.fromisoformat(decided_at_raw)
    best_model = str(payload.get("best_model", "")).strip()
    if not best_model:
        raise ValueError("Brak pola best_model w rekordzie historii pipeline'u")
    evaluations_raw = payload.get("evaluations", [])
    if not isinstance(evaluations_raw, Iterable):
        raise TypeError("Pole evaluations musi być iterowalne")
    evaluations = tuple(_deserialize_model_evaluation(ev) for ev in evaluations_raw)
    drift_report = _deserialize_drift_report(payload.get("drift_report"))
    return PipelineExecutionRecord(
        symbol=symbol,
        decided_at=decided_at,
        best_model=best_model,
        evaluations=evaluations,
        drift_report=drift_report,
        prediction_count=int(payload.get("prediction_count", 0) or 0),
        prediction_mean=(None if payload.get("prediction_mean") is None else float(payload["prediction_mean"])),
        prediction_std=(None if payload.get("prediction_std") is None else float(payload["prediction_std"])),
        prediction_min=(None if payload.get("prediction_min") is None else float(payload["prediction_min"])),
        prediction_max=(None if payload.get("prediction_max") is None else float(payload["prediction_max"])),
    )


def pipeline_history_snapshot_to_dict(snapshot: PipelineHistorySnapshot) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(snapshot, PipelineHistorySnapshot):
        raise TypeError("Oczekiwano instancji PipelineHistorySnapshot")
    result: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, records in snapshot.records.items():
        result[symbol] = [_serialize_pipeline_execution_record(record) for record in records]
    return result


def pipeline_history_snapshot_from_dict(payload: Mapping[str, Any]) -> PipelineHistorySnapshot:
    if not isinstance(payload, Mapping):
        raise TypeError("Dane wejściowe muszą być mapowaniem symbol→historia")
    records: Dict[str, Tuple[PipelineExecutionRecord, ...]] = {}
    for symbol, entries in payload.items():
        if entries is None:
            continue
        if not isinstance(symbol, str):
            raise TypeError("Klucze słownika migawek muszą być tekstowe")
        if not isinstance(entries, Iterable):
            raise TypeError("Lista rekordów historii musi być iterowalna")
        records[symbol] = tuple(_deserialize_pipeline_execution_record(entry) for entry in entries)
    return PipelineHistorySnapshot(records)


def pipeline_history_snapshot_to_json(snapshot: PipelineHistorySnapshot, *, indent: Optional[int] = None) -> str:
    data = pipeline_history_snapshot_to_dict(snapshot)
    return json.dumps(data, indent=indent, sort_keys=True)


def pipeline_history_snapshot_from_json(payload: Union[str, bytes, bytearray]) -> PipelineHistorySnapshot:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("Payload JSON musi być tekstem lub bajtami")
    data = json.loads(payload)
    return pipeline_history_snapshot_from_dict(data)


def pipeline_history_snapshot_to_file(snapshot: PipelineHistorySnapshot, path: PathInput, *, indent: Optional[int] = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pipeline_history_snapshot_to_json(snapshot, indent=indent)
    target.write_text(payload, encoding="utf-8")


def pipeline_history_snapshot_from_file(path: PathInput) -> PipelineHistorySnapshot:
    source = Path(path)
    data = source.read_text(encoding="utf-8")
    return pipeline_history_snapshot_from_json(data)


def diff_pipeline_history_snapshots(
    before: PipelineHistorySnapshot, after: PipelineHistorySnapshot
) -> PipelineHistoryDiff:
    if not isinstance(before, PipelineHistorySnapshot):
        raise TypeError("before musi być instancją PipelineHistorySnapshot")
    if not isinstance(after, PipelineHistorySnapshot):
        raise TypeError("after musi być instancją PipelineHistorySnapshot")

    diff = PipelineHistoryDiff()

    before_records = {symbol: tuple(records) for symbol, records in before.records.items() if records}
    after_records = {symbol: tuple(records) for symbol, records in after.records.items() if records}

    for symbol in before_records.keys() - after_records.keys():
        diff.removed[symbol] = before_records[symbol]

    for symbol in after_records.keys() - before_records.keys():
        diff.added[symbol] = after_records[symbol]

    for symbol in before_records.keys() & after_records.keys():
        previous = before_records[symbol]
        current = after_records[symbol]
        if previous != current:
            diff.changed[symbol] = (previous, current)

    return diff


def pipeline_history_diff_to_dict(diff: PipelineHistoryDiff) -> Dict[str, Any]:
    if not isinstance(diff, PipelineHistoryDiff):
        raise TypeError("Oczekiwano instancji PipelineHistoryDiff")

    result: Dict[str, Any] = {
        "added": {},
        "removed": {},
        "changed": {},
    }

    for symbol, records in diff.added.items():
        result["added"][symbol] = [_serialize_pipeline_execution_record(record) for record in records]

    for symbol, records in diff.removed.items():
        result["removed"][symbol] = [_serialize_pipeline_execution_record(record) for record in records]

    for symbol, (before_records, after_records) in diff.changed.items():
        result["changed"][symbol] = {
            "before": [_serialize_pipeline_execution_record(record) for record in before_records],
            "after": [_serialize_pipeline_execution_record(record) for record in after_records],
        }

    return result


def pipeline_history_diff_from_dict(payload: Mapping[str, Any]) -> PipelineHistoryDiff:
    if not isinstance(payload, Mapping):
        raise TypeError("Dane wejściowe muszą być mapowaniem diffu")

    diff = PipelineHistoryDiff()

    added_raw = payload.get("added", {})
    if not isinstance(added_raw, Mapping):
        raise TypeError("Sekcja 'added' musi być mapowaniem symbol→lista")
    for symbol, records in added_raw.items():
        if not isinstance(symbol, str):
            raise TypeError("Symbol w diffie musi być tekstowy")
        if records is None:
            continue
        if not isinstance(records, Iterable):
            raise TypeError("Lista rekordów dodanych musi być iterowalna")
        diff.added[symbol] = tuple(_deserialize_pipeline_execution_record(record) for record in records)

    removed_raw = payload.get("removed", {})
    if not isinstance(removed_raw, Mapping):
        raise TypeError("Sekcja 'removed' musi być mapowaniem symbol→lista")
    for symbol, records in removed_raw.items():
        if not isinstance(symbol, str):
            raise TypeError("Symbol w diffie musi być tekstowy")
        if records is None:
            continue
        if not isinstance(records, Iterable):
            raise TypeError("Lista rekordów usuniętych musi być iterowalna")
        diff.removed[symbol] = tuple(_deserialize_pipeline_execution_record(record) for record in records)

    changed_raw = payload.get("changed", {})
    if not isinstance(changed_raw, Mapping):
        raise TypeError("Sekcja 'changed' musi być mapowaniem symbol→dane")
    for symbol, records in changed_raw.items():
        if not isinstance(symbol, str):
            raise TypeError("Symbol w diffie musi być tekstowy")
        if records is None:
            continue
        if not isinstance(records, Mapping):
            raise TypeError("Dane zmienionych rekordów muszą być mapowaniem")
        before_entries = records.get("before", [])
        after_entries = records.get("after", [])
        if not isinstance(before_entries, Iterable) or not isinstance(after_entries, Iterable):
            raise TypeError("Sekcje 'before' i 'after' muszą być iterowalne")
        before_records = tuple(_deserialize_pipeline_execution_record(record) for record in before_entries)
        after_records = tuple(_deserialize_pipeline_execution_record(record) for record in after_entries)
        diff.changed[symbol] = (before_records, after_records)

    return diff


def pipeline_history_diff_to_json(diff: PipelineHistoryDiff, *, indent: Optional[int] = None) -> str:
    data = pipeline_history_diff_to_dict(diff)
    return json.dumps(data, indent=indent, sort_keys=True)


def pipeline_history_diff_from_json(payload: Union[str, bytes, bytearray]) -> PipelineHistoryDiff:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("Payload JSON musi być tekstem lub bajtami")
    data = json.loads(payload)
    return pipeline_history_diff_from_dict(data)


def pipeline_history_diff_to_file(diff: PipelineHistoryDiff, path: PathInput, *, indent: Optional[int] = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pipeline_history_diff_to_json(diff, indent=indent)
    target.write_text(payload, encoding="utf-8")


def pipeline_history_diff_from_file(path: PathInput) -> PipelineHistoryDiff:
    source = Path(path)
    data = source.read_text(encoding="utf-8")
    return pipeline_history_diff_from_json(data)


def _format_optional_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "n/d"
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensywne
        return "n/d"
    return f"{numeric:.{precision}f}"


def format_pipeline_execution_record(
    record: PipelineExecutionRecord,
    *,
    include_evaluations: bool = False,
    indent: str = "  ",
) -> str:
    if not isinstance(record, PipelineExecutionRecord):
        raise TypeError("record musi być instancją PipelineExecutionRecord")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    lines = [
        f"Symbol: {record.symbol}",
        f"Decyzja: {record.decided_at.isoformat()}",
        f"Najlepszy model: {record.best_model}",
    ]

    lines.append("Statystyki predykcji:")
    lines.append(f"{indent}Liczba: {int(record.prediction_count)}")
    lines.append(f"{indent}Średnia: {_format_optional_float(record.prediction_mean)}")
    lines.append(f"{indent}Odchylenie standardowe: {_format_optional_float(record.prediction_std)}")
    lines.append(f"{indent}Minimum: {_format_optional_float(record.prediction_min)}")
    lines.append(f"{indent}Maksimum: {_format_optional_float(record.prediction_max)}")

    if record.drift_report is not None:
        drift = record.drift_report
        lines.append("Dryf danych:")
        lines.append(f"{indent}Wskaźnik cech: {_format_optional_float(drift.feature_drift)}")
        lines.append(f"{indent}Zmiana zmienności: {_format_optional_float(drift.volatility_shift)}")
        lines.append(f"{indent}Próg: {_format_optional_float(drift.threshold)}")
        lines.append(f"{indent}Wykryto: {'tak' if drift.triggered else 'nie'}")

    if include_evaluations and record.evaluations:
        lines.append("Ewaluacje modeli:")
        for evaluation in record.evaluations:
            cv_scores = ", ".join(f"{float(score):.4f}" for score in evaluation.cv_scores) or "brak"
            lines.append(
                f"{indent}- {evaluation.model_type}: hit={_format_optional_float(evaluation.hit_rate)} "
                f"pnl={_format_optional_float(evaluation.pnl)} sharpe={_format_optional_float(evaluation.sharpe)}"
            )
            lines.append(f"{indent}{indent}CV: {cv_scores}")
            if evaluation.model_path:
                lines.append(f"{indent}{indent}Model: {evaluation.model_path}")

    return "\n".join(lines)


def format_pipeline_history_snapshot(
    snapshot: PipelineHistorySnapshot,
    *,
    include_evaluations: bool = False,
    indent: str = "  ",
) -> str:
    if not isinstance(snapshot, PipelineHistorySnapshot):
        raise TypeError("snapshot musi być instancją PipelineHistorySnapshot")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    if not snapshot.records:
        return "Historia pipeline'u jest pusta"

    lines = [
        f"Liczba symboli: {snapshot.total_symbols()}",
        f"Łączna liczba rekordów: {snapshot.total_records()}",
    ]

    for symbol in snapshot.symbols():
        records = snapshot.records[symbol]
        lines.append(f"Symbol {symbol}: {len(records)} rekordów")
        for entry in records:
            formatted = format_pipeline_execution_record(
                entry,
                include_evaluations=include_evaluations,
                indent=indent,
            )
            for line in formatted.splitlines():
                lines.append(f"{indent}{line}")

    return "\n".join(lines)


def format_pipeline_history_diff(
    diff: PipelineHistoryDiff,
    *,
    include_evaluations: bool = False,
    indent: str = "  ",
) -> str:
    if not isinstance(diff, PipelineHistoryDiff):
        raise TypeError("diff musi być instancją PipelineHistoryDiff")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    if diff.is_empty():
        return "Brak zmian w historii pipeline'u"

    lines = ["Zmiany w historii pipeline'u:"]

    if diff.added:
        lines.append("Dodane symbole:")
        for symbol in diff.added_symbols():
            records = diff.added[symbol]
            lines.append(f"{indent}{symbol}: {len(records)} rekordów")
            for record in records:
                formatted = format_pipeline_execution_record(
                    record,
                    include_evaluations=include_evaluations,
                    indent=indent,
                )
                for line in formatted.splitlines():
                    lines.append(f"{indent}{indent}{line}")

    if diff.removed:
        lines.append("Usunięte symbole:")
        for symbol in diff.removed_symbols():
            records = diff.removed[symbol]
            lines.append(f"{indent}{symbol}: {len(records)} rekordów")
            for record in records:
                formatted = format_pipeline_execution_record(
                    record,
                    include_evaluations=include_evaluations,
                    indent=indent,
                )
                for line in formatted.splitlines():
                    lines.append(f"{indent}{indent}{line}")

    if diff.changed:
        lines.append("Zmienione symbole:")
        for symbol in diff.changed_symbols():
            before_records, after_records = diff.changed[symbol]
            lines.append(f"{indent}{symbol} (przed → po):")
            for label, records in (("przed", before_records), ("po", after_records)):
                lines.append(f"{indent}{indent}{label}:")
                for record in records:
                    formatted = format_pipeline_execution_record(
                        record,
                        include_evaluations=include_evaluations,
                        indent=indent,
                    )
                    for line in formatted.splitlines():
                        lines.append(f"{indent}{indent}{indent}{line}")

    return "\n".join(lines)


def log_pipeline_execution_record(
    record: PipelineExecutionRecord,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    include_evaluations: bool = False,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    message = format_pipeline_execution_record(
        record,
        include_evaluations=include_evaluations,
        indent=indent,
    )
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


def log_pipeline_history_snapshot(
    snapshot: PipelineHistorySnapshot,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    include_evaluations: bool = False,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    message = format_pipeline_history_snapshot(
        snapshot,
        include_evaluations=include_evaluations,
        indent=indent,
    )
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


def log_pipeline_history_diff(
    diff: PipelineHistoryDiff,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    include_evaluations: bool = False,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    message = format_pipeline_history_diff(
        diff,
        include_evaluations=include_evaluations,
        indent=indent,
    )
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


def format_ensemble_definition(
    definition: EnsembleDefinition,
    *,
    indent: str = "  ",
) -> str:
    """Zbuduj czytelną reprezentację definicji zespołu modeli."""

    if not isinstance(definition, EnsembleDefinition):
        raise TypeError("definition musi być instancją EnsembleDefinition")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    lines = [
        f"Zespół: {definition.name}",
        f"Agregacja: {definition.aggregation}",
        f"Liczba komponentów: {len(definition.components)}",
    ]

    if definition.components:
        lines.append("Komponenty:")
        weights = definition.weights or ()
        for idx, component in enumerate(definition.components):
            entry = f"{indent}- {component}"
            if idx < len(weights):
                entry += f" (waga: {_format_optional_float(float(weights[idx]))})"
            lines.append(entry)
    else:
        lines.append("Komponenty: brak")

    return "\n".join(lines)


def format_ensemble_registry_snapshot(
    snapshot: EnsembleRegistrySnapshot,
    *,
    indent: str = "  ",
) -> str:
    """Zbuduj raport tekstowy opisujący zarejestrowane zespoły modeli."""

    if not isinstance(snapshot, EnsembleRegistrySnapshot):
        raise TypeError("snapshot musi być instancją EnsembleRegistrySnapshot")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    if not snapshot.ensembles:
        return "Rejestr zespołów jest pusty"

    lines = [f"Liczba zespołów: {snapshot.total_ensembles()}"]

    for name in snapshot.names():
        definition = snapshot.ensembles[name]
        formatted = format_ensemble_definition(definition, indent=indent)
        for line in formatted.splitlines():
            lines.append(f"{indent}{line}")

    return "\n".join(lines)


def format_ensemble_registry_diff(
    diff: EnsembleRegistryDiff,
    *,
    indent: str = "  ",
) -> str:
    """Zbuduj raport tekstowy opisujący różnice w rejestrze zespołów modeli."""

    if not isinstance(diff, EnsembleRegistryDiff):
        raise TypeError("diff musi być instancją EnsembleRegistryDiff")
    if not isinstance(indent, str):
        raise TypeError("indent musi być tekstem")

    if diff.is_empty():
        return "Brak zmian w rejestrze zespołów"

    lines = ["Zmiany w rejestrze zespołów:"]

    if diff.added:
        lines.append("Dodane zespoły:")
        for name in diff.added_names():
            formatted = format_ensemble_definition(diff.added[name], indent=indent)
            for line in formatted.splitlines():
                lines.append(f"{indent}{line}")

    if diff.removed:
        lines.append("Usunięte zespoły:")
        for name in diff.removed_names():
            formatted = format_ensemble_definition(diff.removed[name], indent=indent)
            for line in formatted.splitlines():
                lines.append(f"{indent}{line}")

    if diff.changed:
        lines.append("Zmienione zespoły:")
        for name in diff.changed_names():
            before_definition, after_definition = diff.changed[name]
            lines.append(f"{indent}{name} (przed → po):")
            for label, definition in (("przed", before_definition), ("po", after_definition)):
                lines.append(f"{indent}{indent}{label}:")
                formatted = format_ensemble_definition(definition, indent=indent)
                for line in formatted.splitlines():
                    lines.append(f"{indent}{indent}{indent}{line}")

    return "\n".join(lines)


def log_ensemble_definition(
    definition: EnsembleDefinition,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    """Zaloguj pojedynczą definicję zespołu modeli."""

    message = format_ensemble_definition(definition, indent=indent)
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


def log_ensemble_registry_snapshot(
    snapshot: EnsembleRegistrySnapshot,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    """Zaloguj stan rejestru zespołów modeli."""

    message = format_ensemble_registry_snapshot(snapshot, indent=indent)
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


def log_ensemble_registry_diff(
    diff: EnsembleRegistryDiff,
    *,
    level: int = logging.INFO,
    logger_like: Optional[LoggerLike] = None,
    indent: str = "  ",
    extra: Optional[Mapping[str, Any]] = None,
    stacklevel: int = 2,
) -> None:
    """Zaloguj różnice pomiędzy dwiema migawkami rejestru zespołów."""

    message = format_ensemble_registry_diff(diff, indent=indent)
    _emit_history_log(
        message,
        level=level,
        logger_like=logger_like,
        extra=extra,
        stacklevel=stacklevel,
    )


__all__ = [
    "AIManager",
    "MarketRegimeClassifier",
    "MarketRegimeAssessment",
    "TrainResult",
    "ModelEvaluation",
    "StrategySelectionResult",
    "EnsembleDefinition",
    "EnsembleRegistrySnapshot",
    "EnsembleRegistryDiff",
    "DriftReport",
    "TrainingSchedule",
    "PipelineSchedule",
    "PipelineExecutionRecord",
    "PipelineHistorySnapshot",
    "PipelineHistoryDiff",
    "ensemble_registry_snapshot_to_dict",
    "ensemble_registry_snapshot_from_dict",
    "ensemble_registry_snapshot_to_json",
    "ensemble_registry_snapshot_from_json",
    "ensemble_registry_snapshot_to_file",
    "ensemble_registry_snapshot_from_file",
    "diff_ensemble_snapshots",
    "ensemble_registry_diff_to_dict",
    "ensemble_registry_diff_from_dict",
    "ensemble_registry_diff_to_json",
    "ensemble_registry_diff_from_json",
    "ensemble_registry_diff_to_file",
    "ensemble_registry_diff_from_file",
    "format_ensemble_definition",
    "format_ensemble_registry_snapshot",
    "format_ensemble_registry_diff",
    "log_ensemble_definition",
    "log_ensemble_registry_snapshot",
    "log_ensemble_registry_diff",
    "pipeline_history_snapshot_to_dict",
    "pipeline_history_snapshot_from_dict",
    "pipeline_history_snapshot_to_json",
    "pipeline_history_snapshot_from_json",
    "pipeline_history_snapshot_to_file",
    "pipeline_history_snapshot_from_file",
    "diff_pipeline_history_snapshots",
    "pipeline_history_diff_to_dict",
    "pipeline_history_diff_from_dict",
    "pipeline_history_diff_to_json",
    "pipeline_history_diff_from_json",
    "pipeline_history_diff_to_file",
    "pipeline_history_diff_from_file",
    "format_pipeline_execution_record",
    "format_pipeline_history_snapshot",
    "format_pipeline_history_diff",
    "log_pipeline_execution_record",
    "log_pipeline_history_snapshot",
    "log_pipeline_history_diff",
    "_AIModels",
    "_windowize",
]
