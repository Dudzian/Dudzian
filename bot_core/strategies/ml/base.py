"""Bazowe interfejsy oraz wspólna logika dla strategii ML."""
from __future__ import annotations

import abc
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Protocol, Sequence

import numpy as np

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


FeatureVector = np.ndarray


class FeaturePipeline(Protocol):
    """Kontrakt na pipeline przygotowujący cechy i target."""

    feature_names: Sequence[str]

    def fit(self, snapshots: Sequence[MarketSnapshot]) -> None:
        ...

    def transform_features(self, snapshot: MarketSnapshot) -> FeatureVector:
        ...

    def build_training_set(
        self, history: Sequence[MarketSnapshot]
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(slots=True)
class MLModelAdapter(abc.ABC):
    """Abstrakcja nad modelem ML niezależna od frameworka."""

    name: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - sanity check
        self.hyperparameters = dict(self.hyperparameters)

    @abc.abstractmethod
    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """Trenuje model na przekazanym zbiorze."""

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Zwraca prognozy w tej samej liczbie wierszy co wejście."""

    def save(self, path: Path) -> None:
        """Domyślna serializacja modelem pickle/joblib."""

        path = Path(path)
        if path.suffix == ".json":
            payload = {
                "name": self.name,
                "hyperparameters": self.hyperparameters,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        try:
            import joblib
        except ImportError:  # pragma: no cover - środowiska bez joblib
            with path.open("wb") as handle:
                pickle.dump(self, handle)
        else:  # pragma: no cover - joblib testowany tylko gdy dostępny
            joblib.dump(self, path)

    def load(self, path: Path) -> None:
        """Odtwarza parametry z pliku json."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix != ".json":
            raise ValueError("Obsługiwane jest tylko ładowanie metadanych JSON dla adaptera")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):  # pragma: no cover - walidacja defensywna
            raise ValueError("Nieprawidłowy format metadanych modelu")
        if payload.get("name") != self.name:
            raise ValueError("Niezgodność typu modelu w metadanych")
        hyperparams = payload.get("hyperparameters", {})
        if isinstance(hyperparams, Mapping):
            self.hyperparameters = dict(hyperparams)


class ClassicalModelAdapter(MLModelAdapter):
    """Adapter dla klasycznych modeli z API scikit-learn."""

    def __init__(self, estimator: Any, name: str, hyperparameters: Mapping[str, Any] | None = None):
        super().__init__(name=name, hyperparameters=hyperparameters or {})
        self._estimator = estimator

    @property
    def estimator(self) -> Any:
        return self._estimator

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        if not hasattr(self._estimator, "fit"):
            raise TypeError("Przekazany obiekt nie implementuje metody fit")
        self._estimator.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self._estimator, "predict"):
            raise TypeError("Przekazany obiekt nie implementuje metody predict")
        prediction = self._estimator.predict(features)
        return np.asarray(prediction, dtype=float)

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "name": self.name,
            "hyperparameters": self.hyperparameters,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SequentialModelAdapter(MLModelAdapter):
    """Adapter bazowy dla modeli sekwencyjnych (np. sieci neuronowe)."""

    def __init__(self, trainer: Any, name: str, hyperparameters: Mapping[str, Any] | None = None):
        super().__init__(name=name, hyperparameters=hyperparameters or {})
        self._trainer = trainer

    @property
    def trainer(self) -> Any:
        return self._trainer

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        if not hasattr(self._trainer, "fit"):
            raise TypeError("Przekazany obiekt trenera nie implementuje fit")
        self._trainer.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if hasattr(self._trainer, "predict"):
            output = self._trainer.predict(features)
        elif hasattr(self._trainer, "__call__"):
            output = self._trainer(features)
        else:
            raise TypeError("Trener nie udostępnia metody predict ani call")
        return np.asarray(output, dtype=float)


@dataclass(slots=True)
class MLStrategyEngine(StrategyEngine):
    """Silnik strategii oparty na dowolnym adapterze modelu ML."""

    model: MLModelAdapter
    feature_pipeline: FeaturePipeline
    threshold: float = 0.5
    buffer: MutableSequence[MarketSnapshot] = field(default_factory=list)

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        self.buffer.clear()
        self.buffer.extend(history)
        if history:
            self.feature_pipeline.fit(history)
            features, target = self.feature_pipeline.build_training_set(history)
            if len(features) and len(target):
                self.model.fit(features, target)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        if not self.feature_pipeline.feature_names:
            raise RuntimeError("Pipeline nie został poprawnie zainicjalizowany")
        features = self.feature_pipeline.transform_features(snapshot)
        prediction = float(self.model.predict(features.reshape(1, -1))[0])
        side = "BUY" if prediction >= self.threshold else "SELL"
        signal = StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            confidence=abs(prediction - self.threshold),
            metadata={
                "prediction": prediction,
                "threshold": self.threshold,
            },
        )
        return (signal,)


__all__ = [
    "FeatureVector",
    "FeaturePipeline",
    "MLModelAdapter",
    "ClassicalModelAdapter",
    "SequentialModelAdapter",
    "MLStrategyEngine",
]
