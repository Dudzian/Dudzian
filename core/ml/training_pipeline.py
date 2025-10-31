"""Lekkie pipeline'y treningowe z obsługą fallbacku backendów ML."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableSequence, Protocol, Sequence, runtime_checkable

from bot_core.ai import backends
from bot_core.ai.feature_engineering import FeatureDataset

from core.data.validators import DatasetValidationError, DatasetValidator

from .factory import build_backend

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SupportsTraining(Protocol):
    """Minimalny interfejs wymagany od modelu zwracanego przez pipeline."""

    def fit(self, samples: Sequence[Mapping[str, float]], targets: Sequence[float]) -> None:
        """Uczenie modelu na przekazanych próbkach."""

    def batch_predict(self, samples: Sequence[Mapping[str, float]]) -> Sequence[float]:
        """Wykonuje predykcję na zbiorze próbek."""


@dataclass(slots=True)
class TrainingPipelineResult:
    """Podsumowanie pojedynczego uruchomienia pipeline'u treningowego."""

    backend: str
    model: SupportsTraining
    fallback_chain: tuple[Mapping[str, str], ...]
    log_path: Path | None
    validation_log_path: Path | None


class TrainingPipeline:
    """Orchestrator wyboru backendu ML z obsługą fallbacku i logowaniem."""

    def __init__(
        self,
        *,
        preferred_backends: Sequence[str] | None = None,
        config_path: Path | None = None,
        fallback_log_dir: Path | None = None,
        validation_log_dir: Path | None = None,
        dataset_validator: DatasetValidator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        order = [candidate.strip().lower() for candidate in preferred_backends or () if candidate.strip()]
        if "reference" not in order:
            order.append("reference")
        self._preferred_backends: tuple[str, ...] = tuple(order) or ("reference",)
        self._config_path = config_path
        self._fallback_log_dir = Path(fallback_log_dir or Path("logs/ml/fallback"))
        self._validation_log_dir = Path(validation_log_dir or Path("logs/data/validation"))
        self._dataset_validator = dataset_validator or DatasetValidator()
        self._logger = logger or LOGGER

    def train(self, dataset: FeatureDataset) -> TrainingPipelineResult:
        """Trenuje model używając pierwszego dostępnego backendu z listy preferencji."""

        if not isinstance(dataset, FeatureDataset):
            raise TypeError("dataset musi być instancją FeatureDataset")
        samples = dataset.features
        targets = dataset.targets
        if not samples or not targets:
            raise ValueError("Dataset treningowy jest pusty")

        report = self._dataset_validator.validate(dataset)
        validation_log_path = self._dataset_validator.log_report(
            report, self._validation_log_dir
        )
        if report.has_errors:
            raise DatasetValidationError(report, validation_log_path)

        fallback_chain: MutableSequence[Mapping[str, str]] = []
        last_error: backends.BackendUnavailableError | None = None
        for candidate in self._preferred_backends:
            try:
                backend_name, model = build_backend(
                    preferred=(candidate,),
                    config_path=self._config_path,
                )
            except backends.BackendUnavailableError as exc:
                payload = self._serialize_error(candidate, exc)
                fallback_chain.append(payload)
                last_error = exc
                self._logger.warning(
                    "Backend %s jest niedostępny: %s", candidate, payload["message"]
                )
                continue
            except ModuleNotFoundError as exc:
                wrapped = backends.BackendUnavailableError(candidate, exc.name)
                payload = self._serialize_error(candidate, wrapped)
                fallback_chain.append(payload)
                last_error = wrapped
                self._logger.warning(
                    "Backend %s nie może zostać załadowany: %s", candidate, payload["message"]
                )
                continue

            model.fit(samples, targets)
            selected_backend = backend_name
            if backend_name != candidate:
                resolved = self._resolve_backend_error(candidate)
                if resolved is not None:
                    fallback_chain.append(resolved)
                    last_error = backends.BackendUnavailableError(
                        candidate,
                        resolved.get("module"),
                        install_hint=resolved.get("install_hint"),
                    )
                self._logger.warning(
                    "Aktywowano fallback backendu %s → %s", candidate, backend_name
                )
            elif fallback_chain:
                self._logger.info(
                    "Backend %s został użyty po wcześniejszych fallbackach", backend_name
                )
            else:
                self._logger.info("Backend %s wybrany bez konieczności fallbacku", backend_name)

            log_path = self._write_fallback_log(selected_backend, tuple(fallback_chain))
            return TrainingPipelineResult(
                backend=selected_backend,
                model=model,
                fallback_chain=tuple(fallback_chain),
                log_path=log_path,
                validation_log_path=validation_log_path,
            )

        if last_error is not None:
            raise last_error
        raise backends.BackendUnavailableError("brak dostępnych backendów", None)

    def _resolve_backend_error(self, backend: str) -> Mapping[str, str] | None:
        try:
            backends.require_backend(backend, config_path=self._config_path)
        except backends.BackendUnavailableError as exc:
            return self._serialize_error(backend, exc)
        except ModuleNotFoundError as exc:
            wrapped = backends.BackendUnavailableError(backend, exc.name)
            return self._serialize_error(backend, wrapped)
        return None

    def _serialize_error(
        self, backend: str, exc: backends.BackendUnavailableError
    ) -> Mapping[str, str]:
        payload = {
            "backend": backend,
            "message": str(exc),
        }
        module_name = getattr(exc, "module_name", None)
        install_hint = getattr(exc, "install_hint", None)
        if module_name:
            payload["module"] = module_name
        if install_hint:
            payload["install_hint"] = install_hint
        return payload

    def _write_fallback_log(
        self,
        backend: str,
        fallback_chain: Sequence[Mapping[str, str]],
    ) -> Path | None:
        if not fallback_chain:
            return None
        self._fallback_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._fallback_log_dir / f"fallback_{timestamp}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "preferred_backends": list(self._preferred_backends),
            "selected_backend": backend,
            "errors": list(fallback_chain),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

