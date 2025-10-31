"""Prosty rejestr modeli ML przechowujący metadane i wersjonowanie."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from bot_core.runtime.state_manager import RuntimeStateManager

LOGGER = logging.getLogger(__name__)


class ModelRegistryError(RuntimeError):
    """Ogólny błąd rejestru modeli."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadane pojedynczego modelu ML."""

    model_id: str
    backend: str
    artifact_path: str
    sha256: str
    created_at: datetime
    dataset_metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "backend": self.backend,
            "artifact_path": self.artifact_path,
            "sha256": self.sha256,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
        }
        if self.dataset_metadata:
            payload["dataset_metadata"] = dict(self.dataset_metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelMetadata":
        created_at_raw = payload.get("created_at")
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError as exc:  # pragma: no cover - defensywne
                raise ModelRegistryError("Niepoprawny format pola 'created_at'") from exc
        else:
            raise ModelRegistryError("Pole 'created_at' jest wymagane i musi być ISO-8601")
        metadata = payload.get("dataset_metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, Mapping):  # pragma: no cover - defensywne
            raise ModelRegistryError("Pole 'dataset_metadata' musi być mapą")
        return cls(
            model_id=str(payload.get("model_id", "")),
            backend=str(payload.get("backend", "")),
            artifact_path=str(payload.get("artifact_path", "")),
            sha256=str(payload.get("sha256", "")),
            created_at=created_at.astimezone(timezone.utc),
            dataset_metadata=dict(metadata),
        )


class ModelRegistry:
    """Rejestr modeli zapisujący metadane oraz aktywną wersję."""

    def __init__(
        self,
        root: str | Path = "var/models",
        *,
        filename: str = "registry.json",
        state_manager: RuntimeStateManager | None = None,
    ) -> None:
        self._root = Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / filename
        self._state_manager = state_manager

    # -- API publiczne ------------------------------------------------- #
    def publish_model(
        self,
        artifact_path: str | Path,
        *,
        backend: str,
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> ModelMetadata:
        """Dodaje model do rejestru i oznacza go jako aktywny."""

        artifact = Path(artifact_path).expanduser()
        if not artifact.exists():
            raise ModelRegistryError(f"Plik modelu {artifact} nie istnieje")

        digest = self._compute_sha256(artifact)
        created_at = datetime.now(timezone.utc)
        model_id = f"{created_at.strftime('%Y%m%d%H%M%S')}_{digest[:8]}"
        metadata = ModelMetadata(
            model_id=model_id,
            backend=str(backend),
            artifact_path=str(artifact),
            sha256=digest,
            created_at=created_at,
            dataset_metadata=dict(dataset_metadata or {}),
        )

        registry = self._load_registry()
        existing = {entry["model_id"]: entry for entry in registry.get("models", [])}
        existing[model_id] = metadata.to_dict()
        registry["models"] = list(existing.values())
        registry["active_model_id"] = model_id
        self._write_registry(registry)
        self._sync_state_manager(metadata)
        LOGGER.info("Opublikowano model %s (backend=%s)", model_id, backend)
        return metadata

    def list_models(self) -> Sequence[ModelMetadata]:
        """Zwraca listę zarejestrowanych modeli posortowaną malejąco po dacie."""

        registry = self._load_registry()
        entries: Iterable[ModelMetadata] = (
            ModelMetadata.from_dict(item) for item in registry.get("models", [])
        )
        return tuple(sorted(entries, key=lambda item: item.created_at, reverse=True))

    def get_active_model(self) -> ModelMetadata | None:
        """Pobiera aktywny model (jeśli został ustawiony)."""

        registry = self._load_registry()
        active_id = registry.get("active_model_id")
        if not active_id:
            return None
        for entry in registry.get("models", []):
            if entry.get("model_id") == active_id:
                return ModelMetadata.from_dict(entry)
        return None

    def rollback(self, model_id: str) -> ModelMetadata:
        """Aktywuje wskazany model i aktualizuje stan runtime."""

        registry = self._load_registry()
        candidates = {
            entry.get("model_id"): entry for entry in registry.get("models", [])
        }
        if model_id not in candidates:
            raise ModelRegistryError(f"Model {model_id} nie istnieje w rejestrze")
        registry["active_model_id"] = model_id
        self._write_registry(registry)
        metadata = ModelMetadata.from_dict(candidates[model_id])
        self._sync_state_manager(metadata)
        LOGGER.info("Aktywowano model %s", model_id)
        return metadata

    # -- Metody pomocnicze -------------------------------------------- #
    def _load_registry(self) -> MutableMapping[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"models": [], "active_model_id": None}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelRegistryError("Plik rejestru modeli jest uszkodzony") from exc
        if "models" not in data:
            data["models"] = []
        return data

    def _write_registry(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _sync_state_manager(self, metadata: ModelMetadata) -> None:
        if self._state_manager is None:
            return
        payload = metadata.to_dict()
        payload["synced_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self._state_manager.set_active_model(payload)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            LOGGER.warning("Nie udało się zaktualizować RuntimeStateManager: %s", exc)


__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "ModelRegistryError",
]

