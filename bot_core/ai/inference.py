"""Moduł inference Decision Engine korzystający z wytrenowanych artefaktów."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Mapping, MutableMapping, Sequence

from bot_core.alerts import DriftAlertPayload, emit_model_drift_alert

from ._license import ensure_ai_signals_enabled
from .models import ModelArtifact, ModelScore


@dataclass(slots=True)
class ModelRepository:
    """Odpowiada za ładowanie i wersjonowanie artefaktów modeli."""

    base_path: Path
    manifest_name: str = "manifest.json"
    _manifest_cache: dict[str, object] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._manifest_cache = None

    # ------------------------------------------------------------------ manifest --
    @property
    def _manifest_path(self) -> Path:
        return self.base_path / self.manifest_name

    def _default_manifest(self) -> dict[str, object]:
        return {"versions": {}, "aliases": {}, "active": None}

    def _ensure_manifest_maps(
        self, manifest: MutableMapping[str, object]
    ) -> tuple[MutableMapping[str, object], MutableMapping[str, str]]:
        versions_obj = manifest.get("versions")
        if isinstance(versions_obj, MutableMapping):
            versions: MutableMapping[str, object] = versions_obj
        elif isinstance(versions_obj, Mapping):
            versions = {
                str(key): dict(value)
                for key, value in versions_obj.items()
                if isinstance(key, str) and isinstance(value, Mapping)
            }
            manifest["versions"] = versions
        else:
            versions = {}
            manifest["versions"] = versions

        aliases_obj = manifest.get("aliases")
        if isinstance(aliases_obj, MutableMapping):
            aliases: MutableMapping[str, str] = aliases_obj  # type: ignore[assignment]
        elif isinstance(aliases_obj, Mapping):
            aliases = {
                str(key): str(value)
                for key, value in aliases_obj.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            manifest["aliases"] = aliases
        else:
            aliases = {}
            manifest["aliases"] = aliases

        return versions, aliases

    def _load_manifest(self) -> dict[str, object]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        manifest = self._default_manifest()
        try:
            if self._manifest_path.exists():
                with self._manifest_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, Mapping):
                    versions = loaded.get("versions", {})
                    aliases = loaded.get("aliases", {})
                    active = loaded.get("active")
                    manifest["versions"] = {
                        str(version): dict(entry)
                        for version, entry in dict(versions).items()
                        if isinstance(version, str) and isinstance(entry, Mapping)
                    }
                    manifest["aliases"] = {
                        str(alias): str(version)
                        for alias, version in dict(aliases).items()
                        if isinstance(alias, str) and isinstance(version, str)
                    }
                    manifest["active"] = str(active) if isinstance(active, str) else None
        except (OSError, json.JSONDecodeError):
            manifest = self._default_manifest()
        self._ensure_manifest_maps(manifest)
        self._synchronise_aliases(manifest)
        self._manifest_cache = manifest
        return manifest

    def _write_manifest(self, manifest: Mapping[str, object]) -> None:
        payload = {
            "versions": dict(manifest.get("versions", {})),
            "aliases": dict(manifest.get("aliases", {})),
            "active": manifest.get("active"),
        }
        tmp_path = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.replace(self._manifest_path)
        self._manifest_cache = dict(payload)

    def get_manifest(self) -> dict[str, object]:
        manifest = self._load_manifest()
        return json.loads(json.dumps(manifest))  # deep copy defensywny

    def list_versions(self) -> tuple[str, ...]:
        manifest = self._load_manifest()
        versions = list(manifest.get("versions", {}).keys())
        versions.sort(key=self._version_sort_key)
        return tuple(versions)

    def get_active_version(self) -> str | None:
        manifest = self._load_manifest()
        active = manifest.get("active")
        return str(active) if isinstance(active, str) else None

    def set_active_version(self, version: str) -> None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions = manifest.get("versions", {})
        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")
        manifest["active"] = version
        self._write_manifest(manifest)

    # ------------------------------------------------------------------ helpers --
    def _version_sort_key(self, version: str) -> tuple[Any, ...]:
        parts: list[Any] = []
        for chunk in str(version).replace("-", ".").split("."):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.isdigit():
                parts.append(int(chunk))
            else:
                parts.append(chunk)
        return tuple(parts) or (str(version),)

    def _json_safe(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _extract_version(self, artifact: ModelArtifact, provided: str | None) -> str | None:
        if provided:
            return provided.strip()
        meta_version = artifact.metadata.get("model_version")
        if isinstance(meta_version, str) and meta_version.strip():
            return meta_version.strip()
        return None

    def _update_manifest_entry(
        self,
        *,
        version: str,
        destination: Path,
        artifact: ModelArtifact,
        aliases: Sequence[str] | None,
        activate: bool | None,
    ) -> None:
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)

        relative_path = destination.relative_to(self.base_path)
        now = datetime.now(timezone.utc).isoformat()
        entry_aliases = tuple({str(alias).strip() for alias in (aliases or ()) if str(alias).strip()})
        versions[version] = {
            "file": str(relative_path),
            "saved_at": now,
            "metrics": self._json_safe(artifact.metrics),
            "metadata": self._json_safe(artifact.metadata),
            "aliases": list(entry_aliases),
        }
        for alias in entry_aliases:
            aliases_map[alias] = version

        if activate is True or (
            activate is None and (manifest.get("active") in (None, version))
        ):
            manifest["active"] = version

        self._synchronise_aliases(manifest)

        self._write_manifest(manifest)

    def _resolve_reference(self, reference: str) -> Path:
        reference = reference.strip()
        manifest = self._load_manifest()
        if reference in {"active", "@active"}:
            active = manifest.get("active")
            if not isinstance(active, str) or not active:
                raise KeyError("Brak aktywnej wersji modelu w repozytorium")
            reference = active

        versions = manifest.get("versions", {})
        aliases = manifest.get("aliases", {})
        mapped_version = aliases.get(reference) if isinstance(aliases, Mapping) else None
        version = reference if reference in versions else mapped_version
        if isinstance(version, str) and version in versions:
            entry = versions[version]
            if isinstance(entry, Mapping):
                file_path = entry.get("file")
                if isinstance(file_path, str):
                    path = Path(file_path)
                    return path if path.is_absolute() else (self.base_path / path)
            raise KeyError(f"Manifest nie zawiera ścieżki dla wersji '{version}'")

        path = Path(reference)
        return path if path.is_absolute() else (self.base_path / path)

    # ------------------------------------------------------------------ API --
    def load(self, artifact: str | Path | Mapping[str, object]) -> ModelArtifact:
        if isinstance(artifact, Mapping):
            return ModelArtifact.from_dict(artifact)
        if isinstance(artifact, Path):
            path = artifact if artifact.is_absolute() else self.base_path / artifact
        else:
            path = self._resolve_reference(str(artifact))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ModelArtifact.from_dict(payload)

    def save(
        self,
        artifact: ModelArtifact,
        name: str,
        *,
        version: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path:
        destination = self.base_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2)
        tmp_path.replace(destination)

        resolved_version = self._extract_version(artifact, version)
        if resolved_version:
            self._update_manifest_entry(
                version=resolved_version,
                destination=destination,
                artifact=artifact,
                aliases=aliases,
                activate=activate,
            )
        return destination

    def publish(
        self,
        artifact: ModelArtifact,
        *,
        version: str,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool = True,
    ) -> Path:
        if not version or not version.strip():
            raise ValueError("version must be a non-empty string")
        filename = filename or f"model-{version}.json"
        base_metadata = dict(artifact.metadata)
        base_metadata.setdefault("model_version", version)
        published = ModelArtifact(
            feature_names=artifact.feature_names,
            model_state=artifact.model_state,
            trained_at=artifact.trained_at,
            metrics=artifact.metrics,
            metadata=base_metadata,
            backend=artifact.backend,
        )
        publish_aliases = tuple(aliases) if aliases is not None else ("latest",)
        return self.save(
            published,
            filename,
            version=version,
            aliases=publish_aliases,
            activate=activate,
        )

    def resolve(self, reference: str | None = None) -> Path:
        if reference is None:
            active = self.get_active_version()
            if active is None:
                raise KeyError("Brak aktywnej wersji modelu")
            reference = active
        else:
            reference = str(reference)
        path = self._resolve_reference(reference)
        return path

    def get_version_entry(self, version: str) -> Mapping[str, object] | None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, _ = self._ensure_manifest_maps(manifest)
        payload = versions.get(version)
        if isinstance(payload, Mapping):
            return json.loads(json.dumps(payload))
        return None

    def get_alias_target(self, alias: str) -> str | None:
        alias = str(alias).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        manifest = self._load_manifest()
        _, aliases_map = self._ensure_manifest_maps(manifest)
        target = aliases_map.get(alias)
        return str(target) if isinstance(target, str) and target else None

    def assign_alias(self, alias: str, version: str) -> None:
        alias = str(alias).strip()
        version = str(version).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")
        current = aliases_map.get(alias)
        if current == version:
            return
        aliases_map[alias] = version
        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def promote_version(
        self,
        version: str,
        *,
        aliases: Sequence[str] | None = None,
    ) -> None:
        """Promuje wybraną wersję do aktywnej oraz aktualizuje aliasy."""

        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")

        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)

        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")

        manifest["active"] = version

        if aliases is not None:
            normalized_aliases = {
                str(alias).strip()
                for alias in aliases
                if str(alias).strip()
            }
            if normalized_aliases:
                for alias in normalized_aliases:
                    aliases_map[alias] = version
            else:
                for alias, mapped in list(aliases_map.items()):
                    if mapped == version:
                        aliases_map.pop(alias, None)

        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def remove_alias(self, alias: str, *, missing_ok: bool = False) -> None:
        alias = str(alias).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        manifest = self._load_manifest()
        _, aliases_map = self._ensure_manifest_maps(manifest)
        if alias not in aliases_map:
            if missing_ok:
                return
            raise KeyError(f"Alias '{alias}' nie istnieje w repozytorium modeli")
        aliases_map.pop(alias, None)
        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def remove_version(
        self,
        version: str,
        *,
        delete_file: bool = False,
        missing_ok: bool = False,
    ) -> None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        payload = versions.get(version)
        if payload is None:
            if missing_ok:
                return
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")

        versions.pop(version, None)
        for alias, mapped in list(aliases_map.items()):
            if mapped == version:
                aliases_map.pop(alias, None)
        if manifest.get("active") == version:
            manifest["active"] = None

        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

        if delete_file and isinstance(payload, Mapping):
            file_path = payload.get("file")
            if isinstance(file_path, str) and file_path:
                path = Path(file_path)
                resolved = path if path.is_absolute() else (self.base_path / path)
                try:
                    resolved.unlink()
                except FileNotFoundError:
                    pass

    def _synchronise_aliases(self, manifest: MutableMapping[str, object]) -> None:
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        for ver, payload in list(versions.items()):
            if not isinstance(payload, Mapping):
                continue
            current_aliases = [
                alias
                for alias, mapped in aliases_map.items()
                if mapped == ver
            ]
            payload = dict(payload)
            payload["aliases"] = current_aliases
            versions[ver] = payload


@dataclass(slots=True)
class _FeatureDriftMonitor:
    """Simple drift monitor based on moving average of feature z-scores."""

    threshold: float = 3.0
    window: int = 50
    min_observations: int = 10
    cooldown: int = 25
    backend: str = "decision_engine"
    _values: Deque[float] = field(init=False, repr=False)
    _last_alert_score: float | None = field(init=False, default=None, repr=False)
    _last_alert_time: float | None = field(init=False, default=None, repr=False)
    _model_name: str = field(init=False, default="unknown", repr=False)
    _enabled: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self._values = deque(maxlen=max(self.window, 1))

    def configure(
        self,
        *,
        model_name: str,
        threshold: float | None = None,
        window: int | None = None,
        min_observations: int | None = None,
        cooldown: int | None = None,
        backend: str | None = None,
    ) -> None:
        self._model_name = model_name
        if threshold is not None:
            self.threshold = max(float(threshold), 0.0)
        if window is not None:
            self.window = max(int(window), 1)
        if min_observations is not None:
            self.min_observations = max(int(min_observations), 1)
        if cooldown is not None:
            self.cooldown = max(int(cooldown), 1)
        if backend is not None:
            self.backend = str(backend)
        self._values = deque(self._values, maxlen=self.window)

    def disable(self) -> None:
        self._enabled = False

    def observe(
        self,
        features: Mapping[str, float],
        scalers: Mapping[str, tuple[float, float]],
    ) -> float | None:
        if not self._enabled or not scalers:
            return None
        total = 0.0
        count = 0
        for name, (mean, stdev) in scalers.items():
            if stdev <= 0:
                continue
            value = float(features.get(name, mean))
            total += abs(value - mean) / stdev
            count += 1
        if count == 0:
            return None
        score = total / count
        self._values.append(score)
        avg = sum(self._values) / len(self._values)
        if len(self._values) >= self.min_observations and avg > self.threshold:
            now = time.monotonic()
            should_alert = False
            if self._last_alert_score is None:
                should_alert = True
            elif abs(avg - self._last_alert_score) >= 0.1:
                should_alert = True
            elif self._last_alert_time is not None and now - self._last_alert_time > self.cooldown:
                should_alert = True
            if should_alert:
                payload = DriftAlertPayload(
                    model_name=self._model_name,
                    drift_score=avg,
                    threshold=self.threshold,
                    window=len(self._values),
                    backend=self.backend,
                    extra={"recent_score": score},
                )
                emit_model_drift_alert(payload)
                self._last_alert_score = avg
                self._last_alert_time = now
        return avg


class DecisionModelInference:
    """Wykonuje scoring kandydatów Decision Engine."""

    def __init__(self, repository: ModelRepository) -> None:
        ensure_ai_signals_enabled("inference modeli AI")
        self._repository = repository
        self._artifact: ModelArtifact | None = None
        self._model: Any | None = None
        self._target_scale: float = 1.0
        self._feature_scalers: dict[str, tuple[float, float]] = {}
        self._calibration: tuple[float, float] | None = None
        self._model_label: str = "unknown"
        self._drift_monitor = _FeatureDriftMonitor()
        self._last_drift_score: float | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_label(self) -> str:
        return self._model_label

    @model_label.setter
    def model_label(self, value: str) -> None:
        self._model_label = str(value)
        self._drift_monitor.configure(model_name=self._model_label)

    @property
    def last_drift_score(self) -> float | None:
        return self._last_drift_score

    def load_weights(self, artifact: str | Path | Mapping[str, object]) -> None:
        self._artifact = self._repository.load(artifact)
        self._model = self._artifact.build_model()
        metadata = dict(self._artifact.metadata)
        self._target_scale = float(metadata.get("target_scale", 1.0))
        self._feature_scalers = self._extract_scalers(metadata)
        self._calibration = self._extract_calibration(metadata)
        drift_config = metadata.get("drift_monitor")
        if isinstance(drift_config, Mapping):
            self._drift_monitor.configure(
                model_name=getattr(self, "_model_label", "unknown"),
                threshold=drift_config.get("threshold"),
                window=drift_config.get("window"),
                min_observations=drift_config.get("min_observations"),
                cooldown=drift_config.get("cooldown"),
                backend=drift_config.get("backend"),
            )
        else:
            self._drift_monitor.configure(model_name=getattr(self, "_model_label", "unknown"))
        self._last_drift_score = None
        if hasattr(self._model, "feature_scalers"):
            model_scalers = getattr(self._model, "feature_scalers")
            if not self._feature_scalers and isinstance(model_scalers, Mapping):
                self._feature_scalers = {
                    str(name): (float(pair[0]), float(pair[1]))
                    for name, pair in model_scalers.items()
                }
            elif self._feature_scalers:
                self._model.feature_scalers = dict(self._feature_scalers)

    def score(self, features: Mapping[str, float]) -> ModelScore:
        if self._model is None:
            raise RuntimeError("Model inference nie został załadowany")
        prepared = self._prepare_features(features)
        prediction = float(self._model.predict(prepared))
        if self._calibration is not None:
            slope, intercept = self._calibration
            prediction = prediction * slope + intercept
        probability = self._to_probability(prediction)
        drift_score = self._drift_monitor.observe(prepared, self._feature_scalers)
        if drift_score is not None:
            self._last_drift_score = drift_score
        return ModelScore(expected_return_bps=prediction, success_probability=probability)

    def explain(self, features: Mapping[str, float]) -> Mapping[str, float]:
        if self._model is None:
            raise RuntimeError("Model inference nie został załadowany")
        importances: MutableMapping[str, float] = {}
        prepared = self._prepare_features(features)
        baseline = float(self._model.predict(prepared))
        for name in self._model.feature_names:
            perturbed = dict(prepared)
            mean = self._feature_scalers.get(name, (0.0, 0.0))[0]
            perturbed[name] = mean
            delta = baseline - float(self._model.predict(perturbed))
            importances[name] = delta
        return dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))

    def _to_probability(self, value: float) -> float:
        scale = self._target_scale if self._target_scale > 0 else 1.0
        normalized = max(min(value / (scale * 6.0), 10.0), -10.0)
        prob = 1.0 / (1.0 + math.exp(-normalized))
        return max(0.0, min(1.0, prob))

    def _prepare_features(self, features: Mapping[str, float]) -> Mapping[str, float]:
        if self._model is None:
            return features
        prepared: MutableMapping[str, float] = {}
        provided = {str(key): value for key, value in features.items()}
        for name in getattr(self._model, "feature_names", ()):  # type: ignore[attr-defined]
            raw = provided.get(name)
            if raw is None:
                mean = self._feature_scalers.get(name, (0.0, 0.0))[0]
                prepared[name] = float(mean)
            else:
                try:
                    prepared[name] = float(raw)
                except (TypeError, ValueError):
                    prepared[name] = float(
                        self._feature_scalers.get(name, (0.0, 0.0))[0]
                    )
        for key, value in provided.items():
            if key not in prepared:
                try:
                    prepared[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return prepared

    def _extract_scalers(
        self, metadata: Mapping[str, object]
    ) -> dict[str, tuple[float, float]]:
        raw = metadata.get("feature_scalers")
        if not isinstance(raw, Mapping):
            return {}
        scalers: dict[str, tuple[float, float]] = {}
        for name, payload in raw.items():
            if not isinstance(payload, Mapping):
                continue
            mean = float(payload.get("mean", 0.0))
            stdev = float(payload.get("stdev", 0.0))
            scalers[str(name)] = (mean, stdev)
        return scalers

    def _extract_calibration(
        self, metadata: Mapping[str, object]
    ) -> tuple[float, float] | None:
        payload = metadata.get("calibration")
        if not isinstance(payload, Mapping):
            return None
        slope = float(payload.get("slope", 1.0))
        intercept = float(payload.get("intercept", 0.0))
        return slope, intercept


__all__ = ["DecisionModelInference", "ModelRepository"]
