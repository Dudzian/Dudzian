"""Katalog strategii i wspólne interfejsy fabryk."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.guards import get_capability_guard
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature

from .base import StrategyEngine
from .cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from .day_trading import DayTradingSettings, DayTradingStrategy
from .daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from .grid import GridTradingSettings, GridTradingStrategy
from .mean_reversion import MeanReversionSettings, MeanReversionStrategy
from .options import OptionsIncomeSettings, OptionsIncomeStrategy
from .scalping import ScalpingSettings, ScalpingStrategy
from .statistical_arbitrage import (
    StatisticalArbitrageSettings,
    StatisticalArbitrageStrategy,
)
from .volatility_target import VolatilityTargetSettings, VolatilityTargetStrategy


class StrategyFactory(Protocol):
    """Fabryka budująca `StrategyEngine` na podstawie parametrów."""

    def __call__(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        ...


def _normalize_non_empty_str(value: str, *, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _normalize_str_sequence(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not normalized:
        raise ValueError(f"{field_name} must define at least one entry")
    return normalized


def _normalize_optional_str_sequence(values: Any) -> tuple[str, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, str):
        candidates = (values,)
    elif isinstance(values, Iterable):
        candidates = tuple(values)
    else:
        raise TypeError("Expected iterable of strings or string")
    return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(candidate)
        except ValueError as exc:  # pragma: no cover - defensywne
            raise ValueError(f"Invalid ISO timestamp: {value!r}") from exc
    else:
        raise TypeError("Expected ISO8601 string or datetime instance")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def _format_iso(timestamp: datetime | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _match_fingerprint(candidate: str, pattern: str) -> bool:
    normalized_candidate = candidate.strip().lower()
    normalized_pattern = pattern.strip().lower()
    if not normalized_pattern:
        return False
    if normalized_pattern.endswith("*"):
        prefix = normalized_pattern[:-1]
        return normalized_candidate.startswith(prefix)
    return normalized_candidate == normalized_pattern


class StrategyPresetProfile(str, Enum):
    GRID = "grid"
    DCA = "dca"
    AI = "ai"
    HYBRID = "hybrid"

    @classmethod
    def from_value(cls, value: Any) -> "StrategyPresetProfile":
        if isinstance(value, StrategyPresetProfile):
            return value
        if value in (None, ""):
            return cls.HYBRID
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        aliases: dict[str, StrategyPresetProfile] = {
            "grid_trading": cls.GRID,
            "dollar_cost_average": cls.DCA,
            "dollar-cost-averaging": cls.DCA,
            "ai-ml": cls.AI,
            "machine_learning": cls.AI,
        }
        return aliases.get(normalized, cls.HYBRID)


class PresetLicenseState(str, Enum):
    UNLICENSED = "unlicensed"
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    SIGNATURE_INVALID = "signature_invalid"
    DEACTIVATED = "deactivated"


@dataclass(slots=True)
class PresetLicenseStatus:
    preset_id: str
    module_id: str | None
    status: PresetLicenseState
    fingerprint: str | None
    fingerprint_candidates: tuple[str, ...]
    fingerprint_verified: bool
    activated_at: datetime | None
    expires_at: datetime | None
    edition: str | None
    capability: str | None
    signature_verified: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "preset_id": self.preset_id,
            "module_id": self.module_id,
            "status": self.status.value,
            "fingerprint": self.fingerprint,
            "fingerprint_candidates": list(self.fingerprint_candidates),
            "fingerprint_verified": self.fingerprint_verified,
            "activated_at": _format_iso(self.activated_at),
            "expires_at": _format_iso(self.expires_at),
            "edition": self.edition,
            "capability": self.capability,
            "signature_verified": self.signature_verified,
            "issues": list(self.issues),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class StrategyPresetDescriptor:
    preset_id: str
    name: str
    profile: StrategyPresetProfile
    strategies: tuple[Mapping[str, Any], ...]
    required_parameters: Mapping[str, tuple[str, ...]]
    license_status: PresetLicenseStatus
    signature_verified: bool
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_strategies: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "preset_id": self.preset_id,
            "name": self.name,
            "profile": self.profile.value,
            "required_parameters": {key: list(values) for key, values in self.required_parameters.items()},
            "license": self.license_status.as_dict(),
            "signature_verified": self.signature_verified,
            "metadata": dict(self.metadata),
        }
        if include_strategies:
            payload["strategies"] = [dict(entry) for entry in self.strategies]
        if self.source_path is not None:
            payload["source_path"] = str(self.source_path)
        return payload


@dataclass(slots=True)
class StrategyDefinition:
    """Opis pojedynczej strategii dostępnej w katalogu."""

    name: str
    engine: str
    license_tier: str
    risk_classes: Sequence[str]
    required_data: Sequence[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    risk_profile: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_non_empty_str(self.name, field_name="name"))
        object.__setattr__(self, "engine", _normalize_non_empty_str(self.engine, field_name="engine"))
        object.__setattr__(self, "license_tier", _normalize_non_empty_str(self.license_tier, field_name="license_tier"))
        object.__setattr__(self, "risk_classes", _normalize_str_sequence(self.risk_classes, field_name="risk_classes"))
        object.__setattr__(self, "required_data", _normalize_str_sequence(self.required_data, field_name="required_data"))
        object.__setattr__(
            self,
            "tags",
            tuple(_normalize_optional_str_sequence(self.tags)),
        )
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", dict(self.metadata))
        else:
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class StrategyEngineSpec:
    """Opis silnika strategii wraz z wymaganą licencją."""

    key: str
    factory: StrategyFactory
    license_tier: str
    risk_classes: Sequence[str]
    required_data: Sequence[str]
    capability: str | None = None
    default_tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _normalize_non_empty_str(self.key, field_name="key"))
        object.__setattr__(self, "license_tier", _normalize_non_empty_str(self.license_tier, field_name="license_tier"))
        object.__setattr__(self, "risk_classes", _normalize_str_sequence(self.risk_classes, field_name="risk_classes"))
        object.__setattr__(self, "required_data", _normalize_str_sequence(self.required_data, field_name="required_data"))
        object.__setattr__(
            self,
            "default_tags",
            tuple(_normalize_optional_str_sequence(self.default_tags)),
        )

    def build(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        return self.factory(name=name, parameters=parameters, metadata=metadata)


def _ensure_capability_allowed(spec: StrategyEngineSpec, *, strategy_name: str | None = None) -> None:
    guard = get_capability_guard()
    if guard is None or not spec.capability:
        return
    message = (
        f"Strategia '{strategy_name}' wymaga aktywnej licencji {spec.capability}."
        if strategy_name
        else f"Silnik '{spec.key}' wymaga aktywnej licencji {spec.capability}."
    )
    guard.require_strategy(spec.capability, message=message)


def _is_capability_allowed(spec: StrategyEngineSpec) -> bool:
    guard = get_capability_guard()
    if guard is None or not spec.capability:
        return True
    try:
        return guard.capabilities.is_strategy_enabled(spec.capability)
    except AttributeError:
        return True


class StrategyCatalog:
    """Rejestr zarejestrowanych silników strategii i presetów marketplace."""

    def __init__(self, *, hwid_provider: HwIdProvider | None = None) -> None:
        self._registry: MutableMapping[str, StrategyEngineSpec] = {}
        self._presets: MutableMapping[str, StrategyPresetDescriptor] = {}
        self._license_overrides: MutableMapping[str, Mapping[str, Any]] = {}
        self._hwid_provider = hwid_provider

    def set_hwid_provider(self, provider: HwIdProvider | None) -> None:
        """Aktualizuje provider fingerprintu wykorzystywany przy walidacji licencji."""

        self._hwid_provider = provider

    # ------------------------------------------------------------------
    # Wewnętrzne helpery presetów
    # ------------------------------------------------------------------

    def _resolve_hwid_provider(self, provider: HwIdProvider | None) -> HwIdProvider | None:
        return provider or self._hwid_provider

    @staticmethod
    def _collect_fingerprint_candidates(license_payload: Mapping[str, Any]) -> tuple[str, ...]:
        candidates: list[str] = []
        fingerprint_value = license_payload.get("fingerprint")
        if isinstance(fingerprint_value, str) and fingerprint_value.strip():
            candidates.append(fingerprint_value.strip())
        alternate = license_payload.get("fingerprints") or license_payload.get("allowed_fingerprints")
        if isinstance(alternate, Iterable) and not isinstance(alternate, (str, bytes)):
            for item in alternate:
                text = str(item).strip()
                if text:
                    candidates.append(text)
        return tuple(dict.fromkeys(candidates))

    def _compute_license_status(
        self,
        preset_id: str,
        *,
        metadata: Mapping[str, Any],
        license_payload: Mapping[str, Any] | None,
        signature_verified: bool,
        hwid_provider: HwIdProvider | None,
        additional_issues: Sequence[str] = (),
    ) -> PresetLicenseStatus:
        issues = list(additional_issues)
        module_id: str | None = None
        edition: str | None = None
        capability: str | None = None
        activated_at: datetime | None = None
        expires_at: datetime | None = None
        fingerprint_candidates: tuple[str, ...] = ()
        fingerprint_verified = False
        status = PresetLicenseState.UNLICENSED
        payload: Mapping[str, Any] = license_payload or {}

        if payload:
            module_id_value = payload.get("module_id") or metadata.get("module_id") or preset_id
            module_id = str(module_id_value).strip() or preset_id

            edition_value = payload.get("edition") or metadata.get("license_tier")
            if isinstance(edition_value, str) and edition_value.strip():
                edition = edition_value.strip()

            capability_value = payload.get("capability") or metadata.get("capability")
            if isinstance(capability_value, str) and capability_value.strip():
                capability = capability_value.strip()

            activated_at = _parse_iso_datetime(payload.get("activated_at"))
            expires_at = _parse_iso_datetime(payload.get("expires_at"))

            fingerprint_candidates = self._collect_fingerprint_candidates(payload)
            provider = self._resolve_hwid_provider(hwid_provider)
            hwid_value: str | None = None
            if fingerprint_candidates:
                if provider is None:
                    issues.append("fingerprint-provider-missing")
                else:
                    try:
                        hwid_value = provider.read()
                    except HwIdProviderError as exc:  # pragma: no cover - zależne od środowiska
                        issues.append(f"hwid-error:{exc}")
            else:
                provider = self._resolve_hwid_provider(hwid_provider)
                if provider is not None:
                    try:
                        hwid_value = provider.read()
                    except HwIdProviderError as exc:  # pragma: no cover
                        issues.append(f"hwid-error:{exc}")

            if fingerprint_candidates:
                if hwid_value is None:
                    issues.append("fingerprint-unavailable")
                    status = PresetLicenseState.PENDING
                else:
                    fingerprint_verified = any(
                        _match_fingerprint(hwid_value, candidate) for candidate in fingerprint_candidates
                    )
                    status = (
                        PresetLicenseState.ACTIVE
                        if fingerprint_verified
                        else PresetLicenseState.FINGERPRINT_MISMATCH
                    )
            else:
                status = PresetLicenseState.PENDING
                fingerprint_verified = hwid_value is None or bool(hwid_value)

            if payload.get("disabled") or payload.get("revoked"):
                status = PresetLicenseState.DEACTIVATED

            if expires_at is not None and expires_at <= datetime.now(timezone.utc):
                status = PresetLicenseState.EXPIRED
        else:
            status = PresetLicenseState.UNLICENSED

        if not signature_verified:
            issues.append("preset-signature-unverified")
            if payload and payload.get("signature_required"):
                status = PresetLicenseState.SIGNATURE_INVALID
            elif status == PresetLicenseState.ACTIVE:
                status = PresetLicenseState.PENDING

        fingerprint_primary = fingerprint_candidates[0] if fingerprint_candidates else None

        normalized_metadata = dict(payload) if isinstance(payload, Mapping) and payload else {}

        return PresetLicenseStatus(
            preset_id=preset_id,
            module_id=module_id,
            status=status,
            fingerprint=fingerprint_primary,
            fingerprint_candidates=fingerprint_candidates,
            fingerprint_verified=fingerprint_verified,
            activated_at=activated_at,
            expires_at=expires_at,
            edition=edition,
            capability=capability,
            signature_verified=signature_verified,
            issues=tuple(dict.fromkeys(issues)),
            metadata=normalized_metadata,
        )

    def _descriptor_with_overrides(
        self,
        descriptor: StrategyPresetDescriptor,
        *,
        hwid_provider: HwIdProvider | None,
        additional_issues: Sequence[str] | None = None,
    ) -> StrategyPresetDescriptor:
        metadata = dict(descriptor.metadata)
        license_payload: Mapping[str, Any] | None = None
        raw_license = metadata.get("license")
        if isinstance(raw_license, Mapping):
            license_payload = dict(raw_license)

        override = self._license_overrides.get(descriptor.preset_id)
        if override:
            combined = dict(license_payload or {})
            combined.update(dict(override))
            license_payload = combined
            metadata["license"] = combined
        elif license_payload is None:
            metadata.pop("license", None)

        status = self._compute_license_status(
            descriptor.preset_id,
            metadata=metadata,
            license_payload=license_payload,
            signature_verified=descriptor.signature_verified,
            hwid_provider=hwid_provider,
            additional_issues=additional_issues or descriptor.license_status.issues,
        )
        return replace(descriptor, metadata=metadata, license_status=status)

    def _parse_preset_strategies(
        self, payload: Mapping[str, Any]
    ) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, tuple[str, ...]], list[str]]:
        entries = payload.get("strategies")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("Preset must define strategies as an array of mappings")

        strategies: list[Mapping[str, Any]] = []
        required_parameters: dict[str, tuple[str, ...]] = {}
        issues: list[str] = []

        for index, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, Mapping):
                issues.append(f"strategy-entry-{index}-invalid")
                continue

            engine = str(raw_entry.get("engine") or "").strip()
            if not engine:
                issues.append(f"strategy-entry-{index}-missing-engine")
                continue

            name = str(raw_entry.get("name") or engine).strip() or engine
            parameters_raw = raw_entry.get("parameters") or {}
            if isinstance(parameters_raw, Mapping):
                parameters = {str(key): parameters_raw[key] for key in parameters_raw.keys()}
                parameter_keys = tuple(sorted(parameters.keys()))
            else:
                parameters = {}
                parameter_keys = ()

            required_parameters[name] = parameter_keys

            entry_payload: dict[str, Any] = {
                "name": name,
                "engine": engine,
                "license_tier": str(raw_entry.get("license_tier") or "").strip(),
                "risk_classes": list(_normalize_optional_str_sequence(raw_entry.get("risk_classes"))),
                "required_data": list(_normalize_optional_str_sequence(raw_entry.get("required_data"))),
                "parameters": parameters,
                "tags": list(_normalize_optional_str_sequence(raw_entry.get("tags"))),
            }

            risk_profile = raw_entry.get("risk_profile")
            if risk_profile not in (None, ""):
                entry_payload["risk_profile"] = str(risk_profile)

            metadata_payload = raw_entry.get("metadata")
            if isinstance(metadata_payload, Mapping):
                entry_payload["metadata"] = dict(metadata_payload)

            strategies.append(entry_payload)

        if not strategies:
            issues.append("no-strategies-defined")

        return tuple(strategies), required_parameters, issues

    def _build_preset_descriptor(
        self,
        preset_payload: Mapping[str, Any],
        *,
        signature_verified: bool,
        source_path: Path | None,
        hwid_provider: HwIdProvider | None,
        additional_issues: Sequence[str] = (),
    ) -> StrategyPresetDescriptor:
        if not isinstance(preset_payload, Mapping):
            raise TypeError("Preset payload must be a mapping")

        metadata_payload = preset_payload.get("metadata") or {}
        metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}

        raw_name = preset_payload.get("name") or metadata.get("name") or metadata.get("id")
        if raw_name is None:
            raise ValueError("Preset must define a name")
        name = _normalize_non_empty_str(str(raw_name), field_name="preset.name")

        raw_preset_id = metadata.get("id") or metadata.get("preset_id") or name
        preset_id = _normalize_non_empty_str(str(raw_preset_id), field_name="preset_id")

        profile = StrategyPresetProfile.from_value(metadata.get("profile") or metadata.get("preset_profile"))

        strategies, required_parameters, strategy_issues = self._parse_preset_strategies(preset_payload)

        issues = list(additional_issues)
        issues.extend(strategy_issues)

        license_payload = metadata.get("license") if isinstance(metadata.get("license"), Mapping) else None

        status = self._compute_license_status(
            preset_id,
            metadata=metadata,
            license_payload=license_payload,
            signature_verified=signature_verified,
            hwid_provider=hwid_provider,
            additional_issues=issues,
        )

        if license_payload is not None:
            metadata["license"] = dict(license_payload)
        elif "license" in metadata:
            metadata.pop("license")

        return StrategyPresetDescriptor(
            preset_id=preset_id,
            name=name,
            profile=profile,
            strategies=strategies,
            required_parameters=required_parameters,
            license_status=status,
            signature_verified=signature_verified,
            source_path=source_path,
            metadata=metadata,
        )

    def _load_preset_file(
        self,
        path: Path,
        *,
        signing_keys: Mapping[str, bytes] | None,
        hwid_provider: HwIdProvider | None,
    ) -> StrategyPresetDescriptor:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise

        try:
            document = json.loads(raw_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - walidowane w testach integracyjnych
            raise ValueError(f"Preset file {path} contains invalid JSON: {exc}") from exc

        signature_doc: Mapping[str, Any] | None = None
        if isinstance(document, Mapping) and "preset" in document:
            preset_payload = document.get("preset")  # type: ignore[assignment]
            signature_candidate = document.get("signature")
            if isinstance(signature_candidate, Mapping):
                signature_doc = signature_candidate
        elif isinstance(document, Mapping):
            preset_payload = document
        else:
            raise ValueError(f"Preset file {path} must contain JSON object")

        if not isinstance(preset_payload, Mapping):
            raise ValueError(f"Preset file {path} must define mapping payload")

        signature_verified = False
        issues: list[str] = []
        if signature_doc is not None:
            key_id = str(signature_doc.get("key_id") or "").strip()
            algorithm = str(signature_doc.get("algorithm") or "HMAC-SHA256")
            key_bytes = signing_keys.get(key_id) if signing_keys else None
            if key_bytes is None:
                issues.append("missing-signing-key")
            else:
                try:
                    signature_verified = verify_hmac_signature(
                        preset_payload,
                        signature_doc,
                        key=key_bytes,
                        algorithm=algorithm,
                    )
                    if not signature_verified:
                        issues.append("signature-mismatch")
                except Exception as exc:  # pragma: no cover - defensywne logowanie
                    LOGGER.debug("Signature verification error for preset %s", path, exc_info=True)
                    issues.append(f"signature-error:{exc}")

        descriptor = self._build_preset_descriptor(
            preset_payload,
            signature_verified=signature_verified,
            source_path=path,
            hwid_provider=hwid_provider,
            additional_issues=issues,
        )

        self._presets[descriptor.preset_id] = replace(
            descriptor,
            metadata=dict(descriptor.metadata),
        )

        return self._descriptor_with_overrides(
            descriptor,
            hwid_provider=hwid_provider,
            additional_issues=descriptor.license_status.issues,
        )

    # ------------------------------------------------------------------
    # API publiczne presetów
    # ------------------------------------------------------------------

    def load_presets_from_directory(
        self,
        directory: str | Path,
        *,
        signing_keys: Mapping[str, bytes] | None = None,
        hwid_provider: HwIdProvider | None = None,
    ) -> tuple[StrategyPresetDescriptor, ...]:
        base_path = Path(directory)
        if not base_path.exists():
            raise FileNotFoundError(f"Preset directory does not exist: {directory}")

        descriptors: list[StrategyPresetDescriptor] = []
        for candidate in sorted(base_path.glob("*.json")):
            descriptor = self._load_preset_file(
                candidate,
                signing_keys=signing_keys,
                hwid_provider=hwid_provider,
            )
            descriptors.append(descriptor)
        return tuple(descriptors)

    def preset(
        self,
        preset_id: str,
        *,
        hwid_provider: HwIdProvider | None = None,
    ) -> StrategyPresetDescriptor:
        if preset_id not in self._presets:
            raise KeyError(f"Nie znaleziono presetu: {preset_id}")
        base = self._presets[preset_id]
        return self._descriptor_with_overrides(
            base,
            hwid_provider=hwid_provider,
            additional_issues=base.license_status.issues,
        )

    def list_presets(
        self,
        *,
        hwid_provider: HwIdProvider | None = None,
    ) -> tuple[StrategyPresetDescriptor, ...]:
        provider = self._resolve_hwid_provider(hwid_provider)
        result: list[StrategyPresetDescriptor] = []
        for preset_id in sorted(self._presets):
            base = self._presets[preset_id]
            result.append(
                self._descriptor_with_overrides(
                    base,
                    hwid_provider=provider,
                    additional_issues=base.license_status.issues,
                )
            )
        return tuple(result)

    def describe_presets(
        self,
        *,
        profile: StrategyPresetProfile | str | None = None,
        include_strategies: bool = True,
        hwid_provider: HwIdProvider | None = None,
    ) -> Sequence[Mapping[str, object]]:
        resolved_profile: StrategyPresetProfile | None
        if profile is None:
            resolved_profile = None
        else:
            resolved_profile = StrategyPresetProfile.from_value(profile)

        summaries: list[Mapping[str, object]] = []
        for descriptor in self.list_presets(hwid_provider=hwid_provider):
            if resolved_profile and descriptor.profile != resolved_profile:
                continue
            summaries.append(descriptor.as_dict(include_strategies=include_strategies))
        return summaries

    def install_license_override(
        self,
        preset_id: str,
        override: Mapping[str, Any],
        *,
        hwid_provider: HwIdProvider | None = None,
    ) -> StrategyPresetDescriptor:
        if preset_id not in self._presets:
            raise KeyError(f"Nie znaleziono presetu: {preset_id}")
        self._license_overrides[preset_id] = dict(override)
        return self.preset(preset_id, hwid_provider=hwid_provider)

    def clear_license_override(
        self,
        preset_id: str,
        *,
        hwid_provider: HwIdProvider | None = None,
    ) -> StrategyPresetDescriptor:
        if preset_id not in self._presets:
            raise KeyError(f"Nie znaleziono presetu: {preset_id}")
        self._license_overrides.pop(preset_id, None)
        return self.preset(preset_id, hwid_provider=hwid_provider)

    def register(self, spec: StrategyEngineSpec) -> None:
        key = spec.key.lower()
        self._registry[key] = spec

    def get(self, engine: str) -> StrategyEngineSpec:
        key = engine.lower()
        if key not in self._registry:
            raise KeyError(f"Nie znaleziono silnika strategii: {engine}")
        return self._registry[key]

    def create(self, definition: StrategyDefinition) -> StrategyEngine:
        spec = self.get(definition.engine)
        _ensure_capability_allowed(spec, strategy_name=definition.name)
        if definition.license_tier != spec.license_tier:
            raise ValueError(
                f"Strategy '{definition.name}' requires license tier '{definition.license_tier}' "
                f"but engine '{spec.key}' is registered for '{spec.license_tier}'"
            )
        tags = tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))
        risk_classes = tuple(dict.fromkeys((*spec.risk_classes, *definition.risk_classes)))
        required_data = tuple(dict.fromkeys((*spec.required_data, *definition.required_data)))
        metadata = dict(definition.metadata)
        if tags and "tags" not in metadata:
            metadata["tags"] = tags
        if spec.capability and "capability" not in metadata:
            metadata["capability"] = spec.capability
        metadata.setdefault("license_tier", spec.license_tier)
        metadata.setdefault("risk_classes", risk_classes)
        metadata.setdefault("required_data", required_data)
        engine = spec.build(
            name=definition.name,
            parameters=definition.parameters,
            metadata=metadata,
        )
        try:
            setattr(engine, "metadata", dict(metadata))
        except Exception:
            # Nie wszystkie strategie muszą wspierać przypięcie metadanych.
            pass
        return engine

    def merge_tags(self, definition: StrategyDefinition) -> tuple[str, ...]:
        """Zwraca połączone tagi katalogu oraz definicji strategii."""

        spec = self.get(definition.engine)
        return tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))

    def describe_engines(self) -> Sequence[Mapping[str, object]]:
        """Buduje listę zarejestrowanych silników wraz z metadanymi."""

        summary: list[Mapping[str, object]] = []
        for key in sorted(self._registry):
            spec = self._registry[key]
            if not _is_capability_allowed(spec):
                continue
            payload: dict[str, object] = {
                "engine": spec.key,
                "capability": spec.capability,
                "default_tags": list(spec.default_tags),
                "license_tier": spec.license_tier,
                "risk_classes": list(spec.risk_classes),
                "required_data": list(spec.required_data),
            }
            summary.append(payload)
        return summary

    def describe_definitions(
        self,
        definitions: Mapping[str, StrategyDefinition],
        *,
        include_metadata: bool = False,
    ) -> Sequence[Mapping[str, object]]:
        """Tworzy opis strategii na podstawie przekazanych definicji."""

        summary: list[Mapping[str, object]] = []
        for name in sorted(definitions):
            definition = definitions[name]
            try:
                merged_tags = self.merge_tags(definition)
            except KeyError:
                merged_tags = tuple(dict.fromkeys(definition.tags))
            payload: dict[str, object] = {
                "name": name,
                "engine": definition.engine,
                "tags": list(merged_tags),
                "license_tier": definition.license_tier,
                "risk_classes": list(definition.risk_classes),
                "required_data": list(definition.required_data),
            }
            if definition.risk_profile:
                payload["risk_profile"] = definition.risk_profile
            try:
                spec = self.get(definition.engine)
                if not _is_capability_allowed(spec):
                    continue
                if spec.capability:
                    payload["capability"] = spec.capability
                payload["license_tier"] = spec.license_tier
                payload["risk_classes"] = list(
                    dict.fromkeys((*spec.risk_classes, *definition.risk_classes))
                )
                payload["required_data"] = list(
                    dict.fromkeys((*spec.required_data, *definition.required_data))
                )
            except KeyError:
                pass
            if "capability" not in payload and "capability" in definition.metadata:
                payload["capability"] = definition.metadata["capability"]
            if include_metadata and definition.metadata:
                payload["metadata"] = dict(definition.metadata)
            if definition.parameters:
                payload["parameters"] = dict(definition.parameters)
            summary.append(payload)
        return summary


def _build_daily_trend_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = DailyTrendMomentumSettings(
        fast_ma=int(parameters.get("fast_ma", 20)),
        slow_ma=int(parameters.get("slow_ma", 100)),
        breakout_lookback=int(parameters.get("breakout_lookback", 55)),
        momentum_window=int(parameters.get("momentum_window", 20)),
        atr_window=int(parameters.get("atr_window", 14)),
        atr_multiplier=float(parameters.get("atr_multiplier", 2.0)),
        min_trend_strength=float(parameters.get("min_trend_strength", 0.005)),
        min_momentum=float(parameters.get("min_momentum", 0.0)),
    )
    return DailyTrendMomentumStrategy(settings)


def _build_mean_reversion_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = MeanReversionSettings(
        lookback=int(parameters.get("lookback", 96)),
        entry_zscore=float(parameters.get("entry_zscore", 2.0)),
        exit_zscore=float(parameters.get("exit_zscore", 0.5)),
        max_holding_period=int(parameters.get("max_holding_period", 12)),
        volatility_cap=float(parameters.get("volatility_cap", 0.04)),
        min_volume_usd=float(parameters.get("min_volume_usd", 1000.0)),
    )
    return MeanReversionStrategy(settings)


def _build_grid_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = GridTradingSettings(
        grid_size=int(parameters.get("grid_size", 5)),
        grid_spacing=float(parameters.get("grid_spacing", 0.004)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.001)),
        max_inventory=float(parameters.get("max_inventory", 1.0)),
    )
    return GridTradingStrategy(settings)


def _build_volatility_target_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = VolatilityTargetSettings(
        target_volatility=float(parameters.get("target_volatility", 0.1)),
        lookback=int(parameters.get("lookback", 60)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.1)),
        min_allocation=float(parameters.get("min_allocation", 0.1)),
        max_allocation=float(parameters.get("max_allocation", 1.0)),
        floor_volatility=float(parameters.get("floor_volatility", 0.02)),
    )
    return VolatilityTargetStrategy(settings)


def _build_cross_exchange_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = CrossExchangeArbitrageSettings(
        primary_exchange=str(parameters.get("primary_exchange", "")),
        secondary_exchange=str(parameters.get("secondary_exchange", "")),
        spread_entry=float(parameters.get("spread_entry", 0.0015)),
        spread_exit=float(parameters.get("spread_exit", 0.0005)),
        max_notional=float(parameters.get("max_notional", 50_000.0)),
        max_open_seconds=int(parameters.get("max_open_seconds", 120)),
    )
    return CrossExchangeArbitrageStrategy(settings)


def _build_scalping_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = ScalpingSettings.from_parameters(parameters)
    return ScalpingStrategy(settings)


def _build_options_income_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = OptionsIncomeSettings.from_parameters(parameters)
    return OptionsIncomeStrategy(settings)


def _build_statistical_arbitrage_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = StatisticalArbitrageSettings.from_parameters(parameters)
    return StatisticalArbitrageStrategy(settings)


def _build_day_trading_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = DayTradingSettings.from_parameters(parameters)
    return DayTradingStrategy(settings)


def build_default_catalog() -> StrategyCatalog:
    catalog = StrategyCatalog()
    catalog.register(
        StrategyEngineSpec(
            key="daily_trend_momentum",
            factory=_build_daily_trend_strategy,
            license_tier="standard",
            risk_classes=("directional", "momentum"),
            required_data=("ohlcv", "technical_indicators"),
            capability="trend_d1",
            default_tags=("trend", "momentum"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="mean_reversion",
            factory=_build_mean_reversion_strategy,
            license_tier="professional",
            risk_classes=("statistical", "mean_reversion"),
            required_data=("ohlcv", "spread_history"),
            capability="mean_reversion",
            default_tags=("mean_reversion", "stat_arbitrage"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="grid_trading",
            factory=_build_grid_strategy,
            license_tier="professional",
            risk_classes=("market_making",),
            required_data=("order_book", "ohlcv"),
            capability="grid_trading",
            default_tags=("grid", "market_making"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="volatility_target",
            factory=_build_volatility_target_strategy,
            license_tier="enterprise",
            risk_classes=("risk_control", "volatility"),
            required_data=("ohlcv", "realized_volatility"),
            capability="volatility_target",
            default_tags=("volatility", "risk"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="cross_exchange_arbitrage",
            factory=_build_cross_exchange_strategy,
            license_tier="enterprise",
            risk_classes=("arbitrage", "liquidity"),
            required_data=("order_book", "latency_monitoring"),
            capability="cross_exchange",
            default_tags=("arbitrage", "liquidity"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="scalping",
            factory=_build_scalping_strategy,
            license_tier="professional",
            risk_classes=("intraday", "scalping"),
            required_data=("ohlcv", "order_book"),
            capability="scalping",
            default_tags=("intraday", "scalping"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="options_income",
            factory=_build_options_income_strategy,
            license_tier="enterprise",
            risk_classes=("derivatives", "income"),
            required_data=("options_chain", "greeks", "ohlcv"),
            capability="options_income",
            default_tags=("options", "income"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="statistical_arbitrage",
            factory=_build_statistical_arbitrage_strategy,
            license_tier="professional",
            risk_classes=("statistical", "mean_reversion"),
            required_data=("ohlcv", "spread_history"),
            capability="stat_arbitrage",
            default_tags=("stat_arbitrage", "pairs_trading"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="day_trading",
            factory=_build_day_trading_strategy,
            license_tier="standard",
            risk_classes=("intraday", "momentum"),
            required_data=("ohlcv", "technical_indicators"),
            capability="day_trading",
            default_tags=("intraday", "momentum"),
        )
    )
    return catalog


DEFAULT_STRATEGY_CATALOG = build_default_catalog()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StrategyPresetWizard:
    """Buduje i podpisuje presety strategii na podstawie katalogu."""

    def __init__(self, catalog: StrategyCatalog | None = None) -> None:
        self._catalog = catalog or DEFAULT_STRATEGY_CATALOG

    def build_preset(
        self,
        name: str,
        entries: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("Preset name is required")
        if not entries:
            raise ValueError("At least one strategy entry is required")

        strategies: list[dict[str, Any]] = []
        for entry in entries:
            strategies.append(self._build_entry(entry))

        payload: dict[str, Any] = {
            "name": name,
            "created_at": _now_iso(),
            "strategies": strategies,
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    def _build_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        engine_name = str(entry.get("engine") or "").strip()
        if not engine_name:
            raise ValueError("Preset entry must define an engine")
        spec = self._catalog.get(engine_name)

        name = str(entry.get("name") or spec.key)
        _ensure_capability_allowed(spec, strategy_name=name)
        parameters = dict(entry.get("parameters") or {})
        risk_profile = entry.get("risk_profile")
        user_tags = tuple(entry.get("tags") or ())
        merged_tags = tuple(dict.fromkeys((*spec.default_tags, *user_tags)))
        license_tier = str(entry.get("license_tier") or spec.license_tier).strip()
        if license_tier != spec.license_tier:
            raise ValueError(
                f"Preset entry '{name}' declares license tier '{license_tier}' incompatible with engine '{spec.key}'"
            )
        risk_classes = tuple(
            dict.fromkeys((*spec.risk_classes, *_normalize_optional_str_sequence(entry.get("risk_classes"))))
        )
        required_data = tuple(
            dict.fromkeys((*spec.required_data, *_normalize_optional_str_sequence(entry.get("required_data"))))
        )
        metadata = dict(entry.get("metadata") or {})
        metadata.setdefault("license_tier", spec.license_tier)
        metadata.setdefault("risk_classes", risk_classes)
        metadata.setdefault("required_data", required_data)
        if merged_tags and "tags" not in metadata:
            metadata["tags"] = merged_tags
        if spec.capability:
            metadata.setdefault("capability", spec.capability)

        payload: dict[str, Any] = {
            "name": name,
            "engine": spec.key,
            "parameters": parameters,
            "tags": list(merged_tags),
            "license_tier": spec.license_tier,
            "risk_classes": list(risk_classes),
            "required_data": list(required_data),
        }
        if risk_profile:
            payload["risk_profile"] = str(risk_profile)
        if spec.capability:
            payload["capability"] = spec.capability
        if metadata:
            payload["metadata"] = metadata
        return payload

    def build_document(
        self,
        preset: Mapping[str, Any],
        *,
        signing_key: bytes,
        key_id: str | None = None,
        algorithm: str = "HMAC-SHA256",
    ) -> Mapping[str, Any]:
        if not isinstance(signing_key, (bytes, bytearray)):
            raise TypeError("signing_key must be raw bytes")
        preset_payload = dict(preset)
        signature = build_hmac_signature(preset_payload, key=bytes(signing_key), key_id=key_id, algorithm=algorithm)
        return {"preset": preset_payload, "signature": signature}

    def export_signed(
        self,
        preset: Mapping[str, Any],
        *,
        signing_key: bytes,
        path: str | Path,
        key_id: str | None = None,
        algorithm: str = "HMAC-SHA256",
    ) -> Path:
        document = self.build_document(preset, signing_key=signing_key, key_id=key_id, algorithm=algorithm)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


__all__ = [
    "StrategyCatalog",
    "StrategyDefinition",
    "StrategyEngineSpec",
    "StrategyFactory",
    "DEFAULT_STRATEGY_CATALOG",
    "StrategyPresetWizard",
    "build_default_catalog",
    "StrategyPresetDescriptor",
    "StrategyPresetProfile",
    "PresetLicenseStatus",
    "PresetLicenseState",
]
LOGGER = logging.getLogger(__name__)
