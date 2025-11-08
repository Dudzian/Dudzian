"""Struktury danych opisujące możliwości wynikające z licencji offline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Mapping


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            try:
                return datetime.strptime(text, "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _mapping_proxy(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        return MappingProxyType({})
    cleaned = {str(key): value for key, value in data.items() if isinstance(key, str)}
    return MappingProxyType(cleaned)


def _bool_mapping(data: Mapping[str, Any] | None) -> Mapping[str, bool]:
    if not isinstance(data, Mapping):
        return MappingProxyType({})
    cleaned = {str(key): _coerce_bool(value) for key, value in data.items() if isinstance(key, str)}
    return MappingProxyType(cleaned)


@dataclass(frozen=True, slots=True)
class LicenseLimits:
    """Ograniczenia licencyjne."""

    max_paper_controllers: int | None = None
    max_live_controllers: int | None = None
    max_concurrent_bots: int | None = None
    max_alert_channels: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "LicenseLimits":
        if not isinstance(payload, Mapping):
            return cls()

        def _coerce_int(key: str) -> int | None:
            value = payload.get(key)
            if value in (None, ""):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return cls(
            max_paper_controllers=_coerce_int("max_paper_controllers"),
            max_live_controllers=_coerce_int("max_live_controllers"),
            max_concurrent_bots=_coerce_int("max_concurrent_bots"),
            max_alert_channels=_coerce_int("max_alert_channels"),
        )


@dataclass(frozen=True, slots=True)
class LicenseTrialInfo:
    enabled: bool
    expires_at: date | None

    def is_active(self, today: date) -> bool:
        if not self.enabled:
            return False
        if self.expires_at is None:
            return True
        return today <= self.expires_at


@dataclass(frozen=True, slots=True)
class LicenseMaintenanceWindow:
    until: date | None

    def is_active(self, today: date) -> bool:
        if self.until is None:
            return True
        return today <= self.until


@dataclass(frozen=True, slots=True)
class LicenseCapabilities:
    """Zestaw aktywnych modułów i limitów wynikających z licencji."""

    edition: str
    environments: frozenset[str]
    exchanges: Mapping[str, bool]
    strategies: Mapping[str, bool]
    runtime: Mapping[str, bool]
    modules: Mapping[str, bool]
    limits: LicenseLimits
    trial: LicenseTrialInfo
    maintenance: LicenseMaintenanceWindow
    issued_at: date | None
    maintenance_until: date | None
    effective_date: date
    seats: int | None
    holder: Mapping[str, Any]
    metadata: Mapping[str, Any]
    license_id: str | None
    hwid: str | None
    raw_payload: Mapping[str, Any]
    require_hardware_wallet_for_outgoing: bool

    def is_module_enabled(self, name: str) -> bool:
        return bool(self.modules.get(name))

    def is_runtime_enabled(self, name: str) -> bool:
        return bool(self.runtime.get(name))

    def is_exchange_enabled(self, exchange_id: str) -> bool:
        return bool(self.exchanges.get(exchange_id))

    def is_strategy_enabled(self, strategy_id: str) -> bool:
        return bool(self.strategies.get(strategy_id))

    def is_environment_allowed(self, environment: str) -> bool:
        return environment in self.environments

    def is_trial_active(self, today: date | None = None) -> bool:
        return self.trial.is_active(today or self.effective_date)

    def is_maintenance_active(self, today: date | None = None) -> bool:
        return self.maintenance.is_active(today or self.effective_date)


def build_capabilities_from_payload(
    payload: Mapping[str, Any], *, effective_date: date
) -> LicenseCapabilities:
    edition = str(payload.get("edition") or "community").strip().lower()
    environments = frozenset(
        str(value).strip().lower()
        for value in payload.get("environments", [])
        if isinstance(value, str) and value.strip()
    )
    exchanges = _bool_mapping(payload.get("exchanges"))
    strategies = _bool_mapping(payload.get("strategies"))
    runtime = _bool_mapping(payload.get("runtime"))
    modules = _bool_mapping(payload.get("modules"))
    limits = LicenseLimits.from_payload(_mapping_proxy(payload.get("limits")))

    trial_raw = payload.get("trial")
    trial_payload = trial_raw if isinstance(trial_raw, Mapping) else {}
    trial = LicenseTrialInfo(
        enabled=_coerce_bool(trial_payload.get("enabled")),
        expires_at=_parse_date(trial_payload.get("expires_at")),
    )

    maintenance_until = _parse_date(payload.get("maintenance_until"))
    maintenance = LicenseMaintenanceWindow(until=maintenance_until)

    issued_at = _parse_date(payload.get("issued_at"))
    seats_value = payload.get("seats")
    try:
        seats = int(seats_value) if seats_value is not None else None
    except (TypeError, ValueError):
        seats = None

    holder = _mapping_proxy(payload.get("holder"))
    metadata = _mapping_proxy(payload.get("metadata"))
    security_raw = payload.get("security")
    security_section = security_raw if isinstance(security_raw, Mapping) else {}
    require_hw_wallet = bool(
        security_section.get("require_hardware_wallet_for_outgoing")
        or security_section.get("require_hardware_wallet")
    )

    return LicenseCapabilities(
        edition=edition,
        environments=environments,
        exchanges=exchanges,
        strategies=strategies,
        runtime=runtime,
        modules=modules,
        limits=limits,
        trial=trial,
        maintenance=maintenance,
        issued_at=issued_at,
        maintenance_until=maintenance_until,
        effective_date=effective_date,
        seats=seats,
        holder=holder,
        metadata=metadata,
        license_id=str(payload.get("license_id") or payload.get("licenseId") or "") or None,
        hwid=str(payload.get("hwid") or "").strip() or None,
        raw_payload=_mapping_proxy(payload),
        require_hardware_wallet_for_outgoing=require_hw_wallet,
    )


__all__ = [
    "LicenseCapabilities",
    "LicenseLimits",
    "LicenseMaintenanceWindow",
    "LicenseTrialInfo",
    "build_capabilities_from_payload",
]
