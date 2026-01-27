"""Pomocnicze metadane punktów wejścia runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence, Tuple

_CANONICAL_MODULE = "bot_core.runtime.metadata"
if __name__ != _CANONICAL_MODULE:
    sys.modules[_CANONICAL_MODULE] = sys.modules[__name__]

from bot_core.risk.settings import RiskManagerSettings, derive_risk_manager_settings
from bot_core.runtime.paths import resolve_core_config_path

try:  # pragma: no cover - opcjonalny import w środowiskach bez pełnego runtime
    from bot_core.runtime.bootstrap import resolve_runtime_entrypoint as _resolve_runtime_entrypoint
except Exception:  # pragma: no cover - brak modułów bootstrapu w środowisku
    _resolve_runtime_entrypoint = None  # type: ignore

try:  # pragma: no cover - środowiska bez pełnego loadera configu
    from bot_core.config.loader import load_core_config as _load_core_config
except Exception:  # pragma: no cover - brak modułu loadera
    _load_core_config = None  # type: ignore


_RUNTIME_RESOLVER_ATTEMPTED = _resolve_runtime_entrypoint is not None
_TYPED_LOADER_ATTEMPTED = _load_core_config is not None


def _require_yaml() -> ModuleType:
    try:
        return importlib.import_module("yaml")
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Brak opcjonalnej zależności 'PyYAML'. Zainstaluj pakiet PyYAML, "
            "aby korzystać z konfiguracji YAML."
        ) from exc


@dataclass(frozen=True, slots=True)
class RuntimeEntrypointMetadata:
    """Wybrane, lekkie metadane udostępniane aplikacjom desktopowym."""

    environment: str
    risk_profile: str
    controller: str | None
    strategy: str | None
    tags: tuple[str, ...]
    compliance_live_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Zwraca metadane w postaci słownika przyjaznego serializacji."""

        return {
            "environment": self.environment,
            "risk_profile": self.risk_profile,
            "controller": self.controller,
            "strategy": self.strategy,
            "tags": list(self.tags),
            "compliance_live_allowed": self.compliance_live_allowed,
        }


def _normalize_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Mapping):
        # traktujemy mapowania jako niepoprawne listy – lepiej zwrócić pustą krotkę
        return ()
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        return ()
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _extract_compliance_mapping(payload: Any) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload
    attrs = {}
    for key in ("live_allowed", "risk_profiles", "signoffs", "signatures", "signed", "require_signoff"):
        if hasattr(payload, key):
            attrs[key] = getattr(payload, key)
    return attrs or None


def _resolve_compliance_live_allowed(entrypoint_decl: Any, *, logger: logging.Logger | None = None) -> bool:
    if isinstance(entrypoint_decl, Mapping):
        compliance_source = entrypoint_decl.get("compliance")
        risk_profile = entrypoint_decl.get("risk_profile")
    else:
        compliance_source = getattr(entrypoint_decl, "compliance", None)
        risk_profile = getattr(entrypoint_decl, "risk_profile", None)

    compliance = _extract_compliance_mapping(compliance_source)
    if not compliance:
        return False
    if not bool(compliance.get("live_allowed")):
        return False

    reasons: list[str] = []

    signed_flag = bool(compliance.get("signed"))
    if not signed_flag:
        reasons.append("signed flag is not confirmed")

    allowed_profiles = _normalize_sequence(compliance.get("risk_profiles"))
    if allowed_profiles and risk_profile not in allowed_profiles:
        reasons.append("risk profile not permitted for live mode")

    signoff_entries = _normalize_sequence(
        compliance.get("signoffs") or compliance.get("signatures")
    )
    require_signoff = compliance.get("require_signoff")
    if require_signoff is None:
        require_signoff = True
    if bool(require_signoff) and not signoff_entries:
        reasons.append("missing compliance sign-off entries")

    if reasons:
        if logger is not None:
            logger.debug(
                "Runtime compliance guard disabled live trading: %s", ", ".join(reasons)
            )
        return False
    return True


def load_runtime_entrypoint_metadata(
    entrypoint: str,
    *,
    config_path: str | Path | None = None,
    bootstrap: bool = False,
    logger: logging.Logger | None = None,
) -> RuntimeEntrypointMetadata | None:
    """Pobiera metadane deklaratywnego punktu wejścia runtime.

    Funkcja zwraca ``None`` w sytuacjach, gdy runtime nie jest dostępny
    (brak modułów ``bot_core``) albo wpis nie istnieje. Dzięki temu UI może
    działać w trybie degrade-friendly bez podwójnej logiki fallback.
    """

    _ensure_runtime_resolver()
    resolver = _resolve_runtime_entrypoint

    effective_config_path: Path | None
    if config_path is None:
        try:
            effective_config_path = resolve_core_config_path()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiskowa
            if logger is not None:
                logger.debug(
                    "Nie udało się ustalić ścieżki configu runtime dla %s: %r",
                    entrypoint,
                    exc,
                )
            return None
    else:
        effective_config_path = Path(config_path)

    if resolver is None:  # pragma: no cover - środowiska bez runtime
        raw_entry = _load_entrypoint_from_raw_config(
            effective_config_path, entrypoint, logger=logger
        )
        if raw_entry is None:
            if logger is not None:
                logger.debug(
                    "Runtime entrypoint %s pominięty – brak deklaracji w konfiguracji",
                    entrypoint,
                )
            return None
        environment = raw_entry.get("environment")
        risk_profile = raw_entry.get("risk_profile")
        if not isinstance(environment, str) or not environment:
            return None
        if not isinstance(risk_profile, str) or not risk_profile:
            return None
        controller = raw_entry.get("controller")
        strategy = raw_entry.get("strategy")
        tags = _normalize_sequence(raw_entry.get("tags"))
        return RuntimeEntrypointMetadata(
            environment=environment,
            risk_profile=risk_profile,
            controller=str(controller) if isinstance(controller, str) and controller.strip() else None,
            strategy=str(strategy) if isinstance(strategy, str) and strategy.strip() else None,
            tags=tuple(tags),
            compliance_live_allowed=_resolve_compliance_live_allowed(
                raw_entry, logger=logger
            ),
        )

    try:
        entrypoint_decl, _ = resolver(
            entrypoint,
            config_path=effective_config_path,
            bootstrap=bootstrap,
        )
    except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
        if logger is not None:
            logger.debug(
                "Nie udało się pobrać metadanych runtime %s: %r",
                entrypoint,
                exc,
            )
        return None

    tags: Sequence[str] | Iterable[str] = getattr(entrypoint_decl, "tags", ())
    compliance_allowed = _resolve_compliance_live_allowed(
        entrypoint_decl, logger=logger
    )
    if not compliance_allowed:
        raw_entry = _load_entrypoint_from_raw_config(
            effective_config_path, entrypoint, logger=logger
        )
        if raw_entry is not None:
            compliance_allowed = _resolve_compliance_live_allowed(
                raw_entry, logger=logger
            )
    return RuntimeEntrypointMetadata(
        environment=getattr(entrypoint_decl, "environment"),
        risk_profile=getattr(entrypoint_decl, "risk_profile"),
        controller=getattr(entrypoint_decl, "controller", None),
        strategy=getattr(entrypoint_decl, "strategy", None),
        tags=tuple(tags),
        compliance_live_allowed=compliance_allowed,
    )


def load_risk_profile_config(
    entrypoint: str,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[str | None, object | None]:
    """Zwraca konfigurację profilu ryzyka dla wskazanego punktu wejścia runtime.

    Funkcja korzysta z typowanego loadera ``bot_core``. Jeśli nie jest on
    dostępny lub zgłosi wyjątek, następuje degradacja do lekkiego odczytu YAML.
    W przypadku braku profilu zwraca ``(resolved_name, None)``.
    """

    effective_path: Path | None
    if config_path is None:
        try:
            effective_path = resolve_core_config_path()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiskowa
            if logger is not None:
                logger.debug(
                    "Nie udało się ustalić ścieżki konfiguracji core dla %s: %r",
                    entrypoint,
                    exc,
                )
            return profile_name, None
    else:
        effective_path = Path(config_path)

    resolved_name = profile_name

    _ensure_typed_loader()
    if _load_core_config is not None:
        try:
            core_config = _load_core_config(effective_path)
        except Exception as exc:  # pragma: no cover - fallback YAML
            if logger is not None:
                logger.debug(
                    "Typowany loader konfiguracji nie powiódł się dla %s: %r",
                    entrypoint,
                    exc,
                )
        else:
            if not resolved_name:
                entry_decl = core_config.runtime_entrypoints.get(entrypoint)
                if entry_decl is not None:
                    resolved_name = getattr(entry_decl, "risk_profile", None)
            if resolved_name:
                profile = core_config.risk_profiles.get(resolved_name)
                if profile is not None:
                    return resolved_name, profile
                if logger is not None:
                    logger.warning(
                        "Profil ryzyka %s nie istnieje w konfiguracji core (%s)",
                        resolved_name,
                        effective_path,
                    )
            return resolved_name, None

    raw_config = _read_raw_core_config(effective_path, logger=logger)
    if raw_config is None:
        return resolved_name, None

    if not resolved_name:
        runtime_entrypoints = raw_config.get("runtime_entrypoints")
        if isinstance(runtime_entrypoints, Mapping):
            entry = runtime_entrypoints.get(entrypoint)
            if isinstance(entry, Mapping):
                candidate = entry.get("risk_profile")
                if isinstance(candidate, str) and candidate.strip():
                    resolved_name = candidate.strip()

    profiles_section = raw_config.get("risk_profiles")
    if isinstance(profiles_section, Mapping) and resolved_name:
        profile_entry = profiles_section.get(resolved_name)
        if isinstance(profile_entry, Mapping):
            return resolved_name, dict(profile_entry)
        if logger is not None:
            logger.warning(
                "Profil ryzyka %s nie istnieje w konfiguracji core (%s)",
                resolved_name,
                effective_path,
            )
    return resolved_name, None


def _read_raw_core_config(
    path: Path,
    *,
    logger: logging.Logger | None = None,
) -> Mapping[str, object] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            yaml = _require_yaml()
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - fallback diagnostyczny
        if logger is not None:
            logger.debug("Nie udało się wczytać surowego YAML %s: %r", path, exc)
        return None
    return payload if isinstance(payload, Mapping) else None


def _load_entrypoint_from_raw_config(
    path: Path,
    entrypoint: str,
    *,
    logger: logging.Logger | None = None,
) -> Mapping[str, Any] | None:
    raw_config = _read_raw_core_config(path, logger=logger)
    if raw_config is None:
        return None
    runtime_entrypoints = raw_config.get("runtime_entrypoints")
    if not isinstance(runtime_entrypoints, Mapping):
        return None
    entry_payload = runtime_entrypoints.get(entrypoint)
    if not isinstance(entry_payload, Mapping):
        return None
    return entry_payload


def load_risk_manager_settings(
    entrypoint: str,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    defaults: RiskManagerSettings | Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str | None, object | None, RiskManagerSettings]:
    """Łączy odczyt profilu ryzyka z wyprowadzeniem ustawień menedżera ryzyka."""

    resolved_name, profile_payload = load_risk_profile_config(
        entrypoint,
        profile_name=profile_name,
        config_path=config_path,
        logger=logger,
    )
    effective_name = resolved_name or profile_name
    settings = derive_risk_manager_settings(
        profile_payload,
        profile_name=effective_name,
        defaults=defaults,
    )
    return effective_name, profile_payload, settings


__all__ = [
    "RuntimeEntrypointMetadata",
    "load_risk_manager_settings",
    "load_runtime_entrypoint_metadata",
    "load_risk_profile_config",
]

def _ensure_runtime_resolver() -> None:
    global _resolve_runtime_entrypoint  # noqa: PLW0603
    global _RUNTIME_RESOLVER_ATTEMPTED  # noqa: PLW0603
    if _resolve_runtime_entrypoint is not None or _RUNTIME_RESOLVER_ATTEMPTED:
        return
    _RUNTIME_RESOLVER_ATTEMPTED = True
    try:
        from bot_core.runtime.bootstrap import (  # type: ignore
            resolve_runtime_entrypoint as _runtime_resolver,
        )
    except Exception:  # pragma: no cover - brak zależności
        return
    _resolve_runtime_entrypoint = _runtime_resolver


def _ensure_typed_loader() -> None:
    global _load_core_config  # noqa: PLW0603
    global _TYPED_LOADER_ATTEMPTED  # noqa: PLW0603
    if _load_core_config is not None or _TYPED_LOADER_ATTEMPTED:
        return
    _TYPED_LOADER_ATTEMPTED = True
    try:
        from bot_core.config.loader import load_core_config as _loader  # type: ignore
    except Exception:  # pragma: no cover - brak zależności
        return
    _load_core_config = _loader
