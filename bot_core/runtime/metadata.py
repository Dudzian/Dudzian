"""Pomocnicze metadane punktów wejścia runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Any
import logging

import yaml

from bot_core.runtime.paths import resolve_core_config_path

try:  # pragma: no cover - opcjonalny import w środowiskach bez pełnego runtime
    from bot_core.runtime.bootstrap import resolve_runtime_entrypoint as _resolve_runtime_entrypoint
except Exception:  # pragma: no cover - brak modułów bootstrapu w środowisku
    _resolve_runtime_entrypoint = None  # type: ignore

try:  # pragma: no cover - środowiska bez pełnego loadera configu
    from bot_core.config.loader import load_core_config as _load_core_config
except Exception:  # pragma: no cover - brak modułu loadera
    _load_core_config = None  # type: ignore


@dataclass(frozen=True, slots=True)
class RuntimeEntrypointMetadata:
    """Wybrane, lekkie metadane udostępniane aplikacjom desktopowym."""

    environment: str
    risk_profile: str
    controller: str | None
    strategy: str | None
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Zwraca metadane w postaci słownika przyjaznego serializacji."""

        return {
            "environment": self.environment,
            "risk_profile": self.risk_profile,
            "controller": self.controller,
            "strategy": self.strategy,
            "tags": list(self.tags),
        }


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

    resolver = _resolve_runtime_entrypoint
    if resolver is None:  # pragma: no cover - środowiska bez runtime
        if logger is not None:
            logger.debug(
                "Runtime entrypoint %s pominięty – brak funkcji resolve_runtime_entrypoint",
                entrypoint,
            )
        return None

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
    return RuntimeEntrypointMetadata(
        environment=getattr(entrypoint_decl, "environment"),
        risk_profile=getattr(entrypoint_decl, "risk_profile"),
        controller=getattr(entrypoint_decl, "controller", None),
        strategy=getattr(entrypoint_decl, "strategy", None),
        tags=tuple(tags),
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
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - fallback diagnostyczny
        if logger is not None:
            logger.debug("Nie udało się wczytać surowego YAML %s: %r", path, exc)
        return None
    return payload if isinstance(payload, Mapping) else None


@dataclass(frozen=True, slots=True)
class RiskManagerSettings:
    """Zestaw wartości konfiguracyjnych dla adaptera menedżera ryzyka."""

    max_risk_per_trade: float
    max_daily_loss_pct: float
    max_portfolio_risk: float
    max_positions: int
    emergency_stop_drawdown: float
    confidence_level: float | None = None
    target_volatility: float | None = None
    profile_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializuje ustawienia do słownika kompatybilnego z adapterem GUI."""

        payload: dict[str, Any] = {
            "max_risk_per_trade": float(self.max_risk_per_trade),
            "max_portfolio_risk": float(self.max_portfolio_risk),
            "max_positions": int(self.max_positions),
            "emergency_stop_drawdown": float(self.emergency_stop_drawdown),
        }
        if self.confidence_level is not None:
            payload["confidence_level"] = float(self.confidence_level)
        return payload

    def risk_service_kwargs(self) -> dict[str, float | int]:
        """Zwraca parametry oczekiwane przez ``RiskService`` AutoTradera."""

        return {
            "max_position_notional_pct": float(self.max_risk_per_trade),
            "max_daily_loss_pct": float(self.max_daily_loss_pct),
            "max_portfolio_risk_pct": float(self.max_portfolio_risk),
            "emergency_stop_drawdown_pct": float(self.emergency_stop_drawdown),
            "max_positions": int(self.max_positions),
        }


_DEFAULT_RISK_SETTINGS = RiskManagerSettings(
    max_risk_per_trade=0.02,
    max_daily_loss_pct=0.10,
    max_portfolio_risk=0.10,
    max_positions=10,
    emergency_stop_drawdown=0.15,
)


def derive_risk_manager_settings(
    profile: object | None,
    *,
    profile_name: str | None = None,
    defaults: RiskManagerSettings | Mapping[str, Any] | None = None,
) -> RiskManagerSettings:
    """Buduje ustawienia menedżera ryzyka na podstawie profilu runtime."""

    base = {
        "max_risk_per_trade": _DEFAULT_RISK_SETTINGS.max_risk_per_trade,
        "max_daily_loss_pct": _DEFAULT_RISK_SETTINGS.max_daily_loss_pct,
        "max_portfolio_risk": _DEFAULT_RISK_SETTINGS.max_portfolio_risk,
        "max_positions": _DEFAULT_RISK_SETTINGS.max_positions,
        "emergency_stop_drawdown": _DEFAULT_RISK_SETTINGS.emergency_stop_drawdown,
        "confidence_level": _DEFAULT_RISK_SETTINGS.confidence_level,
        "target_volatility": _DEFAULT_RISK_SETTINGS.target_volatility,
    }

    if isinstance(defaults, RiskManagerSettings):
        base.update(
            max_risk_per_trade=defaults.max_risk_per_trade,
            max_daily_loss_pct=defaults.max_daily_loss_pct,
            max_portfolio_risk=defaults.max_portfolio_risk,
            max_positions=defaults.max_positions,
            emergency_stop_drawdown=defaults.emergency_stop_drawdown,
        )
        base["confidence_level"] = defaults.confidence_level
        base["target_volatility"] = defaults.target_volatility
    elif isinstance(defaults, Mapping):
        for key in base:
            if key in defaults and defaults[key] is not None:
                value = defaults[key]
                if key == "max_positions":
                    try:
                        base[key] = int(value)  # type: ignore[assignment]
                    except Exception:
                        continue
                else:
                    try:
                        base[key] = float(value)  # type: ignore[assignment]
                    except Exception:
                        continue

    def _read_value(name: str) -> Any:
        if profile is None:
            return None
        if hasattr(profile, name):
            return getattr(profile, name)
        if isinstance(profile, Mapping):
            return profile.get(name)
        return None

    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            coerced = float(value)
        except Exception:
            return fallback
        if not coerced and coerced != 0.0:
            return fallback
        return coerced

    per_trade = _coerce_float(_read_value("max_position_pct"), base["max_risk_per_trade"])
    if per_trade <= 0:
        per_trade = base["max_risk_per_trade"]
    base["max_risk_per_trade"] = max(0.0, min(1.0, per_trade))

    daily_loss = _coerce_float(_read_value("max_daily_loss_pct"), base["max_daily_loss_pct"])
    if daily_loss <= 0:
        daily_loss = base["max_daily_loss_pct"]
    base["max_daily_loss_pct"] = max(0.0, min(1.0, daily_loss))

    portfolio_limit = _coerce_float(
        _read_value("max_portfolio_risk"),
        base["max_portfolio_risk"] if base["max_portfolio_risk"] else daily_loss,
    )
    if portfolio_limit <= 0:
        portfolio_limit = base["max_portfolio_risk"] if base["max_portfolio_risk"] else daily_loss
    portfolio_limit = max(base["max_risk_per_trade"] + 1e-4, portfolio_limit)
    base["max_portfolio_risk"] = min(1.0, portfolio_limit)

    positions = _read_value("max_open_positions")
    try:
        positions_int = int(positions) if positions is not None else base["max_positions"]
    except Exception:
        positions_int = base["max_positions"]
    if positions_int > 0:
        base["max_positions"] = positions_int

    drawdown = _coerce_float(_read_value("hard_drawdown_pct"), base["emergency_stop_drawdown"])
    if drawdown > 0:
        base["emergency_stop_drawdown"] = min(1.0, max(drawdown, base["max_risk_per_trade"]))

    target_vol = _coerce_float(_read_value("target_volatility"), base["target_volatility"] or 0.0)
    if target_vol > 0:
        base["target_volatility"] = min(1.0, target_vol)
        base["confidence_level"] = max(0.5, min(0.99, 1.0 - min(target_vol, 0.5) / 2))

    return RiskManagerSettings(
        max_risk_per_trade=float(base["max_risk_per_trade"]),
        max_daily_loss_pct=float(base["max_daily_loss_pct"]),
        max_portfolio_risk=float(base["max_portfolio_risk"]),
        max_positions=int(base["max_positions"]),
        emergency_stop_drawdown=float(base["emergency_stop_drawdown"]),
        confidence_level=None if base["confidence_level"] is None else float(base["confidence_level"]),
        target_volatility=None if base["target_volatility"] is None else float(base["target_volatility"]),
        profile_name=profile_name,
    )


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
    "RiskManagerSettings",
    "derive_risk_manager_settings",
    "load_risk_manager_settings",
    "load_runtime_entrypoint_metadata",
    "load_risk_profile_config",
]

