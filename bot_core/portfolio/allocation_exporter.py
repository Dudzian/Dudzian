"""Narzędzia do eksportu domyślnych alokacji z konfiguracji PortfolioGovernora."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

try:  # PyYAML jest opcjonalne – fallback do JSON, jeśli brak pakietu
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from bot_core.config import PortfolioGovernorConfig, load_core_config

LOGGER = logging.getLogger(__name__)


class PortfolioAllocationExportError(RuntimeError):
    """Błąd eksportu alokacji portfela."""


@dataclass(frozen=True, slots=True)
class PortfolioAllocationDocument:
    """Reprezentacja wyeksportowanych alokacji."""

    path: Path
    governor: str
    environment: str | None
    allocations: Mapping[str, float]
    generated_at: datetime


def _extract_assets(config: object) -> Iterable[object]:
    """Zwraca listę definicji aktywów niezależnie od typu obiektu."""

    if hasattr(config, "assets"):
        assets = getattr(config, "assets")
    elif isinstance(config, Mapping):
        assets = config.get("assets")
    else:
        assets = None

    if assets is None:
        return ()

    if isinstance(assets, Mapping):
        return assets.values()

    return tuple(assets)


def _extract_symbol(asset: object) -> str | None:
    if hasattr(asset, "symbol"):
        symbol = getattr(asset, "symbol")
    elif isinstance(asset, Mapping):
        symbol = asset.get("symbol")
    else:
        symbol = None

    if symbol in (None, ""):
        return None
    return str(symbol)


def _extract_weight(asset: object) -> float | None:
    if hasattr(asset, "target_weight"):
        weight = getattr(asset, "target_weight")
    elif isinstance(asset, Mapping):
        weight = asset.get("target_weight")
    else:
        weight = None

    if weight in (None, ""):
        return None

    try:
        return float(weight)
    except (TypeError, ValueError):  # pragma: no cover - ochrona przed dziwnymi typami
        return None


def _build_allocation_map(config: object) -> dict[str, float]:
    allocations: dict[str, float] = {}

    for asset in _extract_assets(config):
        symbol = _extract_symbol(asset)
        if not symbol:
            raise PortfolioAllocationExportError("Wykryto aktywo bez symbolu w konfiguracji PortfolioGovernora")

        weight = _extract_weight(asset)
        if weight is None:
            raise PortfolioAllocationExportError(f"Brak poprawnej wagi target dla aktywa {symbol}")

        if symbol in allocations:
            raise PortfolioAllocationExportError(f"Zduplikowany symbol aktywa: {symbol}")

        allocations[symbol] = float(weight)

    if not allocations:
        raise PortfolioAllocationExportError("Konfiguracja PortfolioGovernora nie zawiera aktywów do eksportu")

    return dict(sorted(allocations.items()))


def _dump_payload(path: Path, payload: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"} and yaml is not None:
        path.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        return

    if suffix in {".yaml", ".yml"} and yaml is None:
        LOGGER.warning(
            "Brak pakietu PyYAML – zapisuję alokacje Stage6 jako JSON w pliku %s",
            path,
        )

    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def export_allocations_for_governor_config(
    config: object,
    output_path: Path,
    *,
    governor_name: str | None = None,
    environment: str | None = None,
) -> PortfolioAllocationDocument:
    """Eksportuje mapę symbol -> target_weight do wskazanego pliku."""

    allocations = _build_allocation_map(config)
    _dump_payload(output_path, allocations)

    portfolio_id = getattr(config, "portfolio_id", None)
    if portfolio_id is None and isinstance(config, Mapping):
        portfolio_id = config.get("portfolio_id")

    name = governor_name or getattr(config, "name", None) or portfolio_id
    if name is None and isinstance(config, Mapping):
        name = config.get("name")

    generated_at = datetime.now(timezone.utc)

    total_weight = sum(allocations.values())

    LOGGER.info(
        "Zapisano alokacje portfela %s (%d symboli, suma wag=%.6f) do %s",
        name or "portfolio",
        len(allocations),
        total_weight,
        output_path,
    )

    if not 0.99 <= total_weight <= 1.01:
        LOGGER.warning(
            "Suma wag alokacji dla %s wynosi %.6f – rozważ normalizację lub aktualizację konfiguracji",
            name or "portfolio",
            total_weight,
        )

    return PortfolioAllocationDocument(
        path=output_path,
        governor=str(name or "portfolio"),
        environment=environment,
        allocations=allocations,
        generated_at=generated_at,
    )


def export_allocations_from_core_config(
    core_config_path: Path,
    governor: str,
    *,
    output_path: Path | None = None,
    environment: str | None = None,
) -> PortfolioAllocationDocument:
    """Ładuje `core.yaml`, wybiera governora i eksportuje jego alokacje."""

    core_config = load_core_config(core_config_path)

    if environment is not None and environment not in core_config.environments:
        raise PortfolioAllocationExportError(
            f"Środowisko {environment} nie istnieje w konfiguracji core ({core_config_path})"
        )

    try:
        governor_cfg: PortfolioGovernorConfig | Mapping[str, object] = core_config.portfolio_governors[governor]
    except KeyError as exc:  # pragma: no cover - ochrona przed regresją konfiguracji
        raise PortfolioAllocationExportError(
            f"PortfolioGovernor {governor} nie istnieje w konfiguracji core ({core_config_path})"
        ) from exc

    if output_path is None:
        suffix = "yaml" if yaml is not None else "json"
        output_path = Path("var/audit/portfolio") / f"allocations_{governor}.{suffix}"

    document = export_allocations_for_governor_config(
        governor_cfg,
        output_path,
        governor_name=governor,
        environment=environment,
    )

    return document


__all__ = [
    "PortfolioAllocationDocument",
    "PortfolioAllocationExportError",
    "export_allocations_for_governor_config",
    "export_allocations_from_core_config",
]

