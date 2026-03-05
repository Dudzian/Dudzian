"""Integracja runtime z warstwą portfolio i raportami Stage6."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery, MarketIntelSnapshot
from bot_core.portfolio import (
    PortfolioDecisionLog,
    PortfolioGovernor,
    PortfolioScheduler,
    StrategyPortfolioGovernor,
)
from bot_core.risk import StressOverrideRecommendation
from bot_core.runtime.portfolio_inputs import (
    build_slo_status_provider,
    build_stress_override_provider,
    load_stress_overrides,
)
from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator
from bot_core.market_intel import load_market_intel_report

_LOGGER = logging.getLogger(__name__)


def _to_path(value: object | None) -> Path | None:
    if value in (None, "", False):
        return None
    try:
        path = Path(str(value)).expanduser()
    except Exception:
        return None
    return path


def _unique_paths(paths: Sequence[Path | None]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in paths:
        if candidate is None:
            continue
        normalized = candidate.expanduser()
        key = str(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def _resolve_latest_report(
    directories: Sequence[Path | None], *, prefix: str, suffix: str = ".json"
) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = float("-inf")
    for base in directories:
        if base is None:
            continue
        directory = base.expanduser()
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            candidates = list(directory.glob(f"{prefix}*{suffix}"))
        except Exception:  # pragma: no cover
            _LOGGER.debug("Nie udało się przeszukać katalogu %s", directory, exc_info=True)
            continue
        for candidate in candidates:
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_path = candidate
                latest_mtime = mtime
    return latest_path


def load_market_intel_snapshots_from_reports(
    governor_name: str,
    config: object | None,
    directories: Sequence[Path],
) -> Mapping[str, MarketIntelSnapshot]:
    search_dirs: list[Path | None] = list(directories)
    output_dir = _to_path(getattr(config, "output_directory", None)) if config else None
    if output_dir is not None:
        search_dirs.insert(0, output_dir)
    normalized_dirs = _unique_paths(search_dirs)
    if not normalized_dirs:
        return {}

    name_slug = governor_name.strip().lower().replace(" ", "_") or "portfolio"
    prefixes = (
        f"market_intel_{name_slug}_",
        f"marketintel_{name_slug}_",
        "market_intel_",
    )

    for prefix in prefixes:
        report_path = _resolve_latest_report(normalized_dirs, prefix=prefix)
        if report_path is None:
            continue
        try:
            snapshots, _ = load_market_intel_report(report_path)
        except Exception:  # pragma: no cover
            _LOGGER.exception(
                "PortfolioGovernor: błąd wczytania raportu Market Intel %s", report_path
            )
            continue
        if snapshots:
            _LOGGER.debug(
                "PortfolioGovernor: użyto fallbackowego raportu Market Intel %s",
                report_path,
            )
            return snapshots
    return {}


def _resolve_latest_stress_report(governor_name: str, directories: Sequence[Path]) -> Path | None:
    name_slug = governor_name.strip().lower().replace(" ", "_") or "portfolio"
    prefixes = (
        f"stress_lab_{name_slug}_",
        "stress_lab_",
    )
    normalized_dirs = _unique_paths(directories)
    for prefix in prefixes:
        report_path = _resolve_latest_report(normalized_dirs, prefix=prefix)
        if report_path is not None:
            return report_path
    return None


def load_stress_overrides_from_reports(
    governor_name: str,
    directories: Sequence[Path],
    *,
    max_age: timedelta | None = None,
) -> tuple[StressOverrideRecommendation, ...]:
    report_path = _resolve_latest_stress_report(governor_name, directories)
    if report_path is None:
        return ()
    try:
        overrides = load_stress_overrides(report_path, max_age=max_age)
    except Exception:  # pragma: no cover
        _LOGGER.exception("PortfolioGovernor: błąd wczytania raportu Stress Lab %s", report_path)
        return ()
    if overrides:
        _LOGGER.debug("PortfolioGovernor: użyto fallbackowego raportu Stress Lab %s", report_path)
    return overrides


def build_portfolio_runtime(
    *,
    scheduler_cfg,
    core_config,
    bootstrap_ctx,
    market_intel: MarketIntelAggregator,
    environment,
    paper_settings,
    resolved_scheduler_name: str,
):
    portfolio_coordinator: PortfolioScheduler | None = None
    governor_name = getattr(scheduler_cfg, "portfolio_governor", None)
    if not governor_name:
        return portfolio_coordinator, getattr(bootstrap_ctx, "portfolio_governor", None)

    governor_cfg = core_config.portfolio_governors.get(governor_name)
    if governor_cfg is None:
        raise KeyError(
            f"Scheduler {resolved_scheduler_name} wskazuje PortfolioGovernora '{governor_name}', którego nie ma w konfiguracji"
        )

    decision_log = (
        bootstrap_ctx.portfolio_decision_log
        if getattr(bootstrap_ctx, "portfolio_decision_log", None) is not None
        else PortfolioDecisionLog()
    )
    governor = PortfolioGovernor(governor_cfg, decision_log=decision_log)

    asset_symbols = [asset.symbol for asset in governor_cfg.assets]
    interval = governor_cfg.market_intel_interval or "1h"
    lookback = int(getattr(governor_cfg, "market_intel_lookback_bars", 168) or 168)
    inputs_cfg = getattr(scheduler_cfg, "portfolio_inputs", None)
    data_cache_root = Path(environment.data_cache_path).expanduser()
    fallback_candidates: tuple[Path | None, ...] = (
        data_cache_root,
        data_cache_root.parent,
    )
    market_intel_cfg = getattr(core_config, "market_intel", None)
    stress_lab_cfg = getattr(core_config, "stress_lab", None)
    stage6_dirs: list[Path | None] = []
    if market_intel_cfg is not None:
        stage6_dirs.append(_to_path(getattr(market_intel_cfg, "output_directory", None)))
    if stress_lab_cfg is not None:
        stage6_dirs.append(_to_path(getattr(stress_lab_cfg, "report_directory", None)))
    fallback_directories = _unique_paths((*fallback_candidates, *stage6_dirs))

    slo_provider = None
    stress_provider = None
    slo_age: timedelta | None = None
    stress_age: timedelta | None = timedelta(minutes=240.0)
    if inputs_cfg is not None:
        slo_age = _minutes_to_timedelta(
            getattr(inputs_cfg, "slo_max_age_minutes", None),
            default_minutes=120.0,
        )
        stress_age = _minutes_to_timedelta(
            getattr(inputs_cfg, "stress_max_age_minutes", None),
            default_minutes=240.0,
        )
        slo_path = getattr(inputs_cfg, "slo_report_path", None)
        if slo_path:
            slo_provider = build_slo_status_provider(
                slo_path,
                fallback_directories=fallback_directories,
                max_age=slo_age,
            )
        stress_path = getattr(inputs_cfg, "stress_lab_report_path", None)
        if stress_path:
            stress_provider = build_stress_override_provider(
                stress_path,
                fallback_directories=fallback_directories,
                max_age=stress_age,
            )
    if stress_provider is None:
        overrides = load_stress_overrides_from_reports(
            governor_name,
            fallback_directories,
            max_age=stress_age,
        )
        if overrides:
            cached_overrides = tuple(overrides)

            def _fallback_stress_provider() -> Sequence[StressOverrideRecommendation]:
                return cached_overrides

            stress_provider = _fallback_stress_provider

    def _market_data_provider() -> Mapping[str, MarketIntelSnapshot]:
        if not asset_symbols:
            return {}
        queries = [
            MarketIntelQuery(symbol=symbol, interval=interval, lookback_bars=lookback)
            for symbol in asset_symbols
        ]
        snapshots: dict[str, MarketIntelSnapshot] = {}
        fallback_snapshots: Mapping[str, MarketIntelSnapshot] | None = None
        try:
            snapshots.update(market_intel.build_many(queries))
        except Exception:  # pragma: no cover
            _LOGGER.exception("PortfolioGovernor: błąd budowania metryk Market Intel")
        missing_symbols = [query.symbol for query in queries if query.symbol not in snapshots]
        if missing_symbols and market_intel_cfg is not None:
            fallback_snapshots = load_market_intel_snapshots_from_reports(
                governor_name,
                market_intel_cfg,
                fallback_directories,
            )
            for symbol in missing_symbols:
                if fallback_snapshots and symbol in fallback_snapshots:
                    snapshots[symbol] = fallback_snapshots[symbol]
        return snapshots

    def _allocation_provider():
        if not hasattr(governor, "latest_market_intel"):
            return 1.0, {}
        latest = getattr(governor, "latest_market_intel")() or {}
        return 1.0, latest

    def _metadata_provider() -> Mapping[str, object]:
        return {
            "environment": environment.name,
            "scheduler": resolved_scheduler_name,
            "governor": governor_name,
        }

    portfolio_coordinator = PortfolioRuntimeCoordinator(
        governor,
        allocation_provider=_allocation_provider,
        market_data_provider=_market_data_provider,
        stress_override_provider=stress_provider,
        slo_status_provider=slo_provider,
        metadata_provider=_metadata_provider,
    )

    return portfolio_coordinator, governor


def _minutes_to_timedelta(value: float | int | None, default_minutes: float):
    minutes = default_minutes if value in (None, "") else float(value)
    if minutes <= 0:
        return None
    return timedelta(minutes=minutes)
