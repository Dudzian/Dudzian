"""Preflight weryfikacji przed uruchomieniem smoke testu paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, EnvironmentConfig
from bot_core.config.validation import validate_core_config
from bot_core.data.intervals import normalize_interval_token
from bot_core.data.ohlcv import (
    CoverageStatus,
    coerce_summary_mapping,
    evaluate_coverage,
    summarize_coverage,
    summarize_issues,
)
from bot_core.data.ohlcv.coverage_check import SummaryThresholdResult, evaluate_summary_thresholds
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.factory import build_risk_profile_from_config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia podstawowe walidacje przed smoke testem paper – "
            "kontrolę konfiguracji oraz pokrycia danych w manifeście."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Środowisko docelowe z sekcji environments",
    )
    parser.add_argument(
        "--manifest",
        help="Opcjonalna ścieżka do manifestu SQLite (domyślnie katalog cache środowiska)",
    )
    parser.add_argument(
        "--as-of",
        help="Znacznik czasu ISO8601 wykorzystywany przy ocenie świeżości danych (domyślnie teraz, UTC)",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        help=(
            "Filtruj kontrolę pokrycia do wskazanych instrumentów. "
            "Można używać zarówno nazw z konfiguracji (np. BTC_USDT) jak i symboli giełdowych."
        ),
    )
    parser.add_argument(
        "--interval",
        dest="intervals",
        action="append",
        default=None,
        help="Filtruj kontrolę pokrycia do wskazanych interwałów (np. 1d, D1, 1h).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=None,
        help="Maksymalna dopuszczalna luka czasowa (minuty) – nadpisuje ustawienie środowiska.",
    )
    parser.add_argument(
        "--min-ok-ratio",
        type=float,
        default=None,
        help="Minimalny udział poprawnych wpisów manifestu (0-1) – nadpisuje konfigurację środowiska.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwróć wynik w formacie JSON zamiast tekstowego podsumowania.",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka pliku, do którego zostanie zapisany wynik w formacie JSON.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Zakończ komendę kodem błędu, jeśli pojawią się ostrzeżenia.",
    )
    parser.add_argument(
        "--skip-risk-check",
        action="store_true",
        help="Pomiń sanity-check silnika ryzyka (tylko do debugowania).",
    )
    return parser.parse_args(argv)


def _parse_as_of(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_field(obj, name: str):
    """Pomocnicze: bezpiecznie odczytaj atrybut albo klucz ze słownika."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _filter_statuses_by_symbols(
    statuses: Iterable[CoverageStatus],
    *,
    universe,
    exchange: str,
    filters: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not filters:
        return list(statuses), []

    alias_map: dict[str, str] = {}
    for instrument in getattr(universe, "instruments", ()):  # type: ignore[attr-defined]
        symbol = instrument.exchange_symbols.get(exchange)
        if not symbol:
            continue
        alias_map[instrument.name.upper()] = symbol
        alias_map[symbol.upper()] = symbol

    available_upper = {
        str(_get_field(status, "symbol")).upper(): str(_get_field(status, "symbol"))
        for status in statuses
        if _get_field(status, "symbol") is not None
    }

    resolved_upper: set[str] = set()
    unknown: list[str] = []
    for raw in filters:
        token = str(raw).upper()
        symbol = alias_map.get(token)
        if symbol is None:
            if token in available_upper:
                symbol = available_upper[token]
        if symbol is None:
            unknown.append(raw)
            continue
        resolved_upper.add(str(symbol).upper())

    if not resolved_upper:
        return [], unknown

    filtered = [
        status
        for status in statuses
        if (_get_field(status, "symbol") is not None)
        and str(_get_field(status, "symbol")).upper() in resolved_upper
    ]
    return filtered, unknown


def _filter_statuses_by_intervals(
    statuses: Iterable[CoverageStatus],
    *,
    filters: Sequence[str] | None,
) -> tuple[list[CoverageStatus], list[str]]:
    if not filters:
        return list(statuses), []

    normalized_filters: list[tuple[str, str]] = []
    invalid: list[str] = []
    for raw in filters:
        normalized = normalize_interval_token(raw)
        if not normalized:
            invalid.append(str(raw))
            continue
        normalized_filters.append((str(raw), normalized))

    if invalid:
        return [], invalid

    available_map: dict[str, set[str]] = {}
    for status in statuses:
        interval_value = _get_field(status, "interval")
        normalized = normalize_interval_token(interval_value)
        if not normalized:
            continue
        available_map.setdefault(normalized, set()).add(str(interval_value))

    resolved_variants: set[str] = set()
    missing: list[str] = []
    for raw, normalized in normalized_filters:
        variants = available_map.get(normalized)
        if not variants:
            missing.append(raw)
            continue
        resolved_variants.update(variants)

    if missing:
        return [], missing

    filtered = [
        status
        for status in statuses
        if str(_get_field(status, "interval")) in resolved_variants
    ]
    return filtered, []








def _breakdown_by_key(statuses: Sequence[object], key_name: str) -> dict[str, dict[str, int]]:
    """
    Agregacja bez obiektów niestabializowalnych:
    Zwraca {key_value: {"total": n, "ok": x, "warning": y, "error": z, "unknown": k, ...}}
    """
    out: dict[str, dict[str, int]] = {}
    for st in statuses or []:
        key = str(_get_field(st, key_name) or "unknown")
        status = str(_get_field(st, "status") or "unknown")
        bucket = out.setdefault(key, {})
        bucket["total"] = bucket.get("total", 0) + 1
        bucket[status] = bucket.get(status, 0) + 1
    return out


def _breakdown_by_interval(statuses: Sequence[object]) -> dict[str, dict[str, int]]:
    return _breakdown_by_key(statuses, "interval")


def _breakdown_by_symbol(statuses: Sequence[object]) -> dict[str, dict[str, int]]:
    return _breakdown_by_key(statuses, "symbol")




def _resolve_reference_symbol(config: CoreConfig, environment: EnvironmentConfig) -> str:
    """Próbuje wyznaczyć reprezentatywny symbol giełdowy dla sanity-checku ryzyka."""

    universe_name = getattr(environment, "instrument_universe", None)
    if not universe_name:
        return "BTCUSDT"

    universes = getattr(config, "instrument_universes", {}) or {}
    universe = universes.get(universe_name)
    if universe is None:
        return "BTCUSDT"

    exchange = environment.exchange
    for instrument in getattr(universe, "instruments", ()):
        symbols = getattr(instrument, "exchange_symbols", None)
        if isinstance(symbols, Mapping):
            candidate = symbols.get(exchange)
            if candidate:
                return str(candidate)
    return "BTCUSDT"


def _risk_sanity_payload(
    *,
    config: CoreConfig,
    environment: EnvironmentConfig,
) -> Mapping[str, object]:
    """Weryfikuje bazowe zachowanie silnika ryzyka dla profilu środowiska."""

    payload: dict[str, object] = {
        "profile": environment.risk_profile,
        "issues": [],
        "warnings": [],
    }

    profiles = getattr(config, "risk_profiles", {}) or {}
    profile_config = profiles.get(environment.risk_profile)
    if profile_config is None:
        payload["status"] = "error"
        payload["issues"] = ["profile_not_defined"]
        return payload

    try:
        profile = build_risk_profile_from_config(profile_config)
    except Exception as exc:  # pragma: no cover - defensywnie
        payload["status"] = "error"
        payload["issues"] = [f"profile_build_failed:{exc}"]
        return payload

    payload["profile"] = profile.name

    engine = ThresholdRiskEngine()
    engine.register_profile(profile)

    equity = 120_000.0
    price = 30_000.0
    target_vol = float(profile.target_volatility())
    max_position_pct = float(profile.max_position_exposure())
    min_multiple_raw = float(profile.stop_loss_atr_multiple())
    min_multiple = max(min_multiple_raw, 0.0)

    checks: dict[str, object] = {}
    issues: list[str] = []
    warnings: list[str] = []

    symbol = _resolve_reference_symbol(config, environment)

    if min_multiple <= 0:
        warnings.append("stop_multiple_not_positive")

    tight_multiple = 0.0
    if min_multiple > 0:
        tight_multiple = max(0.05, min_multiple * 0.5)
        if tight_multiple >= min_multiple:
            tight_multiple = max(0.05, min_multiple - 0.05)

    wide_multiple = 1.0
    if min_multiple > 0:
        wide_multiple = max(min_multiple * 1.5, min_multiple + 0.5, 1.0)
        if wide_multiple <= min_multiple:
            wide_multiple = min_multiple + 0.25

    atr = max(1.0, price * 0.005)
    if max_position_pct <= 0:
        warnings.append("max_position_pct_not_positive")

    snapshot = AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )

    observations: dict[str, object] = {
        "symbol": symbol,
        "price": price,
        "atr": atr,
        "equity": equity,
        "target_volatility": target_vol,
        "max_position_pct": max_position_pct,
        "min_stop_multiple": min_multiple,
        "tight_stop_multiple": tight_multiple,
        "wide_stop_multiple": wide_multiple,
    }

    enforce_profile_rules = target_vol > 0 and min_multiple > 0

    def _build_request(quantity: float, stop_multiple: float) -> OrderRequest:
        stop_price = price - atr * stop_multiple
        metadata = {"atr": atr, "stop_price": stop_price}
        return OrderRequest(
            symbol=symbol,
            side="buy",
            quantity=float(quantity),
            order_type="limit",
            price=price,
            stop_price=stop_price,
            atr=atr,
            metadata=metadata,
        )

    if enforce_profile_rules and tight_multiple > 0:
        tight_result = engine.apply_pre_trade_checks(
            _build_request(0.1, tight_multiple),
            account=snapshot,
            profile_name=profile.name,
        )
        rejected = not tight_result.allowed
        checks["tight_stop_rejected"] = rejected
        checks["tight_stop_reason"] = tight_result.reason
        if not rejected:
            issues.append("tight_stop_not_rejected")
    elif target_vol <= 0 and "target_volatility_not_positive" not in warnings:
        warnings.append("target_volatility_not_positive")

    if enforce_profile_rules and wide_multiple > min_multiple:
        stop_distance = atr * wide_multiple
        if stop_distance <= 0:
            issues.append("wide_stop_distance_non_positive")
        else:
            risk_budget = target_vol * equity
            observations["risk_budget"] = risk_budget
            quantities: list[float] = []
            if risk_budget > 0 and stop_distance > 0:
                risk_budget_qty = risk_budget / stop_distance
                observations["risk_budget_quantity"] = risk_budget_qty
                if risk_budget_qty > 0:
                    quantities.append(risk_budget_qty)
            if max_position_pct > 0 and price > 0:
                position_limit_qty = (max_position_pct * equity) / price
                observations["position_limit_quantity"] = position_limit_qty
                if position_limit_qty > 0:
                    quantities.append(position_limit_qty)

            if not quantities:
                issues.append("unable_to_compute_allowed_quantity")
            else:
                allowed_total_quantity = min(quantities)
                observations["allowed_quantity"] = allowed_total_quantity
                if allowed_total_quantity <= 0:
                    issues.append("allowed_quantity_not_positive")
                else:
                    allowed_result = engine.apply_pre_trade_checks(
                        _build_request(allowed_total_quantity * 0.95, wide_multiple),
                        account=snapshot,
                        profile_name=profile.name,
                    )
                    checks["wide_stop_allowed"] = allowed_result.allowed
                    if not allowed_result.allowed:
                        issues.append("wide_stop_not_accepted")

                    oversized_result = engine.apply_pre_trade_checks(
                        _build_request(allowed_total_quantity * 1.5, wide_multiple),
                        account=snapshot,
                        profile_name=profile.name,
                    )
                    blocked = not oversized_result.allowed
                    checks["oversized_blocked"] = blocked
                    if not blocked:
                        issues.append("oversized_not_blocked")
                    else:
                        adjustments = oversized_result.adjustments or {}
                        max_quantity = 0.0
                        if isinstance(adjustments, Mapping):
                            raw = adjustments.get("max_quantity")
                            if raw is not None:
                                try:
                                    max_quantity = float(raw)
                                except (TypeError, ValueError):  # pragma: no cover - defensywnie
                                    max_quantity = 0.0
                        observations["max_quantity_suggestion"] = max_quantity
                        if not math.isclose(
                            max_quantity,
                            allowed_total_quantity,
                            rel_tol=1e-3,
                            abs_tol=1e-6,
                        ):
                            warnings.append("max_quantity_adjustment_differs")
    elif target_vol <= 0 and "target_volatility_not_positive" not in warnings:
        # brak docelowej zmienności uniemożliwia walidację limitów wolumenu
        warnings.append("target_volatility_not_positive")

    status = "error" if issues else ("warning" if warnings else "ok")
    payload["status"] = status
    payload["issues"] = issues
    payload["warnings"] = warnings
    payload["checks"] = checks
    payload["observations"] = observations
    return payload



def _coverage_payload(
    *,
    manifest_path: Path,
    universe,
    exchange: str,
    as_of: datetime,
    symbols: Sequence[str] | None,
    intervals: Sequence[str] | None,
    max_gap_minutes: float | None,
    min_ok_ratio: float | None,
) -> Mapping[str, object]:
    # --- Patched: kompatybilne wywołanie evaluate_coverage dla różnych wersji API ---
    try:
        statuses = list(
            evaluate_coverage(
                manifest_path=manifest_path,
                universe=universe,
                exchange_name=exchange,
                as_of=as_of,
                intervals=intervals,
            )
        )
    except TypeError:
        # starsza wersja bez 'intervals'
        try:
            statuses = list(
                evaluate_coverage(
                    manifest_path=manifest_path,
                    universe=universe,
                    exchange_name=exchange,
                    as_of=as_of,
                )
            )
        except TypeError:
            # jeszcze starsza – tylko manifest_path
            try:
                statuses = list(evaluate_coverage(manifest_path=manifest_path))
            except TypeError:
                # ultimate fallback – pozycjonalny
                statuses = list(evaluate_coverage(manifest_path))
    # --- koniec patcha ---

    statuses, unknown_symbols = _filter_statuses_by_symbols(
        statuses,
        universe=universe,
        exchange=exchange,
        filters=symbols,
    )
    if unknown_symbols:
        return {
            "status": "error",
            "issues": [
                "unknown_symbols:" + ",".join(sorted(str(token) for token in unknown_symbols))
            ],
            "summary": {"status": "error"},
        }
    statuses, interval_errors = _filter_statuses_by_intervals(statuses, filters=intervals)
    if interval_errors:
        return {
            "status": "error",
            "issues": [
                "unknown_intervals:" + ",".join(sorted(str(token) for token in interval_errors))
            ],
            "summary": {"status": "error"},
        }

    # JSON-safe issues + summary
    issues = list(summarize_issues(statuses))
    summary_payload = coerce_summary_mapping(summarize_coverage(statuses))

    # Własne breakdowny (zawsze JSON-safe)
    summary_payload["by_interval"] = _breakdown_by_interval(statuses)
    summary_payload["by_symbol"] = _breakdown_by_symbol(statuses)

    threshold_result = evaluate_summary_thresholds(
        summary_payload,
        max_gap_minutes=max_gap_minutes,
        min_ok_ratio=min_ok_ratio,
    )
    issues.extend(threshold_result.issues)
    thresholds = dict(threshold_result.thresholds)
    observed = {key: value for key, value in threshold_result.observed.items() if value is not None}

    status = "error" if issues else summary_payload.get("status", "ok")
    payload: dict[str, object] = {
        "status": status,
        "issues": issues,
        "summary": summary_payload,
        "thresholds": thresholds,
    }
    if observed:
        payload["threshold_observed"] = observed
    return payload


def run_precheck(
    *,
    environment_name: str,
    config: CoreConfig | None = None,
    config_path: str | Path | None = None,
    manifest_path: Path | None = None,
    as_of: datetime | None = None,
    symbols: Sequence[str] | None = None,
    intervals: Sequence[str] | None = None,
    max_gap_minutes: float | None = None,
    min_ok_ratio: float | None = None,
    fail_on_warnings: bool = False,
    skip_risk_check: bool = False,
) -> tuple[dict[str, object], int]:
    """Wykonuje sanitarne kontrole paper_precheck i zwraca payload wraz z kodem zakończenia."""

    if config is None:
        if config_path is None:
            raise ValueError("Wymagany jest config lub config_path")
        config = load_core_config(Path(config_path))
    validation = validate_core_config(config)

    environment = config.environments.get(environment_name)

    resolved_as_of = as_of or datetime.now(timezone.utc)
    resolved_manifest = manifest_path

    coverage_result: Mapping[str, object] | None = None
    coverage_status = "skipped"
    coverage_warnings: list[str] = []

    risk_result: Mapping[str, object] | None = None
    risk_status = "skipped"

    if not environment.instrument_universe:
        coverage_warnings.append("environment_missing_universe")
    else:
        universe = config.instrument_universes.get(environment.instrument_universe)
        if universe is None:
            coverage_warnings.append("universe_not_defined")
        elif not resolved_manifest.exists():
            coverage_warnings.append("manifest_missing")
        else:
            max_gap = max_gap_minutes
            min_ok = min_ok_ratio
            data_quality = getattr(environment, "data_quality", None)
            if data_quality is not None:
                if max_gap is None:
                    candidate = getattr(data_quality, "max_gap_minutes", None)
                    if candidate is not None:
                        try:
                            max_gap = float(candidate)
                        except (TypeError, ValueError):
                            coverage_warnings.append("invalid_max_gap_in_config")
                if min_ok is None:
                    candidate_ratio = getattr(data_quality, "min_ok_ratio", None)
                    if candidate_ratio is not None:
                        try:
                            min_ok = float(candidate_ratio)
                        except (TypeError, ValueError):
                            coverage_warnings.append("invalid_min_ok_ratio_in_config")

            if min_ok is not None and not 0 <= float(min_ok) <= 1:
                payload: dict[str, object] = {
                    "status": "error",
                    "config": {
                        "valid": validation.is_valid(),
                        "errors": list(validation.errors),
                        "warnings": list(validation.warnings),
                    },
                    "coverage": None,
                    "coverage_warnings": coverage_warnings + ["invalid_min_ok_ratio"],
                    "risk": None,
                    "environment": environment.name,
                    "manifest_path": str(resolved_manifest),
                    "as_of": resolved_as_of.isoformat(),
                    "coverage_status": "error",
                    "risk_status": "skipped",
                }
                payload["error_reason"] = "invalid_min_ok_ratio"
                return payload, 2

            coverage_result = _coverage_payload(
                manifest_path=resolved_manifest,
                universe=universe,
                exchange=environment.exchange,
                as_of=resolved_as_of,
                symbols=symbols,
                intervals=intervals,
                max_gap_minutes=max_gap,
                min_ok_ratio=min_ok,
            )
            coverage_status = str(coverage_result.get("status", "unknown"))

    if not args.skip_risk_check:
        risk_result = _risk_sanity_payload(config=config, environment=environment)
        if isinstance(risk_result, Mapping):
            risk_status = str(risk_result.get("status", "unknown"))
        else:
            risk_status = "unknown"
    else:
        risk_status = "skipped"

    config_payload = {
        "valid": validation.is_valid(),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }

    result_payload: dict[str, object] = {
        "status": "ok",
        "config": config_payload,
        "coverage": coverage_result,
        "coverage_warnings": coverage_warnings,
        "risk": risk_result,
        "environment": environment.name,
        "manifest_path": str(resolved_manifest),
        "as_of": resolved_as_of.isoformat(),
        "coverage_status": coverage_status,
        "risk_status": risk_status,
    }

    final_status = "ok"
    exit_code = 0

    if not validation.is_valid():
        final_status = "error"
        exit_code = 2
    elif coverage_result is not None and coverage_status == "error":
        final_status = "error"
        exit_code = 3
    elif risk_result is not None and risk_status == "error":
        final_status = "error"
        exit_code = 5
    elif args.fail_on_warnings and (
        validation.warnings
        or coverage_status == "warning"
        or coverage_warnings
        or risk_status == "warning"
    ):
        final_status = "error"
        exit_code = 4
    elif (
        validation.warnings
        or coverage_status == "warning"
        or coverage_warnings
        or risk_status == "warning"
    ):
        final_status = "warning"

    result_payload["status"] = final_status
    result_payload["coverage_status"] = coverage_status
    result_payload["risk_status"] = risk_status

    if args.json or args.output:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized + "\n", encoding="utf-8")
        if args.json:
            print(serialized)

    if not args.json:
        config_payload = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        config_valid = bool(config_payload.get("valid"))
        print(f"Config: {'OK' if config_valid else 'ERROR'}")
        for entry in config_payload.get("errors", []) or []:
            print(f"  ✖ {entry}")
        for entry in config_payload.get("warnings", []) or []:
            print(f"  ⚠ {entry}")

        coverage_result = payload.get("coverage") if isinstance(payload, Mapping) else None
        coverage_status = str(payload.get("coverage_status", "skipped"))
        coverage_warnings = list(payload.get("coverage_warnings", []) or [])

        if coverage_result is None:
            print("Pokrycie danych: pominięto (brak manifestu lub uniwersum)")
        else:
            summary = coverage_result.get("summary", {}) if isinstance(coverage_result, Mapping) else {}
            print(f"Pokrycie danych: {coverage_status}")
            if isinstance(summary, Mapping):
                total = summary.get("total", 0)
                ok = summary.get("ok", 0)
                error = summary.get("error", 0)
                print(f"  łącznie={total} ok={ok} błędy={error}")
                ok_ratio = summary.get("ok_ratio")
                if isinstance(ok_ratio, (int, float)):
                    print(f"  ok_ratio={float(ok_ratio):.4f}")
                manifest_counts = summary.get("manifest_status_counts", {})
                if isinstance(manifest_counts, Mapping) and manifest_counts:
                    counts = ", ".join(
                        f"{status}={count}" for status, count in sorted(manifest_counts.items())
                    )
                    print(f"  statusy manifestu: {counts}")
                worst_gap = summary.get("worst_gap")
                if isinstance(worst_gap, Mapping):
                    print(
                        "  największa luka: {symbol}/{interval} ({gap} min)".format(
                            symbol=worst_gap.get("symbol", "?"),
                            interval=worst_gap.get("interval", "?"),
                            gap=worst_gap.get("gap_minutes", "?"),
                        )
                    )
            issues = coverage_result.get("issues") if isinstance(coverage_result, Mapping) else None
            if issues:
                print("  Problemy:")
                for issue in issues:  # type: ignore[assignment]
                    print(f"    - {issue}")

        if coverage_warnings:
            print("Ostrzeżenia pokrycia:")
            for warning in coverage_warnings:
                print(f"  - {warning}")

        if risk_result is None:
            if args.skip_risk_check:
                print("Sanity-check ryzyka: pominięto (--skip-risk-check)")
            else:
                print(f"Sanity-check ryzyka: {risk_status}")
        else:
            print(f"Sanity-check ryzyka: {risk_status}")
            issues = []
            warnings = []
            if isinstance(risk_result, Mapping):
                issues = list(risk_result.get("issues", []))
                warnings = list(risk_result.get("warnings", []))
            if issues:
                print("  Problemy ryzyka:")
                for issue in issues:
                    print(f"    - {issue}")
            if warnings:
                print("  Ostrzeżenia ryzyka:")
                for warning in warnings:
                    print(f"    - {warning}")

        print(f"Status końcowy: {final_status.upper()}")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
