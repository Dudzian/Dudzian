"""Uruchamia lokalny backend bota (AutoTrader + serwer gRPC) na potrzeby UI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from bot_core.api.server import build_local_runtime_context, LocalRuntimeServer
from bot_core.exchanges.base import Environment as ExchangeEnvironment
from bot_core.logging.config import install_metrics_logging_handler
from bot_core.runtime.state_manager import RuntimeStateError, RuntimeStateManager
from bot_core.security.base import SecretStorageError
from core.reporting import DemoPaperReport, GuardrailReport

_LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _write_ready_payload(
    address: str,
    *,
    ready_file: Optional[str],
    emit_stdout: bool,
    metrics_url: str | None = None,
) -> None:
    payload = {"event": "ready", "address": address, "pid": os.getpid()}
    if metrics_url:
        payload["metrics_url"] = metrics_url
    serialized = json.dumps(payload, ensure_ascii=False)
    if ready_file:
        Path(ready_file).expanduser().write_text(serialized, encoding="utf-8")
    if emit_stdout:
        print(serialized, flush=True)


def _determine_entrypoint_name(config: Any, entrypoint_obj: Any, fallback: str | None) -> str:
    if fallback:
        return fallback
    entrypoints = getattr(getattr(config, "trading", None), "entrypoints", {}) or {}
    for name, candidate in entrypoints.items():
        if candidate is entrypoint_obj:
            return str(name)
    return fallback or "unknown"


def _normalize_environment(value: Any) -> ExchangeEnvironment:
    if isinstance(value, ExchangeEnvironment):
        return value
    try:
        return ExchangeEnvironment(str(value).lower())
    except Exception:
        return ExchangeEnvironment.PAPER


def _validate_runtime_context(context: Any, mode: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"mode": mode}

    entrypoint_cfg = getattr(context, "entrypoint", None)
    details["environment"] = getattr(entrypoint_cfg, "environment", None)
    details["strategy"] = getattr(entrypoint_cfg, "strategy", None)
    details["risk_profile"] = getattr(entrypoint_cfg, "risk_profile", None)

    pipeline = getattr(context, "pipeline", None)
    bootstrap = getattr(pipeline, "bootstrap", None)
    runtime_environment = getattr(bootstrap, "environment", None)
    keychain_key = getattr(runtime_environment, "keychain_key", None)
    credential_purpose = getattr(runtime_environment, "credential_purpose", "trading")
    expected_env = _normalize_environment(getattr(runtime_environment, "environment", ExchangeEnvironment.PAPER))
    details["keychain_key"] = keychain_key
    details["expected_environment"] = expected_env.value

    if keychain_key:
        try:
            context.secret_manager.load_exchange_credentials(
                keychain_key,
                expected_environment=expected_env,
                purpose=str(credential_purpose or "trading"),
                required_permissions=getattr(runtime_environment, "required_permissions", None),
                forbidden_permissions=getattr(runtime_environment, "forbidden_permissions", None),
            )
            details["credentials_validated"] = True
        except SecretStorageError as exc:
            errors.append(f"Walidacja poświadczeń '{keychain_key}' nie powiodła się: {exc}")
        except Exception as exc:  # pragma: no cover - defensywne
            errors.append(f"Nie udało się zweryfikować poświadczeń '{keychain_key}': {exc}")
    else:
        warnings.append("Środowisko nie określa klucza API – pomijam walidację poświadczeń.")

    controller = getattr(pipeline, "controller", None)
    symbols = tuple(getattr(controller, "symbols", ()) or ())
    details["symbols"] = list(symbols)
    if not symbols:
        errors.append("Kontroler strategii nie udostępnia żadnych symboli handlowych.")

    return {"errors": errors, "warnings": warnings, "details": details}


def _collect_paper_exchange_metrics(context: Any) -> dict[str, Any]:
    config = getattr(context, "config", None)
    trading_cfg = getattr(config, "trading", None)
    entrypoints = getattr(trading_cfg, "entrypoints", {}) if trading_cfg is not None else {}
    if not isinstance(entrypoints, Mapping):
        return {}

    execution_cfg = getattr(config, "execution", None) if config is not None else None
    paper_profiles_cfg = getattr(execution_cfg, "paper_profiles", {}) if execution_cfg is not None else {}
    profile_metadata: dict[str, dict[str, Any]] = {}
    if isinstance(paper_profiles_cfg, Mapping):
        for profile in paper_profiles_cfg.values():
            if not isinstance(profile, Mapping):
                continue
            entrypoint_name = str(profile.get("entrypoint") or "")
            metrics_cfg = profile.get("metrics") if isinstance(profile.get("metrics"), Mapping) else {}
            metric_names = {
                key: str(value)
                for key, value in {
                    "rate_limit": metrics_cfg.get("rate_limit"),
                    "network_errors": metrics_cfg.get("network_errors"),
                    "health": metrics_cfg.get("health"),
                }.items()
                if value
            }
            thresholds_cfg = metrics_cfg.get("thresholds") if isinstance(metrics_cfg, Mapping) else {}
            thresholds = {str(key): value for key, value in dict(thresholds_cfg or {}).items()}
            if entrypoint_name:
                profile_metadata[entrypoint_name] = {
                    "metric_names": metric_names,
                    "thresholds": thresholds,
                }

    metrics_registry = getattr(context, "metrics_registry", None)
    metric_cache: dict[str, Any] = {}

    def _get_metric(metric_name: str | None):
        if metrics_registry is None or not metric_name:
            return None
        cached = metric_cache.get(metric_name)
        if metric_name in metric_cache:
            return cached
        try:
            metric_cache[metric_name] = metrics_registry.get(metric_name)
        except KeyError:  # pragma: no cover - rejestr bez wskazanej metryki
            metric_cache[metric_name] = None
        return metric_cache[metric_name]

    summary: dict[str, Any] = {}
    status_counts: dict[str, int] = {"ok": 0, "unknown": 0, "breached": 0}
    total_breaches = 0
    total_missing = 0
    total_invalid = 0
    ok_entrypoint_names: list[str] = []
    breached_entrypoint_names: list[str] = []
    unknown_entrypoint_names: list[str] = []
    monitored_entrypoint_names: list[str] = []
    missing_metrics_entrypoints: dict[str, list[str]] = {}
    invalid_thresholds_entrypoints: dict[str, list[str]] = {}
    breached_thresholds_entrypoints: dict[str, list[str]] = {}
    breach_counts_by_metric: dict[str, int] = {}
    threshold_breach_counts: dict[str, int] = {}
    missing_metric_counts: dict[str, int] = {}
    invalid_threshold_counts: dict[str, int] = {}
    metric_coverage_entrypoints: dict[str, set[str]] = {}
    threshold_coverage_entrypoints: dict[str, set[str]] = {}
    network_error_severity_coverage_entrypoints: dict[str, set[str]] = {}
    monitored_metric_names: set[str] = set()
    monitored_threshold_names: set[str] = set()
    network_error_severity_totals: dict[str, float] = {}
    missing_error_severity_counts: dict[str, int] = {}
    missing_error_severity_entrypoints: dict[str, set[str]] = {}
    total_missing_error_severities = 0
    severities = ("warning", "error", "critical")
    for name, entrypoint in entrypoints.items():
        environment = str(getattr(entrypoint, "environment", "") or "")
        if "paper" not in environment.lower():
            continue

        entrypoint_name = str(name)
        exchange = environment.split("_")[0].lower() if environment else entrypoint_name.lower()
        metadata = profile_metadata.get(entrypoint_name, {})
        metric_names = metadata.get("metric_names", {}) if isinstance(metadata, Mapping) else {}

        rate_metric_name = metric_names.get("rate_limit") or "bot_exchange_rate_limited_total"
        error_metric_name = metric_names.get("network_errors") or "bot_exchange_errors_total"
        health_metric_name = metric_names.get("health") or "bot_exchange_health_status"

        for metric_identifier in (rate_metric_name, error_metric_name, health_metric_name):
            if metric_identifier:
                monitored_metric_names.add(str(metric_identifier))

        rate_metric = _get_metric(str(rate_metric_name))
        error_metric = _get_metric(str(error_metric_name))
        health_metric = _get_metric(str(health_metric_name))

        missing_metrics: list[str] = []
        invalid_thresholds: list[str] = []
        missing_error_severities: list[str] = []

        def _mark_missing(metric_name: str | None) -> None:
            if metric_name and metric_name not in missing_metrics:
                missing_metrics.append(metric_name)

        rate_value: float | None = None
        if rate_metric is not None:
            try:
                rate_value = float(rate_metric.value(labels={"exchange": exchange}))
            except Exception:  # pragma: no cover - defensywne
                rate_value = None
        elif rate_metric_name:
            _mark_missing(str(rate_metric_name))
        if rate_value is not None and rate_metric_name:
            metric_coverage_entrypoints.setdefault(str(rate_metric_name), set()).add(
                entrypoint_name
            )

        error_value: float | None = None
        error_severity_breakdown: dict[str, float] = {}
        if error_metric is not None:
            aggregated = 0.0
            severity_values = getattr(error_metric, "_values", None)
            for severity in severities:
                labels = {"exchange": exchange, "severity": severity}
                has_sample = True
                if isinstance(severity_values, Mapping):
                    normalized = tuple(sorted((str(key), str(value)) for key, value in labels.items()))
                    has_sample = normalized in severity_values
                if not has_sample:
                    missing_error_severities.append(severity)
                    error_severity_breakdown[severity] = 0.0
                    continue
                try:
                    value = float(error_metric.value(labels=labels))
                except Exception:  # pragma: no cover - defensywne
                    value = 0.0
                aggregated += value
                error_severity_breakdown[severity] = value
                network_error_severity_coverage_entrypoints.setdefault(severity, set()).add(
                    entrypoint_name
                )
            error_value = aggregated
        elif error_metric_name:
            _mark_missing(str(error_metric_name))
            missing_error_severities.extend(severities)
        for severity in severities:
            error_severity_breakdown.setdefault(severity, 0.0)
        if error_value is not None and error_metric_name:
            metric_coverage_entrypoints.setdefault(str(error_metric_name), set()).add(
                entrypoint_name
            )

        health_value: float | None = None
        if health_metric is not None:
            labels = {"exchange": exchange}
            has_sample = True
            values = getattr(health_metric, "_values", None)
            if isinstance(values, Mapping):
                normalized = tuple(sorted((str(key), str(value)) for key, value in labels.items()))
                has_sample = normalized in values
            if not has_sample:
                health_value = None
                _mark_missing(str(health_metric_name))
            else:
                try:
                    health_value = float(health_metric.value(labels=labels))
                except Exception:  # pragma: no cover - defensywne
                    health_value = None
        elif health_metric_name:
            _mark_missing(str(health_metric_name))
        if health_value is not None and health_metric_name:
            metric_coverage_entrypoints.setdefault(str(health_metric_name), set()).add(
                entrypoint_name
            )

        summary_entry: dict[str, Any] = {
            "environment": environment,
            "exchange": exchange,
            "rate_limited_events": rate_value if rate_value is not None else 0.0,
            "network_errors": error_value if error_value is not None else 0.0,
            "health_status": health_value,
        }
        if error_severity_breakdown:
            summary_entry["network_errors_by_severity"] = {
                severity: error_severity_breakdown.get(severity, 0.0)
                for severity in severities
            }
        if metric_names:
            summary_entry["metric_names"] = dict(metric_names)
        thresholds = metadata.get("thresholds") if isinstance(metadata, Mapping) else {}
        if thresholds:
            summary_entry["thresholds"] = dict(thresholds)

        for severity in severities:
            value = float(error_severity_breakdown.get(severity, 0.0))
            network_error_severity_coverage_entrypoints.setdefault(severity, set())
            network_error_severity_totals[severity] = (
                network_error_severity_totals.get(severity, 0.0) + value
            )

        if missing_error_severities:
            normalized_missing_severities = sorted(
                dict.fromkeys(str(severity) for severity in missing_error_severities)
            )
            summary_entry["missing_error_severities"] = normalized_missing_severities
            total_missing_error_severities += len(normalized_missing_severities)
            for severity in normalized_missing_severities:
                missing_error_severity_counts[severity] = missing_error_severity_counts.get(
                    severity, 0
                ) + 1
                missing_error_severity_entrypoints.setdefault(severity, set()).add(
                    entrypoint_name
                )

        breaches: list[dict[str, Any]] = []

        def _as_float(value: Any) -> float | None:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        thresholds_mapping = dict(thresholds) if isinstance(thresholds, Mapping) else {}

        def _record_invalid(threshold_key: str) -> None:
            if threshold_key not in invalid_thresholds:
                invalid_thresholds.append(threshold_key)

        def _register_breach(
            metric_key: str,
            threshold_key: str,
            observed: float,
            limit: float,
            comparator: str,
        ) -> None:
            breaches.append(
                {
                    "metric": metric_key,
                    "threshold": threshold_key,
                    "observed": observed,
                    "limit": limit,
                    "comparator": comparator,
                }
            )
            metric_identifier = str(metric_key)
            threshold_identifier = str(threshold_key)
            if metric_identifier:
                breach_counts_by_metric[metric_identifier] = breach_counts_by_metric.get(
                    metric_identifier, 0
                ) + 1
            if threshold_identifier:
                threshold_breach_counts[threshold_identifier] = threshold_breach_counts.get(
                    threshold_identifier, 0
                ) + 1

        rate_limit_threshold = thresholds_mapping.get("rate_limit_max")
        if rate_limit_threshold is not None:
            monitored_threshold_names.add("rate_limit_max")
            limit = _as_float(rate_limit_threshold)
            if limit is None:
                _record_invalid("rate_limit_max")
            else:
                threshold_coverage_entrypoints.setdefault("rate_limit_max", set()).add(
                    entrypoint_name
                )
                if rate_value is not None and rate_value > limit:
                    _register_breach("rate_limit", "rate_limit_max", rate_value, limit, "<=")

        network_error_threshold = thresholds_mapping.get("network_errors_max")
        if network_error_threshold is not None:
            monitored_threshold_names.add("network_errors_max")
            limit = _as_float(network_error_threshold)
            if limit is None:
                _record_invalid("network_errors_max")
            else:
                threshold_coverage_entrypoints.setdefault("network_errors_max", set()).add(
                    entrypoint_name
                )
                if error_value is not None and error_value > limit:
                    _register_breach(
                        "network_errors", "network_errors_max", error_value, limit, "<="
                    )

        health_threshold = thresholds_mapping.get("health_min")
        if health_threshold is not None:
            monitored_threshold_names.add("health_min")
            limit = _as_float(health_threshold)
            if limit is None:
                _record_invalid("health_min")
            else:
                threshold_coverage_entrypoints.setdefault("health_min", set()).add(
                    entrypoint_name
                )
                if health_value is not None and health_value < limit:
                    _register_breach("health", "health_min", health_value, limit, ">=")

        status = "ok"
        if breaches:
            status = "breached"
        elif missing_metrics or invalid_thresholds:
            status = "unknown"

        summary_entry["status"] = status
        if breaches:
            summary_entry["breaches"] = breaches
            total_breaches += len(breaches)
            unique_thresholds = sorted(
                {
                    str(breach.get("threshold"))
                    for breach in breaches
                    if isinstance(breach, Mapping) and breach.get("threshold")
                }
            )
            if unique_thresholds:
                breached_thresholds_entrypoints[entrypoint_name] = unique_thresholds
        if missing_metrics:
            normalized_missing = sorted(dict.fromkeys(str(metric) for metric in missing_metrics))
            summary_entry["missing_metrics"] = normalized_missing
            total_missing += len(normalized_missing)
            if normalized_missing:
                missing_metrics_entrypoints[entrypoint_name] = normalized_missing
                for metric_name in normalized_missing:
                    missing_metric_counts[metric_name] = missing_metric_counts.get(
                        metric_name, 0
                    ) + 1
        if invalid_thresholds:
            normalized_invalid = sorted(
                dict.fromkeys(str(threshold) for threshold in invalid_thresholds)
            )
            summary_entry["invalid_thresholds"] = normalized_invalid
            total_invalid += len(normalized_invalid)
            if normalized_invalid:
                invalid_thresholds_entrypoints[entrypoint_name] = normalized_invalid
                for threshold_name in normalized_invalid:
                    invalid_threshold_counts[threshold_name] = invalid_threshold_counts.get(
                        threshold_name, 0
                    ) + 1

        if status in status_counts:
            status_counts[status] += 1
        else:  # pragma: no cover - defensywny fallback
            status_counts[status] = 1

        if status == "ok":
            ok_entrypoint_names.append(entrypoint_name)
        elif status == "breached":
            breached_entrypoint_names.append(entrypoint_name)
        else:
            unknown_entrypoint_names.append(entrypoint_name)

        summary[entrypoint_name] = summary_entry
        monitored_entrypoint_names.append(entrypoint_name)

    monitored_entrypoints = sum(status_counts.values())
    if monitored_entrypoints == 0:
        overall_status = "not_configured"
    elif status_counts.get("breached", 0):
        overall_status = "breached"
    elif status_counts.get("unknown", 0):
        overall_status = "unknown"
    else:
        overall_status = "ok"

    summary_overview: dict[str, Any] = {
        "status": overall_status,
        "entrypoints": monitored_entrypoints,
        "ok_entrypoints": status_counts.get("ok", 0),
        "breached_entrypoints": status_counts.get("breached", 0),
        "unknown_entrypoints": status_counts.get("unknown", 0),
        "total_breaches": total_breaches,
        "missing_metrics": total_missing,
        "invalid_thresholds": total_invalid,
        "missing_error_severities": total_missing_error_severities,
        "ok_entrypoint_names": sorted(ok_entrypoint_names),
        "breached_entrypoint_names": sorted(breached_entrypoint_names),
        "unknown_entrypoint_names": sorted(unknown_entrypoint_names),
        "monitored_entrypoint_names": sorted({*monitored_entrypoint_names}),
        "missing_metrics_entrypoints": dict(sorted(missing_metrics_entrypoints.items())),
        "invalid_thresholds_entrypoints": dict(
            sorted(invalid_thresholds_entrypoints.items())
        ),
        "breached_thresholds_entrypoints": dict(
            sorted(breached_thresholds_entrypoints.items())
        ),
    }

    def _ratio_mapping(source: Mapping[str, set[str]]) -> dict[str, float]:
        if monitored_entrypoints <= 0:
            return {}
        return {
            key: round(len(names) / monitored_entrypoints, 4)
            for key, names in sorted(source.items())
        }

    metric_coverage_ratios = _ratio_mapping(metric_coverage_entrypoints)
    threshold_coverage_ratios = _ratio_mapping(threshold_coverage_entrypoints)

    def _coverage_score(names: set[str], coverage: Mapping[str, set[str]]) -> float:
        if monitored_entrypoints <= 0 or not names:
            return 0.0
        total = 0.0
        for name in sorted(names):
            covered = len(coverage.get(name, set()))
            total += covered / monitored_entrypoints
        return round(total / len(names), 4)

    summary_overview.update(
        {
            "breach_counts_by_metric": dict(sorted(breach_counts_by_metric.items())),
            "threshold_breach_counts": dict(sorted(threshold_breach_counts.items())),
            "missing_metric_counts": dict(sorted(missing_metric_counts.items())),
            "invalid_threshold_counts": dict(sorted(invalid_threshold_counts.items())),
            "metric_coverage_entrypoints": {
                metric: sorted(names)
                for metric, names in sorted(metric_coverage_entrypoints.items())
            },
            "threshold_coverage_entrypoints": {
                threshold: sorted(names)
                for threshold, names in sorted(threshold_coverage_entrypoints.items())
            },
            "network_error_severity_coverage_entrypoints": {
                severity: sorted(names)
                for severity, names in sorted(
                    network_error_severity_coverage_entrypoints.items()
                )
            },
            "metric_coverage_ratio": metric_coverage_ratios,
            "threshold_coverage_ratio": threshold_coverage_ratios,
            "network_error_severity_coverage_ratio": _ratio_mapping(
                network_error_severity_coverage_entrypoints
            ),
            "metric_coverage_score": _coverage_score(
                set(monitored_metric_names), metric_coverage_entrypoints
            ),
            "threshold_coverage_score": _coverage_score(
                set(monitored_threshold_names), threshold_coverage_entrypoints
            ),
            "monitored_metric_names": sorted(monitored_metric_names),
            "monitored_threshold_names": sorted(monitored_threshold_names),
            "network_error_severity_totals": dict(
                sorted(network_error_severity_totals.items())
            ),
            "missing_error_severity_counts": dict(
                sorted(missing_error_severity_counts.items())
            ),
            "missing_error_severity_entrypoints": {
                severity: sorted(names)
                for severity, names in sorted(missing_error_severity_entrypoints.items())
            },
        }
    )

    summary["summary"] = summary_overview

    return summary


def _write_report(report_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    report_path = Path(report_dir).expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"run_local_bot_{payload.get('mode', 'unknown')}_{timestamp}.json"
    destination = report_path / filename
    sanitized = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    destination.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime.yaml", help="Ścieżka do pliku runtime.yaml")
    parser.add_argument("--entrypoint", help="Nazwa punktu wejścia z sekcji trading.entrypoints")
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasłuchiwania serwera gRPC")
    parser.add_argument("--port", default=0, type=int, help="Port serwera gRPC (0 = automatyczny)")
    parser.add_argument("--log-level", default="INFO", help="Poziom logowania")
    parser.add_argument("--ready-file", help="Opcjonalny plik, do którego zostanie zapisany adres serwera")
    parser.add_argument(
        "--mode",
        choices=("demo", "paper", "live"),
        default="paper",
        help="Tryb uruchomienia runtime (demo → paper → live).",
    )
    parser.add_argument(
        "--state-dir",
        default="var/runtime",
        help="Katalog przechowujący checkpointy scenariuszy E2E.",
    )
    parser.add_argument(
        "--report-dir",
        default="logs/e2e",
        help="Katalog, w którym zostanie zapisany raport końcowy scenariusza.",
    )
    parser.add_argument(
        "--report-markdown-dir",
        default="reports/e2e",
        help="Katalog docelowy raportu Markdown generowanego na koniec scenariusza.",
    )
    parser.add_argument(
        "--no-ready-stdout",
        action="store_true",
        help="Nie wypisuj komunikatu gotowości na stdout (użyteczne przy uruchomieniach z QProcess)",
    )
    parser.add_argument(
        "--manual-confirm",
        action="store_true",
        help="Nie aktywuj automatycznie auto-tradingu (wymaga ręcznego potwierdzenia w UI)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        _LOGGER.error("Nie odnaleziono pliku konfiguracji runtime: %s", config_path)
        return 2

    state_manager = RuntimeStateManager(args.state_dir)
    report_payload: dict[str, Any] = {
        "mode": args.mode,
        "config_path": str(config_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
    }

    context = None
    server: LocalRuntimeServer | None = None
    exit_code = 0
    runtime_started = False
    checkpoint = None
    decision_events: tuple[Mapping[str, Any], ...] = ()
    paper_metrics_summary: dict[str, Any] | None = None

    try:
        context = build_local_runtime_context(
            config_path=str(config_path),
            entrypoint=args.entrypoint,
        )

        entrypoint_name = _determine_entrypoint_name(context.config, context.entrypoint, args.entrypoint)
        report_payload["entrypoint"] = entrypoint_name

        validation = _validate_runtime_context(context, args.mode)
        report_payload["validation"] = validation["details"]
        report_payload["warnings"].extend(validation["warnings"])
        if validation["errors"]:
            report_payload["errors"].extend(validation["errors"])
            report_payload["status"] = "validation_failed"
            exit_code = 3

        if exit_code == 0 and args.mode in {"paper", "live"}:
            try:
                checkpoint = state_manager.require_checkpoint(
                    target_mode=args.mode,
                    entrypoint=entrypoint_name,
                )
                report_payload["checkpoint"] = checkpoint.to_dict()
            except RuntimeStateError as exc:
                message = str(exc)
                report_payload["errors"].append(message)
                report_payload["status"] = "blocked"
                exit_code = 3

        if exit_code == 0 and args.mode == "live":
            try:
                execution_settings = getattr(context.config, "execution", None)
                if execution_settings is not None:
                    setattr(execution_settings, "default_mode", "live")
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się wymusić trybu live w konfiguracji runtime", exc_info=True)

        if exit_code == 0:
            observability_cfg = getattr(context.config, "observability", None)
            enable_metrics_handler = True
            if observability_cfg is not None:
                enable_metrics_handler = getattr(observability_cfg, "enable_log_metrics", True)
            if enable_metrics_handler:
                install_metrics_logging_handler()

            context.start(auto_confirm=not args.manual_confirm)
            runtime_started = True
            server = LocalRuntimeServer(context, host=args.host, port=args.port)
            server.start()
            metrics_url = context.metrics_endpoint
            report_payload["metrics_endpoint"] = metrics_url
            _write_ready_payload(
                server.address,
                ready_file=args.ready_file,
                emit_stdout=not args.no_ready_stdout,
                metrics_url=metrics_url,
            )

            stop_event = threading.Event()

            def _handle_signal(signum, frame):  # noqa: D401, ANN001
                del signum, frame
                stop_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, _handle_signal)

            try:
                while not stop_event.is_set():
                    time.sleep(0.5)
            except KeyboardInterrupt:  # pragma: no cover - alternatywna obsługa sygnałów
                stop_event.set()

    except Exception as exc:  # pragma: no cover - defensywne logowanie
        if exit_code == 0:
            exit_code = 2
        report_payload["errors"].append(str(exc))
        report_payload.setdefault("status", "error")
        _LOGGER.error("Błąd podczas uruchamiania lokalnego runtime: %s", exc)
    finally:
        if context is not None:
            try:
                pipeline = getattr(context, "pipeline", None)
                bootstrap = getattr(pipeline, "bootstrap", None)
                journal = getattr(bootstrap, "decision_journal", None)
                if journal is not None and hasattr(journal, "export"):
                    exported = tuple(journal.export())
                    decision_events = exported
                    report_payload["decision_events"] = len(exported)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się pobrać zdarzeń z decision journal", exc_info=True)

        if paper_metrics_summary is None and context is not None:
            try:
                paper_metrics_summary = _collect_paper_exchange_metrics(context)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug(
                    "Nie udało się zebrać metryk papierowych scenariuszy giełdowych", exc_info=True
                )
                paper_metrics_summary = {}

        if server is not None:
            try:
                server.stop(0.5)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Błąd podczas zatrzymywania serwera gRPC", exc_info=True)
        if context is not None and runtime_started:
            try:
                context.stop()
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Błąd podczas zatrzymywania kontekstu runtime", exc_info=True)

        if exit_code == 0 and args.mode == "demo":
            try:
                checkpoint = state_manager.record_checkpoint(
                    entrypoint=report_payload.get("entrypoint", args.entrypoint or ""),
                    mode="demo",
                    config_path=str(config_path),
                    metadata={
                        "metrics_endpoint": report_payload.get("metrics_endpoint"),
                    },
                )
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się zapisać checkpointu po scenariuszu demo", exc_info=True)
        if checkpoint is not None:
            report_payload["checkpoint"] = checkpoint.to_dict()

        report_payload.setdefault("status", "success" if exit_code == 0 else "error")
        report_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        guardrail_payload: dict[str, Any] | None = None
        try:
            demo_report = DemoPaperReport.from_payload(report_payload, decision_events=decision_events)
            markdown_path = demo_report.write_markdown(args.report_markdown_dir)
            report_payload["report_markdown"] = str(markdown_path)
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się wygenerować raportu Markdown", exc_info=True)
        try:
            registry = getattr(context, "metrics_registry", None) if context is not None else None
            io_queue_cfg = getattr(getattr(context, "config", None), "io_queue", None) if context is not None else None
            log_directory = getattr(io_queue_cfg, "log_directory", None) if io_queue_cfg is not None else None
            if not log_directory:
                log_directory = "logs/guardrails"
            validation_details = report_payload.get("validation", {}) if isinstance(report_payload.get("validation"), Mapping) else {}
            environment_hint = (
                str(validation_details.get("expected_environment") or validation_details.get("environment") or "")
            )
            guardrail_report = GuardrailReport.from_sources(
                registry=registry,
                log_directory=log_directory,
                environment_hint=environment_hint or None,
            )
            guardrail_dir = Path("reports/guardrails")
            guardrail_path = guardrail_report.write_markdown(guardrail_dir)
            guardrail_payload = {
                "report_markdown": str(guardrail_path),
                "summary": guardrail_report.to_dict(),
                "log_directory": str(Path(log_directory).expanduser()),
            }
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się wygenerować raportu guardrail", exc_info=True)
        if guardrail_payload:
            report_payload["guardrails"] = guardrail_payload
        if paper_metrics_summary is not None:
            report_payload["paper_exchange_metrics"] = paper_metrics_summary
            threshold_warnings: list[str] = []
            for entrypoint_name, details in paper_metrics_summary.items():
                if entrypoint_name == "summary":
                    continue
                breaches = details.get("breaches") if isinstance(details, Mapping) else None
                if isinstance(breaches, Sequence):
                    for breach in breaches:
                        if not isinstance(breach, Mapping):
                            continue
                        metric_key = breach.get("metric") or breach.get("threshold")
                        threshold_key = breach.get("threshold")
                        observed = breach.get("observed")
                        limit = breach.get("limit")
                        comparator = breach.get("comparator", "")
                        threshold_warnings.append(
                            "[paper:{entry}] Metryka '{metric}' naruszyła próg '{threshold}' – wartość {observed} "
                            "nie spełnia warunku {comparator} {limit}.".format(
                                entry=entrypoint_name,
                                metric=metric_key,
                                threshold=threshold_key,
                                observed=observed,
                                comparator=comparator,
                                limit=limit,
                            )
                        )
                missing_metrics = details.get("missing_metrics") if isinstance(details, Mapping) else None
                if isinstance(missing_metrics, Sequence):
                    for metric_name in missing_metrics:
                        threshold_warnings.append(
                            f"[paper:{entrypoint_name}] Nie odnaleziono metryki '{metric_name}' w rejestrze telemetrii."
                        )
                invalid_thresholds = details.get("invalid_thresholds") if isinstance(details, Mapping) else None
                if isinstance(invalid_thresholds, Sequence):
                    for threshold_name in invalid_thresholds:
                        threshold_warnings.append(
                            f"[paper:{entrypoint_name}] Nieprawidłowa definicja progu '{threshold_name}' w konfiguracji."
                        )
            if threshold_warnings:
                report_payload["warnings"].extend(threshold_warnings)
        try:
            report_json = _write_report(args.report_dir, report_payload)
            report_payload["report_json"] = str(report_json)
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się zapisać raportu E2E", exc_info=True)

    return exit_code


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
