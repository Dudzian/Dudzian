#!/usr/bin/env python3
"""Eksportuje listę adapterów giełdowych do raportu benchmarkowego."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - konfiguracja ścieżki wykonywana raz
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - środowiska testowe mogą nie mieć packaging
    import packaging.version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback minimalny
    import types

    packaging_module = types.ModuleType("packaging")
    version_module = types.ModuleType("version")

    class Version(str):  # type: ignore
        def __new__(cls, value: str) -> "Version":
            return str.__new__(cls, value)

    class InvalidVersion(Exception):
        """Minimalna implementacja na potrzeby loadera konfiguracji."""

    version_module.Version = Version
    version_module.InvalidVersion = InvalidVersion
    packaging_module.version = version_module
    sys.modules.setdefault("packaging", packaging_module)
    sys.modules.setdefault("packaging.version", version_module)

from bot_core.config.loader import load_core_config

LongPollMetrics = dict[tuple[str, str, str], dict[str, Any]]

_DEFAULT_LONG_POLL_METRICS_PATH = "var/metrics/long_poll_snapshots.json"
_DEFAULT_LONG_POLL_TTL_MINUTES = 720.0
_MONITORED_LONG_POLL_ADAPTERS = {
    "deribit_futures",
    "bitmex_futures",
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stream_base_url(stream_settings: Mapping[str, Any], *, environment: str) -> str | None:
    candidates = (
        stream_settings.get(f"{environment}_base_url"),
        stream_settings.get("base_url"),
        stream_settings.get("live_base_url"),
        stream_settings.get("testnet_base_url"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _extract_margin_mode(
    adapter_settings: Mapping[str, Any],
    default_settings: Mapping[str, Any],
) -> str | None:
    candidate_sources: list[Mapping[str, Any]] = []
    native_entry = _as_mapping(adapter_settings.get("native_adapter"))
    candidate_sources.append(_as_mapping(native_entry.get("settings")))
    default_native = _as_mapping(_as_mapping(default_settings).get("native_adapter", {}))
    candidate_sources.append(_as_mapping(default_native.get("settings")))
    candidate_sources.append(_as_mapping(default_settings))

    for source in candidate_sources:
        for key in ("margin_mode", "marginMode", "margin_type", "marginType"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
        # niektóre adaptery zapisują hedge mode jako bool
        hedge_value = source.get("hedgeMode")
        if isinstance(hedge_value, bool):
            return "hedge" if hedge_value else "one-way"
    return None


def _extract_liquidation_feed(
    adapter_settings: Mapping[str, Any],
    stream_settings: Mapping[str, Any],
    environment: str,
) -> str | None:
    stream_entry = _as_mapping(adapter_settings.get("stream")) or stream_settings
    if not isinstance(stream_entry, Mapping):
        return None
    base_url = _stream_base_url(stream_entry, environment=environment)
    liquidation_path = stream_entry.get("liquidation_path")
    if not isinstance(liquidation_path, str) or not liquidation_path:
        liquidation_path = stream_entry.get("private_path") or stream_entry.get("public_path")
    if not isinstance(liquidation_path, str) or not liquidation_path:
        return base_url
    if base_url:
        return f"{base_url.rstrip('/')}{liquidation_path}" if liquidation_path.startswith("/") else f"{base_url}/{liquidation_path}"
    return liquidation_path


def _hypercare_checklist_status(live_readiness: Any) -> tuple[bool | None, str]:
    if live_readiness is None:
        return None, "not_configured"

    documents: Sequence[Any] = tuple(getattr(live_readiness, "documents", ()) or ())
    required: Sequence[str] = tuple(getattr(live_readiness, "required_documents", ()) or ())
    hypercare_required = any(doc_name == "hypercare_runbook" for doc_name in required)
    hypercare_doc = None
    for doc in documents:
        if getattr(doc, "name", "").lower() == "hypercare_runbook":
            hypercare_doc = doc
            break
    if hypercare_doc is None and not hypercare_required:
        return None, "not_required"
    if hypercare_doc is None:
        return False, "missing_document"
    if bool(getattr(hypercare_doc, "signed", False)) and getattr(hypercare_doc, "signature_path", None):
        return True, "signed"
    return False, "missing_signature"


def _missing_required_documents(live_readiness: Any) -> str:
    if live_readiness is None:
        return ""
    documents: dict[str, Any] = {
        getattr(doc, "name", ""): doc for doc in getattr(live_readiness, "documents", ()) or ()
    }
    missing: list[str] = []
    for required in getattr(live_readiness, "required_documents", ()) or ():
        entry = documents.get(required)
        if entry is None:
            missing.append(required)
            continue
        if not bool(getattr(entry, "signed", False)) or not getattr(entry, "signature_path", None):
            missing.append(required)
    return ",".join(sorted(missing))


def _push_dashboard_snapshot(
    report_path: Path,
    dashboard_dir: Path,
    endpoint: str | None,
) -> None:
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    target_path = dashboard_dir / report_path.name
    shutil.copy2(report_path, target_path)
    if endpoint:
        data = report_path.read_bytes()
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "text/csv"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.URLError as exc:  # pragma: no cover - zależy od środowiska CI
            raise RuntimeError(f"Nie udało się wypchnąć CSV do endpointu dashboardu: {exc}") from exc


def _futures_checklist_state(exchange: str, env_cfg: Any) -> tuple[str | None, bool | None]:
    normalized = exchange.strip().lower()
    if normalized not in {"deribit", "bitmex"}:
        return None, None
    readiness = getattr(env_cfg, "live_readiness", None)
    if readiness is None:
        return None, None
    checklist_id = getattr(readiness, "checklist_id", None)
    signed = bool(getattr(readiness, "signed", False))
    missing_docs = _missing_required_documents(readiness)
    ready = signed and not missing_docs
    return checklist_id, ready


def _hypercare_status_from_config(doc: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(doc, Mapping):
        return {
            "failover": "not_configured",
            "latency": "not_configured",
            "cost": "not_configured",
        }

    def _ready(required: Sequence[str | None]) -> bool:
        return all(bool(entry) for entry in required)

    resilience = doc.get("resilience") or {}
    failover = resilience.get("failover") or {}
    if _ready((failover.get("plan"), failover.get("signature"))):
        failover_status = "ready"
    elif failover.get("plan"):
        failover_status = "missing_signature"
    elif failover:
        failover_status = "missing_plan"
    else:
        failover_status = "not_configured"

    observability = doc.get("observability") or {}
    slo = observability.get("slo") or {}
    metrics_path = observability.get("metrics")
    if _ready((metrics_path, slo.get("signature"))):
        latency_status = "ready"
    elif metrics_path:
        latency_status = "missing_signature"
    elif slo:
        latency_status = "missing_metrics"
    else:
        latency_status = "not_configured"

    portfolio = doc.get("portfolio") or {}
    inputs = portfolio.get("inputs") or {}
    output = portfolio.get("output") or {}
    if inputs.get("portfolio_value") and output.get("summary"):
        cost_status = "ready"
    elif not inputs.get("portfolio_value"):
        cost_status = "missing_budget"
    elif not output.get("summary"):
        cost_status = "missing_summary"
    else:
        cost_status = "not_configured"

    return {
        "failover": failover_status,
        "latency": latency_status,
        "cost": cost_status,
    }


def _load_hypercare_status(path: str | None) -> dict[str, str]:
    if not path:
        return _hypercare_status_from_config(None)
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _hypercare_status_from_config(payload)


def _coerce_datetime(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        cleaned = text.replace("Z", "+00:00")
        try:
            parsed = _dt.datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)
    return None


def _load_long_poll_metrics(path: str | None) -> LongPollMetrics:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - diagnostyka wejścia
        raise RuntimeError(f"Niepoprawny plik metryk long-pollowych ({path}): {exc}") from exc

    default_timestamp = None
    entries: Sequence[Any]
    if isinstance(payload, Mapping):
        default_timestamp = _coerce_datetime(payload.get("collected_at"))
        entries_candidate = payload.get("snapshots") or payload.get("entries") or payload.get("data")
        if isinstance(entries_candidate, Sequence) and not isinstance(entries_candidate, (str, bytes, bytearray)):
            entries = entries_candidate
        else:
            entries = (payload,)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        entries = payload
    else:
        return {}

    fallback_timestamp = _dt.datetime.fromtimestamp(
        metrics_path.stat().st_mtime,
        tz=_dt.timezone.utc,
    )

    metrics: LongPollMetrics = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        labels = _as_mapping(entry.get("labels"))
        adapter = str(labels.get("adapter") or labels.get("exchange") or "").strip()
        scope = str(labels.get("scope") or labels.get("channel") or "").strip()
        environment = str(labels.get("environment") or labels.get("env") or "").strip()
        if not adapter or not scope or not environment:
            continue
        normalized_labels = {
            "adapter": adapter,
            "scope": scope,
            "environment": environment,
        }
        snapshot = {key: value for key, value in entry.items()}
        snapshot["labels"] = normalized_labels
        timestamp = _coerce_datetime(entry.get("collected_at")) or default_timestamp or fallback_timestamp
        snapshot["_collected_at"] = timestamp
        metrics[(adapter, scope, environment)] = snapshot
    return metrics


def _resolve_long_poll_snapshot(
    *,
    adapter_id: str | None,
    environment: str,
    metrics: LongPollMetrics,
) -> tuple[str | None, dict[str, Any] | None]:
    if not adapter_id:
        return None, None
    for scope in ("private", "public"):
        snapshot = metrics.get((adapter_id, scope, environment))
        if snapshot:
            return scope, snapshot
    return None, None


def _empty_long_poll_summary(*, status: str) -> dict[str, Any]:
    return {
        "long_poll_scope": None,
        "long_poll_latency_p95": None,
        "long_poll_delivery_lag_p95": None,
        "long_poll_http_errors": None,
        "long_poll_reconnect_success": None,
        "long_poll_reconnect_failure": None,
        "long_poll_snapshot_age_minutes": None,
        "long_poll_metrics_status": status,
    }


def _summarize_long_poll_metrics(
    *,
    adapter_id: str | None,
    environment: str,
    metrics: LongPollMetrics,
    ttl_minutes: float,
) -> dict[str, Any]:
    if adapter_id not in _MONITORED_LONG_POLL_ADAPTERS:
        return _empty_long_poll_summary(status="not_applicable")

    scope, snapshot = _resolve_long_poll_snapshot(
        adapter_id=adapter_id,
        environment=environment,
        metrics=metrics,
    )
    if snapshot is None:
        return _empty_long_poll_summary(status="missing")

    summary = _empty_long_poll_summary(status="unknown")
    summary["long_poll_scope"] = scope

    latency = _as_mapping(snapshot.get("requestLatency"))
    summary["long_poll_latency_p95"] = latency.get("p95") if latency else None
    delivery_lag = _as_mapping(snapshot.get("deliveryLag"))
    summary["long_poll_delivery_lag_p95"] = delivery_lag.get("p95") if delivery_lag else None

    http_errors = _as_mapping(snapshot.get("httpErrors"))
    if http_errors:
        total = http_errors.get("total")
        try:
            summary["long_poll_http_errors"] = int(total) if total is not None else None
        except (TypeError, ValueError):
            summary["long_poll_http_errors"] = None

    reconnects = _as_mapping(snapshot.get("reconnects"))
    if reconnects:
        for key in ("success", "failure"):
            value = reconnects.get(key)
            try:
                summary[f"long_poll_reconnect_{key}"] = int(value)
            except (TypeError, ValueError):
                summary[f"long_poll_reconnect_{key}"] = None

    collected_at = snapshot.get("_collected_at")
    if isinstance(collected_at, _dt.datetime):
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        age_minutes = max(0.0, (now - collected_at).total_seconds() / 60.0)
        summary["long_poll_snapshot_age_minutes"] = round(age_minutes, 2)
        if ttl_minutes <= 0:
            summary["long_poll_metrics_status"] = "fresh"
        else:
            summary["long_poll_metrics_status"] = "fresh" if age_minutes <= ttl_minutes else "stale"
    else:
        summary["long_poll_metrics_status"] = "unknown"

    return summary


def build_rows(
    config,
    hypercare_summary: Mapping[str, str] | None = None,
    *,
    long_poll_metrics: LongPollMetrics | None = None,
    long_poll_ttl_minutes: float | None = None,
) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    rows: list[dict[str, Any]] = []
    metrics = dict(long_poll_metrics or {})
    ttl_minutes = (
        float(long_poll_ttl_minutes)
        if isinstance(long_poll_ttl_minutes, (int, float)) and math.isfinite(long_poll_ttl_minutes)
        else _DEFAULT_LONG_POLL_TTL_MINUTES
    )

    for exchange, profiles in config.exchange_accounts.items():
        for profile_name, account in profiles.items():
            env_name = account.environment
            env_cfg = config.environments.get(env_name)
            if env_cfg is None:
                continue

            mode_key = env_cfg.exchange.split("_")[-1]
            adapter_entry = None
            adapter_class = "ccxt"
            supports_testnet = False
            for key, entry in config.exchange_adapters.get(exchange, {}).items():
                key_name = getattr(key, "value", str(key))
                if key_name == mode_key:
                    adapter_entry = entry
                    adapter_class = entry.class_path
                    supports_testnet = bool(entry.supports_testnet)
                    break

            default_settings = adapter_entry.default_settings if adapter_entry else {}
            stream_settings = default_settings.get("stream", {}) if isinstance(default_settings, Mapping) else {}
            retry_policy = default_settings.get("retry_policy", {}) if isinstance(default_settings, Mapping) else {}
            adapter_settings = env_cfg.adapter_settings if isinstance(env_cfg.adapter_settings, Mapping) else {}
            futures_margin_mode = _extract_margin_mode(adapter_settings, default_settings)
            liquidation_feed = _extract_liquidation_feed(
                adapter_settings,
                stream_settings,
                environment=env_cfg.environment.value,
            )
            hypercare_signed, hypercare_status = _hypercare_checklist_status(
                getattr(env_cfg, "live_readiness", None)
            )
            missing_docs = _missing_required_documents(getattr(env_cfg, "live_readiness", None))
            checklist_id, checklist_ready = _futures_checklist_state(exchange, env_cfg)

            live_readiness = getattr(env_cfg, "live_readiness", None)
            readiness_signed = bool(getattr(live_readiness, "signed", False))
            readiness_signed_by: list[str] = []
            signed_by_attr = getattr(live_readiness, "signed_by", None)
            if isinstance(signed_by_attr, (list, tuple)):
                readiness_signed_by = [str(entry) for entry in signed_by_attr]
            if isinstance(live_readiness, Mapping):  # pragma: no cover - kompatybilność starych schematów
                readiness_signed = bool(live_readiness.get("signed", False))
                readiness_signed_by = [str(entry) for entry in live_readiness.get("signed_by", ())]

            adapter_id = getattr(env_cfg, "exchange", exchange)
            long_poll_summary = _summarize_long_poll_metrics(
                adapter_id=str(adapter_id) if adapter_id else None,
                environment=env_cfg.environment.value,
                metrics=metrics,
                ttl_minutes=ttl_minutes,
            )

            rows.append(
                {
                    "exchange": exchange,
                    "profile": profile_name,
                    "mode": mode_key,
                    "environment": env_cfg.environment.value,
                    "adapter": adapter_class,
                    "supports_testnet": supports_testnet,
                    "stream_base_url": _stream_base_url(
                        stream_settings,
                        environment=env_cfg.environment.value,
                    ),
                    "retry_max_attempts": retry_policy.get("max_attempts"),
                    "retry_max_delay": retry_policy.get("max_delay"),
                    "live_readiness_signed": readiness_signed,
                    "live_readiness_signed_by": ",".join(readiness_signed_by),
                    "futures_margin_mode": futures_margin_mode,
                    "liquidation_feed": liquidation_feed,
                    "hypercare_checklist_signed": hypercare_signed,
                    "hypercare_checklist_status": hypercare_status,
                    "missing_required_documents": missing_docs,
                    "futures_checklist_id": checklist_id,
                    "futures_checklist_ready": checklist_ready,
                    "hypercare_failover_status": hypercare_summary.get("failover") if hypercare_summary else None,
                    "hypercare_latency_status": hypercare_summary.get("latency") if hypercare_summary else None,
                    "hypercare_cost_status": hypercare_summary.get("cost") if hypercare_summary else None,
                    **long_poll_summary,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku config/core.yaml")
    parser.add_argument("--output", help="Plik CSV do zapisania raportu (""-"" = stdout)")
    parser.add_argument(
        "--report-date",
        default=_dt.date.today().isoformat(),
        help="Data raportu używana do nazwy pliku (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/exchanges",
        help="Domyślny katalog raportów (jeśli nie podano --output)",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="reports/exchanges/signal_quality",
        help="Katalog, do którego kopiowany jest snapshot dashboardu",
    )
    parser.add_argument(
        "--push-dashboard",
        action="store_true",
        help="Wypchnij snapshot do dashboardu po wygenerowaniu CSV",
    )
    parser.add_argument(
        "--dashboard-endpoint",
        help="Opcjonalny endpoint HTTP (np. Prometheus/Grafana datasource) do publikacji CSV",
    )
    parser.add_argument(
        "--hypercare-config",
        help="Ścieżka do pliku config/stage6/hypercare.yaml z metadanymi failover/latency/cost.",
    )
    parser.add_argument(
        "--long-poll-metrics",
        default=_DEFAULT_LONG_POLL_METRICS_PATH,
        help="Ścieżka do pliku JSON z metrykami long-polla (domyślnie var/metrics/long_poll_snapshots.json).",
    )
    parser.add_argument(
        "--long-poll-ttl-minutes",
        type=float,
        default=_DEFAULT_LONG_POLL_TTL_MINUTES,
        help="Maksymalny wiek metryk long-polla (w minutach) zanim status stanie się 'stale'.",
    )
    args = parser.parse_args(argv)

    config = load_core_config(Path(args.config))
    hypercare_summary = _load_hypercare_status(args.hypercare_config)
    long_poll_metrics = _load_long_poll_metrics(args.long_poll_metrics)
    rows = build_rows(
        config,
        hypercare_summary=hypercare_summary,
        long_poll_metrics=long_poll_metrics,
        long_poll_ttl_minutes=args.long_poll_ttl_minutes,
    )

    fieldnames = [
        "exchange",
        "profile",
        "mode",
        "environment",
        "adapter",
        "supports_testnet",
        "stream_base_url",
        "retry_max_attempts",
        "retry_max_delay",
        "live_readiness_signed",
        "live_readiness_signed_by",
        "futures_margin_mode",
        "liquidation_feed",
        "hypercare_checklist_signed",
        "hypercare_checklist_status",
        "missing_required_documents",
        "futures_checklist_id",
        "futures_checklist_ready",
        "hypercare_failover_status",
        "hypercare_latency_status",
        "hypercare_cost_status",
        "long_poll_scope",
        "long_poll_latency_p95",
        "long_poll_delivery_lag_p95",
        "long_poll_http_errors",
        "long_poll_reconnect_success",
        "long_poll_reconnect_failure",
        "long_poll_snapshot_age_minutes",
        "long_poll_metrics_status",
    ]

    close_handle = False
    output_path: Path | None = None
    if args.output:
        if args.output.strip() == "-":
            handle = sys.stdout
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = output_path.open("w", newline="", encoding="utf-8")
            close_handle = True
    else:
        output_path = Path(args.report_dir) / f"{args.report_date}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8")
        close_handle = True

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close_handle:
            handle.close()

    if args.push_dashboard:
        if output_path is None:
            raise SystemError("Eksport na dashboard wymaga zapisu do pliku (ustaw --output lub katalog raportów)")
        dashboard_dir = Path(args.dashboard_dir)
        _push_dashboard_snapshot(output_path, dashboard_dir, args.dashboard_endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

