#!/usr/bin/env python3
"""Eksportuje listę adapterów giełdowych do raportu benchmarkowego."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def build_rows(config) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    rows: list[dict[str, Any]] = []

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

            live_readiness = getattr(env_cfg, "live_readiness", None)
            readiness_signed = bool(getattr(live_readiness, "signed", False))
            readiness_signed_by: list[str] = []
            signed_by_attr = getattr(live_readiness, "signed_by", None)
            if isinstance(signed_by_attr, (list, tuple)):
                readiness_signed_by = [str(entry) for entry in signed_by_attr]
            if isinstance(live_readiness, Mapping):  # pragma: no cover - kompatybilność starych schematów
                readiness_signed = bool(live_readiness.get("signed", False))
                readiness_signed_by = [str(entry) for entry in live_readiness.get("signed_by", ())]

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
    args = parser.parse_args(argv)

    config = load_core_config(Path(args.config))
    rows = build_rows(config)

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

