"""Generuje override alertów Stage6 na podstawie raportu SLO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.observability import (  # noqa: E402 - import po modyfikacji sys.path
    AlertOverrideBuilder,
    AlertOverrideManager,
    SLODefinition,
    SLOStatus,
    load_overrides_document,
)
from bot_core.security.signing import build_hmac_signature  # noqa: E402


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/audit/observability") / f"alert_overrides_{timestamp}.json"


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_definitions(data: Mapping[str, Any]) -> dict[str, SLODefinition]:
    definitions: dict[str, SLODefinition] = {}
    entries = data.get("definitions")
    if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            indicator = entry.get("indicator")
            target = entry.get("target")
            if indicator in (None, "") or target is None:
                continue
            name = str(entry.get("name") or indicator)
            definition = SLODefinition(
                name=name,
                indicator=str(indicator),
                target=float(target),
                comparison=str(entry.get("comparison", ">=")),
                warning_threshold=(
                    float(entry.get("warning_threshold"))
                    if entry.get("warning_threshold") is not None
                    else None
                ),
                severity=str(entry.get("severity", "critical")),
                description=(
                    str(entry.get("description")) if entry.get("description") not in (None, "") else None
                ),
                tags=tuple(str(tag) for tag in (entry.get("tags") or ())),
            )
            definitions[name] = definition
    return definitions


def _to_float_mapping(data: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(data, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _parse_statuses(
    data: Mapping[str, Any],
    definitions: Mapping[str, SLODefinition],
) -> dict[str, SLOStatus]:
    results = data.get("results")
    statuses: dict[str, SLOStatus] = {}
    items: Iterable[tuple[str, Mapping[str, Any]]] = []
    if isinstance(results, Mapping):
        items = [
            (str(name), entry)
            for name, entry in results.items()
            if isinstance(entry, Mapping)
        ]
    elif isinstance(results, Iterable) and not isinstance(results, (str, bytes)):
        temp: list[tuple[str, Mapping[str, Any]]] = []
        for entry in results:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or entry.get("indicator") or len(temp))
            temp.append((name, entry))
        items = temp
    for name, entry in items:
        definition = definitions.get(name)
        indicator = entry.get("indicator") or (definition.indicator if definition else name)
        target_value = entry.get("target")
        if target_value is None and definition is not None:
            target_value = definition.target
        target = float(target_value) if target_value is not None else 0.0
        warning_threshold = entry.get("warning_threshold")
        warning = float(warning_threshold) if warning_threshold is not None else (
            definition.warning_threshold if definition else None
        )
        statuses[name] = SLOStatus(
            name=name,
            indicator=str(indicator),
            value=_as_float(entry.get("value")),
            target=target,
            comparison=str(entry.get("comparison", definition.comparison if definition else ">=")),
            status=str(entry.get("status", "unknown")),
            severity=str(entry.get("severity", definition.severity if definition else "warning")),
            warning_threshold=warning,
            error_budget_pct=_as_float(entry.get("error_budget_pct")),
            window_start=_parse_dt(entry.get("window_start")),
            window_end=_parse_dt(entry.get("window_end")),
            sample_size=int(entry.get("sample_size") or 0),
            reason=(
                str(entry.get("reason"))
                if entry.get("reason") not in (None, "")
                else None
            ),
            metadata=_to_float_mapping(entry.get("metadata")),
        )
    _merge_composite_statuses(data, statuses)
    return statuses


def _merge_composite_statuses(
    data: Mapping[str, Any],
    statuses: MutableMapping[str, SLOStatus],
) -> None:
    composites = data.get("composites")
    if not isinstance(composites, Mapping):
        return
    entries = composites.get("results")
    if isinstance(entries, Mapping):
        iterable = entries.items()
    elif isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
        iterable = []
        temp: list[tuple[str, Mapping[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or entry.get("indicator") or len(temp))
            temp.append((name, entry))
        iterable = temp
    else:
        iterable = []

    for name, entry in iterable:
        if not isinstance(entry, Mapping):
            continue
        counts = entry.get("counts") if isinstance(entry.get("counts"), Mapping) else {}
        if not isinstance(counts, Mapping):
            counts = {}
        objectives = entry.get("objectives")
        if isinstance(objectives, Iterable) and not isinstance(objectives, (str, bytes)):
            total = len(list(objectives))
        else:
            total = 0
        if not total:
            total = sum(
                int(value)
                for key, value in counts.items()
                if key in {"ok", "warning", "breach", "unknown"}
                and isinstance(value, (int, float))
            )
        total = int(total)
        ok_count = int(counts.get("ok", 0)) if isinstance(counts, Mapping) else 0
        value_ratio: float | None = None
        if total > 0:
            value_ratio = ok_count / float(total)
        metadata = _to_float_mapping(entry.get("metadata"))
        if isinstance(counts, Mapping):
            for key in ("ok", "warning", "breach", "unknown"):
                val = counts.get(key)
                if isinstance(val, (int, float)):
                    metadata.setdefault(f"counts_{key}", float(val))
        if value_ratio is not None:
            metadata.setdefault("ok_ratio", float(value_ratio))
        metadata.setdefault("composite", 1.0)
        statuses[name] = SLOStatus(
            name=name,
            indicator=str(entry.get("indicator") or name),
            value=value_ratio,
            target=1.0 if value_ratio is not None else 0.0,
            comparison=">=",
            status=str(entry.get("status", "unknown")),
            severity=str(entry.get("severity", "warning")),
            warning_threshold=None,
            error_budget_pct=None,
            window_start=_parse_dt(entry.get("window_start")),
            window_end=_parse_dt(entry.get("window_end")),
            sample_size=total,
            reason=(
                str(entry.get("reason"))
                if entry.get("reason") not in (None, "")
                else None
            ),
            metadata=metadata,
        )


def _load_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key and args.signing_key_path:
        raise ValueError("Nie można podać jednocześnie klucza HMAC i pliku z kluczem")
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_env:
        value = os.environ.get(args.signing_key_env)
        if value:
            return value.encode("utf-8"), args.signing_key_id
    if args.signing_key_path:
        path = Path(args.signing_key_path)
        if not path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {path}")
        return path.read_bytes().strip(), args.signing_key_id
    return None, None


def _parse_severity_map(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not values:
        return mapping
    for item in values:
        if "=" not in item:
            raise ValueError("Mapa severities musi mieć format status=severity")
        status, severity = item.split("=", 1)
        status = status.strip()
        severity = severity.strip()
        if not status or not severity:
            raise ValueError("Status i severity w mapie nie mogą być puste")
        mapping[status] = severity
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generuje override alertów Stage6 korzystając z raportu SLO",
    )
    parser.add_argument("--slo-report", required=True, help="Plik JSON z raportem SLO")
    parser.add_argument(
        "--output",
        help="Plik wynikowy z override'ami (domyślnie var/audit/observability/...)",
    )
    parser.add_argument(
        "--expires-in",
        type=int,
        default=120,
        help="Czas ważności override'ów w minutach (0 = bezterminowo)",
    )
    parser.add_argument(
        "--severity",
        action="append",
        help="Mapowanie statusu na severity (np. breach=critical)",
    )
    parser.add_argument(
        "--requested-by",
        default="PortfolioGovernor",
        help="Identyfikator zgłaszającego override",
    )
    parser.add_argument(
        "--source",
        default="slo_monitor",
        help="Źródło override'u zapisywane w raporcie",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Dodatkowe tagi dodawane do każdego override'u",
    )
    parser.add_argument(
        "--existing",
        help="Opcjonalny plik z istniejącymi override'ami do połączenia",
    )
    parser.add_argument("--skip-warnings", action="store_true", help="Pomiń statusy warning")
    parser.add_argument("--pretty", action="store_true", help="Formatuj JSON z wcięciami")
    parser.add_argument("--signature", help="Plik podpisu HMAC")
    parser.add_argument("--signing-key", help="Klucz HMAC podany wprost")
    parser.add_argument("--signing-key-env", help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    parser.add_argument("--signing-key-path", help="Ścieżka do pliku z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza w podpisie")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        slo_path = Path(args.slo_report)
        slo_data = json.loads(slo_path.read_text(encoding="utf-8"))
        definitions = _parse_definitions(slo_data)
        statuses = _parse_statuses(slo_data, definitions)
        builder = AlertOverrideBuilder(definitions)
        ttl_minutes = max(0, int(args.expires_in))
        ttl = timedelta(minutes=ttl_minutes)
        overrides = builder.build_from_statuses(
            statuses,
            include_warning=not args.skip_warnings,
            default_ttl=ttl,
            severity_overrides=_parse_severity_map(args.severity),
            requested_by=args.requested_by,
            source=args.source,
            extra_tags=[str(tag) for tag in (args.tag or [])],
        )
        manager = AlertOverrideManager()
        if args.existing:
            existing_path = Path(args.existing)
            existing_data = json.loads(existing_path.read_text(encoding="utf-8"))
            existing_overrides = load_overrides_document(existing_data)
            manager.extend(existing_overrides)
            manager.prune_expired()
        manager.merge(overrides)
        payload = manager.to_payload()
        output_path = Path(args.output) if args.output else _default_output()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            if args.pretty:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        key, key_id = _load_hmac_key(args)
        if key:
            signature_payload = build_hmac_signature(payload, key=key, key_id=key_id)
            signature_path = Path(args.signature) if args.signature else output_path.with_suffix(".sig")
            signature_path.parent.mkdir(parents=True, exist_ok=True)
            with signature_path.open("w", encoding="utf-8") as handle:
                json.dump(signature_payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            print(
                f"Zapisano override'y alertów do {output_path} wraz z podpisem {signature_path}",
            )
        else:
            print(f"Zapisano override'y alertów do {output_path} (bez podpisu HMAC)")
    except Exception as exc:  # noqa: BLE001 - komunikat dla operatora
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run())

