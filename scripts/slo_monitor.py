#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLO Monitor – merged CLI (HEAD + main)

Subcommands:
  - evaluate : Stage6-style — definicje SLO (YAML/JSON) + pomiary (JSON) → raport + (opcjonalnie) CSV i podpis HMAC
  - scan     : Stage5-style — JSONL z metrykami + progi z core.yaml → raport + (opcjonalnie) podpis HMAC
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence, Optional, List

# ────────────────────────────────────────────────────────────────────────────────
# HEAD-style helpers (evaluate)
# ────────────────────────────────────────────────────────────────────────────────
from bot_core.observability import (
    evaluate_slo,
    load_slo_definitions,
    load_slo_measurements,
)

from bot_core.security.signing import build_hmac_signature

# ────────────────────────────────────────────────────────────────────────────────
# main-style helpers (scan)
# ────────────────────────────────────────────────────────────────────────────────
from bot_core.config import load_core_config
from bot_core.config.models import ObservabilityConfig, SLOThresholdConfig


# ==============================================================================
# Common utilities
# ==============================================================================
def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ==============================================================================
# Subcommand: evaluate (HEAD)
# ==============================================================================
def _default_output_path() -> Path:
    return Path("var/audit/observability") / f"slo_report_{_now_ts()}.json"


def _load_signing_key_evaluate(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    # priority: --signing-key > --signing-key-env > --signing-key-path
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_env:
        val = os.environ.get(args.signing_key_env)
        if val:
            return val.encode("utf-8"), args.signing_key_id
    if args.signing_key_path:
        p = Path(args.signing_key_path)
        if p.exists():
            return p.read_bytes().strip(), args.signing_key_id
    return None, None


def _handle_evaluate(args: argparse.Namespace) -> int:
    definitions_path = Path(args.definitions)
    metrics_path = Path(args.metrics)

    definitions, composites = load_slo_definitions(definitions_path)
    if not definitions:
        print("Brak prawidłowych definicji SLO", file=sys.stderr)
        return 2

    measurements = load_slo_measurements(metrics_path)
    report = evaluate_slo(definitions, measurements, composites=composites)
    payload = report.to_payload()

    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(output_path, pretty=args.pretty)

    csv_path: Optional[Path] = None
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        report.write_csv(csv_path)

    key, key_id = _load_signing_key_evaluate(args)
    signature_path = Path(args.signature) if args.signature else output_path.with_suffix(".sig")
    if key:
        sig = build_hmac_signature(payload, key=key, key_id=key_id)
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(json.dumps(sig, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    msg = f"Zapisano raport SLO do {output_path}" + (" wraz z podpisem " + str(signature_path) if key else " (bez podpisu HMAC)")
    if csv_path:
        msg += f"; raport CSV: {csv_path}"
    print(msg)
    return 0


def _build_evaluate_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "evaluate",
        help="Oblicz SLO z definicji i pomiarów (Stage6-style).",
        description="Wczytuje definicje SLO (YAML/JSON) i pomiary (JSON), generuje raport JSON (+ opcjonalnie CSV) i podpis HMAC."
    )
    p.add_argument("--definitions", required=True, help="Plik z definicjami SLO (YAML/JSON)")
    p.add_argument("--metrics", required=True, help="Plik z pomiarami metryk (JSON)")
    p.add_argument("--output", help="Ścieżka raportu JSON (domyślnie var/audit/observability/slo_report_*.json)")
    p.add_argument("--output-csv", help="Ścieżka raportu CSV (opcjonalnie)")
    p.add_argument("--pretty", action="store_true", help="Formatowanie JSON z wcięciami")
    p.add_argument("--signature", help="Ścieżka pliku z podpisem HMAC (domyślnie <output>.sig)")
    p.add_argument("--signing-key", help="Klucz HMAC (inline)")
    p.add_argument("--signing-key-env", help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    p.add_argument("--signing-key-path", help="Plik zawierający klucz HMAC")
    p.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    p.set_defaults(_handler=_handle_evaluate)
    return p


# ==============================================================================
# Subcommand: scan (main)
# ==============================================================================
@dataclass(frozen=True)
class MetricSample:
    metric: str
    value: float
    timestamp: datetime
    labels: Mapping[str, str]


def _parse_metadata(entries: Iterable[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Metadana musi mieć format klucz=wartość: {item}")
        k, v = item.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def _parse_timestamp(raw: object, *, path: Path, line_number: int) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Brak znacznika czasu w {path}:{line_number}")
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"Nieprawidłowy format czasu w {path}:{line_number}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _load_samples(paths: Sequence[Path]) -> List[MetricSample]:
    samples: List[MetricSample] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            raise ValueError(f"Plik metryk nie może być katalogiem: {path}")
        contents = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Nieprawidłowy JSON w {path}:{line_number}") from exc
            metric = str(payload.get("metric", "")).strip()
            if not metric:
                raise ValueError(f"Brak pola metric w {path}:{line_number}")
            try:
                value = float(payload.get("value"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Nieprawidłowa wartość dla {metric} w {path}:{line_number}") from exc
            timestamp = _parse_timestamp(payload.get("timestamp"), path=path, line_number=line_number)
            labels_raw = payload.get("labels") or {}
            if not isinstance(labels_raw, Mapping):
                raise ValueError(f"Pole labels musi być mapą w {path}:{line_number}")
            labels = {str(k): str(v) for k, v in labels_raw.items()}
            samples.append(MetricSample(metric=metric, value=value, timestamp=timestamp, labels=labels))
    return samples


def _labels_match(sample_labels: Mapping[str, str], required: Mapping[str, str]) -> bool:
    for key, value in required.items():
        if sample_labels.get(key) != value:
            return False
    return True


def _aggregate(values: Sequence[float], mode: str) -> float:
    if not values:
        raise ValueError("Brak wartości do agregacji")
    mode = mode.lower()
    if mode in {"average", "avg"}:
        return float(mean(values))
    if mode == "max":
        return float(max(values))
    if mode == "min":
        return float(min(values))
    if mode == "p95":
        ordered = sorted(values)
        index = max(0, int(round(0.95 * (len(ordered) - 1))))
        return float(ordered[index])
    raise ValueError(f"Nieobsługiwany tryb agregacji: {mode}")


def _compare(value: float, comparator: str, objective: float) -> bool:
    if comparator == "<=":
        return value <= objective
    if comparator == ">=":
        return value >= objective
    if comparator == "<":
        return value < objective
    if comparator == ">":
        return value > objective
    raise ValueError(f"Nieobsługiwany komparator: {comparator}")


def _evaluate_slo_main(
    definition: SLOThresholdConfig,
    samples: Sequence[MetricSample],
    *,
    now: datetime,
) -> dict[str, object]:
    window_start = now - timedelta(minutes=definition.window_minutes)
    filtered = [
        s for s in samples
        if s.metric == definition.metric
        and s.timestamp >= window_start
        and _labels_match(s.labels, definition.label_filters)
    ]
    filtered.sort(key=lambda s: s.timestamp)
    observed_value: float | None = None
    status = "insufficient_data"
    violations = 0
    if len(filtered) >= definition.min_samples and filtered:
        values = [s.value for s in filtered]
        observed_value = _aggregate(values, definition.aggregation)
        status = "pass" if _compare(observed_value, definition.comparator, definition.objective) else "fail"
        violations = sum(0 if _compare(s.value, definition.comparator, definition.objective) else 1 for s in filtered)
    return {
        "name": definition.name,
        "metric": definition.metric,
        "objective": definition.objective,
        "comparator": definition.comparator,
        "aggregation": definition.aggregation,
        "window_minutes": definition.window_minutes,
        "labels": dict(definition.label_filters),
        "samples": len(filtered),
        "min_samples": definition.min_samples,
        "violations": violations,
        "status": status,
        "observed": observed_value,
        "window_start": window_start.isoformat().replace("+00:00", "Z"),
        "window_end": now.isoformat().replace("+00:00", "Z"),
    }


def _read_signing_key_path(path: Path) -> bytes:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        raise ValueError("Ścieżka klucza nie może być katalogiem")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    data = resolved.read_bytes()
    if len(data) < 32:
        raise ValueError("Klucz HMAC musi mieć co najmniej 32 bajty")
    if resolved.is_symlink():
        raise ValueError("Ścieżka klucza nie może być symlinkiem")
    if os.name != "nt" and resolved.stat().st_mode & 0o077:
        raise ValueError("Plik klucza powinien mieć uprawnienia 600")
    return data


def _handle_scan(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    core_config = load_core_config(config_path)
    observability: ObservabilityConfig | None = getattr(core_config, "observability", None)
    if observability is None or not observability.slo:
        print("Konfiguracja nie zawiera definicji observability.slo", file=sys.stderr)
        return 2

    slo_definitions = list(observability.slo.values())
    metric_paths = [Path(item) for item in args.metrics]
    samples = _load_samples(metric_paths)
    now = datetime.now(timezone.utc)
    metadata = _parse_metadata(args.metadata)
    metadata.setdefault("source", "slo_monitor")
    metadata.setdefault("inputs", [str(path) for path in metric_paths])

    results = [_evaluate_slo_main(defn, samples, now=now) for defn in slo_definitions]
    report = {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path),
        "metadata": metadata,
        "results": results,
    }

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / (f"{args.basename}.json" if args.basename else f"slo_report_{_now_ts()}.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.signing_key_path:
        signing_key = _read_signing_key_path(Path(args.signing_key_path))
        signature = build_hmac_signature(report, key=signing_key, key_id=args.signing_key_id)
        sig_path = report_path.with_suffix(report_path.suffix + ".sig")
        sig_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Raport SLO zapisany w {report_path}")
    return 0


def _build_scan_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "scan",
        help="Skanuj JSONL z metrykami i oceń SLO z core.yaml (Stage5-style).",
        description="Wczytuje JSONL z metrykami, bierze progi SLO z config/core.yaml i generuje raport JSON + (opcjonalny) podpis."
    )
    p.add_argument("--metrics", action="append", required=True, help="Plik JSONL z metrykami (można podać wiele razy)")
    p.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    p.add_argument("--output-dir", default="var/audit/slo", help="Katalog docelowy raportu (domyślnie var/audit/slo)")
    p.add_argument("--basename", default=None, help="Opcjonalna nazwa bazowa plików (bez rozszerzeń)")
    p.add_argument("--metadata", action="append", default=[], help="Dodatkowe metadane w formacie klucz=wartość")
    p.add_argument("--signing-key-path", dest="signing_key_path", help="Ścieżka do klucza HMAC podpisującego raport")
    p.add_argument("--signing-key-id", dest="signing_key_id", default=None, help="Identyfikator klucza podpisującego")
    p.set_defaults(_handler=_handle_scan)
    return p


# ==============================================================================
# Root parser & entrypoint
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLO Monitor – łączy tryby evaluate (Stage6) i scan (Stage5)."
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{evaluate|scan}", required=True)
    _build_evaluate_parser(sub)
    _build_scan_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
