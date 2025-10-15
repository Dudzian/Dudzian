"""CLI do monitorowania SLO dla Etapu 5 (observability & compliance)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config
from bot_core.config.models import ObservabilityConfig, SLOThresholdConfig
from bot_core.security import build_hmac_signature


@dataclass(frozen=True)
class MetricSample:
    metric: str
    value: float
    timestamp: datetime
    labels: Mapping[str, str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Plik JSONL z metrykami (można podać wiele razy)",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "core.yaml"),
        help="Ścieżka do pliku konfiguracji core.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="var/audit/slo",
        help="Katalog docelowy raportu (domyślnie var/audit/slo)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Opcjonalna nazwa bazowa plików raportu (bez rozszerzeń)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Dodatkowe metadane w formacie klucz=wartość",
    )
    parser.add_argument(
        "--signing-key-path",
        dest="signing_key_path",
        help="Ścieżka do klucza HMAC podpisującego raport",
    )
    parser.add_argument(
        "--signing-key-id",
        dest="signing_key_id",
        default=None,
        help="Identyfikator klucza podpisującego",
    )
    return parser


def _parse_metadata(entries: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Metadana musi mieć format klucz=wartość: {item}")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _parse_timestamp(raw: object, *, path: Path, line_number: int) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Brak znacznika czasu w {path}:{line_number}")
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # noqa: PERF203 - chcemy pełny komunikat
        raise ValueError(f"Nieprawidłowy format czasu w {path}:{line_number}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _load_samples(paths: Sequence[Path]) -> list[MetricSample]:
    samples: list[MetricSample] = []
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
            value_raw = payload.get("value")
            try:
                value = float(value_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Nieprawidłowa wartość metryki dla {metric} w {path}:{line_number}"
                ) from exc
            timestamp = _parse_timestamp(payload.get("timestamp"), path=path, line_number=line_number)
            labels_raw = payload.get("labels") or {}
            if not isinstance(labels_raw, Mapping):
                raise ValueError(f"Pole labels musi być mapą w {path}:{line_number}")
            labels = {str(key): str(value) for key, value in labels_raw.items()}
            samples.append(
                MetricSample(metric=metric, value=value, timestamp=timestamp, labels=labels)
            )
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


def _evaluate_slo(
    definition: SLOThresholdConfig,
    samples: Sequence[MetricSample],
    *,
    now: datetime,
) -> dict[str, object]:
    window_start = now - timedelta(minutes=definition.window_minutes)
    filtered = [
        sample
        for sample in samples
        if sample.metric == definition.metric
        and sample.timestamp >= window_start
        and _labels_match(sample.labels, definition.label_filters)
    ]
    filtered.sort(key=lambda sample: sample.timestamp)
    observed_value: float | None = None
    status = "insufficient_data"
    violations = 0
    if len(filtered) >= definition.min_samples and filtered:
        values = [sample.value for sample in filtered]
        observed_value = _aggregate(values, definition.aggregation)
        status = "pass" if _compare(observed_value, definition.comparator, definition.objective) else "fail"
        violations = sum(
            0
            if _compare(sample.value, definition.comparator, definition.objective)
            else 1
            for sample in filtered
        )
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


def _read_signing_key(path: Path) -> bytes:
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


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    core_config = load_core_config(config_path)
    observability: ObservabilityConfig | None = getattr(core_config, "observability", None)
    if observability is None or not observability.slo:
        raise ValueError("Konfiguracja nie zawiera definicji observability.slo")

    slo_definitions = list(observability.slo.values())
    metric_paths = [Path(item) for item in args.metrics]
    samples = _load_samples(metric_paths)
    now = datetime.now(timezone.utc)
    metadata = _parse_metadata(args.metadata)
    metadata.setdefault("source", "slo_monitor")
    metadata.setdefault("inputs", [str(path) for path in metric_paths])

    results = [_evaluate_slo(definition, samples, now=now) for definition in slo_definitions]
    report = {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path),
        "metadata": metadata,
        "results": results,
    }

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.basename:
        report_path = output_root / f"{args.basename}.json"
    else:
        report_path = output_root / f"slo_report_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.signing_key_path:
        signing_key = _read_signing_key(Path(args.signing_key_path))
        signature = build_hmac_signature(report, key=signing_key, key_id=args.signing_key_id)
        sig_path = report_path.with_suffix(report_path.suffix + ".sig")
        sig_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Raport SLO zapisany w {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())
