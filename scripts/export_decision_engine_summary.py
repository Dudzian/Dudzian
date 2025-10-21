"""Eksportuje podsumowanie jakości Decision Engine na podstawie dziennika decyzji."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.decision import summarize_evaluation_payloads
from bot_core.decision.utils import coerce_float


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True, help="Ścieżka do katalogu lub pliku JSONL z decyzjami")
    parser.add_argument("--output", type=Path, required=True, help="Plik wyjściowy z podsumowaniem")
    parser.add_argument("--pattern", default="*.jsonl", help="Wzorzec plików w katalogu dziennika (domyślnie *.jsonl)")
    parser.add_argument("--environment")
    parser.add_argument("--portfolio")
    parser.add_argument("--risk-profile")
    parser.add_argument("--strategy")
    parser.add_argument("--schedule")
    parser.add_argument("--symbol")
    parser.add_argument("--history-limit", type=int, default=512, help="Maksymalna liczba ewaluacji branych do podsumowania")
    parser.add_argument("--include-history", action="store_true", help="Dołącz szczegóły ostatnich ewaluacji")
    parser.add_argument("--history-size", type=int, default=50, help="Liczba rekordów historii dołączenia przy --include-history")
    parser.add_argument("--pretty", action="store_true", help="Formatuj JSON z wcięciami")
    parser.add_argument("--require-evaluations", action="store_true", help="Zwróć kod wyjścia 2, gdy nie znaleziono żadnych ewaluacji")
    return parser.parse_args(argv)
def _split_field(value: object) -> Sequence[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(part for part in value.split(";") if part)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def _parse_json_object(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            return {str(k): v for k, v in payload.items()}
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _iter_ledger_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Ścieżka dziennika nie istnieje: {path}")
    yield from sorted(path.glob(pattern))


def _load_events(path: Path, pattern: str) -> Iterable[Mapping[str, object]]:
    for file_path in _iter_ledger_files(path, pattern):
        if not file_path.is_file():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, Mapping):
                        yield event
        except OSError:
            continue


def _event_matches_filters(event: Mapping[str, object], args: argparse.Namespace) -> bool:
    filters = {
        "environment": args.environment,
        "portfolio": args.portfolio,
        "risk_profile": args.risk_profile,
        "strategy": args.strategy,
        "schedule": args.schedule,
        "symbol": args.symbol,
    }
    for field, expected in filters.items():
        if expected is None:
            continue
        value = event.get(field)
        if value is None:
            return False
        if str(value) != str(expected):
            return False
    return True


def _event_to_payload(event: Mapping[str, object]) -> tuple[datetime | None, Mapping[str, object]] | None:
    if str(event.get("event")) != "decision_evaluation":
        return None

    status = str(event.get("status", "")).lower()
    accepted = status == "accepted"
    reasons = list(_split_field(event.get("decision_reasons")))
    risk_flags = list(_split_field(event.get("risk_flags")))
    stress_failures = list(_split_field(event.get("stress_failures")))

    payload: MutableMapping[str, object] = {
        "accepted": accepted,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "stress_failures": stress_failures,
    }

    cost = coerce_float(event.get("cost_bps"))
    if cost is not None:
        payload["cost_bps"] = cost
    net_edge = coerce_float(event.get("net_edge_bps"))
    if net_edge is not None:
        payload["net_edge_bps"] = net_edge
    model_return = coerce_float(event.get("model_expected_return_bps"))
    if model_return is not None:
        payload["model_expected_return_bps"] = model_return
    model_probability = coerce_float(event.get("model_success_probability"))
    if model_probability is not None:
        payload["model_success_probability"] = model_probability
    latency = coerce_float(event.get("latency_ms"))
    if latency is not None:
        payload["latency_ms"] = latency

    model_name = event.get("model_name")
    if model_name is not None:
        payload["model_name"] = model_name

    thresholds = _parse_json_object(event.get("decision_thresholds"))
    if thresholds:
        payload["thresholds"] = thresholds

    selection = _parse_json_object(event.get("model_selection"))
    if selection:
        payload["model_selection"] = selection

    candidate: MutableMapping[str, object] = {}
    for key, field in ("symbol", "symbol"), ("action", "side"), ("strategy", "strategy"):
        value = event.get(field)
        if value is not None:
            candidate[key] = value
    expected_probability = coerce_float(event.get("expected_probability"))
    if expected_probability is not None:
        candidate["expected_probability"] = expected_probability
    expected_return = coerce_float(event.get("expected_return_bps"))
    if expected_return is not None:
        candidate["expected_return_bps"] = expected_return
    notional = coerce_float(event.get("notional"))
    if notional is not None:
        candidate["notional"] = notional

    metadata_fields = ("generated_at", "signal_id", "schedule_run_id", "strategy_instance_id", "model_request_id")
    metadata = {field: event.get(field) for field in metadata_fields if event.get(field) is not None}
    if metadata:
        candidate["metadata"] = metadata

    if candidate:
        payload["candidate"] = candidate

    timestamp = _parse_timestamp(event.get("evaluated_at")) or _parse_timestamp(event.get("timestamp"))
    return timestamp, payload


def _collect_evaluations(args: argparse.Namespace) -> list[tuple[datetime | None, Mapping[str, object]]]:
    evaluations: list[tuple[datetime | None, Mapping[str, object]]] = []
    for event in _load_events(args.ledger.expanduser(), args.pattern):
        if not _event_matches_filters(event, args):
            continue
        transformed = _event_to_payload(event)
        if transformed is None:
            continue
        evaluations.append(transformed)
    evaluations.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc))
    limit = args.history_limit
    if limit and limit > 0 and len(evaluations) > limit:
        evaluations = evaluations[-limit:]
    return evaluations


def _build_summary(args: argparse.Namespace, evaluations: list[tuple[datetime | None, Mapping[str, object]]]) -> Mapping[str, object]:
    payloads = [payload for _, payload in evaluations]
    summary_metrics = summarize_evaluation_payloads(payloads, history_limit=args.history_limit)
    filters = {
        key: value
        for key, value in {
            "environment": args.environment,
            "portfolio": args.portfolio,
            "risk_profile": args.risk_profile,
            "strategy": args.strategy,
            "schedule": args.schedule,
            "symbol": args.symbol,
        }.items()
        if value is not None
    }
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary: MutableMapping[str, object] = {
        "type": "decision_engine_summary",
        "generated_at": generated_at,
        "source": str(args.ledger.expanduser()),
        **summary_metrics,
    }
    if filters:
        summary["filters"] = filters

    if args.include_history:
        history_size = max(1, int(args.history_size)) if args.history_size else len(evaluations)
        history_slice = evaluations[-history_size:]
        history_payload = []
        for timestamp, payload in history_slice:
            record = dict(payload)
            if timestamp is not None:
                record.setdefault("evaluated_at", timestamp.isoformat())
            history_payload.append(record)
        summary["history"] = history_payload

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    evaluations = _collect_evaluations(args)
    summary = _build_summary(args, evaluations)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.pretty:
        payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n"
    output_path.write_text(payload, encoding="utf-8")

    if summary["total"] == 0 and args.require_evaluations:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
