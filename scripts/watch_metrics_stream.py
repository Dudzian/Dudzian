"""Narzędzie do podglądu strumienia MetricsService.

Skrypt łączy się z serwerem telemetrii `MetricsService` (gRPC) i wypisuje
otrzymane `MetricsSnapshot`.  Może filtrować zdarzenia po polu `event` w
`notes`, ograniczać liczbę pobranych rekordów oraz formatować wynik w postaci
czytelnej tabeli lub surowego JSON-u.

Przykłady użycia::

    # Podgląd ostatnich 5 rekordów w formacie tabeli
    python -m scripts.watch_metrics_stream --limit 5

    # Wyłącznie zdarzenia reduce_motion jako JSON
    python -m scripts.watch_metrics_stream --event reduce_motion --format json

    # Podłączenie do niestandardowego hosta/portu
    python -m scripts.watch_metrics_stream --host 10.0.0.5 --port 50070

    # Połączenie z serwerem wymagającym tokenu Bearer
    python -m scripts.watch_metrics_stream --auth-token secret --limit 10

Skrypt wymaga obecności wygenerowanych stubów gRPC (`trading_pb2*.py`) oraz
pakietu `grpcio`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Iterable


LOGGER = logging.getLogger("bot_core.scripts.watch_metrics_stream")


def _load_grpc_components():
    try:
        import grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - środowisko bez grpcio
        raise SystemExit(
            "Pakiet grpcio jest wymagany do połączenia z MetricsService."
        ) from exc

    try:
        from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - brak wygenerowanych stubów
        raise SystemExit(
            "Brak wygenerowanych stubów trading_pb2*.py. Uruchom scripts/generate_trading_stubs.py"
        ) from exc

    return grpc, trading_pb2, trading_pb2_grpc


def _timestamp_to_iso(timestamp) -> str | None:
    if timestamp is None:
        return None
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if not seconds and not nanos:
        return None
    seconds = seconds or 0
    nanos = nanos or 0
    dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def _parse_notes(notes: str) -> dict[str, Any]:
    if not notes:
        return {}
    try:
        payload = json.loads(notes)
        if isinstance(payload, dict):
            return payload
        return {"value": payload}
    except json.JSONDecodeError:
        return {"raw": notes}


def _format_snapshot(snapshot, *, fmt: str) -> str:
    record = {
        "generated_at": _timestamp_to_iso(snapshot.generated_at)
        if getattr(snapshot, "HasField", None) and snapshot.HasField("generated_at")
        else None,
        "fps": snapshot.fps if getattr(snapshot, "HasField", None) and snapshot.HasField("fps") else None,
        "event": None,
        "notes": _parse_notes(snapshot.notes),
    }
    event = record["notes"].get("event") if isinstance(record["notes"], dict) else None
    record["event"] = event

    if fmt == "json":
        payload = {
            "generated_at": record["generated_at"],
            "fps": record["fps"],
            "notes": record["notes"],
        }
        return json.dumps(payload, ensure_ascii=False)

    notes_preview = record["notes"]
    if isinstance(notes_preview, dict):
        preview = json.dumps(notes_preview, ensure_ascii=False)
    else:
        preview = str(notes_preview)
    if len(preview) > 160:
        preview = preview[:157] + "..."
    timestamp = record["generated_at"] or "-"
    fps = f"{record['fps']:.2f}" if record["fps"] is not None else "-"
    event_label = event or "-"
    return f"{timestamp} | fps={fps} | event={event_label} | notes={preview}"


def _iter_stream(
    stub,
    request,
    *,
    timeout: float | None,
    metadata: list[tuple[str, str]] | None,
) -> Iterable:
    try:
        stream_kwargs = {"timeout": timeout}
        if metadata:
            stream_kwargs["metadata"] = metadata
        yield from stub.StreamMetrics(request, **stream_kwargs)
    except Exception as exc:  # pragma: no cover - obsługa błędów połączenia
        LOGGER.error("Błąd połączenia z MetricsService: %s", exc)
        raise SystemExit(1) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Adres hosta MetricsService")
    parser.add_argument("--port", type=int, default=50051, help="Port MetricsService")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout RPC w sekundach")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba snapshotów do wypisania; domyślnie stream bez końca",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Filtruj snapshoty po polu event w notes (np. reduce_motion)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Format wypisywanych danych",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Opcjonalny token autoryzacyjny (wysyłany w nagłówku authorization)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    grpc, trading_pb2, trading_pb2_grpc = _load_grpc_components()

    address = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(address)
    stub = trading_pb2_grpc.MetricsServiceStub(channel)

    request = trading_pb2.MetricsRequest()
    request.include_ui_metrics = True

    count = 0
    metadata: list[tuple[str, str]] | None = None
    if args.auth_token:
        metadata = [("authorization", f"Bearer {args.auth_token}")]

    for snapshot in _iter_stream(stub, request, timeout=args.timeout, metadata=metadata):
        notes_payload = _parse_notes(snapshot.notes)
        if args.event and notes_payload.get("event") != args.event:
            continue
        print(_format_snapshot(snapshot, fmt=args.format))
        count += 1
        if args.limit is not None and count >= args.limit:
            break

    if count == 0:
        LOGGER.warning("Nie odebrano żadnych snapshotów z %s", address)
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia
    sys.exit(main())
