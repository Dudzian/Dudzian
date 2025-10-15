"""CLI do generowania podpisanych raportów kosztów transakcyjnych (TCO)."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from bot_core.tco import TCOAnalyzer, TCOReportWriter, TradeCostEvent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fills",
        dest="fills",
        action="append",
        required=True,
        help="Ścieżka do pliku JSONL z transakcjami (można podać wiele razy)",
    )
    parser.add_argument(
        "--output-dir",
        default="var/audit/tco",
        help="Katalog docelowy raportu (domyślnie var/audit/tco)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Opcjonalna nazwa bazowa plików (bez rozszerzeń)",
    )
    parser.add_argument(
        "--signing-key-path",
        dest="signing_key_path",
        help="Ścieżka do klucza HMAC używanego do podpisów CSV/PDF/JSON",
    )
    parser.add_argument(
        "--signing-key-id",
        dest="signing_key_id",
        default=None,
        help="Identyfikator klucza podpisującego umieszczany w dokumencie podpisu",
    )
    parser.add_argument(
        "--cost-limit-bps",
        type=float,
        default=None,
        help="Opcjonalny limit kosztów w punktach bazowych używany do generowania alertów",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Dodatkowe metadane w formacie klucz=wartość",
    )
    return parser


def _load_events(path: Path) -> list[TradeCostEvent]:
    contents = path.read_text(encoding="utf-8")
    events: list[TradeCostEvent] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Nieprawidłowy JSON w {path}:{line_number}") from exc
        events.append(TradeCostEvent.from_mapping(payload))
    return events


def _parse_metadata(items: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Wpis metadanych musi mieć postać klucz=wartość: {item}")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _read_signing_key(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise ValueError("Ścieżka do klucza nie może być katalogiem")
    if path.is_symlink():
        raise ValueError("Ścieżka do klucza nie może być symlinkiem")
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError("Klucz HMAC musi mieć co najmniej 32 bajty")
    return data


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    fill_paths = [Path(item) for item in args.fills]
    events: list[TradeCostEvent] = []
    for candidate in fill_paths:
        events.extend(_load_events(candidate))

    metadata = _parse_metadata(args.metadata)
    metadata.setdefault("source", "run_tco_analysis")
    metadata.setdefault("inputs", [str(path) for path in fill_paths])
    cost_limit = Decimal(str(args.cost_limit_bps)) if args.cost_limit_bps is not None else None
    analyzer = TCOAnalyzer(cost_limit_bps=cost_limit)
    report = analyzer.analyze(events, metadata=metadata)

    writer = TCOReportWriter(report)
    output_root = Path(args.output_dir)
    if args.basename is None:
        timestamp_dir = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
        target_dir = output_root / timestamp_dir
    else:
        target_dir = output_root
    artifacts = writer.write_outputs(target_dir, basename=args.basename)

    if args.signing_key_path:
        signing_key = _read_signing_key(Path(args.signing_key_path))
        writer.sign_artifacts(artifacts, signing_key=signing_key, key_id=args.signing_key_id)
    else:
        print("Ostrzeżenie: brak klucza podpisującego – raport zostanie zapisany bez podpisów", file=sys.stderr)

    print(f"Wygenerowano raport TCO w katalogu {target_dir}")
    print(f"Liczba transakcji: {report.metadata.get('events_count', 0)}")
    if report.alerts:
        print("Alerty:")
        for alert in report.alerts:
            print(f" - {alert}")
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())
