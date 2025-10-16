"""CLI generujący raport TCO Stage5 wraz z podpisem HMAC."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bot_core.reporting.tco import (
    TcoUsageMetrics,
    aggregate_costs,
    load_cost_items,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generuje raport TCO (Total Cost of Ownership) na potrzeby hypercare Stage5.",
    )
    parser.add_argument("--input", required=True, help="Plik JSON z listą pozycji kosztowych")
    parser.add_argument(
        "--artifact-root",
        default="var/audit/tco",
        help="Katalog, w którym zostaną zapisane artefakty TCO (domyślnie var/audit/tco)",
    )
    parser.add_argument("--json-name", default="tco_summary.json", help="Nazwa pliku JSON z podsumowaniem")
    parser.add_argument("--csv-name", default="tco_breakdown.csv", help="Nazwa pliku CSV z rozbiciem pozycji")
    parser.add_argument(
        "--signature-name",
        default="tco_summary.signature.json",
        help="Nazwa pliku z podpisem HMAC podsumowania",
    )
    parser.add_argument("--monthly-trades", type=float, help="Średnia miesięczna liczba transakcji")
    parser.add_argument(
        "--monthly-volume",
        type=float,
        help="Średni miesięczny wolumen obrotu (np. w USD)",
    )
    parser.add_argument("--tag", help="Opcjonalny identyfikator raportu dodawany do metadanych")
    parser.add_argument(
        "--signing-key",
        help="Wartość klucza HMAC (ciąg znaków) do podpisu raportu",
    )
    parser.add_argument(
        "--signing-key-file",
        help="Plik zawierający klucz HMAC do podpisu raportu",
    )
    parser.add_argument(
        "--signing-key-id",
        default="tco",
        help="Identyfikator klucza użytego do podpisu (domyślnie 'tco')",
    )
    parser.add_argument(
        "--timestamp",
        help="Znacznik czasu używany jako nazwa katalogu artefaktów (domyślnie bieżący UTC)",
    )
    parser.add_argument("--print-summary", action="store_true", help="Wypisz podsumowanie JSON na stdout")
    return parser


def _load_signing_key(*, value: str | None, path: str | None) -> bytes | None:
    if value and path:
        raise ValueError("Użyj tylko jednej z opcji: --signing-key lub --signing-key-file")
    if value:
        return value.encode("utf-8")
    if path:
        return Path(path).expanduser().read_bytes().strip()
    return None


def _sanitize_timestamp(raw: str | None) -> str:
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = raw.strip()
    if not candidate:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return candidate.replace(":", "").replace("-", "").replace(" ", "_")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.monthly_trades is not None and args.monthly_trades < 0:
        parser.error("Parametr --monthly-trades musi być nieujemny")
    if args.monthly_volume is not None and args.monthly_volume < 0:
        parser.error("Parametr --monthly-volume musi być nieujemny")

    input_path = Path(args.input).expanduser()
    try:
        items = load_cost_items(input_path)
    except Exception as exc:  # pragma: no cover - argparse zakończy proces
        parser.error(f"Nie udało się wczytać pozycji kosztowych: {exc}")

    summary = aggregate_costs(items)

    usage = TcoUsageMetrics(
        monthly_trades=args.monthly_trades,
        monthly_volume=args.monthly_volume,
    )

    artifact_root = Path(args.artifact_root).expanduser()
    timestamp = _sanitize_timestamp(args.timestamp)
    run_dir = artifact_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    metadata = {
        "input_path": str(input_path.resolve()),
        "items_count": len(summary.items),
    }
    if args.tag:
        metadata["tag"] = args.tag

    json_path = run_dir / args.json_name
    csv_path = run_dir / args.csv_name
    payload = write_summary_json(
        summary,
        json_path,
        generated_at=generated_at,
        usage=usage,
        metadata=metadata,
    )
    write_summary_csv(summary, csv_path)

    signature_path = None
    try:
        signing_key = _load_signing_key(value=args.signing_key, path=args.signing_key_file)
    except FileNotFoundError as exc:
        print(f"Błąd: nie znaleziono pliku z kluczem HMAC: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 2

    if signing_key is not None:
        signature_path = run_dir / args.signature_name
        write_summary_signature(payload, signature_path, key=signing_key, key_id=args.signing_key_id)

    if args.print_summary:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    if signature_path:
        print(f"Raport zapisano w {json_path} (podpis: {signature_path})")
    else:
        print(f"Raport zapisano w {json_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
