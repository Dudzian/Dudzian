"""Monitor SLO Stage6 generujący podpisane raporty Observability."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from bot_core.observability import (
    evaluate_slo,
    load_slo_definitions,
    load_slo_measurements,
    write_slo_results_csv,
)
from bot_core.security.signing import build_hmac_signature


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/audit/observability") / f"slo_report_{timestamp}.json"
def _load_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_env:
        env_value = os.environ.get(args.signing_key_env)
        if env_value:
            return env_value.encode("utf-8"), args.signing_key_id
    if args.signing_key_path:
        path = Path(args.signing_key_path)
        if path.exists():
            return path.read_bytes().strip(), args.signing_key_id
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoruje SLO Stage6 i zapisuje raport JSON")
    parser.add_argument("--definitions", required=True, help="Plik z definicjami SLO (YAML/JSON)")
    parser.add_argument("--metrics", required=True, help="Plik z pomiarami metryk (JSON)")
    parser.add_argument("--output", help="Ścieżka raportu JSON (domyślnie var/audit/observability/...)")
    parser.add_argument(
        "--output-csv",
        help="Opcjonalna ścieżka raportu CSV z wynikami SLO i kompozytów",
    )
    parser.add_argument("--pretty", action="store_true", help="Czy formatować JSON z wcięciami")
    parser.add_argument("--signature", help="Ścieżka pliku z podpisem HMAC")
    parser.add_argument("--signing-key", help="Klucz HMAC wprost w CLI")
    parser.add_argument("--signing-key-env", help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    parser.add_argument("--signing-key-path", help="Plik zawierający klucz HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    args = parser.parse_args()

    definitions, composites = load_slo_definitions(Path(args.definitions))
    if not definitions:
        raise SystemExit("Brak prawidłowych definicji SLO")

    measurements = load_slo_measurements(Path(args.metrics))
    report = evaluate_slo(definitions, measurements, composites=composites)
    payload = report.to_payload()

    output_path = Path(args.output) if args.output else _default_output()
    report.write_json(output_path, pretty=args.pretty)

    csv_path: Path | None = None
    if args.output_csv:
        csv_path = Path(args.output_csv)
        report.write_csv(csv_path)

    key, key_id = _load_signing_key(args)
    signature_path = Path(args.signature) if args.signature else output_path.with_suffix(".sig")
    if key:
        signature_payload = build_hmac_signature(payload, key=key, key_id=key_id)
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        with signature_path.open("w", encoding="utf-8") as handle:
            json.dump(signature_payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        message = f"Zapisano raport SLO do {output_path} wraz z podpisem {signature_path}"
    else:
        message = f"Zapisano raport SLO do {output_path} (bez podpisu HMAC)"

    if csv_path is not None:
        message += f"; raport CSV: {csv_path}"
    print(message)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
