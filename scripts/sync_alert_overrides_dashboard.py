"""Publikuje anotacje override'ów SLO do dashboardu Grafana."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.observability import (  # noqa: E402 - import po modyfikacji sys.path
    build_dashboard_annotations_payload,
    load_dashboard_definition,
    load_overrides_from_document,
    save_dashboard_annotations,
)
from bot_core.security.signing import build_hmac_signature  # noqa: E402


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/audit/observability") / f"dashboard_annotations_{timestamp}.json"


def _load_overrides(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_overrides_from_document(data)


def _load_dashboard_uid(path: Path | None) -> str | None:
    if path is None:
        return None
    definition = load_dashboard_definition(path)
    return definition.uid


def _load_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key and args.signing_key_path:
        raise ValueError("Nie można podać jednocześnie klucza HMAC i ścieżki do pliku")
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_env:
        value = os.environ.get(args.signing_key_env)
        if value:
            return value.encode("utf-8"), args.signing_key_id
    if args.signing_key_path:
        key_path = Path(args.signing_key_path)
        if not key_path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {key_path}")
        return key_path.read_bytes().strip(), args.signing_key_id
    return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generuje anotacje do dashboardu Grafana na podstawie override'ów alertów",
    )
    parser.add_argument(
        "--overrides",
        required=True,
        help="Plik JSON wygenerowany przez generate_alert_overrides.py",
    )
    parser.add_argument(
        "--dashboard",
        help="Ścieżka do dashboardu Grafana (do odczytu UID)",
    )
    parser.add_argument(
        "--panel-id",
        type=int,
        help="Opcjonalny identyfikator panelu Grafana powiązanego z anotacjami",
    )
    parser.add_argument(
        "--output",
        help="Plik wyjściowy (domyślnie var/audit/observability/dashboard_annotations_*.json)",
    )
    parser.add_argument("--pretty", action="store_true", help="Formatuj JSON z wcięciami")
    parser.add_argument(
        "--signature",
        help="Opcjonalny plik podpisu HMAC (domyślnie obok pliku wynikowego)",
    )
    parser.add_argument("--signing-key", help="Klucz HMAC podany wprost")
    parser.add_argument(
        "--signing-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC",
    )
    parser.add_argument("--signing-key-path", help="Ścieżka do pliku z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        overrides_path = Path(args.overrides)
        overrides = _load_overrides(overrides_path)
        dashboard_uid = _load_dashboard_uid(Path(args.dashboard)) if args.dashboard else None
        payload = build_dashboard_annotations_payload(
            overrides,
            dashboard_uid=dashboard_uid,
            panel_id=args.panel_id,
        )
        output_path = Path(args.output) if args.output else _default_output()
        save_dashboard_annotations(payload, output_path=output_path, pretty=args.pretty)

        key, key_id = _load_hmac_key(args)
        if key:
            signature_payload = build_hmac_signature(payload, key=key, key_id=key_id)
            signature_path = Path(args.signature) if args.signature else output_path.with_suffix(".sig")
            signature_path.parent.mkdir(parents=True, exist_ok=True)
            with signature_path.open("w", encoding="utf-8") as handle:
                json.dump(signature_payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            print(
                "Zapisano anotacje dashboardu oraz podpis HMAC:",
                output_path,
                signature_path,
            )
        else:
            print("Zapisano anotacje dashboardu:", output_path)
    except Exception as exc:  # noqa: BLE001 - komunikat dla operatora
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run())

