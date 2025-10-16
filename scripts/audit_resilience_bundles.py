"""Automatyczna walidacja paczek odporności Stage6 wraz z raportem CSV/JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience.audit import (
    audit_bundles,
    build_summary,
    write_csv_report,
    write_json_report,
    write_json_report_signature,
)
from bot_core.resilience.policy import ResiliencePolicy, load_policy


def _load_hmac_material(
    *,
    inline_value: str | None,
    file_path: str | None,
    env_name: str | None,
    context: str,
) -> bytes | None:
    provided = sum(1 for option in (inline_value, file_path, env_name) if option)
    if provided > 1:
        raise ValueError(f"{context}: wybierz dokładnie jedno źródło klucza (wartość, plik lub zmienna)")

    if inline_value:
        key = inline_value.encode("utf-8")
    elif file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"{context}: plik z kluczem nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError(f"{context}: plik klucza powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif env_name:
        value = os.getenv(env_name)
        if not value:
            raise ValueError(f"{context}: zmienna środowiskowa {env_name} jest pusta")
        key = value.encode("utf-8")
    else:
        return None

    if len(key) < 16:
        raise ValueError(f"{context}: klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key


def _load_hmac_key(args: argparse.Namespace) -> bytes | None:
    return _load_hmac_material(
        inline_value=args.hmac_key,
        file_path=args.hmac_key_file,
        env_name=args.hmac_key_env,
        context="Weryfikacja paczek",
    )


def _load_report_hmac_key(args: argparse.Namespace) -> bytes | None:
    return _load_hmac_material(
        inline_value=getattr(args, "report_hmac_key", None),
        file_path=getattr(args, "report_hmac_key_file", None),
        env_name=getattr(args, "report_hmac_key_env", None),
        context="Podpis raportu JSON",
    )


def _load_policy(args: argparse.Namespace) -> ResiliencePolicy | None:
    if not args.policy:
        return None
    return load_policy(Path(args.policy))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audytuje paczki odpornościowe Stage6 i generuje raporty.",
    )
    parser.add_argument(
        "--directory",
        default=str(REPO_ROOT / "var" / "resilience"),
        help="Katalog z paczkami ZIP do audytu",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Traktuj brak podpisu manifestu jako błąd krytyczny",
    )
    parser.add_argument(
        "--hmac-key",
        help="Wartość klucza HMAC do weryfikacji podpisów",
    )
    parser.add_argument(
        "--hmac-key-file",
        help="Ścieżka do pliku z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC",
    )
    parser.add_argument(
        "--csv-report",
        help="Ścieżka do raportu CSV (opcjonalnie)",
    )
    parser.add_argument(
        "--json-report",
        help="Ścieżka do raportu JSON z wynikami (opcjonalnie)",
    )
    parser.add_argument(
        "--json-report-signature",
        help="Ścieżka do podpisu raportu JSON (opcjonalnie)",
    )
    parser.add_argument(
        "--policy",
        help="Ścieżka do pliku JSON z polityką wymagań Stage6",
    )
    parser.add_argument(
        "--report-hmac-key",
        help="Wartość klucza HMAC do podpisu raportu JSON",
    )
    parser.add_argument(
        "--report-hmac-key-file",
        help="Plik z kluczem HMAC do podpisu raportu JSON",
    )
    parser.add_argument(
        "--report-hmac-key-env",
        help="Zmienna środowiskowa z kluczem HMAC do podpisu raportu JSON",
    )
    parser.add_argument(
        "--report-hmac-key-id",
        help="Identyfikator klucza podpisującego raport JSON",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.json_report_signature and not args.json_report:
        print("Błąd: podpis raportu JSON wymaga użycia --json-report", file=sys.stderr)
        return 2

    try:
        key = _load_hmac_key(args)
        policy = _load_policy(args)
        results = audit_bundles(
            Path(args.directory),
            hmac_key=key,
            require_signature=args.require_signature,
            policy=policy,
        )
        if args.csv_report:
            write_csv_report(results, Path(args.csv_report))
        if args.json_report:
            summary = write_json_report(results, Path(args.json_report))
        else:
            summary = build_summary(results)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

        if args.json_report_signature:
            report_key = _load_report_hmac_key(args) or key
            if report_key is None:
                raise ValueError(
                    "Brak klucza do podpisu raportu JSON (użyj --report-hmac-key*, lub --hmac-key)"
                )
            write_json_report_signature(
                summary,
                Path(args.json_report_signature),
                key=report_key,
                key_id=getattr(args, "report_hmac_key_id", None),
                target=Path(args.json_report).name,
            )
    except Exception as exc:  # noqa: BLE001 - komunikat CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(run())
