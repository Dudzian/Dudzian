"""Weryfikacja podpisanego raportu hypercare Stage6."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.runtime.stage6_hypercare import (  # noqa: E402
    Stage6HypercareVerificationResult,
    verify_stage6_hypercare_summary,
)


def _load_key(args: argparse.Namespace) -> bytes | None:
    inline = args.hmac_key
    file_path = args.hmac_key_file
    env_name = args.hmac_key_env

    if inline and file_path:
        raise ValueError("Podaj klucz HMAC jako wartość lub plik, nie oba jednocześnie")

    if inline:
        key = inline.encode("utf-8")
    elif file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif env_name:
        value = os.getenv(env_name)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza HMAC")
        key = value.encode("utf-8")
    else:
        return None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weryfikuje raport hypercare Stage6 wraz z podpisem HMAC.",
    )
    parser.add_argument("summary", help="Ścieżka do raportu Stage6 hypercare (JSON)")
    parser.add_argument(
        "--signature",
        help="Ścieżka do podpisu (domyślnie <summary>.sig, jeśli istnieje)",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Wymaga obecności poprawnego podpisu HMAC",
    )
    parser.add_argument("--hmac-key", help="Klucz HMAC w formie tekstowej")
    parser.add_argument("--hmac-key-file", help="Ścieżka do pliku z kluczem HMAC")
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC",
    )
    return parser


def _serialize_result(result: Stage6HypercareVerificationResult) -> dict[str, object]:
    return {
        "summary": result.summary_path.as_posix(),
        "signature": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.overall_status,
        "signature_valid": result.signature_valid,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "components": dict(result.component_statuses),
    }


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).expanduser()
    signature_path = Path(args.signature).expanduser() if args.signature else None

    try:
        key = _load_key(args)
        result = verify_stage6_hypercare_summary(
            summary_path,
            signature_path=signature_path,
            signing_key=key,
            require_signature=args.require_signature,
        )
    except Exception as exc:  # noqa: BLE001 - komunikat CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Ostrzeżenie: {warning}", file=sys.stderr)

    if result.issues:
        for issue in result.issues:
            print(f"Błąd: {issue}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            _serialize_result(result),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())
