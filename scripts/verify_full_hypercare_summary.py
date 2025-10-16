"""Weryfikacja podpisanego raportu pełnego hypercare (Stage5 + Stage6)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.runtime.full_hypercare import (  # noqa: E402
    FullHypercareVerificationResult,
    verify_full_hypercare_summary,
)


def _load_key(value: str | None, file_path: str | None, env_name: str | None) -> bytes | None:
    provided = [bool(value), bool(file_path), bool(env_name)]
    if sum(provided) > 1:
        raise ValueError("Klucz HMAC należy podać jako wartość, plik lub zmienną środowiskową")

    if value:
        key = value.encode("utf-8")
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
        env_value = os.getenv(env_name)
        if not env_value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza HMAC")
        key = env_value.encode("utf-8")
    else:
        return None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key


def _load_primary_key(args: argparse.Namespace) -> bytes | None:
    return _load_key(args.hmac_key, args.hmac_key_file, args.hmac_key_env)


def _load_stage5_key(args: argparse.Namespace) -> bytes | None:
    return _load_key(
        args.stage5_hmac_key,
        args.stage5_hmac_key_file,
        args.stage5_hmac_key_env,
    )


def _load_stage6_key(args: argparse.Namespace) -> bytes | None:
    return _load_key(
        args.stage6_hmac_key,
        args.stage6_hmac_key_file,
        args.stage6_hmac_key_env,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weryfikuje podpisany raport pełnego hypercare oraz opcjonalnie raporty Stage5/Stage6.",
    )
    parser.add_argument("summary", help="Ścieżka do raportu full hypercare (JSON)")
    parser.add_argument(
        "--signature",
        help="Ścieżka do podpisu raportu (domyślnie <summary>.sig, jeśli istnieje)",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Wymaga obecności poprawnego podpisu raportu zbiorczego",
    )
    parser.add_argument("--hmac-key", help="Klucz HMAC w formie tekstowej")
    parser.add_argument("--hmac-key-file", help="Ścieżka do pliku z kluczem HMAC")
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC",
    )

    parser.add_argument(
        "--revalidate-stage5",
        action="store_true",
        help="Ponownie weryfikuje raport Stage5 wykorzystując ścieżki z raportu zbiorczego",
    )
    parser.add_argument("--stage5-hmac-key", help="Klucz HMAC Stage5 w formie tekstowej")
    parser.add_argument(
        "--stage5-hmac-key-file",
        help="Ścieżka do pliku z kluczem HMAC raportu Stage5",
    )
    parser.add_argument(
        "--stage5-hmac-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC Stage5",
    )
    parser.add_argument(
        "--stage5-require-signature",
        action="store_true",
        help="Wymaga poprawnego podpisu raportu Stage5 podczas ponownej weryfikacji",
    )

    parser.add_argument(
        "--revalidate-stage6",
        action="store_true",
        help="Ponownie weryfikuje raport Stage6 wykorzystując ścieżki z raportu zbiorczego",
    )
    parser.add_argument("--stage6-hmac-key", help="Klucz HMAC Stage6 w formie tekstowej")
    parser.add_argument(
        "--stage6-hmac-key-file",
        help="Ścieżka do pliku z kluczem HMAC raportu Stage6",
    )
    parser.add_argument(
        "--stage6-hmac-key-env",
        help="Nazwa zmiennej środowiskowej zawierającej klucz HMAC Stage6",
    )
    parser.add_argument(
        "--stage6-require-signature",
        action="store_true",
        help="Wymaga poprawnego podpisu raportu Stage6 podczas ponownej weryfikacji",
    )
    return parser


def _serialize_stage5(result: object | None) -> dict[str, object] | None:
    from bot_core.runtime.stage5_hypercare import Stage5HypercareVerificationResult  # noqa: E402

    if not isinstance(result, Stage5HypercareVerificationResult):
        return None
    return {
        "summary_path": result.summary_path.as_posix(),
        "signature_path": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.overall_status,
        "signature_valid": result.signature_valid,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "artifacts": dict(result.artifact_statuses),
    }


def _serialize_stage6(result: object | None) -> dict[str, object] | None:
    from bot_core.runtime.stage6_hypercare import Stage6HypercareVerificationResult  # noqa: E402

    if not isinstance(result, Stage6HypercareVerificationResult):
        return None
    return {
        "summary_path": result.summary_path.as_posix(),
        "signature_path": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.overall_status,
        "signature_valid": result.signature_valid,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "components": dict(result.component_statuses),
    }


def _serialize_result(result: FullHypercareVerificationResult) -> dict[str, object]:
    data: dict[str, object] = {
        "summary": result.summary_path.as_posix(),
        "signature": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.overall_status,
        "signature_valid": result.signature_valid,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
        "components": dict(result.component_statuses),
    }
    stage5 = _serialize_stage5(result.stage5)
    stage6 = _serialize_stage6(result.stage6)
    if stage5 is not None:
        data["stage5"] = stage5
    if stage6 is not None:
        data["stage6"] = stage6
    return data


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).expanduser()
    signature_path = Path(args.signature).expanduser() if args.signature else None

    try:
        key = _load_primary_key(args)
        stage5_key = _load_stage5_key(args)
        stage6_key = _load_stage6_key(args)

        result = verify_full_hypercare_summary(
            summary_path,
            signature_path=signature_path,
            signing_key=key,
            require_signature=args.require_signature,
            revalidate_stage5=args.revalidate_stage5,
            revalidate_stage6=args.revalidate_stage6,
            stage5_signing_key=stage5_key,
            stage6_signing_key=stage6_key,
            stage5_require_signature=args.stage5_require_signature,
            stage6_require_signature=args.stage6_require_signature,
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
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())
