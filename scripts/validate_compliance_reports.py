"""Waliduje raporty zgodności Stage5 z podpisami HMAC."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Sequence
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.compliance.reports import validate_compliance_reports

_DEFAULT_CONTROLS = [
    "stage5.training.log",
    "stage5.tco.report",
    "stage5.oem.dry_run",
    "stage5.key_rotation",
    "stage5.compliance.review",
]


def _decode_key(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - walidacja wejścia
        raise SystemExit(f"Nie udało się zdekodować klucza HMAC: {exc}")


def _load_key_from_file(path: str | None) -> bytes | None:
    if not path:
        return None
    return _decode_key(Path(path).read_text(encoding="utf-8"))


def _load_key_from_env(env_name: str | None) -> bytes | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value is None:
        raise SystemExit(f"Zmienna środowiskowa {env_name} nie jest ustawiona")
    return _decode_key(value)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Waliduje raporty Stage5 (JSON) i zwraca status zgodności.",
    )
    parser.add_argument("paths", nargs="+", help="Ścieżki do raportów lub katalogów z raportami")
    parser.add_argument(
        "--expected-control",
        action="append",
        dest="expected_controls",
        help="Identyfikator kontroli wymagany w raporcie (można powtórzyć)",
    )
    parser.add_argument(
        "--no-default-controls",
        action="store_true",
        help="Nie dodawaj domyślnego zestawu kontroli Stage5",
    )
    parser.add_argument(
        "--signing-key",
        help="Klucz HMAC w formacie Base64",
    )
    parser.add_argument(
        "--signing-key-file",
        help="Ścieżka do pliku z kluczem HMAC (Base64)",
    )
    parser.add_argument(
        "--signing-key-env",
        help="Zmienna środowiskowa z kluczem HMAC (Base64)",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Traktuj brak podpisu jako błąd krytyczny",
    )
    parser.add_argument(
        "--output-json",
        help="Ścieżka do zapisu wyniku walidacji w formacie JSON",
    )
    return parser.parse_args(argv)


def _gather_reports(paths: Sequence[str]) -> list[Path]:
    result: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    return result


def _resolve_controls(args: argparse.Namespace) -> list[str] | None:
    controls = [] if args.no_default_controls else list(_DEFAULT_CONTROLS)
    if args.expected_controls:
        controls.extend(args.expected_controls)
    return controls or None


def _resolve_key(args: argparse.Namespace) -> bytes | None:
    for loader in (_decode_key(args.signing_key), _load_key_from_file(args.signing_key_file)):
        if loader:
            return loader
    if args.signing_key_env:
        return _load_key_from_env(args.signing_key_env)
    return None


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reports = _gather_reports(args.paths)
    if not reports:
        raise SystemExit("Nie znaleziono raportów do walidacji")

    signing_key = _resolve_key(args)
    controls = _resolve_controls(args)
    results = validate_compliance_reports(
        reports,
        expected_controls=controls,
        signing_key=signing_key,
        require_signature=args.require_signature,
    )

    summary = []
    exit_code = 0
    for result in results:
        payload = {
            "report": str(result.report_path),
            "issues": result.issues,
            "warnings": result.warnings,
            "failed_controls": result.failed_controls,
            "passed_controls": result.passed_controls,
            "metadata": dict(result.metadata),
        }
        summary.append(payload)
        if result.issues or result.failed_controls:
            exit_code = 1

    output = {
        "reports": summary,
        "total": len(summary),
        "errors": sum(1 for item in summary if item["issues"]),
        "failures": sum(1 for item in summary if item["failed_controls"]),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
