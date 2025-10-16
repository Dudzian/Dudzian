"""Walidacja paczki odporności Stage6 wraz z podpisem HMAC."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience.bundle import (
    ResilienceBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
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
        description="Weryfikuje integralność paczki odpornościowej Stage6.",
    )
    parser.add_argument("bundle", help="Ścieżka do archiwum ZIP paczki odpornościowej")
    parser.add_argument(
        "--manifest",
        help="Ścieżka do manifestu (domyślnie <bundle>.manifest.json)",
    )
    parser.add_argument(
        "--signature",
        help="Ścieżka do podpisu manifestu (domyślnie <bundle>.manifest.sig)",
    )
    parser.add_argument(
        "--hmac-key",
        help="Wartość klucza HMAC do weryfikacji podpisu",
    )
    parser.add_argument(
        "--hmac-key-file",
        help="Plik z kluczem HMAC do weryfikacji podpisu",
    )
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bundle_path = Path(args.bundle).expanduser()
    manifest_path = (
        Path(args.manifest).expanduser()
        if args.manifest
        else bundle_path.with_suffix(".manifest.json")
    )
    signature_path = (
        Path(args.signature).expanduser()
        if args.signature
        else bundle_path.with_suffix(".manifest.sig")
    )

    try:
        manifest = load_manifest(manifest_path)
        verifier = ResilienceBundleVerifier(bundle_path, manifest)
        errors = verifier.verify_files()
        signature_doc = load_signature(signature_path if signature_path.exists() else None)
        key = _load_key(args)
        if signature_doc is not None and key is not None:
            errors.extend(verify_signature(manifest, signature_doc, key=key))
        elif signature_doc is not None and key is None:
            print(
                "Ostrzeżenie: dostarczono podpis bez klucza HMAC – pomijam weryfikację",
                file=sys.stderr,
            )
        if not signature_doc and key is not None:
            print(
                "Ostrzeżenie: dostarczono klucz HMAC, lecz nie znaleziono podpisu",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - komunikat CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    if errors:
        for message in errors:
            print(f"Błąd: {message}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "bundle": bundle_path.as_posix(),
                "manifest": manifest_path.as_posix(),
                "verified_files": manifest.get("file_count"),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(run())

