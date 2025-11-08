"""CLI wspierające podpisywanie i weryfikację pluginów strategii."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (
    PluginSigner,
    PluginVerifier,
    dump_package,
    load_manifest,
    load_package,
)


def _read_key(args: argparse.Namespace) -> bytes:
    if getattr(args, "key", None):
        return str(args.key).encode("utf-8")
    if getattr(args, "key_file", None):
        return Path(args.key_file).read_bytes().strip()
    raise SystemExit("Klucz podpisu należy dostarczyć (--key lub --key-file)")


def _cmd_sign(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    signer = PluginSigner(_read_key(args), key_id=args.key_id)
    package = signer.build_package(manifest, review_notes=args.note)
    if args.output:
        dump_package(package, args.output)
    else:
        sys.stdout.write(package.to_json())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    package = load_package(args.package)
    verifier = PluginVerifier(_read_key(args))
    ok = verifier.verify(package.manifest, package.signature)
    message = "OK" if ok else "INVALID"
    sys.stdout.write(f"{message}\n")
    return 0 if ok else 1


def _cmd_describe(args: argparse.Namespace) -> int:
    package = load_package(args.package)
    manifest = package.manifest
    metadata = manifest.to_dict()
    metadata["review_notes"] = list(package.review_notes)
    for key, value in metadata.items():
        sys.stdout.write(f"{key}: {value}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sign_parser = subparsers.add_parser("sign", help="Podpisz manifest pluginu")
    sign_parser.add_argument("--manifest", required=True, help="Ścieżka do manifestu JSON/YAML")
    sign_parser.add_argument("--key", help="Klucz HMAC w formie tekstowej")
    sign_parser.add_argument("--key-file", help="Plik zawierający klucz HMAC")
    sign_parser.add_argument("--key-id", help="Identyfikator klucza publicznego")
    sign_parser.add_argument("--output", help="Plik docelowy na podpisany pakiet JSON")
    sign_parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Notatka QA dołączana do pakietu (można podać wielokrotnie)",
    )
    sign_parser.set_defaults(func=_cmd_sign)

    verify_parser = subparsers.add_parser("verify", help="Zweryfikuj podpis pakietu")
    verify_parser.add_argument("--package", required=True, help="Podpisany pakiet JSON")
    verify_parser.add_argument("--key", help="Klucz HMAC w formie tekstowej")
    verify_parser.add_argument("--key-file", help="Plik zawierający klucz HMAC")
    verify_parser.set_defaults(func=_cmd_verify)

    describe_parser = subparsers.add_parser("describe", help="Wyświetl metadane pakietu")
    describe_parser.add_argument("--package", required=True, help="Podpisany pakiet JSON")
    describe_parser.set_defaults(func=_cmd_describe)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

