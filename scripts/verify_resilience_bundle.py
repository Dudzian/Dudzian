from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import logging
import os
import stat
import tarfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience.bundle import (
    ResilienceBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)
from bot_core.security.signing import canonical_json_bytes

_LOGGER = logging.getLogger("verify_resilience_bundle")


def _normalise_name(name: str) -> str:
    name = name.lstrip("./")
    return name


def _assert_safe_member(member: tarfile.TarInfo) -> None:
    if member.issym() or member.islnk():
        raise ValueError("Archiwum zawiera niedozwolone linki")
    name = member.name
    if name.startswith("/"):
        raise ValueError(f"Archiwum zawiera niebezpieczną ścieżkę: {name}")
    parts = Path(name).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Archiwum zawiera niebezpieczną ścieżkę: {name}")


def _read_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Mapping[str, object]:
    if member.isdir():
        raise ValueError("Oczekiwano pliku JSON, otrzymano katalog")
    data = archive.extractfile(member)
    if data is None:
        raise ValueError(f"Nie można odczytać danych: {member.name}")
    payload = data.read()
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"Nieprawidłowy JSON w {member.name}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Oczekiwano obiektu JSON (mapy)")
    return parsed


def _compute_digest(archive: tarfile.TarFile, member: tarfile.TarInfo, algorithm: str) -> str:
    if algorithm.lower() != "sha256":
        raise ValueError(f"Nieobsługiwany algorytm digest: {algorithm}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Nie można odczytać {member.name}")
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_signature_embedded(
    *,
    archive: tarfile.TarFile,
    manifest_member: tarfile.TarInfo,
    signature_doc: Mapping[str, object],
    signing_key: bytes,
) -> None:
    payload = signature_doc.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Dokument podpisu nie zawiera pola 'payload'")
    path = str(payload.get("path", ""))
    if _normalise_name(path) != _normalise_name(manifest_member.name):
        raise ValueError("Podpis odnosi się do nieoczekiwanej ścieżki manifestu")

    digests = {k: v for k, v in payload.items() if k.startswith("sha") and k != "path"}
    if len(digests) != 1:
        raise ValueError("Sekcja payload powinna zawierać dokładnie jeden wpis digest")
    digest_name, digest_value = next(iter(digests.items()))
    if not isinstance(digest_value, str) or not digest_value:
        raise ValueError("Digest manifestu powinien być łańcuchem heksadecymalnym")

    computed_digest = _compute_digest(archive, manifest_member, digest_name)
    if computed_digest != digest_value:
        raise ValueError("Digest manifestu nie zgadza się z podpisem")

    signature = signature_doc.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("Dokument podpisu nie zawiera sekcji 'signature'")
    algorithm = signature.get("algorithm")
    if algorithm != "HMAC-SHA256":
        raise ValueError(f"Nieobsługiwany algorytm podpisu: {algorithm}")

    value = signature.get("value")
    if not isinstance(value, str):
        raise ValueError("Wartość podpisu musi być łańcuchem base64")
    try:
        mac = base64.b64decode(value, validate=True)
    except binascii.Error as exc:  # type: ignore[name-defined]
        raise ValueError("Nieprawidłowe base64 podpisu") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Nieprawidłowe base64 podpisu") from exc

    expected = hmac.new(signing_key, canonical_json_bytes(payload), digestmod="sha256").digest()
    if not hmac.compare_digest(expected, mac):
        raise ValueError("Weryfikacja podpisu manifestu nie powiodła się")


def _verify_tar_bundle(*, bundle_path: Path, signing_key: bytes | None) -> Mapping[str, object]:
    with tarfile.open(bundle_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        manifest_member = members.get("manifest.json") or members.get("./manifest.json")
        if manifest_member is None:
            raise ValueError("Archiwum nie zawiera manifest.json")
        manifest = _read_json(archive, manifest_member)

        files = manifest.get("files")
        if not isinstance(files, Iterable):
            raise ValueError("Manifest nie zawiera listy plików")

        verified = 0
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("Pozycja manifestu musi być mapą")
            path_value = entry.get("path")
            sha_value = entry.get("sha256")
            if not isinstance(path_value, str) or not isinstance(sha_value, str):
                raise ValueError("Pozycja wymaga pól 'path' i 'sha256'")
            path = path_value.strip()
            sha = sha_value.strip()
            if not path or not sha:
                raise ValueError("Pozycja wymaga pól 'path' i 'sha256'")
            member = members.get(path) or members.get(f"./{path}")
            if member is None:
                raise ValueError(f"Brak artefaktu w paczce: {path}")
            _assert_safe_member(member)
            digest = _compute_digest(archive, member, "sha256")
            if digest != sha:
                raise ValueError("Niezgodny SHA-256 pliku z manifestem (Digest mismatch)")
            verified += 1

        signature_member = (
            members.get("manifest.json.sig")
            or members.get("./manifest.json.sig")
            or members.get("manifest.sig")
            or members.get("./manifest.sig")
        )
        if signing_key is not None:
            if signature_member is None:
                raise ValueError("Tryb TAR wymaga podpisu manifestu")
            signature_doc = _read_json(archive, signature_member)
            _verify_manifest_signature_embedded(
                archive=archive,
                manifest_member=manifest_member,
                signature_doc=signature_doc,
                signing_key=signing_key,
            )
        elif signature_member is not None:
            _LOGGER.warning("Archiwum zawiera podpis manifestu, ale nie podano klucza HMAC")

    return {
        "bundle": str(bundle_path),
        "manifest": "manifest.json (embedded)",
        "verified_files": verified,
    }


def _verify_external_manifest(
    *,
    bundle_path: Path,
    manifest_path: Path,
    signature_path: Path | None,
    signing_key: bytes | None,
) -> Mapping[str, object]:
    manifest = load_manifest(manifest_path)
    verifier = ResilienceBundleVerifier(bundle_path, manifest)
    errors = verifier.verify_files()
    if errors:
        raise ValueError("Walidacja zewnętrznego manifestu nie powiodła się: " + "; ".join(errors))

    signature_doc = load_signature(signature_path)
    if signature_doc and signing_key:
        signature_errors = verify_signature(manifest, signature_doc, key=signing_key)
        if signature_errors:
            raise ValueError("Weryfikacja podpisu manifestu nie powiodła się: " + "; ".join(signature_errors))
    elif signature_doc and not signing_key:
        message = "Podpis manifestu wykryty, ale nie przekazano klucza HMAC (podpis bez klucza)"
        _LOGGER.warning(message)
        print(message, file=sys.stderr)
    elif signing_key and not signature_doc:
        message = "Przekazano klucz HMAC, ale brak pliku podpisu manifestu"
        _LOGGER.warning(message)
        print(message, file=sys.stderr)

    return {
        "bundle": str(bundle_path),
        "manifest": str(manifest_path),
        "verified_files": len(manifest.get("files", [])),
    }


def _load_signing_key_merged(
    *,
    inline_value: str | None,
    file_path: str | None,
    env_name_primary: str | None,
    env_name_alt: str | None,
) -> bytes | None:
    if inline_value:
        data = inline_value.encode("utf-8")
        if len(data) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return data

    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise ValueError(f"Plik klucza {path} nie istnieje")
        if path.is_dir():
            raise ValueError("Plik klucza nie może być katalogiem")
        if path.is_symlink():
            raise ValueError("Plik klucza nie może być symlinkiem")
        if os.name != "nt":
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                raise ValueError("Plik klucza powinien mieć uprawnienia maks. 600")
        data = path.read_bytes().strip()
        if len(data) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return data

    for env_name in (env_name_primary, env_name_alt):
        if not env_name:
            continue
        value = os.environ.get(env_name)
        if value is None:
            continue
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} jest pusta")
        data = value.encode("utf-8")
        if len(data) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return data

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weryfikacja paczki odporności Stage6")
    parser.add_argument("bundle", nargs="?", help="Ścieżka do archiwum (tar.gz lub zip)")
    parser.add_argument("--bundle", dest="bundle_option", help="Ścieżka do archiwum (tar.gz lub zip)")
    parser.add_argument("--manifest", help="Zewnętrzny manifest JSON (dla plików ZIP)")
    parser.add_argument("--signature", help="Plik podpisu manifestu")
    parser.add_argument("--hmac-key", dest="hmac_key", help="Klucz HMAC podany inline")
    parser.add_argument("--hmac-key-file", dest="hmac_key_file", help="Plik z kluczem HMAC")
    parser.add_argument("--hmac-key-env", dest="hmac_key_env", help="Zmienna środowiskowa z kluczem HMAC")
    parser.add_argument(
        "--signing-key-env",
        dest="hmac_key_env",
        help="Alias zmiennej środowiskowej z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-env-alt",
        dest="hmac_key_env_alt",
        help="Alternatywna zmienna środowiskowa z kluczem HMAC",
    )
    parser.add_argument("--log-level", default="INFO", help="Poziom logowania (domyślnie INFO)")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bundle_arg = args.bundle_option or args.bundle
    if not bundle_arg:
        parser.error("Musisz wskazać ścieżkę do archiwum (--bundle lub argument pozycyjny)")
    return _execute(parser, args, Path(bundle_arg))


def _execute(parser: argparse.ArgumentParser, args: argparse.Namespace, bundle_path: Path) -> int:
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    try:
        signing_key = _load_signing_key_merged(
            inline_value=args.hmac_key,
            file_path=args.hmac_key_file,
            env_name_primary=args.hmac_key_env,
            env_name_alt=args.hmac_key_env_alt,
        )
    except ValueError as exc:
        _LOGGER.error("Błąd odczytu klucza HMAC: %s", exc)
        return 1

    try:
        if bundle_path.suffixes[-2:] == [".tar", ".gz"] or bundle_path.suffix == ".tgz":
            if signing_key is None:
                _LOGGER.error("Tryb TAR wymaga podania klucza HMAC")
                return 2
            summary = _verify_tar_bundle(bundle_path=bundle_path, signing_key=signing_key)
        else:
            summary = _verify_external_manifest(
                bundle_path=bundle_path,
                manifest_path=Path(args.manifest) if args.manifest else bundle_path.with_suffix(".manifest.json"),
                signature_path=Path(args.signature) if args.signature else bundle_path.with_suffix(".manifest.sig"),
                signing_key=signing_key,
            )
    except ValueError as exc:
        _LOGGER.error("Weryfikacja nie powiodła się: %s", exc)
        return 2
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Weryfikacja nie powiodła się")
        return 2

    print(json.dumps(summary, ensure_ascii=False))
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    """Wrapper used by tests to execute the CLI without exiting the interpreter."""

    if argv is None:
        return main(None)

    args = list(argv)
    if not args:
        return main([])

    first = args[0] if args else None
    if "--bundle" not in args and first and not str(first).startswith("-"):
        args = ["--bundle", *args]

    return main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
