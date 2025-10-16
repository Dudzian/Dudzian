#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walidacja paczki odporności Stage6 (HMAC + integralność plików).

Zintegrowana wersja łączy funkcjonalności:
1) Tryb TAR.GZ (wariant 'main'): paczka zawiera `manifest.json` oraz `manifest.json.sig`.
   - Weryfikacja podpisu HMAC manifestu (algorytm z dokumentu podpisu, np. HMAC-SHA384).
   - Weryfikacja integralności plików wskazanych w manifeście (SHA-256).
2) Tryb zewnętrznych artefaktów (wariant 'HEAD'):
   - Ścieżka do paczki (np. archiwum), osobny manifest `<bundle>.manifest.json` i podpis `<bundle>.manifest.sig`.
   - Walidacja zawartości przez `ResilienceBundleVerifier` oraz (opcjonalnie) weryfikacja podpisu `verify_signature`.

Wybór trybu:
- Jeśli `--bundle` ma rozszerzenie `.tar.gz`/`.tgz` → tryb TAR.
- W przeciwnym razie używany jest tryb zewnętrzny (HEAD). Można jawnie podać --manifest/--signature.

Klucz HMAC:
- Priorytet: `--hmac-key` → `--hmac-key-file`/`--signing-key-path` → `--hmac-key-env`/`--signing-key-env`.
- Minimalna długość: 16 bajtów (dla kompatybilności z HEAD). Zalecane ≥32 bajty.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Wariant HEAD (zewnętrzny manifest/podpis) ──────────────────────────────────
from bot_core.resilience.bundle import (  # type: ignore
    ResilienceBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)

# ── Wariant MAIN (paczka tar.gz z osadzonym manifestem/podpisem) ───────────────
from bot_core.security.signing import canonical_json_bytes  # type: ignore
from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_no_symlinks,
    _ensure_windows_safe_component,
    _ensure_windows_safe_tree,
)

LOGGER = logging.getLogger("verify_resilience_bundle")


# ==============================================================================
# Wspólne: odczyt klucza HMAC (połączone opcje HEAD + MAIN)
# ==============================================================================
def _load_signing_key_merged(
    *,
    inline_value: Optional[str],
    file_path: Optional[str],
    env_name_primary: Optional[str],
    env_name_alt: Optional[str],
) -> Optional[bytes]:
    """
    Ładuje klucz HMAC wg priorytetu:
      1) inline_value (--hmac-key)
      2) file_path (--hmac-key-file | --signing-key-path)
      3) env_name_primary (--hmac-key-env)
      4) env_name_alt (--signing-key-env)
    """
    # 1) Wartość inline
    if inline_value:
        key = inline_value.encode("utf-8")
        if len(key) < 16:
            raise ValueError("Klucz HMAC (inline) musi mieć co najmniej 16 bajtów")
        return key

    # 2) Plik z kluczem
    if file_path:
        candidate = Path(file_path).expanduser()
        _ensure_no_symlinks(candidate, label="Ścieżka klucza HMAC")
        resolved = candidate.resolve()
        _ensure_windows_safe_tree(resolved, label="Ścieżka klucza HMAC")
        if not resolved.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {resolved}")
        if os.name != "nt":
            mode = resolved.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        data = resolved.read_bytes()
        if len(data) < 16:
            raise ValueError("Klucz HMAC w pliku musi mieć co najmniej 16 bajtów")
        return data

    # 3) Zmienna środowiskowa (priorytet: --hmac-key-env)
    for env_name in (env_name_primary, env_name_alt):
        if env_name:
            value = os.environ.get(env_name)
            if not value:
                raise ValueError(f"Zmienna środowiskowa {env_name} jest pusta")
            data = value.encode("utf-8")
            if len(data) < 16:
                raise ValueError("Klucz HMAC z ENV musi mieć co najmniej 16 bajtów")
            return data

    return None


# ==============================================================================
# Tryb MAIN (tar.gz): pomocnicze funkcje weryfikacji osadzonych artefaktów
# ==============================================================================
def _is_tar_bundle(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz")


def _normalise_name(name: str) -> str:
    if name.startswith("./"):
        name = name[2:]
    return name


def _load_archive_members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        normalised = _normalise_name(member.name)
        members[normalised] = member
    return members


def _assert_safe_member(member: tarfile.TarInfo) -> None:
    if member.isdir():
        return
    if member.issym() or member.islnk():
        raise ValueError(f"Bundle zawiera nieobsługiwane linki: {member.name}")
    parts = Path(_normalise_name(member.name)).parts
    if any(part in ("..", "") for part in parts):
        raise ValueError(f"Bundle zawiera niebezpieczną ścieżkę: {member.name}")
    _ensure_windows_safe_component(
        component=parts[-1], label="Pozycja w bundle", context=member.name
    )


def _read_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Mapping[str, object]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Nie udało się odczytać {member.name} z archiwum")
    with extracted:
        data = json.load(extracted)
    if not isinstance(data, Mapping):
        raise ValueError(f"Oczekiwano obiektu JSON w {member.name}")
    return data


def _compute_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Nie udało się odczytać {member.name} z archiwum")
    hasher = hashlib.sha256()
    with extracted:
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_digest(archive: tarfile.TarFile, member: tarfile.TarInfo, algorithm: str) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Nie udało się odczytać {member.name} z archiwum")
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Nieobsługiwany algorytm digest w podpisie: {algorithm}") from exc
    with extracted:
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_manifest_signature_embedded(
    *,
    archive: tarfile.TarFile,
    manifest_member: tarfile.TarInfo,
    signature_doc: Mapping[str, object],
    signing_key: bytes,
) -> None:
    signature_section = signature_doc.get("signature")
    payload_section = signature_doc.get("payload")
    if not isinstance(signature_section, Mapping) or not isinstance(payload_section, Mapping):
        raise ValueError("Dokument podpisu musi zawierać obiekty 'signature' i 'payload'")

    payload_path = payload_section.get("path")
    if payload_path != "manifest.json":
        raise ValueError("Podpis odnosi się do nieoczekiwanej ścieżki")

    digest_items = [key for key in payload_section if key != "path"]
    if len(digest_items) != 1:
        raise ValueError("Sekcja payload musi zawierać dokładnie jeden wpis digest")
    digest_key = digest_items[0]
    expected_digest = payload_section[digest_key]
    if not isinstance(expected_digest, str):
        raise ValueError("Wartość digest musi być łańcuchem heksadecymalnym")

    actual_digest = _compute_digest(archive, manifest_member, digest_key)
    if actual_digest != expected_digest:
        raise ValueError("Digest manifestu nie zgadza się z payload w podpisie")

    algorithm = signature_section.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.upper().startswith("HMAC-"):
        raise ValueError("Nieobsługiwany algorytm podpisu")
    digest_name = algorithm.split("-", 1)[1].lower()
    if not hasattr(hashlib, digest_name):
        raise ValueError(f"Nieobsługiwany digest HMAC: {digest_name}")

    signature_value = signature_section.get("value")
    if not isinstance(signature_value, str):
        raise ValueError("Wartość podpisu musi być ciągiem base64")
    try:
        decoded_signature = base64.b64decode(signature_value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Nieprawidłowe base64 podpisu: {exc}") from exc

    expected_mac = hmac.new(
        signing_key,
        canonical_json_bytes(payload_section),
        getattr(hashlib, digest_name),
    ).digest()
    if not hmac.compare_digest(decoded_signature, expected_mac):
        raise ValueError("Weryfikacja podpisu manifestu nie powiodła się")


def _verify_tar_bundle(*, bundle_path: Path, signing_key: bytes) -> dict:
    _ensure_no_symlinks(bundle_path, label="Ścieżka bundle")
    resolved_bundle = bundle_path.resolve()
    _ensure_windows_safe_tree(resolved_bundle, label="Ścieżka bundle")
    if not resolved_bundle.is_file():
        raise FileNotFoundError(f"Bundle nie istnieje: {resolved_bundle}")

    with tarfile.open(resolved_bundle, "r:gz") as archive:
        members = _load_archive_members(archive)
        manifest_member = members.get("manifest.json") or members.get("manifest")
        signature_member = members.get("manifest.json.sig") or members.get("manifest.sig")
        if manifest_member is None or signature_member is None:
            raise ValueError("Brak manifestu lub podpisu w paczce")

        for member in members.values():
            _assert_safe_member(member)

        manifest = _read_json(archive, manifest_member)
        signature_doc = _read_json(archive, signature_member)
        _verify_manifest_signature_embedded(
            archive=archive,
            manifest_member=manifest_member,
            signature_doc=signature_doc,
            signing_key=signing_key,
        )

        files = manifest.get("files")
        if not isinstance(files, Iterable):
            raise ValueError("Manifest nie zawiera listy plików 'files'")

        verified = 0
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("Pozycja manifestu musi być mapą")
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError("Pozycja wymaga pól 'path' i 'sha256'")
            member = members.get(path) or members.get(f"./{path}")
            if member is None:
                raise ValueError(f"Brak artefaktu w paczce: {path}")
            computed = _compute_sha256(archive, member)
            if computed != digest:
                raise ValueError(f"Niezgodny SHA-256 dla {path}: oczekiwano {digest}, obliczono {computed}")
            verified += 1

    return {
        "bundle": resolved_bundle.as_posix(),
        "manifest": "manifest.json (embedded)",
        "verified_files": verified,
    }


# ==============================================================================
# Tryb HEAD (zewnętrzny manifest/podpis): wrapper łączący oryginalną logikę
# ==============================================================================
def _verify_external_manifest(
    *,
    bundle_path: Path,
    manifest_path: Optional[Path],
    signature_path: Optional[Path],
    signing_key: Optional[bytes],
) -> dict:
    bundle_path = bundle_path.expanduser()
    manifest_path = (manifest_path.expanduser() if manifest_path else bundle_path.with_suffix(".manifest.json"))
    signature_path = (signature_path.expanduser() if signature_path else bundle_path.with_suffix(".manifest.sig"))

    manifest = load_manifest(manifest_path)
    verifier = ResilienceBundleVerifier(bundle_path, manifest)
    errors = verifier.verify_files()

    signature_doc = load_signature(signature_path if signature_path.exists() else None)
    if signature_doc is not None and signing_key is not None:
        errors.extend(verify_signature(manifest, signature_doc, key=signing_key))
    elif signature_doc is not None and signing_key is None:
        print("Ostrzeżenie: dostarczono podpis bez klucza HMAC – pomijam weryfikację", file=sys.stderr)
    if not signature_doc and signing_key is not None:
        print("Ostrzeżenie: dostarczono klucz HMAC, lecz nie znaleziono podpisu", file=sys.stderr)

    if errors:
        for message in errors:
            print(f"Błąd: {message}", file=sys.stderr)
        raise ValueError("Walidacja zewnętrznego manifestu nie powiodła się")

    return {
        "bundle": bundle_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "verified_files": manifest.get("file_count") or (len(manifest.get("files", [])) if isinstance(manifest.get("files"), list) else None),
    }


# ==============================================================================
# CLI
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path, help="Ścieżka do paczki (tar.gz/tgz lub inna)")

    # Opcje HEAD (zewnętrzne ścieżki)
    p.add_argument("--manifest", help="Ścieżka do manifestu (domyślnie <bundle>.manifest.json)")
    p.add_argument("--signature", help="Ścieżka do podpisu (domyślnie <bundle>.manifest.sig)")

    # Zbiorczy zestaw opcji klucza HMAC (HEAD + MAIN)
    p.add_argument("--hmac-key", help="Wartość klucza HMAC (inline, UTF-8)")
    p.add_argument("--hmac-key-file", help="Plik z kluczem HMAC (UTF-8)")
    p.add_argument("--hmac-key-env", help="ENV z kluczem HMAC (UTF-8)")
    p.add_argument("--signing-key-path", help="Alias: plik z kluczem HMAC (UTF-8)")
    p.add_argument("--signing-key-env", help="Alias: ENV z kluczem HMAC (UTF-8)")

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Poziom logowania (domyślnie INFO)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Zbiorcze rozstrzygnięcie klucza (priorytet opisany w docstringu)
    try:
        signing_key = _load_signing_key_merged(
            inline_value=args.hmac_key,
            file_path=args.hmac_key_file or args.signing_key_path,
            env_name_primary=args.hmac_key_env,
            env_name_alt=args.signing_key_env,
        )
    except Exception as exc:
        LOGGER.error("Błąd odczytu klucza HMAC: %s", exc)
        return 1

    # Tryb automatyczny wg rozszerzenia bundle
    try:
        if _is_tar_bundle(args.bundle):
            if signing_key is None:
                raise ValueError("Tryb TAR wymaga klucza HMAC (podaj --hmac-key/--hmac-key-file/--signing-key-path/--hmac-key-env/--signing-key-env)")
            summary = _verify_tar_bundle(bundle_path=args.bundle, signing_key=signing_key)
        else:
            summary = _verify_external_manifest(
                bundle_path=args.bundle,
                manifest_path=Path(args.manifest) if args.manifest else None,
                signature_path=Path(args.signature) if args.signature else None,
                signing_key=signing_key,
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Weryfikacja nie powiodła się: %s", exc)
        return 2

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
