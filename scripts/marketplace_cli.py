#!/usr/bin/env python3
"""CLI do zarządzania lokalnym repozytorium konfiguracji Marketplace."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config_marketplace.schema import (  # noqa: E402
    MarketplaceCatalog,
    MarketplaceRepositoryConfig,
    load_catalog,
    load_repository_config,
)
from bot_core.security.marketplace_validator import (  # noqa: E402
    MarketplaceValidator,
    MarketplaceVerificationError,
)
from bot_core.marketplace import (  # noqa: E402
    PresetDocument,
    PresetSignatureVerification,
    load_private_key,
    serialize_preset_document,
    sign_preset_payload,
)

_MARKETPLACE_DIR = REPO_ROOT / "config" / "marketplace"


class MarketplaceRepository:
    """Obsługa lokalnego repozytorium Marketplace."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _MARKETPLACE_DIR
        self.catalog_path = self.root / "catalog.json"
        self.config_path = self.root / "repository.json"

    def ensure_initialized(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            default = MarketplaceRepositoryConfig()
            self.config_path.write_text(default.model_dump_json(indent=2), encoding="utf-8")
        if not self.catalog_path.exists():
            empty = MarketplaceCatalog()
            self.catalog_path.write_text(empty.model_dump_json(indent=2), encoding="utf-8")

    def load_config(self) -> MarketplaceRepositoryConfig:
        return load_repository_config(self.config_path)

    def load_catalog(self) -> MarketplaceCatalog:
        return load_catalog(self.catalog_path)

    def save_catalog(self, catalog: MarketplaceCatalog) -> None:
        self.catalog_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")

    def update_config(self, config: MarketplaceRepositoryConfig) -> None:
        self.config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def sync(self, *, source: str | None = None, force: bool = False, timeout: int = 20) -> MarketplaceCatalog:
        """Pobiera zdalny katalog i zapisuje go lokalnie."""

        self.ensure_initialized()
        config = self.load_config()
        remote = source or (config.remote_index_url if config.remote_index_url else None)
        if not remote:
            raise MarketplaceVerificationError(
                "Nie określono źródła katalogu. Użyj --source lub skonfiguruj remote_index_url."
            )

        payload, etag = self._fetch_remote(remote, etag=None if force else config.etag, timeout=timeout)
        if payload is None:
            print("Lokalny katalog jest aktualny (304 Not Modified).")
            return self.load_catalog()

        catalog = MarketplaceCatalog.model_validate_json(payload.decode("utf-8"))
        config.etag = etag
        config.last_sync_at = datetime.utcnow()
        self.save_catalog(catalog)
        self.update_config(config)
        return catalog

    def _fetch_remote(
        self,
        url: str,
        *,
        etag: str | None,
        timeout: int,
    ) -> tuple[bytes | None, str | None]:
        parsed = urlparse(url)
        if parsed.scheme in {"", "file"}:
            path = Path(parsed.path if parsed.scheme == "file" else url).expanduser()
            data = path.read_bytes()
            return data, None

        headers = {"User-Agent": "marketplace-cli/1.0"}
        if etag:
            headers["If-None-Match"] = etag
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:  # type: ignore[call-arg]
                body = response.read()
                return body, response.headers.get("ETag")
        except HTTPError as exc:  # pragma: no cover - I/O
            if exc.code == 304:
                return None, etag
            raise MarketplaceVerificationError(f"Błąd HTTP podczas pobierania katalogu: {exc}") from exc
        except URLError as exc:  # pragma: no cover - I/O
            raise MarketplaceVerificationError(f"Nie udało się pobrać katalogu: {exc}") from exc


def _load_signing_keys(paths: Sequence[str]) -> Mapping[str, bytes]:
    keys: dict[str, bytes] = {}
    for entry in paths:
        key_id, _, location = entry.partition(":")
        if not key_id or not location:
            raise MarketplaceVerificationError(
                f"Niepoprawny format klucza '{entry}'. Użyj KEY_ID:/sciezka/do/klucza"
            )
        path = Path(location).expanduser()
        keys[key_id] = path.read_bytes()
    return keys


def _load_preset_spec(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - defensywne logowanie
        raise MarketplaceVerificationError(f"Nie znaleziono pliku presetu: {path}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = yaml.safe_load(text)
    if not isinstance(document, Mapping):
        raise MarketplaceVerificationError("Specyfikacja presetu musi być obiektem JSON/YAML.")
    payload = document.get("preset") if isinstance(document.get("preset"), Mapping) else document
    if not isinstance(payload, Mapping):
        raise MarketplaceVerificationError("Preset musi być obiektem JSON/YAML.")
    return dict(payload)


def _cmd_list(repo: MarketplaceRepository, _: argparse.Namespace) -> int:
    catalog = repo.load_catalog()
    if not catalog.packages:
        print("Brak paczek w katalogu.")
        return 0
    for pkg in catalog.packages:
        tags = ", ".join(pkg.tags) if pkg.tags else "brak"
        print(f"- {pkg.package_id} ({pkg.version}) – {pkg.summary} [tagi: {tags}]")
    return 0


def _cmd_show(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    catalog = repo.load_catalog()
    metadata = catalog.find(args.package_id)
    if not metadata:
        print(f"Nie znaleziono paczki: {args.package_id}", file=sys.stderr)
        return 1
    print(json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _cmd_sync(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    catalog = repo.sync(source=args.source, force=args.force)
    print(f"Zapisano katalog ({len(catalog.packages)} paczek).")
    return 0


def _cmd_validate(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    catalog = repo.load_catalog()
    keys = _load_signing_keys(args.key)
    validator = MarketplaceValidator(signing_keys=keys)
    repository_root = repo.root
    results = validator.verify_catalog(catalog, repository_root=repository_root)
    failures = 0
    for result in results:
        if args.allow_fingerprint_mismatch:
            filtered_errors = []
            dropped = False
            for error in result.errors:
                if "fingerprint" in error.lower():
                    result.warnings.append(f"Pominięto błąd fingerprintu: {error}")
                    dropped = True
                else:
                    filtered_errors.append(error)
            result.errors = filtered_errors
            if dropped:
                result.verified = not result.errors
        status = "OK" if result.verified else "FAILED"
        print(f"[{status}] {result.package_id}:{result.artifact}")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
        for error in result.errors:
            failures += 1
            print(f"  ✖ {error}")
    return 0 if failures == 0 else 2


def _cmd_package(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    del repo
    spec_path = Path(args.input).expanduser()
    payload = _load_preset_spec(spec_path)
    private_key = load_private_key(Path(args.private_key).expanduser())
    signature = sign_preset_payload(
        payload,
        private_key=private_key,
        key_id=args.key_id,
        issuer=args.issuer,
        include_public_key=not args.omit_public_key,
    )
    verification = PresetSignatureVerification(
        verified=True,
        issues=(),
        algorithm=signature.algorithm,
        key_id=signature.key_id,
    )
    document = PresetDocument(
        payload=payload,
        signature=signature,
        verification=verification,
        fmt=args.format,
    )
    serialized = serialize_preset_document(document, format=args.format)
    ext = "yaml" if args.format == "yaml" else "json"
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    preset_id = str(metadata.get("id") or "").strip() or spec_path.stem
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else spec_path.with_suffix(f".{ext}")
    )
    output_path.write_bytes(serialized)
    print(f"Zapisano podpisany preset do {output_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(_MARKETPLACE_DIR),
        help="Ścieżka do lokalnego repozytorium Marketplace (domyślnie config/marketplace).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmd_list = sub.add_parser("list", help="Wyświetla dostępne paczki.")
    cmd_list.set_defaults(func=_cmd_list)

    cmd_show = sub.add_parser("show", help="Pokazuje pełne metadane paczki.")
    cmd_show.add_argument("package_id", help="Identyfikator paczki z katalogu.")
    cmd_show.set_defaults(func=_cmd_show)

    cmd_sync = sub.add_parser("sync", help="Pobiera najnowszy katalog z repozytorium zdalnego.")
    cmd_sync.add_argument("--source", help="Adres URL katalogu (opcjonalnie).")
    cmd_sync.add_argument("--force", action="store_true", help="Ignoruj ETag i wymuś pobranie.")
    cmd_sync.set_defaults(func=_cmd_sync)

    cmd_validate = sub.add_parser("validate", help="Weryfikuje podpisy i fingerprint paczek.")
    cmd_validate.add_argument(
        "--key",
        action="append",
        required=True,
        help="Mapa kluczy podpisu w formacie KEY_ID:/sciezka/do/klucza (można powtarzać).",
    )
    cmd_validate.add_argument(
        "--allow-fingerprint-mismatch",
        action="store_true",
        help="Traktuj błędy fingerprintu jako ostrzeżenia (do testów).",
    )
    cmd_validate.set_defaults(func=_cmd_validate)

    cmd_package = sub.add_parser(
        "package",
        help="Generuje podpisany preset ze specyfikacji JSON/YAML.",
    )
    cmd_package.add_argument("input", help="Ścieżka do pliku z definicją presetu (JSON/YAML).")
    cmd_package.add_argument(
        "--output",
        "-o",
        help="Plik wynikowy (domyślnie metadata.id z odpowiednim rozszerzeniem).",
    )
    cmd_package.add_argument("--key-id", required=True, help="Identyfikator klucza podpisu.")
    cmd_package.add_argument(
        "--private-key",
        required=True,
        help="Ścieżka do klucza prywatnego Ed25519 (PEM/base64/hex).",
    )
    cmd_package.add_argument("--issuer", help="Opcjonalny identyfikator wystawcy podpisu.")
    cmd_package.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Format wyjściowy pliku (domyślnie JSON).",
    )
    cmd_package.add_argument(
        "--omit-public-key",
        action="store_true",
        help="Nie dołączaj klucza publicznego do podpisu.",
    )
    cmd_package.set_defaults(func=_cmd_package)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo = MarketplaceRepository(Path(args.root))
    repo.ensure_initialized()
    return args.func(repo, args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
