#!/usr/bin/env python3
"""CLI do zarządzania lokalnym repozytorium konfiguracji Marketplace."""
from __future__ import annotations

import argparse
import json
import re
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
    ExchangePresetValidationResult,
    generate_exchange_presets,
    PresetDocument,
    PresetSignatureVerification,
    load_private_key,
    serialize_preset_document,
    sign_preset_payload,
    reconcile_exchange_presets,
    validate_exchange_presets,
)

_PACKAGE_VERSION_REF_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._-]{2,63}@" r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
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


def _validate_release_metadata(
    catalog: MarketplaceCatalog,
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    """Waliduje dodatkowe pola release/versioning katalogu."""

    errors: list[str] = []
    warnings: list[str] = []
    presets_root = repo_root / "config" / "marketplace" / "presets"

    for package in catalog.packages:
        review_status = (package.release.review_status or "").strip().lower()
        if review_status not in {"", "pending", "in_review", "approved", "rejected"}:
            warnings.append(
                f"{package.package_id}: nieznany status recenzji '{package.release.review_status}'."
            )
        if review_status == "approved":
            if not package.release.reviewers:
                errors.append(
                    f"{package.package_id}: zatwierdzona paczka wymaga listy recenzentów."
                )
            if package.release.approved_at is None:
                errors.append(
                    f"{package.package_id}: brak pola approved_at dla zatwierdzonej paczki."
                )

        for entry in package.exchange_compatibility:
            status = (entry.status or "").strip().lower()
            if status in {"certified", "production"} and entry.last_verified_at is None:
                errors.append(
                    f"{package.package_id}: wpis kompatybilności {entry.exchange} w statusie '{entry.status}' wymaga last_verified_at."
                )

        for ref in package.versioning.supersedes + package.versioning.superseded_by:
            if not _PACKAGE_VERSION_REF_PATTERN.match(ref):
                errors.append(
                    f"{package.package_id}: niepoprawny format odwołania wersji '{ref}'."
                )

        source = package.versioning.source
        if source:
            spec_path = presets_root / source
            if not spec_path.exists():
                errors.append(
                    f"{package.package_id}: wskazana ścieżka presetu '{source}' nie istnieje."
                )
        else:
            warnings.append(
                f"{package.package_id}: brak pola versioning.source – nie będzie możliwe automatyczne odtworzenie presetu."
            )

    return errors, warnings


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
    rel_errors, rel_warnings = _validate_release_metadata(catalog, repo.root)
    for warning in rel_warnings:
        print(f"[META] ⚠ {warning}")
    for error in rel_errors:
        failures += 1
        print(f"[META] ✖ {error}")
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


def _cmd_generate_exchange_presets(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    del repo
    exchanges_dir = Path(args.exchanges_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    private_key = Path(args.private_key).expanduser()
    documents = generate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        private_key=private_key,
        key_id=args.key_id,
        issuer=args.issuer,
        version=args.version,
        version_strategy=args.version_strategy,
        selected_exchanges=args.exchange,
    )

    if not documents:
        print("Nie znaleziono konfiguracji giełd do wygenerowania presetów.")
        return 0

    for document in documents:
        status = "OK" if document.verification.verified else "FAILED"
        location = document.path if document.path else output_dir
        print(f"- {document.preset_id} [{status}] → {location}")

    print(f"Wygenerowano {len(documents)} presetów do katalogu {output_dir}")
    return 0


def _cmd_check_exchange_presets(repo: MarketplaceRepository, args: argparse.Namespace) -> int:
    del repo
    exchanges_dir = Path(args.exchanges_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    signing_keys = _load_signing_keys(args.key) if args.key else None

    results = validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version=args.version,
        signing_keys=signing_keys,
        selected_exchanges=args.exchange,
        version_strategy=args.version_strategy,
    )

    if not results:
        print("Brak definicji giełd do walidacji.")
        return 0

    failures = 0
    for result in results:
        failures += _print_exchange_preset_status(result)

    if args.fix:
        if not args.private_key or not args.key_id:
            print(
                "Opcje --private-key oraz --key-id są wymagane podczas użycia --fix.",
                file=sys.stderr,
            )
            return 2
        if failures == 0:
            print("Brak problemów do naprawienia.")
            return 0

        print("\nRozpoczynam naprawę presetów giełdowych...")
        repaired_results = reconcile_exchange_presets(
            exchanges_dir=exchanges_dir,
            output_dir=output_dir,
            private_key=Path(args.private_key).expanduser(),
            key_id=args.key_id,
            issuer=args.issuer,
            version=args.version,
            signing_keys=signing_keys,
            remove_orphans=args.remove_orphans,
            selected_exchanges=args.exchange,
            version_strategy=args.version_strategy,
        )

        print("\nWynik po naprawie:")
        failures = 0
        for result in repaired_results:
            failures += _print_exchange_preset_status(result)

        if failures:
            print(
                "Nie udało się w pełni naprawić presetów giełdowych.",
                file=sys.stderr,
            )
            return 1

        print("Wszystkie presety giełdowe zostały naprawione i zweryfikowane.")
        return 0

    if failures:
        print(f"Wykryto {failures} problemów z presetami giełdowymi.", file=sys.stderr)
        return 1

    print("Wszystkie presety giełdowe są aktualne i poprawnie podpisane.")
    return 0


def _print_exchange_preset_status(result: ExchangePresetValidationResult) -> int:
    problems: list[str] = []
    if not result.exists:
        problems.append("brak pliku")
    if not result.verified:
        problems.append("niepoprawny podpis")
    if not result.up_to_date:
        problems.append("wymaga regeneracji")

    status = "OK" if not problems else ", ".join(problems)
    location = result.preset_path if result.exists else "(brak pliku)"
    version_info = result.current_version or "brak"
    print(
        f"- {result.preset_id} → {location} | aktualna={version_info} | oczekiwana={result.expected_version} | {status}"
    )
    for issue in result.issues:
        print(f"    - {issue}")
    return 1 if problems else 0


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

    cmd_generate = sub.add_parser(
        "generate-exchange-presets",
        help="Generuje podpisane presety dla wszystkich giełd z config/exchanges.",
    )
    cmd_generate.add_argument(
        "--exchanges-dir",
        default="config/exchanges",
        help="Katalog z plikami konfiguracji giełd (domyślnie config/exchanges).",
    )
    cmd_generate.add_argument(
        "--output-dir",
        default="config/marketplace/presets/exchanges",
        help="Katalog docelowy wygenerowanych presetów.",
    )
    cmd_generate.add_argument("--key-id", required=True, help="Identyfikator klucza podpisu.")
    cmd_generate.add_argument(
        "--private-key",
        required=True,
        help="Klucz prywatny Ed25519 używany do podpisania presetów.",
    )
    cmd_generate.add_argument("--issuer", help="Opcjonalny identyfikator wydawcy podpisu.")
    cmd_generate.add_argument("--version", default="1.0.0", help="Wersja generowanych presetów.")
    cmd_generate.add_argument(
        "--version-strategy",
        choices=["static", "spec-hash"],
        default="static",
        help="Strategia nadawania wersji presetom (domyślnie static).",
    )
    cmd_generate.add_argument(
        "--exchange",
        "-e",
        action="append",
        help="Opcjonalnie ogranicz generowanie do wybranych giełd (powtarzalny parametr).",
    )
    cmd_generate.set_defaults(func=_cmd_generate_exchange_presets)

    cmd_check = sub.add_parser(
        "check-exchange-presets",
        help="Waliduje istniejące presety giełdowe względem definicji YAML.",
    )
    cmd_check.add_argument(
        "--exchanges-dir",
        default="config/exchanges",
        help="Katalog z plikami definicji giełd (domyślnie config/exchanges).",
    )
    cmd_check.add_argument(
        "--output-dir",
        default="config/marketplace/presets/exchanges",
        help="Katalog z podpisanymi presetami giełdowymi.",
    )
    cmd_check.add_argument(
        "--version",
        help="Wymuszona wersja metadanych do porównania (domyślnie wartość z pliku).",
    )
    cmd_check.add_argument(
        "--version-strategy",
        choices=["static", "spec-hash"],
        default="static",
        help="Strategia wyliczania oczekiwanej wersji presetu (domyślnie static).",
    )
    cmd_check.add_argument(
        "--key",
        action="append",
        help="Opcjonalne klucze weryfikacji podpisu KEY_ID:/ścieżka/do/klucza.",
    )
    cmd_check.add_argument(
        "--fix",
        action="store_true",
        help="Automatycznie regeneruje brakujące lub przestarzałe presety.",
    )
    cmd_check.add_argument(
        "--private-key",
        help="Klucz prywatny Ed25519 używany do naprawy presetów.",
    )
    cmd_check.add_argument(
        "--key-id",
        help="Identyfikator klucza podpisu używany podczas naprawy.",
    )
    cmd_check.add_argument(
        "--issuer",
        help="Identyfikator wystawcy podpisu używany przy naprawie (opcjonalnie).",
    )
    cmd_check.add_argument(
        "--remove-orphans",
        action="store_true",
        help="Usuwa osierocone pliki presetów z katalogu docelowego.",
    )
    cmd_check.add_argument(
        "--exchange",
        "-e",
        action="append",
        help="Waliduj jedynie wskazane giełdy (powtarzalny parametr).",
    )
    cmd_check.set_defaults(func=_cmd_check_exchange_presets)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo = MarketplaceRepository(Path(args.root))
    repo.ensure_initialized()
    return args.func(repo, args)


validate_release_metadata = _validate_release_metadata


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
