#!/usr/bin/env python3
"""Walidator presetów Marketplace."""

import argparse
import base64
import hmac
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _strip_conflicting_paths() -> None:
    conflict_dirs = {REPO_ROOT / "scripts", REPO_ROOT / "deploy", REPO_ROOT / "tests"}
    cleaned: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() in conflict_dirs:
                continue
        except Exception:
            cleaned.append(entry)
            continue
        cleaned.append(entry)
    sys.path = cleaned


_strip_conflicting_paths()

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# Importujemy moduł bezpośrednio z pliku, aby ominąć ciężkie zależności z
# bot_core.security.__init__, które nie są potrzebne w tym skrypcie (np.
# SQLAlchemy). Ładowanie modułu w ten sposób pozwala uruchamiać walidator w
# środowiskach CI bez instalacji pełnego zestawu zależności.

_CATALOG_SIGNATURES_PATH = REPO_ROOT / "bot_core" / "security" / "catalog_signatures.py"
_catalog_signatures_spec = importlib.util.spec_from_file_location(
    "bot_core.security.catalog_signatures", _CATALOG_SIGNATURES_PATH
)
if _catalog_signatures_spec is None or _catalog_signatures_spec.loader is None:
    raise ImportError(f"Nie można wczytać modułu katalogu z {_CATALOG_SIGNATURES_PATH}")
_catalog_signatures = importlib.util.module_from_spec(_catalog_signatures_spec)
_catalog_signatures_spec.loader.exec_module(_catalog_signatures)

verify_catalog_signature_file = _catalog_signatures.verify_catalog_signature_file

REQUIRED_RELEASE_FIELDS = ("review_status", "approved_at", "reviewers")


def _load_hmac_key(path: Path) -> bytes:
    key = path.read_bytes().strip()
    if not key:
        raise ValueError(f"Plik klucza HMAC {path} jest pusty")
    return key


def _load_ed25519_public_key(source: Path | bytes | bytearray | str) -> ed25519.Ed25519PublicKey:
    if isinstance(source, Path):
        data = source.read_bytes()
        label = str(source)
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        label = "<bytes>"
    else:
        data = str(source).encode("utf-8")
        label = "<inline>"

    try:
        return serialization.load_pem_public_key(data)  # type: ignore[return-value]
    except ValueError:
        try:
            return ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(data))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Nie udało się wczytać klucza publicznego Ed25519 z {label}") from exc


def _verify_signature(
    target: Path,
    *,
    hmac_key: bytes,
    hmac_key_id: str,
    ed25519_key: ed25519.Ed25519PublicKey,
    ed25519_key_id: str,
) -> list[str]:
    problems: list[str] = []
    sig_path = target.with_suffix(target.suffix + ".sig")
    if not sig_path.exists():
        return [f"{target.name}: brak pliku podpisu {sig_path.name}"]

    try:
        signature = json.loads(sig_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"{sig_path.name}: nie udało się wczytać JSON podpisu ({exc})"]

    content = target.read_bytes()
    digest = sha256(content).hexdigest()
    if signature.get("sha256") != digest:
        problems.append(
            f"{sig_path.name}: suma sha256 {signature.get('sha256')} nie zgadza się z {digest}"
        )

    hmac_body = signature.get("hmac") or {}
    if hmac_body.get("key_id") != hmac_key_id:
        problems.append(f"{sig_path.name}: oczekiwano key_id {hmac_key_id} dla HMAC")
    else:
        try:
            expected = base64.b64encode(hmac.new(hmac_key, content, sha256).digest()).decode(
                "ascii"
            )
            if hmac_body.get("value") != expected:
                problems.append(f"{sig_path.name}: niepoprawny podpis HMAC")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{sig_path.name}: błąd weryfikacji HMAC ({exc})")

    ed_body = signature.get("ed25519") or {}
    if ed_body.get("key_id") != ed25519_key_id:
        problems.append(f"{sig_path.name}: oczekiwano key_id {ed25519_key_id} dla Ed25519")
    else:
        try:
            ed_value = base64.b64decode(ed_body.get("value") or b"")
            ed25519_key.verify(ed_value, content)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{sig_path.name}: niepoprawny podpis Ed25519 ({exc})")
    return problems


def _validate_catalog_files(
    *,
    catalog_path: Path,
    markdown_path: Path | None,
    hmac_key: bytes,
    ed25519_key_material: bytes,
    require_signature: bool,
    require_markdown: bool,
) -> list[str]:
    problems: list[str] = []

    def _validate_single(path: Path) -> None:
        if not path.exists():
            problems.append(f"{path.name}: katalog Marketplace nie istnieje")
            return
        if require_signature:
            problems.extend(
                verify_catalog_signature_file(
                    path,
                    hmac_key=hmac_key,
                    ed25519_key=ed25519_key_material,
                )
            )

    _validate_single(catalog_path)
    if require_markdown and markdown_path is not None:
        _validate_single(markdown_path)

    return problems


def _validate_file(
    path: Path,
    *,
    hmac_key: bytes,
    hmac_key_id: str,
    ed25519_key: ed25519.Ed25519PublicKey,
    ed25519_key_id: str,
    require_markdown: bool,
    require_signature: bool,
) -> list[str]:
    problems: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"{path.name}: nie udało się wczytać JSON ({exc})"]

    catalog = data.get("catalog")
    if not isinstance(catalog, dict):
        return [f"{path.name}: brak sekcji catalog"]

    release = catalog.get("release")
    if not isinstance(release, dict):
        problems.append(f"{path.name}: brak sekcji catalog.release")
    else:
        for field in REQUIRED_RELEASE_FIELDS:
            value = release.get(field)
            if not value:
                problems.append(f"{path.name}: pole release.{field} jest puste")

    compatibility = catalog.get("exchange_compatibility")
    if not compatibility:
        problems.append(f"{path.name}: brak sekcji exchange_compatibility")

    if require_markdown:
        md_path = path.with_suffix(".md")
        if not md_path.exists():
            problems.append(f"{path.name}: brak pliku markdown {md_path.name}")
        elif md_path.stat().st_size == 0:
            problems.append(f"{path.name}: markdown {md_path.name} jest pusty")

    if require_signature:
        problems.extend(
            _verify_signature(
                path,
                hmac_key=hmac_key,
                hmac_key_id=hmac_key_id,
                ed25519_key=ed25519_key,
                ed25519_key_id=ed25519_key_id,
            )
        )
        if require_markdown and path.with_suffix(".md").exists():
            problems.extend(
                _verify_signature(
                    path.with_suffix(".md"),
                    hmac_key=hmac_key,
                    hmac_key_id=hmac_key_id,
                    ed25519_key=ed25519_key,
                    ed25519_key_id=ed25519_key_id,
                )
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprawdza kompletność presetów Marketplace")
    parser.add_argument(
        "--presets",
        type=Path,
        default=Path("config/marketplace/presets"),
        help="Katalog ze specyfikacjami presetów",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=15,
        help="Minimalna liczba presetów wymagana do publikacji",
    )
    parser.add_argument(
        "--hmac-key",
        type=Path,
        default=Path("config/marketplace/keys/dev-hmac.key"),
        help="Klucz HMAC do weryfikacji podpisów presetów",
    )
    parser.add_argument(
        "--hmac-key-id",
        default="dev-hmac",
        help="Identyfikator klucza HMAC (spodziewany w podpisach)",
    )
    parser.add_argument(
        "--ed25519-key",
        type=Path,
        default=Path("config/marketplace/keys/dev-presets-ed25519.pub"),
        help="Publiczny klucz Ed25519 do weryfikacji podpisów presetów",
    )
    parser.add_argument(
        "--ed25519-key-id",
        default="dev-presets-ed25519",
        help="Identyfikator klucza Ed25519 (spodziewany w podpisach)",
    )
    parser.add_argument(
        "--allow-missing-signatures",
        action="store_true",
        help="Nie weryfikuj podpisów HMAC/Ed25519 (do lokalnych testów)",
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Pomiń wymóg obecności plików Markdown dla presetów",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("config/marketplace/catalog.json"),
        help="Ścieżka do katalogu Marketplace (JSON) do weryfikacji podpisu",
    )
    parser.add_argument(
        "--catalog-markdown",
        type=Path,
        default=Path("config/marketplace/catalog.md"),
        help="Ścieżka do katalogu Marketplace (Markdown) do weryfikacji podpisu",
    )
    args = parser.parse_args()

    hmac_key = _load_hmac_key(args.hmac_key)
    ed25519_key_material = args.ed25519_key.read_bytes()
    ed25519_key = _load_ed25519_public_key(ed25519_key_material)

    files = sorted(
        path
        for path in args.presets.rglob("*.json")
        if ".meta" not in path.parts and not path.name.endswith(".sig")
    )
    if len(files) < args.min_count:
        print(
            f"ERROR: Znaleziono {len(files)} presetów w {args.presets}, wymagane {args.min_count}",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for path in files:
        failures.extend(
            _validate_file(
                path,
                hmac_key=hmac_key,
                hmac_key_id=args.hmac_key_id,
                ed25519_key=ed25519_key,
                ed25519_key_id=args.ed25519_key_id,
                require_markdown=not args.skip_markdown,
                require_signature=not args.allow_missing_signatures,
            )
        )

    failures.extend(
        _validate_catalog_files(
            catalog_path=args.catalog,
            markdown_path=args.catalog_markdown,
            hmac_key=hmac_key,
            ed25519_key_material=ed25519_key_material,
            require_signature=not args.allow_missing_signatures,
            require_markdown=not args.skip_markdown,
        )
    )

    if failures:
        print("ERROR: Niektóre presety są niekompletne:", file=sys.stderr)
        for problem in failures:
            print(f" - {problem}", file=sys.stderr)
        return 3

    print(f"OK: {len(files)} presetów spełnia wymagania katalogu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
