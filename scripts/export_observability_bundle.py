#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buduje podpisane paczki obserwowalności (Grafana + Prometheus) dla Stage6.

Tryby:
- builder: używa bot_core.observability.bundle.ObservabilityBundleBuilder (jeśli dostępny).
- fallback: samodzielnie pakuje pliki do archiwum ZIP, generuje manifest i podpis HMAC.

Argumenty z obu światów są zachowane:
- --source kategoria=ścieżka (domyślnie dashboards/alerts z repo)
- --include/--exclude (glob)
- --bundle-name oraz --version (dla nazwy artefaktu)
- HMAC: --hmac-key / --hmac-key-file / --hmac-key-env / --hmac-key-id
- overrides: --overrides (JSON Stage6) – z podsumowaniem w metadanych
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no branch - deterministyczne dodanie ścieżki
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.security.signing import build_hmac_signature, canonical_json_bytes

# --- Próbujemy API „HEAD”
try:  # pragma: no cover
    from bot_core.observability import (
        AlertOverrideManager,
        load_overrides_document,
    )
    from bot_core.observability.bundle import (
        AssetSource,
        ObservabilityBundleBuilder,
    )

    _HAS_BUILDER = True
except Exception:  # pragma: no cover
    AssetSource = None  # type: ignore
    ObservabilityBundleBuilder = None  # type: ignore
    AlertOverrideManager = None  # type: ignore
    load_overrides_document = None  # type: ignore
    _HAS_BUILDER = False

# --- Próbujemy pomocników „main”
try:  # pragma: no cover
    from deploy.packaging.build_core_bundle import (  # type: ignore
        SignatureManager,
        _ensure_casefold_safe_tree,
        _ensure_no_symlinks,
        _ensure_windows_safe_component,
        _ensure_windows_safe_tree,
        _validate_bundle_version,
    )
    _HAS_SIG_MANAGER = True
except Exception:  # pragma: no cover
    SignatureManager = None  # type: ignore
    _HAS_SIG_MANAGER = False

_LOGGER = logging.getLogger("stage6.observability.bundle")

DEFAULT_SOURCES = (
    ("dashboards", REPO_ROOT / "deploy" / "grafana" / "provisioning" / "dashboards"),
    ("alert_rules", REPO_ROOT / "deploy" / "prometheus"),
)

# ------------------------- Utils wspólne -------------------------
def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_revision() -> Optional[str]:
    import subprocess

    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.decode("ascii").strip()


def _parse_metadata(values: list[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not values:
        return metadata
    for item in values:
        if "=" not in item:
            raise ValueError("Metadane muszą mieć format klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        metadata[key] = value.strip()
    return metadata


@dataclass(frozen=True)
class _SourceDecl:
    category: str
    root: Path


def _parse_sources(values: list[str] | None) -> list[_SourceDecl]:
    if values:
        out: list[_SourceDecl] = []
        for item in values:
            if "=" not in item:
                raise ValueError("Źródło musi mieć format kategoria=ścieżka")
            category, raw_path = item.split("=", 1)
            category = category.strip()
            if not category:
                raise ValueError("Kategoria źródła nie może być pusta")
            out.append(_SourceDecl(category=category, root=Path(raw_path.strip())))
        return out
    return [ _SourceDecl(category=c, root=p) for c, p in DEFAULT_SOURCES ]


def _load_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    inline = args.hmac_key
    file_path = args.hmac_key_file
    env_name = args.hmac_key_env
    key_id = args.hmac_key_id

    if inline and file_path:
        raise ValueError("Podaj klucz HMAC jako wartość lub plik, nie oba jednocześnie")

    if inline:
        potential_file = Path(inline)
        if potential_file.exists():
            file_path = potential_file
            inline = None
        else:
            key = inline.encode("utf-8")
    if file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif inline:
        key = inline.encode("utf-8")
    elif env_name:
        value = os.getenv(env_name)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza HMAC")
        key = value.encode("utf-8")
    else:
        return None, None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key, key_id


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _glob_match_any(path: str, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return True
    from fnmatch import fnmatch
    return any(fnmatch(path, pat) for pat in patterns)


def _sha256_of(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


# ------------------------- Tryb 1: Builder jeśli jest -------------------------
def _run_with_builder(args: argparse.Namespace) -> dict[str, Any]:
    assert ObservabilityBundleBuilder is not None  # for type-checkers
    assert AssetSource is not None

    sources = [
        AssetSource(category=decl.category, root=decl.root)
        for decl in _parse_sources(args.source)
    ]
    include = args.include or ["stage6*", "**/stage6*"]
    metadata = _parse_metadata(args.metadata)

    if args.overrides:
        overrides_path = Path(args.overrides)
        overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
        if load_overrides_document and AlertOverrideManager:
            overrides = load_overrides_document(overrides_data)
            overrides_manager = AlertOverrideManager(overrides)
            overrides_manager.prune_expired()
            overrides_payload = overrides_manager.to_payload()
            metadata["alert_overrides"] = {
                "path": overrides_path.as_posix(),
                "summary": overrides_payload.get("summary"),
                "annotations": overrides_payload.get("annotations"),
            }
        else:
            metadata["alert_overrides_path"] = overrides_path.as_posix()

    signing_key, key_id = _load_hmac_key(args)

    # Wersja w nazwie paczki (jeśli podano)
    bundle_name = args.bundle_name
    if args.version:
        bundle_name = f"{bundle_name}-{args.version}"

    builder = ObservabilityBundleBuilder(
        sources,
        include=include,
        exclude=args.exclude or None,
    )
    artifacts = builder.build(
        bundle_name=bundle_name,
        output_dir=Path(args.output_dir),
        metadata=metadata,
        signing_key=signing_key,
        signing_key_id=key_id,
    )
    summary = {
        "bundle": artifacts.bundle_path.as_posix(),
        "manifest": artifacts.manifest_path.as_posix(),
        "files": artifacts.manifest.get("file_count"),
        "size_bytes": artifacts.manifest.get("total_size_bytes"),
    }
    if artifacts.signature_path:
        summary["signature"] = artifacts.signature_path.as_posix()
    return summary


# --------------------- Tryb 2: fallback samodzielny ---------------------
@dataclass(frozen=True)
class _Asset:
    virtual_path: PurePosixPath
    source: Path
    kind: str  # "dashboard" | "alert" | "asset"


def _validate_out_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        if path.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {path}")
        if not path.is_dir():
            raise ValueError(f"Output directory is not a directory: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _safe_tree_checks(path: Path, label: str) -> None:
    # Jeżeli mamy helpery z „main”, używamy ich. W przeciwnym razie lekkie sprawdzenia lokalne.
    if _HAS_SIG_MANAGER:
        _ensure_no_symlinks(path, label=label)
        _ensure_windows_safe_tree(path, label=label)
        if path.is_dir():
            _ensure_casefold_safe_tree(path, label=label)
        else:
            _ensure_windows_safe_component(component=path.name, label=label, context=path.name)
    else:
        if path.is_symlink():
            raise ValueError(f"{label}: symlinki są zabronione: {path}")


def _collect_assets_fallback(
    sources: Sequence[_SourceDecl],
    *,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[_Asset]:
    assets: list[_Asset] = []
    seen: set[str] = set()

    def _should_take(rel: str) -> bool:
        if not _glob_match_any(rel, include):
            return False
        if exclude and _glob_match_any(rel, exclude):
            return False
        return True

    for decl in sources:
        root = decl.root
        _safe_tree_checks(root, label=f"{decl.category} root")
        if not root.exists():
            raise FileNotFoundError(f"Nie znaleziono ścieżki źródłowej: {root}")
        if root.is_file():
            rel = root.name
            if not _should_take(rel):
                continue
            kind = "dashboard" if decl.category.lower().startswith("dash") else (
                "alert" if decl.category.lower().startswith("alert") else "asset"
            )
            virt = PurePosixPath(decl.category) / root.name
            key = virt.as_posix().casefold()
            if key in seen:
                raise ValueError(f"Duplikat w paczce: {virt}")
            seen.add(key)
            assets.append(_Asset(virt, root, kind))
            continue

        for candidate in sorted(root.rglob("*")):
            if candidate.is_dir():
                continue
            rel = candidate.relative_to(root).as_posix()
            if not _should_take(rel):
                continue
            kind = "dashboard" if decl.category.lower().startswith("dash") else (
                "alert" if decl.category.lower().startswith("alert") else "asset"
            )
            virt = PurePosixPath(decl.category) / PurePosixPath(rel).name
            key = virt.as_posix().casefold()
            if key in seen:
                raise ValueError(f"Duplikat w paczce: {virt}")
            seen.add(key)
            _safe_tree_checks(candidate, label=f"{decl.category} file")
            assets.append(_Asset(virt, candidate.resolve(), kind))
    if not assets:
        raise ValueError("Nie wykryto żadnych plików do spakowania (sprawdź --include/--exclude)")
    return assets


_FALLBACK_SIGNATURE_SCHEMA = "stage6.observability.bundle.signature"
_FALLBACK_SIGNATURE_VERSION = "1.0"


def _write_signature_document(
    manifest_path: Path,
    *,
    manifest: Mapping[str, object],
    key: bytes,
    digest: str = "sha256",
    key_id: str | None = None,
) -> Path:
    digest_name = digest.strip().upper()
    if digest_name.startswith("HMAC-"):
        digest_name = digest_name[5:]
    if digest_name != "SHA256":
        raise ValueError(
            "Fallback exporter obsługuje wyłącznie HMAC-SHA256 – użyj buildera dla innych algorytmów"
        )

    signature = build_hmac_signature(
        manifest,
        key=key,
        algorithm="HMAC-SHA256",
        key_id=key_id,
    )
    document = {
        "schema": _FALLBACK_SIGNATURE_SCHEMA,
        "schema_version": _FALLBACK_SIGNATURE_VERSION,
        "generated_at": _now_utc_iso(),
        "manifest": manifest_path.name,
        "bundle_name": manifest.get("bundle_name"),
        "version": manifest.get("version"),
        "signature": signature,
    }
    sig_path = manifest_path.with_suffix(".sig")
    sig_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sig_path


def _write_zip_archive(source_dir: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(source_dir)
            arcname = relative.as_posix()
            archive.write(path, arcname)


def _run_fallback(args: argparse.Namespace) -> dict[str, Any]:
    # Wejście
    out_dir = _validate_out_dir(Path(args.output_dir))
    version = args.version or _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if _HAS_SIG_MANAGER:
        version = _validate_bundle_version(version)
    bundle_base = f"{args.bundle_name}-{version}"

    # Zbierz pliki
    sources = _parse_sources(args.source)
    include = args.include or ["stage6*", "**/stage6*"]
    assets = _collect_assets_fallback(sources, include=include, exclude=args.exclude or None)

    # Staging
    staging_root = Path(tempfile.mkdtemp(prefix="observability_bundle_"))
    try:
        # Kopiuj pliki do stagingu
        for asset in assets:
            dst = staging_root / asset.virtual_path.as_posix()
            _ensure_parent(dst)
            shutil.copy2(asset.source, dst)

        # Manifest
        files: List[Mapping[str, object]] = []
        total_size = 0
        for asset in assets:
            target = staging_root / asset.virtual_path.as_posix()
            size = target.stat().st_size
            total_size += size
            files.append(
                {
                    "path": asset.virtual_path.as_posix(),
                    "kind": asset.kind,
                    "sha256": _sha256_of(target),
                    "size_bytes": size,
                }
            )
        dashboards = sorted(
            asset.virtual_path.as_posix() for asset in assets if asset.kind == "dashboard"
        )
        alert_rules = sorted(
            asset.virtual_path.as_posix() for asset in assets if asset.kind == "alert"
        )

        manifest: Dict[str, object] = {
            "schema": "stage6.observability.manifest",
            "version": version,
            "bundle_name": args.bundle_name,
            "generated_at": _now_utc_iso(),
            "git_revision": _git_revision(),
            "file_count": len(files),
            "total_size_bytes": int(total_size),
            "files": sorted(files, key=lambda x: x["path"]),
        }
        if dashboards:
            manifest["dashboards"] = dashboards
        if alert_rules:
            manifest["alert_rules"] = alert_rules

        # Metadane & overrides (opcjonalnie)
        meta = _parse_metadata(args.metadata)
        if args.overrides:
            overrides_path = Path(args.overrides)
            try:
                overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
                if load_overrides_document and AlertOverrideManager:
                    overrides = load_overrides_document(overrides_data)
                    mgr = AlertOverrideManager(overrides)
                    mgr.prune_expired()
                    payload = mgr.to_payload()
                    meta["alert_overrides"] = {
                        "path": overrides_path.as_posix(),
                        "summary": payload.get("summary"),
                        "annotations": payload.get("annotations"),
                    }
                else:
                    meta["alert_overrides_path"] = overrides_path.as_posix()
            except Exception as exc:
                _LOGGER.warning("Nie udało się wczytać overrides: %s", exc)
        if meta:
            manifest["metadata"] = meta

        manifest_path = staging_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Podpis
        signing_key, key_id = _load_hmac_key(args)
        sig_path: Optional[Path] = None
        if signing_key:
            sig_path = _write_signature_document(
                manifest_path,
                manifest=manifest,
                key=signing_key,
                digest=args.digest,
                key_id=key_id,
            )

        # Skopiuj manifest i podpis obok archiwum (wygodniej do weryfikacji offline)
        final_manifest = out_dir / f"{bundle_base}.manifest.json"
        shutil.copy2(manifest_path, final_manifest)
        final_sig: Optional[Path] = None
        if sig_path:
            final_sig = out_dir / f"{bundle_base}.manifest.sig"
            shutil.copy2(sig_path, final_sig)

        # Usuń metadane ze stagingu, aby archiwum zawierało wyłącznie aktywa
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        if sig_path:
            try:
                sig_path.unlink()
            except FileNotFoundError:
                pass

        # Archiwum
        archive_path = out_dir / f"{bundle_base}.zip"
        if archive_path.exists():
            raise FileExistsError(f"Paczka już istnieje: {archive_path}")
        _write_zip_archive(staging_root, archive_path)

        summary = {
            "bundle": archive_path.as_posix(),
            "manifest": final_manifest.as_posix(),
            "files": manifest["file_count"],
            "size_bytes": manifest["total_size_bytes"],
        }
        if final_sig:
            summary["signature"] = final_sig.as_posix()
        return summary
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


# ------------------------- CLI -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Buduje paczkę obserwowalności Stage6 (z manifestem i podpisem HMAC)."
    )
    # wspólne
    p.add_argument("--output-dir", default=str(REPO_ROOT / "var" / "observability"),
                   help="Katalog docelowy dla paczki")
    p.add_argument(
        "--bundle-name",
        default="observability-bundle",
        help="Prefiks nazwy paczki (domyślnie observability-bundle)",
    )
    p.add_argument("--version", help="Identyfikator wersji paczki (np. 2025.10.16)")
    p.add_argument("--metadata", action="append", help="Metadane w formacie klucz=wartość (wielokrotnie)")
    p.add_argument("--source", action="append",
                   help="Źródło w formacie kategoria=ścieżka (domyślnie Stage6 dashboards/alerts)")
    p.add_argument("--include", action="append",
                   help="Wzorce plików do uwzględnienia (glob, wielokrotnie; domyślnie stage6*)")
    p.add_argument("--exclude", action="append",
                   help="Wzorce plików do pominięcia (glob, wielokrotnie)")
    p.add_argument("--overrides", help="Ścieżka do pliku override alertów (JSON Stage6)")
    p.add_argument(
        "--mode",
        choices=["fallback", "builder"],
        default="fallback",
        help="Tryb eksportu: fallback (tar.gz + manifest) lub builder (jeśli dostępny).",
    )
    # podpis
    p.add_argument(
        "--hmac-key",
        "--signing-key",
        dest="hmac_key",
        help="Wartość klucza HMAC",
    )
    p.add_argument(
        "--hmac-key-file",
        "--signing-key-file",
        dest="hmac_key_file",
        help="Plik z kluczem HMAC (UTF-8)",
    )
    p.add_argument(
        "--hmac-key-env",
        "--signing-key-env",
        dest="hmac_key_env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC",
    )
    p.add_argument(
        "--hmac-key-id",
        "--key-id",
        dest="hmac_key_id",
        help="Identyfikator klucza HMAC umieszczony w podpisie",
    )
    p.add_argument(
        "--digest",
        default="sha256",
        help="Algorytm skrótu dla HMAC (sha256/sha384/sha512; domyślnie sha256)",
    )
    # logowanie
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        if args.mode == "builder":
            if not _HAS_BUILDER:
                raise RuntimeError("Tryb builder jest niedostępny – brak ObservabilityBundleBuilder")
            summary = _run_with_builder(args)
        else:
            summary = _run_fallback(args)
    except Exception as exc:  # pragma: no cover
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
