#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buduje podpisane paczki odporności Stage6 (resilience bundle).

Tryby:
- builder: używa bot_core.resilience.bundle.ResilienceBundleBuilder (HEAD).
- fallback: samodzielnie pakuje raport/artefakty/katalog do tar.gz i podpisuje manifest (Main).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- HEAD API (jeśli jest)
try:  # pragma: no cover
    from bot_core.resilience.bundle import ResilienceBundleBuilder  # type: ignore
    _HAS_BUILDER = True
except Exception:  # pragma: no cover
    ResilienceBundleBuilder = None  # type: ignore
    _HAS_BUILDER = False

# --- Main utilsy podpisu (jeśli są)
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

_LOGGER = logging.getLogger("stage6.resilience.bundle")

DEFAULT_REPORT = (
    REPO_ROOT
    / "var"
    / "audit"
    / "stage6"
    / "resilience"
    / "resilience_failover_report.json"
)

# ---------------------------- wspólne utils ----------------------------
def _now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_revision() -> Optional[str]:
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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _sha256_of(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _sign_manifest_fallback(
    manifest_path: Path, *, key: bytes, algorithm: str = "sha384", key_id: str | None = None
) -> Path:
    alg = algorithm.lower().replace("hmac-", "")
    if alg not in {"sha256", "sha384", "sha512"}:
        alg = "sha384"
    digest = hmac.new(key, manifest_path.read_bytes(), getattr(hashlib, alg)).hexdigest()
    doc = {
        "schema": "stage6.resilience.signature",
        "signed_at": _now_utc_iso(),
        "algorithm": f"HMAC-{alg.upper()}",
        "key_id": key_id,
        "target": "manifest.json",
        "digest": digest,
    }
    sig_path = manifest_path.with_suffix(".sig")
    sig_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sig_path


# ------------------------------ fallback: raport/artefakty ------------------------------
@dataclass(frozen=True)
class _Asset:
    virtual_path: PurePosixPath
    source: Path
    kind: str  # "report" | "signature" | "extra" | "artifact"


def _validate_file(path: Path, *, label: str) -> Path:
    _safe_tree_checks(path, label=label)
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"{label} must be a file: {resolved}")
    if _HAS_SIG_MANAGER:
        _ensure_windows_safe_component(resolved.name, label=label, context=resolved.name)
    return resolved


def _collect_from_report(report: Path, signature: Optional[Path], extras: Sequence[Path]) -> list[_Asset]:
    assets: list[_Asset] = []
    seen: set[str] = set()

    def add(source: Path, vpath: PurePosixPath, kind: str) -> None:
        key = vpath.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate entry: {vpath}")
        seen.add(key)
        assets.append(_Asset(vpath, source, kind))

    report_file = _validate_file(report, label="Resilience report")
    add(report_file, PurePosixPath("reports") / report_file.name, "report")

    if signature is None:
        candidate = report_file.with_suffix(report_file.suffix + ".sig")
        signature = candidate if candidate.exists() else None
    if signature is not None:
        sig_file = _validate_file(signature, label="Resilience report signature")
        add(sig_file, PurePosixPath("reports") / f"{report_file.name}.sig", "signature")

    for extra in extras:
        resolved = _validate_file(extra, label="Extra artefact")
        add(resolved, PurePosixPath("extras") / resolved.name, "extra")

    return assets


def _collect_from_source_dir(source: Path, include: Sequence[str] | None, exclude: Sequence[str] | None) -> list[_Asset]:
    source = source.expanduser()
    _safe_tree_checks(source, label="Source directory")
    if not source.exists() or not source.is_dir():
        raise ValueError(f"Source must be an existing directory: {source}")
    from fnmatch import fnmatch
    def take(rel: str) -> bool:
        ok = True
        if include:
            ok = any(fnmatch(rel, pat) for pat in include)
        if exclude and any(fnmatch(rel, pat) for pat in exclude):
            ok = False
        return ok
    assets: list[_Asset] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(source).as_posix()
        if not take(rel):
            continue
        assets.append(_Asset(PurePosixPath("artifacts") / PurePosixPath(rel), path.resolve(), "artifact"))
    if not assets:
        raise ValueError("Brak plików do spakowania (sprawdź --include/--exclude)")
    return assets


def _copy_assets(assets: Iterable[_Asset], dest_root: Path) -> None:
    for a in assets:
        dst = dest_root / a.virtual_path.as_posix()
        _ensure_parent(dst)
        shutil.copy2(a.source, dst)


def _build_manifest(
    *,
    version: str,
    bundle_name: str,
    assets: Sequence[_Asset],
    staging_root: Path,
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, object]:
    files: list[Mapping[str, object]] = []
    total = 0
    for a in assets:
        target = staging_root / a.virtual_path.as_posix()
        size = target.stat().st_size
        total += size
        files.append(
            {
                "path": a.virtual_path.as_posix(),
                "kind": a.kind,
                "sha256": _sha256_of(target),
                "size_bytes": size,
            }
        )
    manifest: dict[str, object] = {
        "schema": "stage6.resilience.manifest",
        "version": version,
        "bundle_name": bundle_name,
        "generated_at": _now_utc_iso(),
        "git_revision": _git_revision(),
        "file_count": len(files),
        "total_size_bytes": int(total),
        "files": sorted(files, key=lambda x: x["path"]),
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    return manifest


def _tar_dir(src: Path, dst: Path) -> None:
    with tarfile.open(dst, "w:gz") as tar:
        tar.add(src, arcname=".")


# ------------------------------ CLI ------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Buduje paczkę odporności Stage6 (manifest + opcjonalny podpis HMAC)."
    )
    # Wejścia (można użyć obu naraz – zostaną zmergowane):
    p.add_argument("--source", help="Katalog z artefaktami (HEAD)", type=str)
    p.add_argument("--include", action="append", help="Wzorce do uwzględnienia w --source (glob, wielokrotnie)")
    p.add_argument("--exclude", action="append", help="Wzorce do pominięcia w --source (glob, wielokrotnie)")

    p.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Ścieżka do JSON raportu failover (Main)")
    p.add_argument("--signature", type=Path, help="Ścieżka do podpisu raportu (domyślnie <report>.sig jeśli istnieje)")
    p.add_argument("--extra", action="append", default=[], type=Path, help="Dodatkowy plik do paczki (wielokrotnie)")

    # Wyjścia / meta
    p.add_argument("--output-dir", default=str(REPO_ROOT / "var" / "resilience"), help="Katalog docelowy")
    p.add_argument("--bundle-name", default="stage6-resilience", help="Prefiks nazwy paczki")
    p.add_argument("--version", help="Wersja paczki (np. 2025.10.16); w builder-mode doklejane do nazwy")
    p.add_argument("--metadata", action="append", help="Metadane klucz=wartość (wielokrotnie)")

    # Podpis (obsługujemy oba style)
    p.add_argument("--hmac-key", help="Wartość klucza HMAC")
    p.add_argument("--hmac-key-file", help="Plik z kluczem HMAC (UTF-8)")
    p.add_argument("--hmac-key-env", help="Zmienna środowiskowa z kluczem HMAC")
    p.add_argument("--signing-key-path", type=Path, help="(alias Main) Plik z kluczem HMAC")
    p.add_argument("--signing-key-env", help="(alias Main) Zmienna środowiskowa z kluczem HMAC")
    p.add_argument("--key-id", help="Identyfikator klucza HMAC w podpisie")
    p.add_argument("--digest", default="sha384", help="Algorytm skrótu HMAC: sha256/sha384/sha512 (domyślnie sha384)")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def _resolve_hmac_key(args: argparse.Namespace) -> Tuple[Optional[bytes], Optional[str]]:
    # Priorytet: hmac-key / hmac-key-file / hmac-key-env, potem aliasy Main
    if args.hmac_key:
        key = args.hmac_key.encode("utf-8")
    elif args.hmac_key_file:
        path = Path(args.hmac_key_file).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif args.hmac_key_env and os.getenv(args.hmac_key_env):
        key = os.environ[args.hmac_key_env].encode("utf-8")
    elif args.signing_key_path:
        path = Path(args.signing_key_path).expanduser()
        if _HAS_SIG_MANAGER:
            _ensure_no_symlinks(path, label="Signing key path")
            _ensure_windows_safe_tree(path, label="Signing key path")
            if not path.is_file():
                raise ValueError(f"Signing key path must reference a file: {path}")
            if os.name != "nt":
                mode = path.stat().st_mode
                if mode & (stat.S_IRWXG | stat.S_IRWXO):
                    raise ValueError("Signing key permissions too permissive; expected 600")
        elif path.is_symlink():
            raise ValueError(f"Signing key path must not be a symlink: {path}")
        key = path.read_bytes()
    elif args.signing_key_env and os.getenv(args.signing_key_env):
        key = os.environ[args.signing_key_env].encode("utf-8")
    else:
        return None, None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key, (args.key_id or None)


def _builder_mode_possible(args: argparse.Namespace) -> bool:
    # Użyj buildera tylko gdy mamy --source i brak specyficznych wejść Main (report/extras).
    return _HAS_BUILDER and args.source and not args.extra and args.report == DEFAULT_REPORT and args.signature is None


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    try:
        out_dir = _validate_out_dir(Path(args.output_dir))
        metadata = _parse_metadata(args.metadata)

        # ------------------- Tryb builder (HEAD) -------------------
        if _builder_mode_possible(args):
            signing_key, key_id = _resolve_hmac_key(args)
            # nazwa paczki z wersją (jeśli podano)
            bundle_name = args.bundle_name if not args.version else f"{args.bundle_name}-{args.version}"
            builder = ResilienceBundleBuilder(
                Path(args.source),
                include=args.include,
                exclude=args.exclude,
            )
            artifacts = builder.build(
                bundle_name=bundle_name,
                output_dir=out_dir,
                metadata=metadata,
                signing_key=signing_key,
                signing_key_id=key_id,
            )
            summary = {
                "bundle": artifacts.bundle_path.as_posix(),
                "manifest": artifacts.manifest_path.as_posix(),
                "files": artifacts.manifest["file_count"],
                "size_bytes": artifacts.manifest["total_size_bytes"],
            }
            if artifacts.signature_path:
                summary["signature"] = artifacts.signature_path.as_posix()
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        # ------------------- Tryb fallback (Main + rozszerzenia) -------------------
        # Zbierz assety: raport + extras
        assets: list[_Asset] = []
        if args.report:
            assets.extend(_collect_from_report(args.report, args.signature, args.extra))

        # Jeżeli podano również source (a nie mamy buildera) – dodaj pliki z katalogu
        if args.source:
            assets.extend(_collect_from_source_dir(Path(args.source), args.include, args.exclude))

        if not assets:
            raise ValueError("Brak wejść. Podaj --report lub --source (albo oba).")

        # Staging
        staging_root = Path(tempfile.mkdtemp(prefix="resilience_bundle_"))
        try:
            _copy_assets(assets, staging_root)

            # Wersja
            version = args.version or _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            if _HAS_SIG_MANAGER:
                version = _validate_bundle_version(version)

            # Manifest
            manifest = _build_manifest(
                version=version,
                bundle_name=args.bundle_name,
                assets=assets,
                staging_root=staging_root,
                metadata=metadata,
            )
            manifest_path = staging_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            # Podpis
            signing_key, key_id = _resolve_hmac_key(args)
            sig_path: Optional[Path] = None
            if signing_key:
                if _HAS_SIG_MANAGER:
                    mgr = SignatureManager(signing_key, digest_algorithm=args.digest.lower(), key_id=key_id)
                    digest_val = mgr.digest_file(manifest_path)
                    sig_path = manifest_path.with_suffix(".sig")
                    mgr.write_signature_document(
                        {"path": "manifest.json", args.digest.lower(): digest_val}, sig_path
                    )
                else:
                    sig_path = _sign_manifest_fallback(manifest_path, key=signing_key, algorithm=args.digest, key_id=key_id)

            # Archiwum
            archive_path = out_dir / f"{args.bundle_name}-{version}.tar.gz"
            if archive_path.exists():
                raise FileExistsError(f"Paczka już istnieje: {archive_path}")
            _tar_dir(staging_root, archive_path)

            # Przenieś manifest/podpis obok archiwum (wygodna weryfikacja)
            final_manifest = out_dir / f"{args.bundle_name}-{version}.manifest.json"
            shutil.copy2(manifest_path, final_manifest)
            final_sig = None
            if sig_path:
                final_sig = out_dir / f"{args.bundle_name}-{version}.manifest.sig"
                shutil.copy2(sig_path, final_sig)

            summary = {
                "bundle": archive_path.as_posix(),
                "manifest": final_manifest.as_posix(),
                "files": manifest["file_count"],
                "size_bytes": manifest["total_size_bytes"],
            }
            if final_sig:
                summary["signature"] = final_sig.as_posix()
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    except Exception as exc:  # pragma: no cover
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
