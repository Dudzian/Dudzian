"""Publikacja podpisanych pakietów strategii Stage4 wraz z wpisem decision logu."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_tree,
    _validate_bundle_version,
)
from deploy.packaging.build_strategy_bundle import (  # type: ignore
    BUNDLE_NAME,
    build_from_cli as build_strategy_bundle,
)
from bot_core.security.signing import build_hmac_signature


DEFAULT_STAGING_DIR = REPO_ROOT / "var/dist/strategies"
DEFAULT_RELEASE_ROOT = REPO_ROOT / "var/releases/strategies"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_digest(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _prepare_directory(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    _ensure_no_symlinks(path, label=label)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{label} musi być katalogiem: {path}")
        _ensure_casefold_safe_tree(path, label=label)
    else:
        path.mkdir(parents=True, exist_ok=True)
    _ensure_windows_safe_tree(path, label=label)
    return path.resolve()


def _prepare_release_directory(root: Path, version: str) -> Path:
    validated_version = _validate_bundle_version(version)
    base = _prepare_directory(root, label="Katalog release'ów")
    target = base / validated_version
    if target.exists():
        raise ValueError(f"Release {validated_version} już istnieje: {target}")
    target.mkdir(parents=False, exist_ok=False)
    return target


def _load_decision_key(args: argparse.Namespace) -> bytes | None:
    inline = args.decision_log_hmac_key
    path = args.decision_log_hmac_key_file
    if inline and path:
        raise ValueError(
            "Podaj klucz decision logu jako --decision-log-hmac-key lub --decision-log-hmac-key-file"
        )
    if inline:
        key = inline.encode("utf-8")
    elif path:
        candidate = Path(path).expanduser()
        _ensure_no_symlinks(candidate, label="Decision log key")
        resolved = candidate.resolve()
        _ensure_windows_safe_tree(resolved, label="Decision log key")
        if not resolved.is_file():
            raise ValueError(f"Plik klucza decision logu nie istnieje: {resolved}")
        if os.name != "nt":
            mode = resolved.stat().st_mode
            if mode & (0o077):
                raise ValueError(
                    "Plik klucza decision logu musi mieć uprawnienia maks. 600"
                )
        key = resolved.read_bytes()
    else:
        return None

    if len(key) < 32:
        raise ValueError("Klucz decision logu musi mieć co najmniej 32 bajty")
    return key


def _append_decision_log(
    *,
    path: Optional[str],
    payload: Mapping[str, Any],
    key: bytes | None,
    key_id: Optional[str],
    allow_unsigned: bool,
) -> None:
    if not path:
        if key and not allow_unsigned:
            return
        if not allow_unsigned:
            raise ValueError("Wpis decision logu jest wymagany – podaj --decision-log-path")
        return

    entry = dict(payload)
    if key is not None:
        entry["signature"] = build_hmac_signature(
            payload,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=key_id,
        )
    elif not allow_unsigned:
        raise ValueError(
            "Brak klucza HMAC decision logu – podaj --decision-log-hmac-key lub --decision-log-hmac-key-file"
        )

    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _copy_release_artifacts(
    archive: Path,
    target_dir: Path,
) -> dict[str, Any]:
    basename = archive.stem
    manifest = archive.with_suffix(".manifest.json")
    signature = archive.with_suffix(".manifest.sig")

    if not manifest.exists() or not signature.exists():
        raise FileNotFoundError(
            "Brak plików manifestu lub podpisu obok zbudowanego archiwum"
        )

    artifacts = {}
    for src in (archive, manifest, signature):
        destination = target_dir / src.name
        if destination.exists():
            raise ValueError(f"Artefakt już istnieje w release: {destination}")
        _ensure_no_symlinks(src, label=f"Artefakt {src.name}")
        _ensure_windows_safe_tree(src, label=f"Artefakt {src.name}")
        _ensure_casefold_safe_tree(src.parent, label=f"Katalog artefaktu {src.name}")
        data = src.read_bytes()
        destination.write_bytes(data)
        digest = _sha256_digest(destination)
        artifacts[src.name] = {
            "path": destination.as_posix(),
            **digest,
        }

    metadata = {
        "schema": "stage4.strategy_release.metadata",
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "bundle_name": BUNDLE_NAME,
        "version": basename.removeprefix(f"{BUNDLE_NAME}-"),
        "artifacts": artifacts,
    }

    metadata_path = target_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _build_decision_payload(
    *,
    metadata: Mapping[str, Any],
    release_dir: Path,
    notes: Optional[str],
    category: Optional[str],
) -> Mapping[str, Any]:
    entry: dict[str, Any] = {
        "schema": "stage4.strategy_release",
        "schema_version": "1.0",
        "timestamp": _now_iso(),
        "version": metadata["version"],
        "bundle_name": metadata["bundle_name"],
        "release_path": release_dir.as_posix(),
        "artifacts": metadata["artifacts"],
    }
    if notes:
        entry["notes"] = notes
    if category:
        entry["category"] = category
    return entry


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buduje i publikuje pakiet strategii Stage4 do katalogu release'ów",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--signing-key-path", required=True)
    parser.add_argument("--signing-key-id")
    parser.add_argument("--digest", default="sha384")
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING_DIR))
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--strategy", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--decision-log-path")
    parser.add_argument("--decision-log-hmac-key")
    parser.add_argument("--decision-log-hmac-key-file")
    parser.add_argument("--decision-log-key-id")
    parser.add_argument("--decision-log-category")
    parser.add_argument("--decision-log-notes")
    parser.add_argument(
        "--decision-log-allow-unsigned",
        action="store_true",
        help="Pozwala na wpis decision logu bez podpisu",
    )

    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    version = _validate_bundle_version(args.version)
    staging_dir = Path(args.staging_dir).expanduser()
    release_root = Path(args.release_dir).expanduser()

    build_args = [
        "--version",
        version,
        "--signing-key-path",
        args.signing_key_path,
        "--output-dir",
        str(staging_dir),
        "--digest",
        args.digest,
    ]
    if args.signing_key_id:
        build_args.extend(["--signing-key-id", args.signing_key_id])
    for strategy_arg in args.strategy:
        build_args.extend(["--strategy", strategy_arg])
    for dataset_arg in args.dataset:
        build_args.extend(["--dataset", dataset_arg])

    archive_path = build_strategy_bundle(build_args)

    release_dir = _prepare_release_directory(release_root, version)
    metadata = _copy_release_artifacts(archive_path, release_dir)

    decision_payload = _build_decision_payload(
        metadata=metadata,
        release_dir=release_dir,
        notes=args.decision_log_notes,
        category=args.decision_log_category,
    )
    key = _load_decision_key(args)
    _append_decision_log(
        path=args.decision_log_path,
        payload=decision_payload,
        key=key,
        key_id=args.decision_log_key_id,
        allow_unsigned=args.decision_log_allow_unsigned,
    )

    logging.info("Opublikowano pakiet strategii Stage4: %s", release_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entrypoint
    try:
        return run(argv)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        logging.error("Publikacja pakietu strategii nie powiodła się: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
