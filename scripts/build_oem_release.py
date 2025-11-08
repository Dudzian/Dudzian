from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from deploy.packaging.pipeline import PackagingContext, build_pipeline_from_mapping


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Plik {path} musi zawierać obiekt JSON")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buduje OEM installer, paczki aktualizacji oraz manifest release'u")
    parser.add_argument("--pipeline-config", required=True, help="Ścieżka do konfiguracji pipeline'u (JSON)")
    parser.add_argument("--manifest", required=True, help="Ścieżka do manifestu bundla (JSON)")
    parser.add_argument("--staging-root", required=True, help="Katalog roboczy z plikami bundla")
    parser.add_argument("--archive", required=True, help="Ścieżka docelowa instalatora (ZIP/TAR)")
    parser.add_argument(
        "--build-installer",
        help="Polecenie budujące instalator; dostępne placeholdery: {archive}, {staging}, {manifest}",
    )
    parser.add_argument(
        "--release-dir",
        default="var/dist/releases",
        help="Katalog docelowy manifestów release'ów",
    )
    parser.add_argument(
        "--output",
        help="Opcjonalny plik z podsumowaniem (JSON)",
    )
    return parser


def _run_build_command(command: str, *, archive: Path, staging: Path, manifest: Path) -> None:
    substitutions = {
        "archive": str(archive),
        "staging": str(staging),
        "manifest": str(manifest),
    }
    args = [part.format(**substitutions) for part in shlex.split(command)]
    subprocess.run(args, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pipeline_config_path = Path(args.pipeline_config).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    staging_root = Path(args.staging_root).expanduser()
    archive_path = Path(args.archive).expanduser()

    if args.build_installer:
        _run_build_command(args.build_installer, archive=archive_path, staging=staging_root, manifest=manifest_path)

    if not archive_path.exists():
        raise SystemExit(f"Instalator {archive_path} nie istnieje po zakończeniu kroku build")

    config_payload = _load_mapping(pipeline_config_path)
    manifest_payload = _load_mapping(manifest_path)

    pipeline = build_pipeline_from_mapping(config_payload, base_dir=pipeline_config_path.parent)
    context = PackagingContext(staging_root=staging_root, archive_path=archive_path, manifest=manifest_payload)
    report = pipeline.execute(context)

    release_dir = Path(args.release_dir).expanduser()
    release_dir.mkdir(parents=True, exist_ok=True)
    release_manifest = {
        "bundle": context.bundle_name,
        "version": context.version,
        "platform": context.platform,
        "archive": str(archive_path),
        "report": report.to_mapping(),
    }
    release_path = release_dir / f"{context.bundle_name}-{context.version}-{context.platform}.json"
    release_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "release_manifest": str(release_path),
        "archive": str(archive_path),
        "update_packages": [entry["package_path"] for entry in release_manifest["report"].get("update_packages", [])],
    }
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    json.dump(summary, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
