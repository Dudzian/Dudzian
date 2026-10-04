"""Fail-closed publisher for the existing operator-owned public trust package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    PUBLIC_PACKAGE_FILENAMES,
    load_production_trust,
    validate_public_package_layout,
)

ARTIFACT_NAME = "stage9-public-production-trust"
SOURCE_ROOT = Path(r"C:\CryptoHunter-Production-Authority\ceremony-results\final")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(source: Path, output: Path, receipt: Path) -> None:
    expected = (SOURCE_ROOT / CEREMONY_ID).resolve()
    if source.resolve() != expected:
        raise RuntimeError("source must be the canonical operator authority ceremony directory")
    validate_public_package_layout(source)
    context = load_production_trust(source)
    if context.ceremony_id != CEREMONY_ID:
        raise RuntimeError("verified ceremony identity differs")
    before = {name: _sha(source / name) for name in PUBLIC_PACKAGE_FILENAMES}
    if output.exists():
        raise RuntimeError("publication output must not already exist")
    package_output = output / CEREMONY_ID
    package_output.mkdir(parents=True)
    for name in PUBLIC_PACKAGE_FILENAMES:
        shutil.copyfile(source / name, package_output / name)
    validate_public_package_layout(package_output)
    load_production_trust(package_output)
    after = {name: _sha(source / name) for name in PUBLIC_PACKAGE_FILENAMES}
    copied = {name: _sha(package_output / name) for name in PUBLIC_PACKAGE_FILENAMES}
    if before != after or copied != before:
        raise RuntimeError("source mutation or non-byte-exact publication detected")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ceremony_id": CEREMONY_ID,
                "package_manifest_sha256": before["package_manifest.json"],
                "source_package_verification": "PASS",
                "artifact_name": ARTIFACT_NAME,
                "source_revision": os.environ["GITHUB_SHA"],
                "github_run_id": os.environ["GITHUB_RUN_ID"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    publish(args.source, args.output, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
