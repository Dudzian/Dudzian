"""Spójność plików backendu i deklarowanych metadanych Stage 9.

To pomoc diagnostyczna: manifest i receipt pochodzą od wywołującego. Funkcja
nie uwierzytelnia zatwierdzonego wydania ani procesu SCM i nie uprawnia do PASS
current_revision_runtime_bound. Właściwe autorytety pozostają w istniejących
artifact preflight oraz qualify_enrolled_runtime(live=True).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


class RuntimeProvenanceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeProvenanceError(f"required provenance file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_runtime_provenance(
    manifest_path: Path,
    clean_receipt_path: Path,
    installed_backend: Path,
    installed_product_version: str,
    *,
    source_revision: str,
    ci_run_id: str,
    ci_provider: str,
) -> dict[str, str]:
    manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt: Any = json.loads(clean_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise RuntimeProvenanceError("runtime provenance inputs must be JSON objects")
    manifest_sha256 = _sha256(manifest_path)
    backend_sha256 = _sha256(installed_backend)
    msi = manifest.get("msi")
    executables = manifest.get("production_executables")
    product_version = manifest.get("product_version")
    if (
        not isinstance(msi, dict)
        or not isinstance(executables, dict)
        or not isinstance(product_version, str)
        or not product_version.strip()
    ):
        raise RuntimeProvenanceError("runtime provenance metadata is malformed")
    msi_sha256 = msi.get("sha256")
    expected_backend_sha256 = executables.get("CryptoHunterBackend.exe")
    if (
        not isinstance(msi_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", msi_sha256) is None
        or not isinstance(expected_backend_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_backend_sha256) is None
    ):
        raise RuntimeProvenanceError("runtime provenance digests are malformed")
    identity_matches = (
        receipt.get("source_revision") == source_revision
        and receipt.get("ci_run_id") == ci_run_id
        and receipt.get("ci_provider") == ci_provider == "https://github.com"
        and receipt.get("runner_os") == "Windows"
        and receipt.get("product_version") == product_version == installed_product_version
        and receipt.get("manifest_sha256") == manifest_sha256
        and receipt.get("msi_sha256") == msi_sha256
        and manifest.get("architecture") == "x64"
        and backend_sha256 == expected_backend_sha256
    )
    if not identity_matches:
        raise RuntimeProvenanceError(
            "installed backend is not the current-run qualified Windows artifact"
        )
    return {
        "source_revision": source_revision,
        "ci_run_id": ci_run_id,
        "product_version": product_version,
        "qualified_manifest_sha256": manifest_sha256,
        "qualified_msi_sha256": msi_sha256,
        "installed_backend_sha256": backend_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind installed Stage-10 runtime to Stage-9")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-install-receipt", type=Path, required=True)
    parser.add_argument("--installed-backend", type=Path, required=True)
    parser.add_argument("--installed-product-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = qualify_runtime_provenance(
            args.manifest,
            args.clean_install_receipt,
            args.installed_backend,
            args.installed_product_version,
            source_revision=os.environ["GITHUB_SHA"],
            ci_run_id=os.environ["GITHUB_RUN_ID"],
            ci_provider=os.environ["GITHUB_SERVER_URL"],
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, json.JSONDecodeError, RuntimeProvenanceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
