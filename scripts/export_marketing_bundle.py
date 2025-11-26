from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
import shutil


def _collect_files(base: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(base.glob(pattern)))
    return files


def _require_signing_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise SystemExit(f"Brak klucza HMAC w zmiennej {env_name}")
    return key


def _serialize_for_signature(manifest: dict) -> bytes:
    body = dict(manifest)
    body.pop("hmac_signature", None)
    return json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _write_manifest(dest: Path, manifest: dict, signing_key: str | None) -> dict:
    manifest_body = dict(manifest)
    if signing_key:
        signature = hmac.new(signing_key.encode("utf-8"), _serialize_for_signature(manifest_body), hashlib.sha256).hexdigest()
        manifest_body["hmac_signature"] = signature
    else:
        manifest_body["hmac_signature"] = None
    dest.write_text(json.dumps(manifest_body, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_body


def _validate_manifest(manifest_path: Path, signing_key: str) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = hmac.new(signing_key.encode("utf-8"), _serialize_for_signature(data), hashlib.sha256).hexdigest()
    if expected != data.get("hmac_signature"):
        raise SystemExit("Podpis HMAC nie zgadza się z zawartością manifestu marketingowego")


def _copy_artifacts(source_files: Iterable[Path], destination_dir: Path) -> list[dict]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for source in source_files:
        if not source.is_file():
            continue
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        results.append(
            {
                "source": str(source.relative_to(Path.cwd())),
                "stored_as": str(destination.relative_to(Path.cwd())),
            }
        )
    return results


def build_marketing_bundle(args: argparse.Namespace) -> dict:
    destination_base = Path(args.destination).resolve()
    stress_lab_source = Path(args.stress_lab_dir).resolve()
    signal_quality_source = Path(args.signal_quality_dir).resolve()

    stress_lab_artifacts = _collect_files(stress_lab_source, ("*.json", "*.sig", "*.manifest.json"))
    signal_quality_artifacts = _collect_files(signal_quality_source, ("*.json", "*.csv"))

    artifacts: list[dict] = []
    artifacts.extend(
        {
            **entry,
            "category": "stress_lab",
            "link": f"{args.link_prefix}/stress_lab/{Path(entry['stored_as']).name}" if args.link_prefix else entry["stored_as"],
        }
        for entry in _copy_artifacts(stress_lab_artifacts, destination_base / "stress_lab")
    )
    artifacts.extend(
        {
            **entry,
            "category": "signal_quality",
            "link": f"{args.link_prefix}/signal_quality/{Path(entry['stored_as']).name}" if args.link_prefix else entry["stored_as"],
        }
        for entry in _copy_artifacts(signal_quality_artifacts, destination_base / "signal_quality")
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_range": args.report_range,
        "artifacts": artifacts,
        "release_artifacts": {
            "stress_lab_report": "stress-lab-report",
            "marketing_bundle": "benchmark-marketing-bundle",
            "destination_dir": str(destination_base.relative_to(Path.cwd())),
        },
    }

    signing_key = _require_signing_key(args.signing_key_env) if args.signing_key_env else None
    manifest_path = destination_base / "manifest.json"
    return _write_manifest(manifest_path, manifest, signing_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksport bundla marketingowego benchmarku")
    parser.add_argument("--destination", default="var/marketing/benchmark", help="Katalog docelowy bundla marketingowego")
    parser.add_argument("--stress-lab-dir", default="reports/stress_lab", help="Katalog z raportami Stress Lab")
    parser.add_argument("--signal-quality-dir", default="reports/exchanges/signal_quality", help="Katalog z checklistami adapterów")
    parser.add_argument("--report-range", default="latest", help="Zakres raportu (np. release tag lub przedział dat)")
    parser.add_argument("--signing-key-env", default=None, help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    parser.add_argument("--link-prefix", default=None, help="Prefiks URL dla artefaktów (np. https://releases.example/artifacts)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Waliduje istniejący manifest bundla marketingowego zamiast kopiować artefakty",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination_base = Path(args.destination).resolve()

    if args.validate_only:
        signing_key = _require_signing_key(args.signing_key_env or "MARKETING_BUNDLE_HMAC")
        manifest_path = destination_base / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit("Brak manifestu do walidacji w katalogu bundla marketingowego")
        _validate_manifest(manifest_path, signing_key)
        print(f"Manifest {manifest_path} ma poprawny podpis HMAC")
        return

    manifest = build_marketing_bundle(args)
    if manifest.get("hmac_signature"):
        print("Utworzono bundel marketingowy z podpisem HMAC")
    else:
        print("Utworzono bundel marketingowy (bez podpisu HMAC)")


if __name__ == "__main__":
    main()
