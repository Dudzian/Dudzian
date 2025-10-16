"""Automatyczna weryfikacja paczek odporności Stage6 i raportowanie wyników."""

from __future__ import annotations

import csv
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .bundle import (
    ResilienceBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)
from .policy import ResiliencePolicy, evaluate_policy
from bot_core.security.signing import build_hmac_signature


def _timestamp() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


_REPORT_SCHEMA = "stage6.resilience.audit.summary"
_REPORT_SIGNATURE_SCHEMA = "stage6.resilience.audit.summary.signature"
_REPORT_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class BundleAuditResult:
    """Rezultat weryfikacji pojedynczej paczki."""

    bundle_path: Path
    manifest_path: Path
    signature_path: Path | None
    manifest: Mapping[str, object] | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    verified_at: str

    def is_successful(self) -> bool:
        return not self.errors

    @property
    def file_count(self) -> int:
        if not self.manifest:
            return 0
        return int(self.manifest.get("file_count", 0))

    @property
    def total_size_bytes(self) -> int:
        if not self.manifest:
            return 0
        return int(self.manifest.get("total_size_bytes", 0))

    @property
    def metadata(self) -> Mapping[str, object]:
        if not self.manifest:
            return {}
        meta = self.manifest.get("metadata")
        if isinstance(meta, Mapping):
            return meta
        return {}

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle": self.bundle_path.as_posix(),
            "manifest": self.manifest_path.as_posix(),
            "signature": self.signature_path.as_posix() if self.signature_path else None,
            "verified_at": self.verified_at,
            "status": "ok" if self.is_successful() else "failed",
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "metadata": self.metadata,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _default_manifest_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(".manifest.json")


def _default_signature_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(".manifest.sig")


def audit_bundle(
    bundle_path: Path,
    *,
    hmac_key: bytes | None = None,
    require_signature: bool = False,
    policy: ResiliencePolicy | None = None,
) -> BundleAuditResult:
    bundle_path = bundle_path.expanduser().resolve()
    manifest_path = _default_manifest_path(bundle_path)
    signature_path = _default_signature_path(bundle_path)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: Mapping[str, object] | None = None

    try:
        manifest = load_manifest(manifest_path)
        verifier = ResilienceBundleVerifier(bundle_path, manifest)
        errors.extend(verifier.verify_files())
    except Exception as exc:  # noqa: BLE001 - komunikaty do raportu
        errors.append(str(exc))
        manifest = None

    signature_doc: Mapping[str, object] | None = None
    if manifest is not None:
        try:
            signature_doc = load_signature(signature_path if signature_path.exists() else None)
        except Exception as exc:  # noqa: BLE001 - komunikat
            errors.append(str(exc))
        if signature_doc is None and require_signature:
            errors.append("Brak podpisu manifestu")
        if signature_doc is not None and hmac_key is not None:
            errors.extend(verify_signature(manifest, signature_doc, key=hmac_key))
        if signature_doc is None and hmac_key is not None:
            errors.append("Dostarczono klucz HMAC, lecz nie znaleziono podpisu")
        if signature_doc is not None and hmac_key is None:
            errors.append("Dostarczono podpis, lecz nie przekazano klucza HMAC do weryfikacji")

        if signature_doc is not None or signature_path.exists():
            # brak dodatkowych działań – podpis zweryfikowany powyżej
            pass

        if manifest is not None and policy is not None:
            policy_errors, policy_warnings = evaluate_policy(manifest, policy)
            errors.extend(policy_errors)
            warnings.extend(policy_warnings)

    return BundleAuditResult(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        signature_path=signature_path if signature_path.exists() else None,
        manifest=manifest,
        errors=tuple(errors),
        warnings=tuple(warnings),
        verified_at=_timestamp(),
    )


def audit_bundles(
    directory: Path,
    *,
    hmac_key: bytes | None = None,
    require_signature: bool = False,
    policy: ResiliencePolicy | None = None,
) -> list[BundleAuditResult]:
    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise ValueError(f"Katalog z paczkami nie istnieje: {directory}")

    results: list[BundleAuditResult] = []
    for bundle_path in sorted(directory.glob("*.zip")):
        results.append(
            audit_bundle(
                bundle_path,
                hmac_key=hmac_key,
                require_signature=require_signature,
                policy=policy,
            )
        )
    if not results:
        raise ValueError(f"Nie znaleziono paczek ZIP w katalogu {directory}")
    return results


def write_csv_report(results: Sequence[BundleAuditResult], output_path: Path) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bundle",
                "manifest",
                "signature",
                "status",
                "verified_at",
                "file_count",
                "total_size_bytes",
                "error_count",
                "errors",
                "warning_count",
                "warnings",
                "metadata",
            ]
        )
        for result in results:
            metadata = json.dumps(result.metadata, ensure_ascii=False, sort_keys=True)
            writer.writerow(
                [
                    result.bundle_path.as_posix(),
                    result.manifest_path.as_posix(),
                    result.signature_path.as_posix() if result.signature_path else "",
                    "ok" if result.is_successful() else "failed",
                    result.verified_at,
                    result.file_count,
                    result.total_size_bytes,
                    len(result.errors),
                    " | ".join(result.errors),
                    len(result.warnings),
                    " | ".join(result.warnings),
                    metadata,
                ]
            )


def build_summary(results: Sequence[BundleAuditResult]) -> dict[str, object]:
    """Buduje spójne podsumowanie JSON z wyników audytu."""

    summary = {
        "schema": _REPORT_SCHEMA,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "generated_at": _timestamp(),
        "audited": len(results),
        "ok": sum(1 for item in results if item.is_successful()),
        "failed": sum(1 for item in results if not item.is_successful()),
        "warnings": sum(len(item.warnings) for item in results),
        "results": [item.to_dict() for item in results],
    }
    return summary


def write_json_report(results: Sequence[BundleAuditResult], output_path: Path) -> dict[str, object]:
    """Zapisuje raport JSON i zwraca jego strukturę."""

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_json_report_signature(
    summary: Mapping[str, object],
    output_path: Path,
    *,
    key: bytes,
    key_id: str | None = None,
    target: str | None = None,
) -> Mapping[str, object]:
    """Podpisuje raport JSON kluczem HMAC i zwraca dokument podpisu."""

    payload = {
        "schema": _REPORT_SIGNATURE_SCHEMA,
        "schema_version": _REPORT_SCHEMA_VERSION,
        "signed_at": _timestamp(),
        "target": target or output_path.name,
        "signature": build_hmac_signature(
            summary,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=key_id,
        ),
    }
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "BundleAuditResult",
    "audit_bundle",
    "audit_bundles",
    "write_csv_report",
    "build_summary",
    "write_json_report",
    "write_json_report_signature",
]
