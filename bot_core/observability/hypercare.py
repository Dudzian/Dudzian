"""Automatyzuje cykl hypercare Observability Stage6."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from bot_core.observability.alert_overrides import (
    AlertOverrideBuilder,
    AlertOverrideManager,
    load_overrides_document,
)
from bot_core.observability.bundle import (
    AssetSource,
    ObservabilityBundleArtifacts,
    ObservabilityBundleBuilder,
    ObservabilityBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)
from bot_core.observability.dashboard_sync import (
    build_dashboard_annotations_payload,
    load_dashboard_definition,
    save_dashboard_annotations,
)
from bot_core.observability.io import load_slo_definitions, load_slo_measurements
from bot_core.observability.slo import (
    SLOReport,
    evaluate_slo,
)
from bot_core.security.signing import build_hmac_signature


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    AssetSource(
        category="dashboards",
        root=REPO_ROOT / "deploy" / "grafana" / "provisioning" / "dashboards",
    ),
    AssetSource(
        category="alerts",
        root=REPO_ROOT / "deploy" / "prometheus",
    ),
)


def _default_signature_path(base: Path, override: Path | None) -> Path:
    if override:
        return override.expanduser()
    return base.with_suffix(".sig")


def _write_signature(
    payload: Mapping[str, Any],
    *,
    path: Path,
    key: bytes,
    key_id: str | None,
) -> Path:
    signature = build_hmac_signature(payload, key=key, key_id=key_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(signature, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return path


@dataclass(slots=True)
class SLOOutputConfig:
    json_path: Path
    csv_path: Path | None = None
    signature_path: Path | None = None
    pretty_json: bool = False


@dataclass(slots=True)
class OverridesOutputConfig:
    json_path: Path
    signature_path: Path | None = None
    include_warning: bool = True
    ttl: timedelta = timedelta(minutes=120)
    requested_by: str | None = None
    source: str = "slo_monitor"
    tags: Sequence[str] = field(default_factory=tuple)
    severity_overrides: Mapping[str, str] = field(default_factory=dict)
    existing_path: Path | None = None


@dataclass(slots=True)
class DashboardSyncConfig:
    dashboard_path: Path
    output_path: Path
    signature_path: Path | None = None
    panel_id: int | None = None
    pretty: bool = False


@dataclass(slots=True)
class BundleConfig:
    output_dir: Path
    bundle_name: str = "stage6-observability"
    sources: Sequence[AssetSource] | None = None
    include: Sequence[str] | None = None
    exclude: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None
    verify: bool = True


@dataclass(slots=True)
class ObservabilityCycleConfig:
    definitions_path: Path
    metrics_path: Path
    slo: SLOOutputConfig
    overrides: OverridesOutputConfig | None = None
    dashboard: DashboardSyncConfig | None = None
    bundle: BundleConfig | None = None
    signing_key: bytes | None = None
    signing_key_id: str | None = None


@dataclass(slots=True)
class ObservabilityCycleResult:
    slo_report_path: Path
    slo_signature_path: Path | None
    slo_csv_path: Path | None
    overrides_path: Path | None
    overrides_signature_path: Path | None
    dashboard_annotations_path: Path | None
    dashboard_signature_path: Path | None
    bundle_path: Path | None
    bundle_manifest_path: Path | None
    bundle_signature_path: Path | None
    bundle_verification: Mapping[str, Any] | None


class ObservabilityHypercareCycle:
    """Wykonuje kompletny cykl observability Stage6."""

    def __init__(self, config: ObservabilityCycleConfig) -> None:
        self._config = config

    def run(self) -> ObservabilityCycleResult:
        definitions, composites = load_slo_definitions(self._config.definitions_path)
        if not definitions:
            raise ValueError("Brak definicji SLO do ewaluacji")
        measurements = load_slo_measurements(self._config.metrics_path)
        report = evaluate_slo(definitions, measurements, composites=composites)

        slo_json = self._config.slo.json_path.expanduser()
        report.write_json(slo_json, pretty=self._config.slo.pretty_json)
        slo_csv_path: Path | None = None
        if self._config.slo.csv_path:
            slo_csv_path = self._config.slo.csv_path.expanduser()
            report.write_csv(slo_csv_path)

        slo_signature_path: Path | None = None
        if self._config.signing_key:
            signature_path = _default_signature_path(
                slo_json,
                self._config.slo.signature_path,
            )
            slo_signature_path = _write_signature(
                report.to_payload(),
                path=signature_path,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
            )

        overrides_path: Path | None = None
        overrides_signature: Path | None = None
        overrides_payload: Mapping[str, Any] | None = None
        overrides_manager = AlertOverrideManager()
        if self._config.overrides:
            overrides = self._build_overrides(report)
            if self._config.overrides.existing_path:
                existing_path = self._config.overrides.existing_path.expanduser()
                if existing_path.exists():
                    data = json.loads(existing_path.read_text(encoding="utf-8"))
                    overrides_manager.extend(load_overrides_document(data))
                    overrides_manager.prune_expired(reference=report.generated_at)
            overrides_manager.merge(overrides)
            overrides_payload = overrides_manager.to_payload(reference=report.generated_at)
            overrides_path = self._config.overrides.json_path.expanduser()
            overrides_path.parent.mkdir(parents=True, exist_ok=True)
            with overrides_path.open("w", encoding="utf-8") as handle:
                json.dump(overrides_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if self._config.signing_key:
                signature_target = _default_signature_path(
                    overrides_path,
                    self._config.overrides.signature_path,
                )
                overrides_signature = _write_signature(
                    overrides_payload,
                    path=signature_target,
                    key=self._config.signing_key,
                    key_id=self._config.signing_key_id,
                )

        dashboard_path: Path | None = None
        dashboard_signature: Path | None = None
        dashboard_payload: Mapping[str, Any] | None = None
        if self._config.dashboard:
            dashboard_definition = load_dashboard_definition(
                self._config.dashboard.dashboard_path.expanduser()
            )
            overrides_for_dashboard = overrides_manager.active(reference=report.generated_at)
            dashboard_payload = build_dashboard_annotations_payload(
                overrides_for_dashboard,
                reference=report.generated_at,
                dashboard_uid=dashboard_definition.uid,
                panel_id=self._config.dashboard.panel_id,
            )
            dashboard_path = self._config.dashboard.output_path.expanduser()
            save_dashboard_annotations(
                dashboard_payload,
                output_path=dashboard_path,
                pretty=self._config.dashboard.pretty,
            )
            if self._config.signing_key:
                signature_target = _default_signature_path(
                    dashboard_path,
                    self._config.dashboard.signature_path,
                )
                dashboard_signature = _write_signature(
                    dashboard_payload,
                    path=signature_target,
                    key=self._config.signing_key,
                    key_id=self._config.signing_key_id,
                )

        bundle_path: Path | None = None
        bundle_manifest_path: Path | None = None
        bundle_signature_path: Path | None = None
        verification: Mapping[str, Any] | None = None
        if self._config.bundle:
            artifacts = self._build_bundle(
                report=report,
                overrides_payload=overrides_payload,
                overrides_path=overrides_path,
                dashboard_payload=dashboard_payload,
                dashboard_path=dashboard_path,
                slo_csv_path=slo_csv_path,
            )
            bundle_path = artifacts.bundle_path
            bundle_manifest_path = artifacts.manifest_path
            bundle_signature_path = artifacts.signature_path
            if self._config.bundle.verify:
                verification = self._verify_bundle(
                    artifacts,
                    key=self._config.signing_key,
                )

        return ObservabilityCycleResult(
            slo_report_path=slo_json,
            slo_signature_path=slo_signature_path,
            slo_csv_path=slo_csv_path,
            overrides_path=overrides_path,
            overrides_signature_path=overrides_signature,
            dashboard_annotations_path=dashboard_path,
            dashboard_signature_path=dashboard_signature,
            bundle_path=bundle_path,
            bundle_manifest_path=bundle_manifest_path,
            bundle_signature_path=bundle_signature_path,
            bundle_verification=verification,
        )

    def _build_overrides(self, report: SLOReport):
        config = self._config.overrides
        assert config is not None  # dla typu
        definitions_map = {definition.name: definition for definition in report.definitions}
        builder = AlertOverrideBuilder(definitions_map)
        ttl = config.ttl if config.ttl.total_seconds() >= 0 else timedelta(0)
        reference = report.generated_at.astimezone(timezone.utc)
        return builder.build_from_statuses(
            report.statuses,
            include_warning=config.include_warning,
            default_ttl=ttl,
            severity_overrides=dict(config.severity_overrides),
            requested_by=config.requested_by,
            source=config.source,
            extra_tags=tuple(config.tags),
            reference=reference,
        )

    def _build_bundle(
        self,
        *,
        report: SLOReport,
        overrides_payload: Mapping[str, Any] | None,
        overrides_path: Path | None,
        dashboard_payload: Mapping[str, Any] | None,
        dashboard_path: Path | None,
        slo_csv_path: Path | None,
    ) -> ObservabilityBundleArtifacts:
        config = self._config.bundle
        assert config is not None
        sources = tuple(config.sources or DEFAULT_SOURCES)
        include = tuple(config.include or ("stage6*", "**/stage6*"))
        exclude = tuple(config.exclude or ())
        builder = ObservabilityBundleBuilder(sources, include=include, exclude=exclude or None)

        metadata: MutableMapping[str, Any] = dict(config.metadata or {})
        slo_metadata: MutableMapping[str, Any] = {
            "json": self._config.slo.json_path.expanduser().as_posix(),
            "generated_at": report.generated_at.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        if slo_csv_path:
            slo_metadata["csv"] = slo_csv_path.expanduser().as_posix()
        metadata["slo_report"] = slo_metadata

        if overrides_payload and overrides_path:
            metadata["alert_overrides"] = {
                "path": overrides_path.as_posix(),
                "summary": overrides_payload.get("summary"),
                "annotations": overrides_payload.get("annotations"),
            }
        if dashboard_payload and dashboard_path:
            metadata["dashboard_annotations"] = {
                "path": dashboard_path.as_posix(),
                "summary": dashboard_payload.get("summary"),
            }

        artifacts = builder.build(
            bundle_name=config.bundle_name,
            output_dir=config.output_dir.expanduser(),
            metadata=metadata,
            signing_key=self._config.signing_key,
            signing_key_id=self._config.signing_key_id,
        )
        return artifacts

    def _verify_bundle(
        self,
        artifacts: ObservabilityBundleArtifacts,
        *,
        key: bytes | None,
    ) -> Mapping[str, Any]:
        manifest = load_manifest(artifacts.manifest_path)
        verifier = ObservabilityBundleVerifier(artifacts.bundle_path, manifest)
        errors = verifier.verify_files()
        signature_verified: bool | None = None
        signature_doc = load_signature(artifacts.signature_path if artifacts.signature_path else None)
        if signature_doc is not None and key is not None:
            sig_errors = verify_signature(manifest, signature_doc, key=key)
            if sig_errors:
                errors.extend(sig_errors)
                signature_verified = False
            else:
                signature_verified = True
        elif signature_doc is not None and key is None:
            signature_verified = None
        if errors:
            raise ValueError("Weryfikacja paczki obserwowalności nie powiodła się: " + "; ".join(errors))
        return {
            "bundle": artifacts.bundle_path.as_posix(),
            "manifest": artifacts.manifest_path.as_posix(),
            "verified_files": manifest.get("file_count"),
            "signature_verified": signature_verified,
        }


__all__ = [
    "SLOOutputConfig",
    "OverridesOutputConfig",
    "DashboardSyncConfig",
    "BundleConfig",
    "ObservabilityCycleConfig",
    "ObservabilityCycleResult",
    "ObservabilityHypercareCycle",
]

