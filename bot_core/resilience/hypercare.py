"""Automatyzuje cykl hypercare odporności Stage6."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from bot_core.resilience.audit import (
    BundleAuditResult,
    audit_bundle,
    write_csv_report,
    write_json_report,
    write_json_report_signature,
)
from bot_core.resilience.bundle import (
    ResilienceBundleArtifacts,
    ResilienceBundleBuilder,
    ResilienceBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)
from bot_core.resilience.drill import (
    FailoverDrillSummary,
    evaluate_failover_drill,
    load_failover_plan,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)
from bot_core.resilience.policy import ResiliencePolicy
from bot_core.resilience.self_healing import (
    SubprocessSelfHealingExecutor,
    build_self_healing_plan,
    execute_self_healing_plan,
    load_self_healing_rules,
    summarize_self_healing_plan,
    write_self_healing_report,
    write_self_healing_signature,
)


@dataclass(slots=True)
class BundleConfig:
    """Konfiguracja budowy paczki odpornościowej."""

    source: Path
    output_dir: Path
    bundle_name: str = "stage6-resilience"
    include: Sequence[str] | None = None
    exclude: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AuditConfig:
    """Parametry audytu paczki odpornościowej."""

    json_path: Path
    csv_path: Path | None = None
    signature_path: Path | None = None
    require_signature: bool = False
    verify_signature: bool = True
    policy: ResiliencePolicy | None = None


@dataclass(slots=True)
class FailoverConfig:
    """Parametry ćwiczenia failover Stage6."""

    plan_path: Path
    json_path: Path
    csv_path: Path | None = None
    signature_path: Path | None = None


@dataclass(slots=True)
class SelfHealingConfig:
    """Parametry planu self-healing dla drillu Stage6."""

    rules_path: Path
    output_path: Path
    signature_path: Path | None = None
    mode: Literal["plan", "execute"] = "plan"


@dataclass(slots=True)
class ResilienceCycleConfig:
    """Łączna konfiguracja cyklu hypercare odporności Stage6."""

    bundle: BundleConfig
    audit: AuditConfig
    failover: FailoverConfig
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    audit_hmac_key: bytes | None = None
    self_healing: SelfHealingConfig | None = None


@dataclass(slots=True)
class ResilienceCycleResult:
    """Artefakty wygenerowane przez cykl odporności."""

    bundle_artifacts: ResilienceBundleArtifacts
    audit_result: BundleAuditResult
    audit_summary: Mapping[str, Any]
    audit_summary_path: Path
    audit_signature_path: Path | None
    audit_csv_path: Path | None
    failover_summary: FailoverDrillSummary
    failover_payload: Mapping[str, Any]
    failover_summary_path: Path
    failover_signature_path: Path | None
    failover_csv_path: Path | None
    self_healing_payload: Mapping[str, Any] | None
    self_healing_report_path: Path | None
    self_healing_signature_path: Path | None
    verification: Mapping[str, Any]


class ResilienceHypercareCycle:
    """Wykonuje kompletny cykl odporności Stage6."""

    def __init__(self, config: ResilienceCycleConfig) -> None:
        self._config = config

    def run(self) -> ResilienceCycleResult:
        bundle_artifacts = self._build_bundle()

        audit_result, audit_summary, audit_signature_path = self._run_audit(bundle_artifacts)
        if audit_result.errors:
            joined = "; ".join(audit_result.errors)
            raise ValueError(f"Audyt paczki zakończył się błędami: {joined}")

        failover_summary, failover_payload, failover_signature_path = self._run_failover(
            bundle_artifacts,
            audit_result,
        )

        self_healing_payload, self_healing_signature_path = self._run_self_healing(failover_summary)

        verification = self._verify_bundle(bundle_artifacts)

        return ResilienceCycleResult(
            bundle_artifacts=bundle_artifacts,
            audit_result=audit_result,
            audit_summary=audit_summary,
            audit_summary_path=self._config.audit.json_path.expanduser(),
            audit_signature_path=audit_signature_path,
            audit_csv_path=self._config.audit.csv_path.expanduser()
            if self._config.audit.csv_path
            else None,
            failover_summary=failover_summary,
            failover_payload=failover_payload,
            failover_summary_path=self._config.failover.json_path.expanduser(),
            failover_signature_path=failover_signature_path,
            failover_csv_path=self._config.failover.csv_path.expanduser()
            if self._config.failover.csv_path
            else None,
            self_healing_payload=self_healing_payload,
            self_healing_report_path=self._config.self_healing.output_path.expanduser()
            if self._config.self_healing
            else None,
            self_healing_signature_path=self_healing_signature_path,
            verification=verification,
        )

    def _build_bundle(self) -> ResilienceBundleArtifacts:
        config = self._config.bundle
        include = tuple(config.include or ("**",))
        exclude = tuple(config.exclude or ())
        builder = ResilienceBundleBuilder(config.source, include=include, exclude=exclude or None)

        metadata: MutableMapping[str, Any] = dict(config.metadata or {})
        metadata.setdefault(
            "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        audit_metadata: MutableMapping[str, Any] = {
            "json": self._config.audit.json_path.expanduser().as_posix(),
            "require_signature": self._config.audit.require_signature,
            "verify_signature": self._config.audit.verify_signature,
        }
        if self._config.audit.csv_path:
            audit_metadata["csv"] = self._config.audit.csv_path.expanduser().as_posix()
        if self._config.audit.signature_path:
            audit_metadata["signature"] = self._config.audit.signature_path.expanduser().as_posix()
        if self._config.audit.policy:
            audit_metadata["policy_enforced"] = True
        metadata["audit_report"] = audit_metadata

        failover_metadata: MutableMapping[str, Any] = {
            "json": self._config.failover.json_path.expanduser().as_posix(),
            "plan": self._config.failover.plan_path.expanduser().as_posix(),
        }
        if self._config.failover.csv_path:
            failover_metadata["csv"] = self._config.failover.csv_path.expanduser().as_posix()
        if self._config.failover.signature_path:
            failover_metadata["signature"] = self._config.failover.signature_path.expanduser().as_posix()
        metadata["failover_summary"] = failover_metadata

        if self._config.self_healing:
            self_heal_config = self._config.self_healing
            self_heal_metadata: MutableMapping[str, Any] = {
                "mode": self_heal_config.mode,
                "rules": self_heal_config.rules_path.expanduser().as_posix(),
                "output": self_heal_config.output_path.expanduser().as_posix(),
            }
            if self_heal_config.signature_path:
                self_heal_metadata["signature"] = self_heal_config.signature_path.expanduser().as_posix()
            metadata["self_healing"] = self_heal_metadata

        artifacts = builder.build(
            bundle_name=config.bundle_name,
            output_dir=config.output_dir.expanduser(),
            metadata=metadata,
            signing_key=self._config.signing_key,
            signing_key_id=self._config.signing_key_id,
        )
        return artifacts

    def _run_audit(
        self, artifacts: ResilienceBundleArtifacts
    ) -> tuple[BundleAuditResult, Mapping[str, Any], Path | None]:
        audit_key = self._config.audit_hmac_key or (
            self._config.signing_key if self._config.audit.verify_signature else None
        )
        result = audit_bundle(
            artifacts.bundle_path,
            hmac_key=audit_key if self._config.audit.verify_signature else None,
            require_signature=self._config.audit.require_signature,
            policy=self._config.audit.policy,
        )
        summary = write_json_report([result], self._config.audit.json_path)
        if self._config.audit.csv_path:
            write_csv_report([result], self._config.audit.csv_path)

        signature_path: Path | None = None
        if self._config.signing_key:
            target = (
                self._config.audit.signature_path
                if self._config.audit.signature_path is not None
                else self._config.audit.json_path.with_suffix(
                    self._config.audit.json_path.suffix + ".sig"
                )
            )
            signature_path = target.expanduser()
            write_json_report_signature(
                summary,
                signature_path,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
                target=self._config.audit.json_path.expanduser().name,
            )
        return result, summary, signature_path

    def _run_failover(
        self,
        artifacts: ResilienceBundleArtifacts,
        audit_result: BundleAuditResult,
    ) -> tuple[FailoverDrillSummary, Mapping[str, Any], Path | None]:
        plan = load_failover_plan(self._config.failover.plan_path)
        manifest = audit_result.manifest or artifacts.manifest
        summary = evaluate_failover_drill(plan, manifest, bundle_audit=audit_result)
        payload = write_summary_json(summary, self._config.failover.json_path)
        if self._config.failover.csv_path:
            write_summary_csv(summary, self._config.failover.csv_path)

        signature_path: Path | None = None
        if self._config.signing_key:
            target = (
                self._config.failover.signature_path
                if self._config.failover.signature_path is not None
                else self._config.failover.json_path.with_suffix(
                    self._config.failover.json_path.suffix + ".sig"
                )
            )
            signature_path = target.expanduser()
            write_summary_signature(
                payload,
                signature_path,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
                target=self._config.failover.json_path.expanduser().name,
            )
        return summary, payload, signature_path

    def _run_self_healing(
        self, summary: FailoverDrillSummary
    ) -> tuple[Mapping[str, Any] | None, Path | None]:
        if not self._config.self_healing:
            return None, None

        rules = load_self_healing_rules(self._config.self_healing.rules_path)
        plan = build_self_healing_plan(summary, rules)
        if self._config.self_healing.mode == "execute":
            executor = SubprocessSelfHealingExecutor()
            report = execute_self_healing_plan(plan, executor)
        else:
            report = summarize_self_healing_plan(plan)
        payload = write_self_healing_report(report, self._config.self_healing.output_path)

        signature_path: Path | None = None
        if self._config.signing_key:
            target = (
                self._config.self_healing.signature_path
                if self._config.self_healing.signature_path is not None
                else self._config.self_healing.output_path.with_suffix(
                    self._config.self_healing.output_path.suffix + ".sig"
                )
            )
            signature_path = target.expanduser()
            write_self_healing_signature(
                payload,
                signature_path,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
                target=self._config.self_healing.output_path.expanduser().name,
            )
        return payload, signature_path

    def _verify_bundle(self, artifacts: ResilienceBundleArtifacts) -> Mapping[str, Any]:
        manifest = load_manifest(artifacts.manifest_path)
        verifier = ResilienceBundleVerifier(artifacts.bundle_path, manifest)
        errors = verifier.verify_files()
        signature_doc = load_signature(artifacts.signature_path if artifacts.signature_path else None)
        signature_verified: bool | None = None
        verification_key = self._config.audit_hmac_key or self._config.signing_key
        if signature_doc is not None and verification_key is not None:
            sig_errors = verify_signature(manifest, signature_doc, key=verification_key)
            if sig_errors:
                errors.extend(sig_errors)
                signature_verified = False
            else:
                signature_verified = True
        elif signature_doc is not None:
            signature_verified = None
        if errors:
            raise ValueError(
                "Weryfikacja paczki odpornościowej nie powiodła się: " + "; ".join(errors)
            )
        return {
            "bundle": artifacts.bundle_path.as_posix(),
            "manifest": artifacts.manifest_path.as_posix(),
            "verified_files": manifest.get("file_count"),
            "signature_verified": signature_verified,
        }


__all__ = [
    "BundleConfig",
    "AuditConfig",
    "FailoverConfig",
    "SelfHealingConfig",
    "ResilienceCycleConfig",
    "ResilienceCycleResult",
    "ResilienceHypercareCycle",
]
