"""Powiadamia kanały compliance o wyniku smoke testu na podstawie summary.json."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from bot_core.alerts import AlertMessage
from bot_core.config.loader import load_core_config
from bot_core.runtime.bootstrap import build_alert_channels
from bot_core.security import SecretManager, SecretStorageError
from scripts._cli_common import create_secret_manager


_LOGGER = logging.getLogger(__name__)
_DEFAULT_CATEGORY = "paper_smoke_compliance"
_SUMMARY_KEY_ENVIRONMENT = "environment"
_SUMMARY_KEY_SEVERITY = "severity"
_SUMMARY_KEY_REPORT = "report"
_SUMMARY_KEY_PRECHECK = "precheck"
_SUMMARY_KEY_JSON = "json_log"
_SUMMARY_KEY_ARCHIVE = "archive"
_SUMMARY_KEY_PUBLISH = "publish"
_SUMMARY_KEY_MANIFEST = "manifest"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wysyła alert compliance na podstawie podsumowania smoke testu paper tradingu."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska, dla którego wysyłamy alert",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Ścieżka do pliku summary.json wygenerowanego przez run_daily_trend.py",
    )
    parser.add_argument(
        "--category",
        default=_DEFAULT_CATEGORY,
        help=f"Kategoria alertu (domyślnie {_DEFAULT_CATEGORY})",
    )
    parser.add_argument(
        "--severity-override",
        default=None,
        help="Wymuszenie poziomu ważności alertu (info/warning/error/critical)",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Opcjonalna nazwa operatora, która nadpisze wartość z summary.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nie wysyłaj alertu – wypisz payload w formacie JSON",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany do odczytu sekretów (keychain / plik szyfrowany)",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do magazynu sekretów w środowiskach headless (Linux)",
    )
    parser.add_argument(
        "--headless-secrets-path",
        default=None,
        help="Ścieżka do zaszyfrowanego magazynu sekretów w trybie headless",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania skryptu",
    )
    return parser.parse_args(argv)


def _load_summary(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku summary.json pod ścieżką {path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywne
        raise ValueError(f"Niepoprawny JSON w pliku {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Plik summary.json powinien zawierać obiekt JSON")
    return payload


def _normalize_severity(value: str | None, *, fallback: str = "info") -> str:
    if not value:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"info", "warning", "error", "critical"}:
        return normalized
    return fallback


def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    return create_secret_manager(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_secrets_path,
    )


def _append_publish_context(
    context: MutableMapping[str, str], publish_payload: Mapping[str, Any] | None
) -> None:
    if not isinstance(publish_payload, Mapping):
        return

    status = str(publish_payload.get("status", "unknown"))
    context["paper_smoke_publish_status"] = status

    exit_code = publish_payload.get("exit_code")
    if exit_code is not None:
        context["paper_smoke_publish_exit_code"] = str(exit_code)

    required = publish_payload.get("required")
    if required is not None:
        context["paper_smoke_publish_required"] = "true" if bool(required) else "false"

    reason = publish_payload.get("reason")
    if reason:
        context["paper_smoke_publish_reason"] = str(reason)

    stdout_snippet = publish_payload.get("raw_stdout")
    if stdout_snippet:
        context["paper_smoke_publish_stdout_snippet"] = str(stdout_snippet)[:2000]

    stderr_snippet = publish_payload.get("raw_stderr")
    if stderr_snippet:
        context["paper_smoke_publish_stderr_snippet"] = str(stderr_snippet)[:2000]

    json_sync = publish_payload.get("json_sync")
    if isinstance(json_sync, Mapping):
        context["paper_smoke_publish_json_status"] = str(json_sync.get("status", "unknown"))
        backend = json_sync.get("backend")
        if backend:
            context["paper_smoke_publish_json_backend"] = str(backend)
        location = json_sync.get("location")
        if location:
            context["paper_smoke_publish_json_location"] = str(location)
        metadata = json_sync.get("metadata")
        if isinstance(metadata, Mapping):
            for key, value in metadata.items():
                context[f"paper_smoke_publish_json_{key}"] = str(value)

    archive_upload = publish_payload.get("archive_upload")
    if isinstance(archive_upload, Mapping):
        context["paper_smoke_publish_archive_status"] = str(
            archive_upload.get("status", "unknown")
        )
        backend = archive_upload.get("backend")
        if backend:
            context["paper_smoke_publish_archive_backend"] = str(backend)
        location = archive_upload.get("location")
        if location:
            context["paper_smoke_publish_archive_location"] = str(location)
        metadata = archive_upload.get("metadata")
        if isinstance(metadata, Mapping):
            for key, value in metadata.items():
                context[f"paper_smoke_publish_archive_{key}"] = str(value)


def _append_telemetry_context(
    context: MutableMapping[str, str], telemetry_payload: Mapping[str, Any] | None
) -> None:
    if not isinstance(telemetry_payload, Mapping):
        return

    def _set(key: str, value: object | None) -> None:
        if value is None:
            return
        context[key] = str(value)

    def _set_bool(key: str, value: object | None) -> None:
        if value is None:
            return
        context[key] = "true" if bool(value) else "false"

    _set("paper_smoke_telemetry_summary_path", telemetry_payload.get("summary_path"))
    _set("paper_smoke_telemetry_decision_log_path", telemetry_payload.get("decision_log_path"))
    _set(
        "paper_smoke_telemetry_metrics_source_path",
        telemetry_payload.get("metrics_source_path"),
    )

    risk_profile = telemetry_payload.get("risk_profile")
    if isinstance(risk_profile, Mapping):
        _set("paper_smoke_risk_profile_name", risk_profile.get("name"))
        _set("paper_smoke_risk_profile_source", risk_profile.get("source"))
        _set(
            "paper_smoke_risk_profile_environment_fallback",
            risk_profile.get("environment_fallback"),
        )
        _set("paper_smoke_risk_profile_profiles_file", risk_profile.get("profiles_file"))

    decision_report = telemetry_payload.get("decision_log_report")
    if isinstance(decision_report, Mapping):
        _set("paper_smoke_decision_log_status", decision_report.get("status"))
        _set("paper_smoke_decision_log_report_path", decision_report.get("path"))
        exit_code = decision_report.get("exit_code")
        if exit_code is not None:
            context["paper_smoke_decision_log_exit_code"] = str(exit_code)

    snippets = telemetry_payload.get("snippets")
    if isinstance(snippets, Mapping):
        _set("paper_smoke_snippet_env_path", snippets.get("env_path"))
        _set("paper_smoke_snippet_yaml_path", snippets.get("yaml_path"))

    required_scopes = telemetry_payload.get("required_auth_scopes")
    if isinstance(required_scopes, (list, tuple, set)) and required_scopes:
        context["paper_smoke_required_auth_scopes"] = ",".join(
            str(scope) for scope in required_scopes
        )

    auth_scope_requirements = telemetry_payload.get("auth_scope_requirements")
    if isinstance(auth_scope_requirements, Mapping):
        for service, info in auth_scope_requirements.items():
            if not isinstance(info, Mapping):
                continue
            scopes = info.get("required_scopes")
            key = f"paper_smoke_{service}_required_scopes"
            if isinstance(scopes, (list, tuple, set)) and scopes:
                context[key] = ",".join(str(scope) for scope in scopes)
            elif isinstance(scopes, str) and scopes:
                context[key] = scopes

    risk_requirements = telemetry_payload.get("risk_service_requirements")
    if isinstance(risk_requirements, Mapping):
        details = risk_requirements.get("details")
        if isinstance(details, Mapping):
            if "require_tls" in details:
                _set_bool("paper_smoke_risk_service_require_tls", details.get("require_tls"))
            materials = details.get("tls_materials")
            if isinstance(materials, (list, tuple, set)) and materials:
                context["paper_smoke_risk_service_tls_materials"] = ",".join(
                    str(item) for item in materials
                )
            expected = details.get("expected_server_sha256")
            if isinstance(expected, (list, tuple, set)) and expected:
                context["paper_smoke_risk_service_expected_sha256"] = ",".join(
                    str(item) for item in expected
                )
            scopes = details.get("required_scopes")
            if isinstance(scopes, (list, tuple, set)) and scopes:
                context["paper_smoke_risk_service_required_scopes"] = ",".join(
                    str(scope) for scope in scopes
                )
            elif isinstance(scopes, str) and scopes:
                context["paper_smoke_risk_service_required_scopes"] = scopes

            # rozszerzenie: wymagane identyfikatory tokenów
            token_ids = details.get("required_token_ids")
            if isinstance(token_ids, (list, tuple, set)) and token_ids:
                context["paper_smoke_risk_service_required_token_ids"] = ",".join(
                    str(token) for token in token_ids
                )
            elif isinstance(token_ids, str) and token_ids:
                context["paper_smoke_risk_service_required_token_ids"] = token_ids

            if "require_auth_token" in details:
                _set_bool(
                    "paper_smoke_risk_service_require_auth_token",
                    details.get("require_auth_token"),
                )

        cli_args = risk_requirements.get("cli_args")
        if isinstance(cli_args, (list, tuple)) and cli_args:
            context["paper_smoke_risk_service_cli_args"] = " ".join(
                str(arg) for arg in cli_args
            )

        combined_metadata = risk_requirements.get("combined_metadata")
        if isinstance(combined_metadata, Mapping):
            if "tls_enabled" in combined_metadata:
                _set_bool(
                    "paper_smoke_risk_service_tls_enabled",
                    combined_metadata.get("tls_enabled"),
                )
            pinned = combined_metadata.get("pinned_fingerprints")
            if isinstance(pinned, (list, tuple, set)) and pinned:
                context["paper_smoke_risk_service_pinned_fingerprints"] = ",".join(
                    str(entry) for entry in pinned
                )


def _build_alert_payload(
    *, summary: Mapping[str, Any], environment: str, operator_override: str | None
) -> tuple[str, str, str, Mapping[str, str]]:
    severity = _normalize_severity(str(summary.get(_SUMMARY_KEY_SEVERITY)))
    env_in_summary = str(summary.get(_SUMMARY_KEY_ENVIRONMENT, environment))
    if env_in_summary != environment:
        _LOGGER.warning(
            "Środowisko w summary.json (%s) różni się od wartości CLI (%s)",
            env_in_summary,
            environment,
        )

    report_section = summary.get(_SUMMARY_KEY_REPORT)
    if not isinstance(report_section, Mapping):
        raise ValueError("Brak sekcji 'report' w summary.json")
    summary_sha256 = str(report_section.get("summary_sha256", ""))
    report_dir = str(report_section.get("directory", ""))

    context: MutableMapping[str, str] = {
        "environment": environment,
        "summary_sha256": summary_sha256,
    }
    if report_dir:
        context["summary_directory"] = report_dir

    window_section = summary.get("window")
    if isinstance(window_section, Mapping):
        start_value = window_section.get("start")
        end_value = window_section.get("end")
        if start_value:
            context["window_start"] = str(start_value)
        if end_value:
            context["window_end"] = str(end_value)

    operator_value = operator_override or summary.get("operator")
    if operator_value:
        context["operator"] = str(operator_value)

    precheck = summary.get(_SUMMARY_KEY_PRECHECK)
    if isinstance(precheck, Mapping):
        context["precheck_status"] = str(precheck.get("status", "unknown"))
        coverage_status = precheck.get("coverage_status")
        if coverage_status:
            context["precheck_coverage_status"] = str(coverage_status)
        risk_status = precheck.get("risk_status")
        if risk_status:
            context["precheck_risk_status"] = str(risk_status)

    telemetry_section = summary.get("telemetry")
    if isinstance(telemetry_section, Mapping):
        _append_telemetry_context(context, telemetry_section)

    json_section = summary.get(_SUMMARY_KEY_JSON)
    if isinstance(json_section, Mapping):
        json_path = json_section.get("path")
        if json_path:
            context["paper_smoke_json_path"] = str(json_path)
        record_id = json_section.get("record_id")
        if record_id:
            context["paper_smoke_json_record_id"] = str(record_id)
        sync_info = json_section.get("sync")
        if isinstance(sync_info, Mapping):
            context["paper_smoke_json_sync_backend"] = str(sync_info.get("backend", ""))
            context["paper_smoke_json_sync_location"] = str(sync_info.get("location", ""))
            metadata = sync_info.get("metadata")
            if isinstance(metadata, Mapping):
                for key, value in metadata.items():
                    context[f"paper_smoke_json_sync_{key}"] = str(value)

    archive_section = summary.get(_SUMMARY_KEY_ARCHIVE)
    if isinstance(archive_section, Mapping):
        archive_path = archive_section.get("path")
        if archive_path:
            context["paper_smoke_archive_path"] = str(archive_path)
        upload_info = archive_section.get("upload")
        if isinstance(upload_info, Mapping):
            context["paper_smoke_archive_backend"] = str(upload_info.get("backend", ""))
            context["paper_smoke_archive_location"] = str(upload_info.get("location", ""))
            metadata = upload_info.get("metadata")
            if isinstance(metadata, Mapping):
                for key, value in metadata.items():
                    context[f"paper_smoke_archive_{key}"] = str(value)

    manifest_section = summary.get(_SUMMARY_KEY_MANIFEST)
    if isinstance(manifest_section, Mapping):
        manifest_path = manifest_section.get("manifest_path")
        if manifest_path:
            context["paper_smoke_manifest_path"] = str(manifest_path)
        metrics_path = manifest_section.get("metrics_path")
        if metrics_path:
            context["paper_smoke_manifest_metrics_path"] = str(metrics_path)
        summary_path = manifest_section.get("summary_path")
        if summary_path:
            context["paper_smoke_manifest_summary_path"] = str(summary_path)
        worst_status = manifest_section.get("worst_status")
        if worst_status:
            context["paper_smoke_manifest_status"] = str(worst_status)
        stage_value = manifest_section.get("stage")
        if stage_value:
            context["paper_smoke_manifest_stage"] = str(stage_value)
        risk_profile = manifest_section.get("risk_profile")
        if risk_profile:
            context["paper_smoke_manifest_risk_profile"] = str(risk_profile)
        deny_status = manifest_section.get("deny_status")
        if isinstance(deny_status, (list, tuple, set)):
            context["paper_smoke_manifest_deny_status"] = ",".join(str(item) for item in deny_status)
        elif deny_status:
            context["paper_smoke_manifest_deny_status"] = str(deny_status)

        status_counts = manifest_section.get("status_counts")
        if isinstance(status_counts, Mapping):
            for status, value in status_counts.items():
                context[f"paper_smoke_manifest_status_count_{status}"] = str(value)

        total_entries = manifest_section.get("total_entries")
        if total_entries is not None:
            context["paper_smoke_manifest_total_entries"] = str(total_entries)

        signature_payload = manifest_section.get("summary_signature")
        if isinstance(signature_payload, Mapping):
            signature_value = signature_payload.get("value")
            if signature_value:
                context["paper_smoke_manifest_signature"] = str(signature_value)
            signature_algorithm = signature_payload.get("algorithm")
            if signature_algorithm:
                context["paper_smoke_manifest_signature_algorithm"] = str(signature_algorithm)
            signature_key_id = signature_payload.get("key_id")
            if signature_key_id:
                context["paper_smoke_manifest_signature_key_id"] = str(signature_key_id)

        exit_code = manifest_section.get("exit_code")
        if exit_code is not None:
            context["paper_smoke_manifest_exit_code"] = str(exit_code)
            try:
                if int(exit_code) != 0:
                    severity = "error"
            except (TypeError, ValueError):
                severity = "error"

    tls_section = summary.get("tls_audit")
    if isinstance(tls_section, Mapping):
        report_path = tls_section.get("report_path")
        status_value = tls_section.get("status")
        exit_code = tls_section.get("exit_code")
        warnings_payload = tls_section.get("warnings")
        errors_payload = tls_section.get("errors")
        if report_path:
            context["paper_smoke_tls_audit_path"] = str(report_path)
        if status_value:
            context["paper_smoke_tls_audit_status"] = str(status_value)
        if exit_code is not None:
            context["paper_smoke_tls_audit_exit_code"] = str(exit_code)
            try:
                if int(exit_code) != 0:
                    severity = "error"
            except (TypeError, ValueError):
                severity = "error"
        if isinstance(warnings_payload, (list, tuple, set)) and warnings_payload:
            context["paper_smoke_tls_audit_warning_count"] = str(len(warnings_payload))
            if severity == "info":
                severity = "warning"
        elif isinstance(warnings_payload, str) and warnings_payload:
            context["paper_smoke_tls_audit_warning_count"] = "1"
            if severity == "info":
                severity = "warning"
        if isinstance(errors_payload, (list, tuple, set)) and errors_payload:
            context["paper_smoke_tls_audit_error_count"] = str(len(errors_payload))
            severity = "error"
        elif isinstance(errors_payload, str) and errors_payload:
            context["paper_smoke_tls_audit_error_count"] = "1"
            severity = "error"

    token_section = summary.get("token_audit")
    if isinstance(token_section, Mapping):
        report_path = token_section.get("report_path")
        status_value = token_section.get("status")
        exit_code = token_section.get("exit_code")
        warnings_payload = token_section.get("warnings")
        errors_payload = token_section.get("errors")
        if report_path:
            context["paper_smoke_token_audit_path"] = str(report_path)
        if status_value:
            context["paper_smoke_token_audit_status"] = str(status_value)
        if exit_code is not None:
            context["paper_smoke_token_audit_exit_code"] = str(exit_code)
            try:
                if int(exit_code) != 0:
                    severity = "error"
            except (TypeError, ValueError):
                severity = "error"
        if isinstance(warnings_payload, (list, tuple, set)) and warnings_payload:
            context["paper_smoke_token_audit_warning_count"] = str(len(warnings_payload))
            if severity == "info":
                severity = "warning"
        elif isinstance(warnings_payload, str) and warnings_payload:
            context["paper_smoke_token_audit_warning_count"] = "1"
            if severity == "info":
                severity = "warning"
        if isinstance(errors_payload, (list, tuple, set)) and errors_payload:
            context["paper_smoke_token_audit_error_count"] = str(len(errors_payload))
            severity = "error"
        elif isinstance(errors_payload, str) and errors_payload:
            context["paper_smoke_token_audit_error_count"] = "1"
            severity = "error"

    # Sekcja audytu baseline bezpieczeństwa (zachowana z drugiej gałęzi)
    security_baseline_section = summary.get("security_baseline")
    if isinstance(security_baseline_section, Mapping):
        report_path = security_baseline_section.get("report_path")
        status_value = security_baseline_section.get("status")
        exit_code = security_baseline_section.get("exit_code")
        warnings_payload = security_baseline_section.get("warnings")
        errors_payload = security_baseline_section.get("errors")
        if report_path:
            context["paper_smoke_security_baseline_path"] = str(report_path)
        if status_value:
            context["paper_smoke_security_baseline_status"] = str(status_value)
        baseline_status_value = security_baseline_section.get("baseline_status")
        if baseline_status_value:
            context["paper_smoke_security_baseline_report_status"] = str(baseline_status_value)
        if exit_code is not None:
            context["paper_smoke_security_baseline_exit_code"] = str(exit_code)
            try:
                if int(exit_code) != 0:
                    severity = "error"
            except (TypeError, ValueError):
                severity = "error"
        if isinstance(warnings_payload, (list, tuple, set)) and warnings_payload:
            context["paper_smoke_security_baseline_warning_count"] = str(len(warnings_payload))
            if severity == "info":
                severity = "warning"
        elif isinstance(warnings_payload, str) and warnings_payload:
            context["paper_smoke_security_baseline_warning_count"] = "1"
            if severity == "info":
                severity = "warning"
        if isinstance(errors_payload, (list, tuple, set)) and errors_payload:
            context["paper_smoke_security_baseline_error_count"] = str(len(errors_payload))
            severity = "error"
        elif isinstance(errors_payload, str) and errors_payload:
            context["paper_smoke_security_baseline_error_count"] = "1"
            severity = "error"
        signature_payload = security_baseline_section.get("summary_signature")
        if isinstance(signature_payload, Mapping):
            signature_value = signature_payload.get("value")
            if signature_value:
                context["paper_smoke_security_baseline_signature"] = str(signature_value)
            signature_algorithm = signature_payload.get("algorithm")
            if signature_algorithm:
                context["paper_smoke_security_baseline_signature_algorithm"] = str(signature_algorithm)
            signature_key_id = signature_payload.get("key_id")
            if signature_key_id:
                context["paper_smoke_security_baseline_signature_key_id"] = str(signature_key_id)

    publish_section = summary.get(_SUMMARY_KEY_PUBLISH)
    if isinstance(publish_section, Mapping):
        _append_publish_context(context, publish_section)
        publish_status = str(publish_section.get("status", "unknown")).lower()
        required = bool(publish_section.get("required"))
        if required and publish_status != "ok":
            severity = "error"

    timestamp_value = summary.get("timestamp")
    if timestamp_value:
        context["summary_timestamp"] = str(timestamp_value)

    body_lines = [
        f"Środowisko: {environment}",
        f"Operator: {context.get('operator', 'unknown')}",
        f"Severity: {severity.upper()}",
        f"Pre-check: {context.get('precheck_status', 'unknown')}",
    ]
    manifest_status = context.get("paper_smoke_manifest_status")
    if manifest_status:
        body_lines.append(f"Manifest OHLCV: {manifest_status}")
    token_status = context.get("paper_smoke_token_audit_status")
    if token_status:
        body_lines.append(f"Audyt tokenów: {token_status}")
    token_warnings = context.get("paper_smoke_token_audit_warning_count")
    if token_warnings:
        body_lines.append(f"Tokeny ostrzeżenia: {token_warnings}")
    token_errors = context.get("paper_smoke_token_audit_error_count")
    if token_errors:
        body_lines.append(f"Tokeny błędy: {token_errors}")
    tls_status = context.get("paper_smoke_tls_audit_status")
    if tls_status:
        body_lines.append(f"Audyt TLS: {tls_status}")
    tls_warnings = context.get("paper_smoke_tls_audit_warning_count")
    if tls_warnings:
        body_lines.append(f"TLS ostrzeżenia: {tls_warnings}")
    tls_errors = context.get("paper_smoke_tls_audit_error_count")
    if tls_errors:
        body_lines.append(f"TLS błędy: {tls_errors}")
    baseline_status = context.get("paper_smoke_security_baseline_status")
    if baseline_status:
        body_lines.append(f"Audyt bezpieczeństwa: {baseline_status}")
    baseline_warnings = context.get("paper_smoke_security_baseline_warning_count")
    if baseline_warnings:
        body_lines.append(f"Bezpieczeństwo ostrzeżenia: {baseline_warnings}")
    baseline_errors = context.get("paper_smoke_security_baseline_error_count")
    if baseline_errors:
        body_lines.append(f"Bezpieczeństwo błędy: {baseline_errors}")
    publish_status = context.get("paper_smoke_publish_status")
    if publish_status:
        body_lines.append(f"Auto-publikacja: {publish_status}")
    json_location = context.get("paper_smoke_json_sync_location")
    if json_location:
        body_lines.append(f"JSONL: {json_location}")
    archive_location = context.get("paper_smoke_archive_location")
    if archive_location:
        body_lines.append(f"Archiwum: {archive_location}")

    title = f"Paper smoke summary ({environment})"
    body = "\n".join(body_lines)
    return severity, title, body, context


def _dispatch_alert(
    *,
    args: argparse.Namespace,
    message: AlertMessage,
    summary: Mapping[str, Any],
) -> None:
    config = load_core_config(args.config)
    if args.environment not in config.environments:
        raise KeyError(f"Środowisko '{args.environment}' nie istnieje w konfiguracji")
    environment_cfg = config.environments[args.environment]

    try:
        secret_manager = _create_secret_manager(args)
    except SecretStorageError as exc:
        raise RuntimeError(f"Nie udało się zainicjalizować magazynu sekretów: {exc}") from exc

    _, router, audit_log = build_alert_channels(
        core_config=config,
        environment=environment_cfg,
        secret_manager=secret_manager,
    )

    _LOGGER.info(
        "Wysyłam alert paper_smoke_compliance: environment=%s severity=%s",
        args.environment,
        message.severity,
    )
    router.dispatch(message)
    try:
        exported = list(audit_log.export())
    except Exception:  # noqa: BLE001 - defensywne
        exported = []

    output_payload = {
        "status": "sent",
        "category": message.category,
        "severity": message.severity,
        "environment": args.environment,
        "summary_sha256": message.context.get("summary_sha256", ""),
        "audit_records": exported,
    }
    print(json.dumps(output_payload, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), stream=sys.stdout)

    summary_path = Path(args.summary_json)
    try:
        summary = _load_summary(summary_path)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Nie udało się odczytać summary.json: %s", exc)
        return 2

    severity, title, body, context = _build_alert_payload(
        summary=summary,
        environment=args.environment,
        operator_override=args.operator,
    )

    if args.severity_override:
        severity = _normalize_severity(args.severity_override, fallback=severity)

    message = AlertMessage(
        category=args.category,
        title=title,
        body=body,
        severity=severity,
        context=context,
    )

    if args.dry_run:
        preview = {
            "status": "dry-run",
            "category": message.category,
            "severity": message.severity,
            "title": message.title,
            "body": message.body,
            "context": dict(message.context),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    try:
        _dispatch_alert(args=args, message=message, summary=summary)
    except FileNotFoundError as exc:
        _LOGGER.error("Błąd konfiguracji: %s", exc)
        return 1
    except KeyError as exc:
        _LOGGER.error(str(exc))
        return 1
    except RuntimeError as exc:
        _LOGGER.error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nieznany błąd podczas wysyłki alertu: %s", exc)
        return 3

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
