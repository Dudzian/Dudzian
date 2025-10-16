#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5 Compliance Validator – merged CLI (HEAD + main)

Subcommands:
  - validate : Skanuje raporty (JSON) z katalogów/ścieżek, sprawdza oczekiwane „kontrole”
               i podpisy HMAC (klucz w Base64), zwraca zbiorcze JSON summary.
  - audit    : Weryfikuje wskazane artefakty (TCO, observability bundle, decision smoke/log,
               SLO/alerts/rotation/compliance) i opcjonalnie sprawdza podpisy przez scripts.verify_signature.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────────────────────────────────
# Imports wspólne z obu wariantów
# ────────────────────────────────────────────────────────────────────────────────
# HEAD-style validator:
from bot_core.compliance.reports import validate_compliance_reports  # type: ignore

# main-style signature verifier:
from scripts import verify_signature as verify_signature_module  # type: ignore

# ==============================================================================
# validate (HEAD) – masowa walidacja raportów
# ==============================================================================
_DEFAULT_CONTROLS = [
    "stage5.training.log",
    "stage5.tco.report",
    "stage5.oem.dry_run",
    "stage5.key_rotation",
    "stage5.compliance.review",
]


def _decode_b64_key(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise SystemExit(f"Nie udało się zdekodować klucza HMAC (Base64): {exc}") from exc


def _load_b64_key_from_file(path: str | None) -> bytes | None:
    if not path:
        return None
    return _decode_b64_key(Path(path).read_text(encoding="utf-8"))


def _load_b64_key_from_env(env_name: str | None) -> bytes | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value is None:
        raise SystemExit(f"Zmienna środowiskowa {env_name} nie jest ustawiona")
    return _decode_b64_key(value)


def _gather_reports(paths: Sequence[str]) -> list[Path]:
    result: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    return result


def _resolve_controls(args: argparse.Namespace) -> list[str] | None:
    controls = [] if args.no_default_controls else list(_DEFAULT_CONTROLS)
    if args.expected_controls:
        controls.extend(args.expected_controls)
    return controls or None


def _resolve_signing_key_validate(args: argparse.Namespace) -> bytes | None:
    # Priorytet: --signing-key (Base64) > --signing-key-file > --signing-key-env
    for candidate in (
        _decode_b64_key(args.signing_key),
        _load_b64_key_from_file(args.signing_key_file),
    ):
        if candidate:
            return candidate
    if args.signing_key_env:
        return _load_b64_key_from_env(args.signing_key_env)
    return None


def _build_validate_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "validate",
        help="Waliduje raporty Stage5 (kontrole + podpisy HMAC w Base64).",
        description="Skanuje pliki/katalogi z raportami JSON, sprawdza oczekiwane 'kontrole' i podpisy HMAC."
    )
    p.add_argument("paths", nargs="+", help="Ścieżki do raportów lub katalogów z raportami")
    p.add_argument(
        "--expected-control",
        action="append",
        dest="expected_controls",
        help="Identyfikator kontroli wymagany w raporcie (można powtórzyć)",
    )
    p.add_argument(
        "--no-default-controls",
        action="store_true",
        help="Nie dodawaj domyślnej listy kontroli Stage5",
    )
    p.add_argument("--signing-key", help="Klucz HMAC (Base64) inline")
    p.add_argument("--signing-key-file", help="Plik z kluczem HMAC (Base64)")
    p.add_argument("--signing-key-env", help="ENV zawierające klucz HMAC (Base64)")
    p.add_argument(
        "--require-signature",
        action="store_true",
        help="Brak podpisu traktuj jako błąd krytyczny",
    )
    p.add_argument("--output-json", help="Zapisz wynik walidacji do pliku JSON")
    p.set_defaults(_handler=_handle_validate)
    return p


def _handle_validate(args: argparse.Namespace) -> int:
    reports = _gather_reports(args.paths)
    if not reports:
        print("Nie znaleziono raportów do walidacji", file=sys.stderr)
        return 2

    signing_key = _resolve_signing_key_validate(args)
    controls = _resolve_controls(args)

    results = validate_compliance_reports(
        reports,
        expected_controls=controls,
        signing_key=signing_key,
        require_signature=args.require_signature,
    )

    summary = []
    exit_code = 0
    for result in results:
        payload = {
            "report": str(result.report_path),
            "issues": result.issues,
            "warnings": result.warnings,
            "failed_controls": result.failed_controls,
            "passed_controls": result.passed_controls,
            "metadata": dict(result.metadata),
        }
        summary.append(payload)
        if result.issues or result.failed_controls:
            exit_code = 1

    output = {
        "reports": summary,
        "total": len(summary),
        "errors": sum(1 for item in summary if item["issues"]),
        "failures": sum(1 for item in summary if item["failed_controls"]),
    }

    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output_json:
        Path(args.output_json).write_text(serialized + "\n", encoding="utf-8")
    return exit_code


# ==============================================================================
# audit (main) – punktowe sprawdzanie artefaktów
# ==============================================================================
@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    details: Mapping[str, Any]


class ComplianceValidationError(RuntimeError):
    """Podnoszony w przypadku krytycznej niezgodności artefaktów."""


def _ensure_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise ComplianceValidationError(f"{label} nie istnieje: {resolved}")
    if resolved.is_dir():
        raise ComplianceValidationError(f"{label} nie może być katalogiem: {resolved}")
    if resolved.is_symlink():
        raise ComplianceValidationError(f"{label} nie może być symlinkiem: {resolved}")
    if os.name != "nt":
        mode = resolved.stat().st_mode
        if mode & 0o077:
            raise ComplianceValidationError(f"{label} musi mieć uprawnienia maks. 600: {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Any:
    resolved = _ensure_regular_file(path, label=label)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComplianceValidationError(f"Nieprawidłowy JSON w {resolved}") from exc


def _verify_signature(manifest: Path, signature: Path, key: Path) -> None:
    # delegujemy do istniejącego skryptu verify_signature, by zachować spójność
    signature_document = json.loads(signature.read_text(encoding="utf-8"))
    payload = signature_document.get("payload", {}) if isinstance(signature_document, dict) else {}
    digest_field = "sha256"
    if isinstance(payload, Mapping):
        for candidate in ("sha256", "sha384"):
            if candidate in payload:
                digest_field = candidate
                break

    argv = [
        "--manifest",
        str(manifest),
        "--signature",
        str(signature),
        "--signing-key",
        str(key),
        "--digest",
        digest_field,
    ]
    exit_code = verify_signature_module.main(argv)
    if exit_code != 0:
        raise ComplianceValidationError(f"Weryfikacja podpisu nie powiodła się dla {manifest}")


def _check_tco(args: argparse.Namespace) -> CheckResult | None:
    if not args.tco_json:
        return None
    payload = _load_json(args.tco_json, label="Raport TCO")
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError("Raport TCO musi być obiektem JSON")
    if "strategies" not in payload or "total" not in payload:
        raise ComplianceValidationError("Raport TCO musi zawierać pola 'strategies' oraz 'total'")
    if args.tco_signature or args.tco_signing_key:
        if not args.tco_signature or not args.tco_signing_key:
            raise ComplianceValidationError("Do weryfikacji podpisu TCO wymagane są --tco-signature oraz --tco-signing-key")
        _verify_signature(args.tco_json, _ensure_regular_file(args.tco_signature, label="Podpis TCO"),
                          _ensure_regular_file(args.tco_signing_key, label="Klucz TCO"))
    return CheckResult("tco_report", "ok", {"path": str(Path(args.tco_json).expanduser().resolve())})


def _check_observability(args: argparse.Namespace) -> CheckResult | None:
    if not args.observability_manifest:
        return None
    manifest_path = _ensure_regular_file(args.observability_manifest, label="Manifest observability")
    _load_json(manifest_path, label="Manifest observability")
    if args.observability_signature or args.observability_signing_key:
        if not args.observability_signature or not args.observability_signing_key:
            raise ComplianceValidationError("Do weryfikacji manifestu observability wymagane są podpis oraz klucz")
        _verify_signature(manifest_path,
                          _ensure_regular_file(args.observability_signature, label="Podpis manifestu observability"),
                          _ensure_regular_file(args.observability_signing_key, label="Klucz manifestu observability"))
    return CheckResult("observability_bundle", "ok", {"manifest": str(manifest_path)})


def _check_decision_smoke(args: argparse.Namespace) -> CheckResult | None:
    if not args.decision_smoke:
        return None
    payload = _load_json(args.decision_smoke, label="Raport decision smoke")
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError("Raport decision smoke musi być obiektem JSON")
    for field in ("accepted", "rejected", "stress_failures"):
        value = payload.get(field)
        if not isinstance(value, int) or value < 0:
            raise ComplianceValidationError(f"Pole '{field}' w raporcie decision smoke musi być nieujemną liczbą całkowitą")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ComplianceValidationError("Raport decision smoke musi zawierać niepustą listę 'evaluations'")
    return CheckResult(
        "decision_smoke",
        "ok",
        {
            "path": str(Path(args.decision_smoke).expanduser().resolve()),
            "accepted": payload.get("accepted", 0),
            "rejected": payload.get("rejected", 0),
            "stress_failures": payload.get("stress_failures", 0),
        },
    )


def _check_decision_log(args: argparse.Namespace) -> CheckResult | None:
    if not args.decision_log_summary:
        return None
    payload = _load_json(args.decision_log_summary, label="Raport decision log")
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError("Raport decision log musi być obiektem JSON")
    status = str(payload.get("status", "")).upper()
    if status != "PASS":
        raise ComplianceValidationError(f"Raport decision log posiada status {status!r}, oczekiwano 'PASS'")
    missing_fields = payload.get("missing_fields", [])
    if missing_fields:
        raise ComplianceValidationError("Raport decision log zgłasza brakujące pola: " + ", ".join(map(str, missing_fields)))
    return CheckResult("decision_log", "ok", {"status": status})


def _check_json_payload(*, path: Optional[Path], label: str, required_keys: Sequence[str], check_name: str) -> CheckResult | None:
    if not path:
        return None
    payload = _load_json(path, label=label)
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError(f"{label} musi być obiektem JSON")
    for key in required_keys:
        if key not in payload:
            raise ComplianceValidationError(f"{label} musi zawierać pole '{key}'")
    return CheckResult(check_name, "ok", {"path": str(Path(path).expanduser().resolve())})


def _dump_summary(results: Sequence[CheckResult]) -> list[Mapping[str, Any]]:
    return [{"check": r.name, "status": r.status, "details": dict(r.details)} for r in results]


def _build_audit_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "audit",
        help="Weryfikuje wskazane artefakty zgodności (podpisy + struktura).",
        description="Punktowa walidacja TCO, manifestu observability, decision smoke/log i podstawowych raportów JSON."
    )
    # TCO
    p.add_argument("--tco-json", type=Path, help="Ścieżka do raportu TCO (JSON)")
    p.add_argument("--tco-signature", type=Path, help="Podpis HMAC raportu TCO (plik JSON)")
    p.add_argument("--tco-signing-key", type=Path, help="Klucz HMAC do weryfikacji raportu TCO")
    # Observability bundle
    p.add_argument("--observability-manifest", type=Path, help="Manifest dashboardów/alertów")
    p.add_argument("--observability-signature", type=Path, help="Podpis manifestu observability")
    p.add_argument("--observability-signing-key", type=Path, help="Klucz HMAC do weryfikacji manifestu")
    # Decision smoke/log
    p.add_argument("--decision-smoke", type=Path, help="Wynik run_decision_engine_smoke.py")
    p.add_argument("--decision-log-summary", type=Path, help="Raport verify_decision_log.py (PASS wymagany)")
    # Inne raporty (kluczowe pola)
    p.add_argument("--slo-report", type=Path, help="Raport monitoringu SLO (musi mieć pola 'metrics' i 'generated_at')")
    p.add_argument("--alerts-report", type=Path, help="Raport walidacji reguł alertowych (musi mieć 'status')")
    p.add_argument("--rotation-report", type=Path, help="Raport planu rotacji kluczy (musi mieć 'plan')")
    p.add_argument("--compliance-report", type=Path, help="Raport audytu compliance (musi mieć 'status')")
    p.add_argument("--summary-output", type=Path, help="Opcjonalna ścieżka do zapisu podsumowania JSON")
    p.set_defaults(_handler=_handle_audit)
    return p


def _handle_audit(args: argparse.Namespace) -> int:
    checks: list[CheckResult] = []
    try:
        res = _check_tco(args)
        if res:
            checks.append(res)
        res = _check_observability(args)
        if res:
            checks.append(res)
        res = _check_decision_smoke(args)
        if res:
            checks.append(res)
        res = _check_decision_log(args)
        if res:
            checks.append(res)
        for name, path, required in (
            ("slo_report", args.slo_report, ("metrics", "generated_at")),
            ("alerts_report", args.alerts_report, ("status",)),
            ("rotation_plan", args.rotation_report, ("plan",)),
            ("compliance_report", args.compliance_report, ("status",)),
        ):
            res = _check_json_payload(path=path, label=name.replace("_", " ").title(), required_keys=required, check_name=name)
            if res:
                checks.append(res)
    except ComplianceValidationError as exc:
        summary = _dump_summary(checks)
        summary.append({"check": "failure", "status": "failed", "details": {"error": str(exc)}})
        if args.summary_output:
            args.summary_output.expanduser().resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[compliance] Błąd walidacji: {exc}")
        return 1

    if not checks:
        print("[compliance] Nie przekazano żadnych artefaktów do walidacji")
        return 1

    summary = _dump_summary(checks)
    if args.summary_output:
        args.summary_output.expanduser().resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for r in checks:
        print(f"[compliance] {r.name}: OK")
    return 0


# ==============================================================================
# Root CLI
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage5 Compliance Validator – łączy tryby validate (masowa walidacja) i audit (artefakty)."
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{validate|audit}", required=True)
    _build_validate_parser(sub)
    _build_audit_parser(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
