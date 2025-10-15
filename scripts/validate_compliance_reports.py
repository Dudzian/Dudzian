"""Walidacja artefaktów compliance Etapu 5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import verify_signature as verify_signature_module  # noqa: E402


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    details: Mapping[str, Any]


class ComplianceValidationError(RuntimeError):
    """Podnoszony w przypadku krytycznej niezgodności artefaktów."""


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--tco-json", type=Path, help="Ścieżka do raportu TCO (JSON)")
    parser.add_argument(
        "--tco-signature",
        type=Path,
        help="Podpis HMAC raportu TCO (plik JSON)",
    )
    parser.add_argument(
        "--tco-signing-key",
        type=Path,
        help="Klucz HMAC używany do weryfikacji raportu TCO",
    )

    parser.add_argument(
        "--observability-manifest",
        type=Path,
        help="Manifest dashboardów/alertów observability",
    )
    parser.add_argument(
        "--observability-signature",
        type=Path,
        help="Podpis manifestu observability",
    )
    parser.add_argument(
        "--observability-signing-key",
        type=Path,
        help="Klucz HMAC do weryfikacji manifestu observability",
    )

    parser.add_argument(
        "--decision-smoke",
        type=Path,
        help="Wynik skryptu run_decision_engine_smoke.py",
    )
    parser.add_argument(
        "--decision-log-summary",
        type=Path,
        help="Raport z verify_decision_log.py dla Etapu 5",
    )
    parser.add_argument(
        "--slo-report",
        type=Path,
        help="Raport monitoringu SLO",
    )
    parser.add_argument(
        "--alerts-report",
        type=Path,
        help="Raport walidacji reguł alertowych",
    )
    parser.add_argument(
        "--rotation-report",
        type=Path,
        help="Raport planu rotacji kluczy Stage5",
    )
    parser.add_argument(
        "--compliance-report",
        type=Path,
        help="Raport audytu compliance Stage5",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Opcjonalna ścieżka do zapisu podsumowania JSON",
    )

    return parser.parse_args(argv)


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
            raise ComplianceValidationError(
                f"{label} musi mieć uprawnienia maksymalnie 600: {resolved}"
            )
    return resolved


def _load_json(path: Path, *, label: str) -> Any:
    resolved = _ensure_regular_file(path, label=label)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - uszkodzony JSON
        raise ComplianceValidationError(f"Nieprawidłowy JSON w {resolved}") from exc


def _verify_signature(manifest: Path, signature: Path, key: Path) -> None:
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
        raise ComplianceValidationError(
            f"Weryfikacja podpisu nie powiodła się dla {manifest}"
        )


def _check_tco(args: argparse.Namespace) -> CheckResult | None:
    if not args.tco_json:
        return None

    payload = _load_json(args.tco_json, label="Raport TCO")
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError("Raport TCO musi być obiektem JSON")
    if "strategies" not in payload or "total" not in payload:
        raise ComplianceValidationError(
            "Raport TCO musi zawierać pola 'strategies' oraz 'total'"
        )

    if args.tco_signature or args.tco_signing_key:
        if not args.tco_signature or not args.tco_signing_key:
            raise ComplianceValidationError(
                "Do weryfikacji podpisu TCO wymagane są --tco-signature oraz --tco-signing-key"
            )
        signature_path = _ensure_regular_file(args.tco_signature, label="Podpis TCO")
        key_path = _ensure_regular_file(args.tco_signing_key, label="Klucz TCO")
        _verify_signature(args.tco_json, signature_path, key_path)

    return CheckResult(
        name="tco_report",
        status="ok",
        details={"path": str(Path(args.tco_json).expanduser().resolve())},
    )


def _check_observability(args: argparse.Namespace) -> CheckResult | None:
    if not args.observability_manifest:
        return None
    manifest_path = _ensure_regular_file(
        args.observability_manifest, label="Manifest observability"
    )
    _load_json(manifest_path, label="Manifest observability")

    if args.observability_signature or args.observability_signing_key:
        if not args.observability_signature or not args.observability_signing_key:
            raise ComplianceValidationError(
                "Do weryfikacji manifestu observability wymagane są podpis oraz klucz"
            )
        signature_path = _ensure_regular_file(
            args.observability_signature, label="Podpis manifestu observability"
        )
        key_path = _ensure_regular_file(
            args.observability_signing_key, label="Klucz manifestu observability"
        )
        _verify_signature(manifest_path, signature_path, key_path)

    return CheckResult(
        name="observability_bundle",
        status="ok",
        details={"manifest": str(manifest_path)},
    )


def _check_decision_smoke(args: argparse.Namespace) -> CheckResult | None:
    if not args.decision_smoke:
        return None

    payload = _load_json(args.decision_smoke, label="Raport decision smoke")
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError("Raport decision smoke musi być obiektem JSON")
    for field in ("accepted", "rejected", "stress_failures"):
        value = payload.get(field)
        if not isinstance(value, int) or value < 0:
            raise ComplianceValidationError(
                f"Pole '{field}' w raporcie decision smoke musi być nieujemną liczbą całkowitą"
            )

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ComplianceValidationError(
            "Raport decision smoke musi zawierać niepustą listę 'evaluations'"
        )

    return CheckResult(
        name="decision_smoke",
        status="ok",
        details={
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
        raise ComplianceValidationError(
            f"Raport decision log posiada status {status!r}, oczekiwano 'PASS'"
        )
    missing_fields = payload.get("missing_fields", [])
    if missing_fields:
        raise ComplianceValidationError(
            "Raport decision log zgłasza brakujące pola: " + ", ".join(map(str, missing_fields))
        )

    return CheckResult(
        name="decision_log",
        status="ok",
        details={"status": status},
    )


def _check_json_payload(
    *,
    path: Optional[Path],
    label: str,
    required_keys: Sequence[str],
    check_name: str,
) -> CheckResult | None:
    if not path:
        return None
    payload = _load_json(path, label=label)
    if not isinstance(payload, Mapping):
        raise ComplianceValidationError(f"{label} musi być obiektem JSON")
    for key in required_keys:
        if key not in payload:
            raise ComplianceValidationError(
                f"{label} musi zawierać pole '{key}'"
            )
    return CheckResult(
        name=check_name,
        status="ok",
        details={"path": str(Path(path).expanduser().resolve())},
    )


def _dump_summary(results: Sequence[CheckResult]) -> list[Mapping[str, Any]]:
    return [
        {
            "check": result.name,
            "status": result.status,
            "details": dict(result.details),
        }
        for result in results
    ]


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    checks: list[CheckResult] = []

    try:
        result = _check_tco(args)
        if result:
            checks.append(result)

        result = _check_observability(args)
        if result:
            checks.append(result)

        result = _check_decision_smoke(args)
        if result:
            checks.append(result)

        result = _check_decision_log(args)
        if result:
            checks.append(result)

        for name, path, required in (
            ("slo_report", args.slo_report, ("metrics", "generated_at")),
            ("alerts_report", args.alerts_report, ("status",)),
            ("rotation_plan", args.rotation_report, ("plan",)),
            ("compliance_report", args.compliance_report, ("status",)),
        ):
            result = _check_json_payload(
                path=path,
                label=name.replace("_", " ").title(),
                required_keys=required,
                check_name=name,
            )
            if result:
                checks.append(result)

    except ComplianceValidationError as exc:
        summary = _dump_summary(checks)
        summary.append(
            {
                "check": "failure",
                "status": "failed",
                "details": {"error": str(exc)},
            }
        )
        if args.summary_output:
            args.summary_output.expanduser().resolve().write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"[compliance] Błąd walidacji: {exc}")
        return 1

    if not checks:
        print("[compliance] Nie przekazano żadnych artefaktów do walidacji")
        return 1

    summary = _dump_summary(checks)
    if args.summary_output:
        args.summary_output.expanduser().resolve().write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for result in checks:
        print(f"[compliance] {result.name}: OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - punkt wejścia CLI
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
