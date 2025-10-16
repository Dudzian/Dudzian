import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience import hypercare  # noqa: E402
from bot_core.resilience.policy import load_policy  # noqa: E402


def _parse_metadata(values: list[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not values:
        return metadata
    for item in values:
        if "=" not in item:
            raise ValueError("Metadane muszą mieć format klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        metadata[key] = value.strip()
    return metadata


def _load_key(value: str | None, path: str | None, env: str | None) -> bytes | None:
    provided = [item for item in (value, path, env) if item]
    if len(provided) > 1:
        raise ValueError("Klucz HMAC podaj jako wartość, plik lub zmienną środowiskową – tylko jedną opcję")
    if value:
        return value.encode("utf-8")
    if env:
        env_value = os.environ.get(env)
        if not env_value:
            raise ValueError(f"Zmienna środowiskowa {env} nie zawiera klucza HMAC")
        return env_value.encode("utf-8")
    if path:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {file_path}")
        return file_path.read_bytes().strip()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wykonuje pełny cykl Resilience Stage6")
    parser.add_argument("--source", required=True, help="Katalog źródłowy artefaktów do paczki")
    parser.add_argument("--plan", required=True, help="Plan failover (JSON)")

    parser.add_argument(
        "--bundle-output-dir",
        default=str(REPO_ROOT / "var" / "resilience"),
        help="Katalog docelowy paczki odpornościowej",
    )
    parser.add_argument("--bundle-name", default="stage6-resilience", help="Prefiks nazwy paczki")
    parser.add_argument("--include", action="append", help="Wzorce plików do uwzględnienia (glob)")
    parser.add_argument("--exclude", action="append", help="Wzorce plików do pominięcia (glob)")
    parser.add_argument("--metadata", action="append", help="Dodatkowe metadane w formacie klucz=wartość")

    parser.add_argument(
        "--audit-json",
        default="var/audit/resilience/audit_summary.json",
        help="Plik wyjściowy z podsumowaniem audytu",
    )
    parser.add_argument("--audit-csv", help="Opcjonalny raport CSV z audytu")
    parser.add_argument("--audit-signature", help="Plik podpisu raportu audytu")
    parser.add_argument("--audit-require-signature", action="store_true", help="Wymagaj podpisu manifestu podczas audytu")
    parser.add_argument("--audit-no-verify", action="store_true", help="Nie weryfikuj podpisu manifestu kluczem HMAC")
    parser.add_argument("--audit-policy", help="Plik polityki wymagań paczki (JSON)")
    parser.add_argument("--audit-hmac-key", help="Klucz HMAC do weryfikacji podpisu manifestu")
    parser.add_argument("--audit-hmac-key-path", help="Plik z kluczem HMAC dla audytu")
    parser.add_argument("--audit-hmac-key-env", help="Zmienna środowiskowa z kluczem HMAC audytu")

    parser.add_argument(
        "--failover-json",
        default="var/audit/resilience/failover_summary.json",
        help="Plik wyjściowy podsumowania failover",
    )
    parser.add_argument("--failover-csv", help="Opcjonalny raport CSV z failover")
    parser.add_argument("--failover-signature", help="Podpis podsumowania failover")

    parser.add_argument("--self-heal-config", help="Konfiguracja self-healing (JSON)")
    parser.add_argument(
        "--self-heal-output",
        default="var/audit/resilience/self_healing_report.json",
        help="Plik raportu self-healing",
    )
    parser.add_argument("--self-heal-signature", help="Podpis raportu self-healing")
    parser.add_argument(
        "--self-heal-mode",
        choices=("plan", "execute"),
        default="plan",
        help="Tryb self-healing: plan lub wykonanie",
    )

    parser.add_argument("--signing-key", help="Wartość klucza HMAC do podpisów")
    parser.add_argument("--signing-key-path", help="Plik z kluczem HMAC do podpisów")
    parser.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        metadata = _parse_metadata(args.metadata)
        signing_key = _load_key(args.signing_key, args.signing_key_path, args.signing_key_env)
        audit_key = _load_key(args.audit_hmac_key, args.audit_hmac_key_path, args.audit_hmac_key_env)
        policy = load_policy(Path(args.audit_policy)) if args.audit_policy else None

        bundle_config = hypercare.BundleConfig(
            source=Path(args.source),
            output_dir=Path(args.bundle_output_dir),
            bundle_name=args.bundle_name,
            include=args.include,
            exclude=args.exclude,
            metadata=metadata,
        )
        audit_config = hypercare.AuditConfig(
            json_path=Path(args.audit_json),
            csv_path=Path(args.audit_csv) if args.audit_csv else None,
            signature_path=Path(args.audit_signature) if args.audit_signature else None,
            require_signature=args.audit_require_signature,
            verify_signature=not args.audit_no_verify,
            policy=policy,
        )
        failover_config = hypercare.FailoverConfig(
            plan_path=Path(args.plan),
            json_path=Path(args.failover_json),
            csv_path=Path(args.failover_csv) if args.failover_csv else None,
            signature_path=Path(args.failover_signature) if args.failover_signature else None,
        )
        self_healing_config = None
        if args.self_heal_config:
            self_healing_config = hypercare.SelfHealingConfig(
                rules_path=Path(args.self_heal_config),
                output_path=Path(args.self_heal_output),
                signature_path=Path(args.self_heal_signature) if args.self_heal_signature else None,
                mode=args.self_heal_mode,
            )

        cycle = hypercare.ResilienceHypercareCycle(
            hypercare.ResilienceCycleConfig(
                bundle=bundle_config,
                audit=audit_config,
                failover=failover_config,
                signing_key=signing_key,
                signing_key_id=args.signing_key_id,
                audit_hmac_key=audit_key,
                self_healing=self_healing_config,
            )
        )
        result = cycle.run()
    except Exception as exc:  # noqa: BLE001 - komunikat dla operatora
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    summary: dict[str, Any] = {
        "bundle": result.bundle_artifacts.bundle_path.as_posix(),
        "audit": {
            "json": result.audit_summary_path.as_posix(),
            "status": result.audit_result.is_successful(),
        },
        "failover": {
            "json": result.failover_summary_path.as_posix(),
            "status": result.failover_summary.status,
        },
        "verification": result.verification,
    }
    if result.audit_csv_path:
        summary["audit"]["csv"] = result.audit_csv_path.as_posix()
    if result.audit_signature_path:
        summary["audit"]["signature"] = result.audit_signature_path.as_posix()
    if result.failover_csv_path:
        summary["failover"]["csv"] = result.failover_csv_path.as_posix()
    if result.failover_signature_path:
        summary["failover"]["signature"] = result.failover_signature_path.as_posix()
    if result.self_healing_report_path:
        summary["self_healing"] = {
            "json": result.self_healing_report_path.as_posix(),
        }
        if result.self_healing_signature_path:
            summary["self_healing"]["signature"] = result.self_healing_signature_path.as_posix()
        if result.self_healing_payload:
            summary["self_healing"]["status"] = result.self_healing_payload.get("status")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(run())
