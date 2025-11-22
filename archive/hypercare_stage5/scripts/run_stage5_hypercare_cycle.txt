"""CLI agregujące artefakty hypercare Stage5 i generujące zbiorczy raport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from bot_core.runtime.stage5_hypercare import (
    Stage5ComplianceConfig,
    Stage5HypercareConfig,
    Stage5HypercareCycle,
    Stage5OemAcceptanceConfig,
    Stage5RotationConfig,
    Stage5SloConfig,
    Stage5TcoConfig,
    Stage5TrainingConfig,
)


def _load_key(value: str | None, path: str | None) -> bytes | None:
    if value and path:
        raise ValueError("Podaj klucz w formie tekstu lub pliku, nie obu naraz.")
    if value:
        return value.encode("utf-8")
    if path:
        return Path(path).expanduser().read_bytes().strip()
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buduje podpisany raport podsumowujący cykl hypercare Stage5.",
    )

    parser.add_argument(
        "--tco-summary",
        required=True,
        help="Ścieżka do pliku JSON z raportem TCO (scripts/run_tco_analysis.py)",
    )
    parser.add_argument("--tco-signature", help="Plik z podpisem HMAC raportu TCO")
    parser.add_argument("--tco-signing-key", help="Klucz do weryfikacji podpisu TCO (tekst)")
    parser.add_argument("--tco-signing-key-file", help="Plik z kluczem weryfikującym TCO")
    parser.add_argument(
        "--tco-require-signature",
        action="store_true",
        help="Wymagaj obecności podpisu HMAC dla raportu TCO",
    )

    parser.add_argument(
        "--rotation-summary",
        required=True,
        help="Raport JSON wygenerowany przez scripts/rotate_keys.py",
    )
    parser.add_argument("--rotation-signing-key", help="Klucz HMAC do weryfikacji raportu rotacji")
    parser.add_argument("--rotation-signing-key-file", help="Plik z kluczem HMAC do raportu rotacji")
    parser.add_argument(
        "--rotation-require-signature",
        action="store_true",
        help="Wymagaj podpisu w raporcie rotacji kluczy",
    )

    parser.add_argument(
        "--compliance-report",
        action="append",
        dest="compliance_reports",
        help="Raport zgodności Stage5 (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--compliance-expected-control",
        action="append",
        dest="compliance_expected_controls",
        help="Identyfikator kontroli oczekiwany w raporcie zgodności",
    )
    parser.add_argument("--compliance-signing-key", help="Klucz do weryfikacji raportów zgodności")
    parser.add_argument("--compliance-signing-key-file", help="Plik z kluczem raportów zgodności")
    parser.add_argument(
        "--compliance-require-signature",
        action="store_true",
        help="Wymagaj podpisów HMAC we wszystkich raportach zgodności",
    )

    parser.add_argument(
        "--training-log",
        action="append",
        dest="training_logs",
        help="Log szkolenia Stage5 (można podać wielokrotnie)",
    )
    parser.add_argument("--training-signing-key", help="Klucz do weryfikacji podpisów szkoleń")
    parser.add_argument("--training-signing-key-file", help="Plik z kluczem podpisów szkoleń")
    parser.add_argument(
        "--training-require-signature",
        action="store_true",
        help="Wymagaj podpisów w logach szkoleniowych",
    )

    parser.add_argument("--slo-report", help="Raport SLO wygenerowany przez scripts/slo_monitor.py")
    parser.add_argument("--slo-signature", help="Podpis HMAC raportu SLO")
    parser.add_argument("--slo-signing-key", help="Klucz do weryfikacji podpisu raportu SLO")
    parser.add_argument("--slo-signing-key-file", help="Plik z kluczem do raportu SLO")
    parser.add_argument(
        "--slo-require-signature",
        action="store_true",
        help="Wymagaj podpisu HMAC raportu SLO",
    )

    parser.add_argument("--oem-summary", help="Podsumowanie kroków scripts/run_oem_acceptance.py")
    parser.add_argument("--oem-signature", help="Podpis HMAC podsumowania akceptacji OEM")
    parser.add_argument("--oem-signing-key", help="Klucz HMAC do weryfikacji podpisu OEM")
    parser.add_argument("--oem-signing-key-file", help="Plik z kluczem HMAC do podpisu OEM")
    parser.add_argument(
        "--oem-require-signature",
        action="store_true",
        help="Wymagaj podpisu HMAC podsumowania akceptacji OEM",
    )

    parser.add_argument(
        "--output",
        default="var/audit/stage5/hypercare/summary.json",
        help="Plik wynikowy z raportem hypercare Stage5",
    )
    parser.add_argument("--signature", help="Plik z podpisem raportu hypercare")
    parser.add_argument("--signing-key", help="Klucz HMAC do podpisu raportu hypercare")
    parser.add_argument("--signing-key-file", help="Plik z kluczem HMAC do podpisu raportu")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego raport")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        tco_config = Stage5TcoConfig(
            summary_path=Path(args.tco_summary),
            signature_path=Path(args.tco_signature).expanduser() if args.tco_signature else None,
            signing_key=_load_key(args.tco_signing_key, args.tco_signing_key_file),
            require_signature=bool(args.tco_require_signature),
        )
        rotation_config = Stage5RotationConfig(
            summary_path=Path(args.rotation_summary),
            signing_key=_load_key(args.rotation_signing_key, args.rotation_signing_key_file),
            require_signature=bool(args.rotation_require_signature),
        )

        compliance_reports = [Path(path) for path in args.compliance_reports or []]
        compliance_config = Stage5ComplianceConfig(
            reports=compliance_reports,
            expected_controls=tuple(args.compliance_expected_controls or ()),
            signing_key=_load_key(args.compliance_signing_key, args.compliance_signing_key_file),
            require_signature=bool(args.compliance_require_signature),
        )

        training_logs = [Path(path) for path in args.training_logs or []]
        training_config = Stage5TrainingConfig(
            logs=training_logs,
            signing_key=_load_key(args.training_signing_key, args.training_signing_key_file),
            require_signature=bool(args.training_require_signature),
        )

        slo_config = None
        if args.slo_report:
            slo_config = Stage5SloConfig(
                report_path=Path(args.slo_report),
                signature_path=Path(args.slo_signature) if args.slo_signature else None,
                signing_key=_load_key(args.slo_signing_key, args.slo_signing_key_file),
                require_signature=bool(args.slo_require_signature),
            )

        oem_config = None
        if args.oem_summary:
            oem_config = Stage5OemAcceptanceConfig(
                summary_path=Path(args.oem_summary),
                signature_path=Path(args.oem_signature) if args.oem_signature else None,
                signing_key=_load_key(args.oem_signing_key, args.oem_signing_key_file),
                require_signature=bool(args.oem_require_signature),
            )

        config = Stage5HypercareConfig(
            output_path=Path(args.output),
            signature_path=Path(args.signature) if args.signature else None,
            signing_key=_load_key(args.signing_key, args.signing_key_file),
            signing_key_id=args.signing_key_id,
            tco=tco_config,
            rotation=rotation_config,
            compliance=compliance_config,
            training=training_config,
            slo=slo_config,
            oem=oem_config,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(json.dumps({"error": f"Nie znaleziono pliku: {exc}"}), file=sys.stderr)
        return 2

    cycle = Stage5HypercareCycle(config)
    result = cycle.run()

    output = {
        "output": str(result.output_path),
        "status": result.payload.get("overall_status"),
        "signature": str(result.signature_path) if result.signature_path else None,
        "issues": result.payload.get("issues", []),
        "warnings": result.payload.get("warnings", []),
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
