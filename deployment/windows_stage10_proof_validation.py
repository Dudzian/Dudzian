"""Jawna odmowa Stage 10 do czasu dostarczenia natywnych autorytetów dowodowych.

To diagnostyka braków, nie drugi Production Trust ani weryfikator zapisanych
deklaracji. Żaden JSON, status operatora czy wynik testu nie dostarcza brakującej
ciągłości reboot/soak. Istniejące wymagania produkcyjne pozostają bez zmian.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, NoReturn
import uuid

from deployment.windows_stage10_diagnostic_output import DiagnosticOutputError, publish_diagnostic

UNAVAILABLE = "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
PROOF_REQUIREMENTS = {
    "WINDOWS_REBOOT_RECOVERY": {
        "boot_changed": "Brak chronionej historii bootu właściwego hosta i próby.",
        "current_revision_runtime_bound": (
            "Wymagane niezależne piny artefaktu i bieżąca kwalifikacja enrolled runtime "
            "przez istniejące autorytety Stage 9."
        ),
        "service_autostarted": "Brak niezależnego potwierdzenia przyczyny startu usługi.",
        "readiness_contract_after_reboot": (
            "Brak świeżej uwierzytelnionej kwalifikacji procesu SCM i PostgreSQL; "
            "backend-readiness.json jest wyłącznie diagnostyką."
        ),
    },
    "WINDOWS_LONG_RUNNING_LIFECYCLE": {
        "duration_at_least_24h": "Brak pomiaru monotonicznego nieprzerwanej próby.",
        "current_revision_runtime_bound": (
            "Wymagane niezależne piny artefaktu i bieżąca kwalifikacja enrolled runtime "
            "przez istniejące autorytety Stage 9."
        ),
        "same_service_instance": (
            "Brak ciągłej obserwacji uchwytu procesu; zgodne próbki PID i czasu startu "
            "nie dowodzą ciągłości."
        ),
        "readiness_contract_before_after": (
            "Brak dwóch świeżych uwierzytelnionych kwalifikacji właściwych instancji "
            "procesu i zatwierdzonego produktu."
        ),
    },
}


class Stage10ProofUnavailable(RuntimeError):
    """Odmowa dowodu fizycznego, a nie niepowodzenie syntetycznego testu."""


def require_native_proof_authority() -> NoReturn:
    """Tymczasowa nieprzechodnia blokada, nie ukończony kwalifikator fizyczny.

    Jej zastąpienie wymaga odrębnego natywnego obserwatora bootu, pochodzenia
    startu, ciągłego uchwytu procesu i czasu monotonicznego oraz bieżącego
    kwalifikatora produktu/SQL z istniejącego Production Trust. Nie implementuje
    tego ani zapis diagnostyki, ani dostarczony przez wywołującego raport.
    """
    raise Stage10ProofUnavailable(UNAVAILABLE)


def blocked_report() -> dict[str, Any]:
    """Zwróć wszystkie braki; nawet kompletne zmienne CI nie uprawniają do PASS."""
    return {
        "schema_version": 1,
        "report_kind": "windows-stage10-proof-blockers",
        "status": "BLOCKED",
        "reason_code": UNAVAILABLE,
        "attempt_id": uuid.uuid4().hex,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "native_qualification_performed": False,
        "results": {
            item: {
                "status": "BLOCKED",
                "proofs": {
                    proof: {"status": "BLOCKED", "reason": reason}
                    for proof, reason in requirements.items()
                },
            }
            for item, requirements in PROOF_REQUIREMENTS.items()
        },
    }


def write_blocked_report(
    output: Path | None = None,
    *,
    approved_directory: Path | None = None,
    protected_inputs: tuple[Path, ...] = (),
) -> Path:
    """Publikuj nową diagnostykę wyłącznie w osobno zatwierdzonym katalogu.

    Historyczne pliki pozostają nietknięte. Próba ma nowy identyfikator i domyślnie
    nową nazwę; każda kolizja odmawia, nawet dla identycznego starego raportu.
    To diagnostyka BLOCKED, której konsument nie uznaje za evidence.
    """
    if approved_directory is None:
        raise DiagnosticOutputError("zapis wymaga osobno zatwierdzonego katalogu wynikowego")
    report = blocked_report()
    raw = (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    destination = output or approved_directory / f"stage10-blocked-{report['attempt_id']}.json"
    publish_diagnostic(
        destination,
        raw,
        approved_directory=approved_directory,
        protected_inputs=(
            *protected_inputs,
            Path(__file__).with_name("windows_stage10_lifecycle_contract.json"),
        ),
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Odczytowa odmowa kwalifikacji Windows Stage 10")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved-output-directory", type=Path)
    parser.add_argument("--protected-input", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.output is None and args.approved_output_directory is None:
            print(json.dumps(blocked_report(), ensure_ascii=False, allow_nan=False, indent=2))
        else:
            path = write_blocked_report(
                args.output,
                approved_directory=args.approved_output_directory,
                protected_inputs=tuple(args.protected_input),
            )
            print(f"Diagnostyka BLOCKED: {path}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(
            f"{UNAVAILABLE}: zapis diagnostyki nie powiódł się ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1
    print(UNAVAILABLE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
