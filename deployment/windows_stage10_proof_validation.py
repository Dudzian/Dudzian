"""Jawna odmowa Stage 10 do czasu dostarczenia natywnych autorytetów dowodowych.

To diagnostyka braków, nie drugi Production Trust ani weryfikator zapisanych
deklaracji. Żaden JSON, status operatora czy wynik testu nie dostarcza brakującej
ciągłości reboot/soak. Istniejące wymagania produkcyjne pozostają bez zmian.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, NoReturn

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
    """Brak implementacji autorytetu; celowo nie przyjmuj deklaracji wywołującego."""
    raise Stage10ProofUnavailable(UNAVAILABLE)


def blocked_report() -> dict[str, Any]:
    """Zwróć wszystkie braki; nawet kompletne zmienne CI nie uprawniają do PASS."""
    return {
        "schema_version": 1,
        "report_kind": "windows-stage10-proof-blockers",
        "status": "BLOCKED",
        "reason_code": UNAVAILABLE,
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


def write_blocked_report(output: Path) -> None:
    """Usuń stary wynik przed próbą zapisu, publikuj wyłącznie pełną diagnostykę.

    Przerwanie lub błąd zapisu nie może pozostawić poprzedniego PASS pod ścieżką
    wyjściową. Plik tymczasowy jest tworzony w tym samym katalogu co wynik.
    """
    output.unlink(missing_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(blocked_report(), stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Odczytowa odmowa kwalifikacji Windows Stage 10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        write_blocked_report(args.output)
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
