"""Narzędzia pomocnicze do sondowania keyring i weryfikacji fingerprintu HWID."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Callable, Iterable

import keyring

try:  # ``bot_core`` jest opcjonalny przy testach jednostkowych.
    from core.security.license_verifier import LicenseVerifier
except Exception:  # pragma: no cover - fallback dla środowisk testowych bez zależności
    LicenseVerifier = None  # type: ignore[assignment]

_orig_get = keyring.get_password


def _probe(service: str, username: str) -> str | None:
    """Proxy na ``keyring.get_password`` wypisujący rezultat."""

    value = _orig_get(service, username)
    print(
        f"[PROBE] get_password(service={service!r}, username={username!r}) ->",
        "HIT" if value else "None",
    )
    return value


def run_daily_trend_probe(argv: Iterable[str] | None = None) -> None:
    """Uruchamia sondę keyring z parametrami jak w historycznym przykładzie."""

    keyring.get_password = _probe

    sys.argv = list(
        argv
        or (
            "run_daily_trend.py",
            "--environment",
            "binance_paper",
            "--secret-namespace",
            "dudzian.trading",
            "--dry-run",
            "--log-level",
            "INFO",
        )
    )

    runpy.run_path("scripts/run_daily_trend.py", run_name="__main__")


class HwidValidationError(RuntimeError):
    """Wyjątek zgłaszany gdy fingerprint HWID nie zgadza się z oczekiwaniem."""


def validate_hwid(
    expected_hwid: str | None,
    *,
    fingerprint_reader: Callable[[], str] | None = None,
) -> str:
    """Weryfikuje fingerprint urządzenia względem oczekiwanej wartości.

    Funkcja korzysta z ``LicenseVerifier`` (jeśli dostępny) lub przekazanego
    ``fingerprint_reader``. W przypadku braku możliwości odczytu HWID lub
    rozbieżności z oczekiwaną wartością zgłaszany jest ``HwidValidationError``.
    Zwracany fingerprint może zostać zapisany w logach instalatora.
    """

    reader: Callable[[], str]
    if fingerprint_reader is not None:
        reader = fingerprint_reader
    elif LicenseVerifier is not None:
        verifier = LicenseVerifier()

        def reader() -> str:
            result = verifier.read_fingerprint()
            if not result.ok:
                raise HwidValidationError(
                    f"Nie udało się odczytać fingerprintu: {result.error_code or 'unknown'}"
                )
            assert result.fingerprint is not None
            return result.fingerprint

    else:  # pragma: no cover - ścieżka awaryjna dla środowisk minimalnych
        raise HwidValidationError("Brak dostępu do modułu LicenseVerifier.")

    try:
        fingerprint = reader().strip()
    except HwidValidationError:
        raise
    except Exception as exc:  # pragma: no cover - zabezpieczenie na wszelki wypadek
        raise HwidValidationError(f"Błąd odczytu fingerprintu: {exc}") from exc

    if not fingerprint:
        raise HwidValidationError("Pusty fingerprint urządzenia.")

    if expected_hwid and fingerprint != expected_hwid:
        raise HwidValidationError(
            f"Fingerprint {fingerprint} nie pasuje do oczekiwanego {expected_hwid}."
        )

    return fingerprint


def install_hook_main(expected_hwid_path: str | None = None) -> str:
    """Punkt wejścia dla instalatora zwracający zweryfikowany fingerprint.

    Jeśli wskazano ``expected_hwid_path`` funkcja odczyta plik JSON zawierający
    klucz ``fingerprint`` i użyje go do weryfikacji.
    """

    fingerprint_expected: str | None = None
    if expected_hwid_path:
        path = Path(expected_hwid_path).expanduser()
        if not path.exists():
            raise HwidValidationError(f"Plik fingerprintu {path} nie istnieje.")
        data = path.read_text(encoding="utf-8").strip()
        if data:
            try:
                import json

                payload = json.loads(data)
                fingerprint_expected = str(payload.get("fingerprint") or "").strip() or None
            except Exception as exc:
                raise HwidValidationError(
                    f"Nie udało się odczytać pliku fingerprintu {path}: {exc}"
                ) from exc

    fake_fingerprint = os.environ.get("KBOT_FAKE_FINGERPRINT")
    if fake_fingerprint is not None:
        fake_value = fake_fingerprint.strip()
        if fake_value:
            return validate_hwid(fingerprint_expected, fingerprint_reader=lambda: fake_value)

    return validate_hwid(fingerprint_expected)


def main() -> None:  # pragma: no cover - zachowuje historyczne zachowanie skryptu
    """Zachowuje dotychczasowy entrypoint uruchamiający sondę keyring."""

    run_daily_trend_probe()


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["HwidValidationError", "install_hook_main", "run_daily_trend_probe", "validate_hwid"]
