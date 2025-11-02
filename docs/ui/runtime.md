# Runtime UI

## Blokada pojedynczej instancji

Bot Trading Shell wymusza pojedynczą aktywną instancję procesu. Podczas startu
aplikacja tworzy blokadę `QLockFile` w lokalizacji
`var/runtime/bot_trading_shell.lock` (lub w ścieżce wskazanej przez
`BOT_CORE_UI_LOCK_FILE`). Jeżeli katalog `var/runtime` nie istnieje, jest
automatycznie tworzony (można go nadpisać zmienną `BOT_CORE_UI_RUNTIME_DIR`).

Gdy zostanie wykryta działająca instancja:

- użytkownik otrzymuje komunikat w oknie dialogowym oraz w standardowym
  wyjściu błędów z informacją o PID/hoscie aktywnego procesu,
- uruchomienie kończy się kodem wyjścia różnym od zera,
- incydent jest raportowany do backendu licencjonowania poprzez wywołanie
  `python3 -m bot_core.security.fingerprint report-single-instance`.
  Zdarzenie `ui_single_instance_conflict` trafia do `logs/security_admin.log`
  razem z PID-em, hostem i kanoniczną (absolutną) ścieżką pliku blokady.

Jeżeli blokada nie może zostać założona z innych powodów (np. brak uprawnień
do katalogu), aplikacja kończy działanie z komunikatem o błędzie, ale nie
raportuje incydentu do backendu.

Interpreter Pythona używany do raportowania może zostać zmieniony przez
argument CLI `--security-python` lub zmienną środowiskową
`BOT_CORE_UI_PYTHON`. Gdy fingerprint nie zostanie przekazany z poziomu UI,
backend samodzielnie pobiera aktualny odcisk urządzenia, a w przypadku
problemu loguje ostrzeżenie i kontynuuje raportowanie.

## Integracja z testami

Moduł testowy `tests/ui/test_single_instance.py` weryfikuje, że funkcja
`bot_core.security.fingerprint.report_single_instance_event` zapisuje
odpowiednie wpisy auditowe. Test korzysta z trybu `QT_QPA_PLATFORM=offscreen`
pozostając kompatybilnym z uruchomieniami CI bez środowiska graficznego.
