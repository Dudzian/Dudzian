# Migracja po usunięciu legacy storage

Od wersji 2024.09 runtime usuwa wsparcie dla automatycznego wczytywania zasobów z dawnych ścieżek (`var/state/ui_settings.json`, `~/.kryptolowca/api_credentials.json`, zaszyfrowane pliki `SecurityManager`). Poniższe kroki pozwalają istniejącym instalacjom przeprowadzić migrację przed aktualizacją.

## 1. Magazyn sekretów (`api_credentials.json`)

1. Zainstaluj pomocniczy pakiet narzędziowy `dudzian-migrate` (dostarczany razem z aktualizacją) i uruchom skrypt `python -m dudzian_migrate.secret_store --input ~/.kryptolowca/api_credentials.json --output ~/.dudzian/secret_index.json`.
2. Skrypt zapisze klucze API w natywnym magazynie (`KeyringSecretStorage`) oraz zarchiwizuje stary plik jako `api_credentials.json.legacy`.
3. Po pomyślnym zakończeniu usuń oryginalny plik `~/.kryptolowca/api_credentials.json`. Runtime zweryfikuje brak pliku przy pierwszym uruchomieniu.
4. Nowa lokalizacja danych konfiguracyjnych to katalog `~/.dudzian/`. Możesz nadpisać go przez zmienną `DUDZIAN_HOME`.
5. Środowiska headless zapisują zaszyfrowany magazyn (`secrets.age`) w `~/.dudzian/` (również respektując `DUDZIAN_HOME`).

## 2. Ustawienia UI

1. Zaktualizuj ścieżkę ustawień do `~/.dudzian/ui_settings.json` (domyślnie wykorzystywana lokalizacja respektująca `DUDZIAN_HOME`) lub wskaż własną lokalizację poprzez zmienną środowiskową `BOT_CORE_UI_SETTINGS_PATH`.
2. Jeśli posiadasz plik `var/state/ui_settings.json`, skopiuj go ręcznie do nowej ścieżki. Aplikacja loguje błąd migracji, gdy wykryje plik w starym miejscu.

## 3. Pliki `SecurityManager`

1. Runtime nie obsługuje już flag `--legacy-security-*` w migratorze Stage6. Użyj narzędzia `python -m dudzian_migrate.security_manager --input /ścieżka/do/api_keys.enc --output secrets/api_keys.vault --passphrase-env LEGACY_PASS`, które konwertuje zaszyfrowany plik do formatu Stage6.
2. Zanotuj użyte źródła haseł (parametr, plik, zmienna środowiskowa) w decision logu. Narzędzie zapisuje raport kompatybilny z formatem `migration_summary.json`.
3. Po migracji usuń zaszyfrowane pliki i powiązaną sól z katalogu użytkownika.

## 4. Weryfikacja po migracji

- Uruchom `python -m bot_core.runtime.stage6_preset_cli --core-config config/core.yaml --preset ... --secrets-output secrets/api_keys.vault --summary-json var/audit/stage6/migration_summary.json` (bez flag `--legacy-security-*`).
- Sprawdź, czy katalog `~/.dudzian/` zawiera `secret_index.json`, `ui_settings.json` i brak w nim artefaktów `api_credentials.json`.
- Upewnij się, że `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` została uaktualniona i decision log zawiera wpis o migracji.

W razie problemów migracyjnych dołącz logi narzędzia `dudzian-migrate` i zgłoś ticket w kanale hypercare.
