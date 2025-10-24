# OEM Runtime – instalacja, aktywacja i troubleshooting

Dokument opisuje procedury operatorskie wymagane przy wdrożeniu bundla OEM.
Skupiamy się na trzech obszarach: instalacji, aktywacji licencji oraz
troubleshootingu fingerprintu i logów bezpieczeństwa.

## Instalacja bundla

1. Rozpakuj archiwum zbudowane przez `deploy/packaging/build_core_bundle.py` i
   wejdź do katalogu `bootstrap/`.
2. Eksportuj zmienną `OEM_BUNDLE_HMAC_KEY` zawierającą klucz HMAC (Base64).
3. Jeśli fingerprint urządzenia nie jest jeszcze zapisany, ustaw `OEM_FINGERPRINT`
   na wartość referencyjną lub upewnij się, że moduł
   `bot_core.security.fingerprint` jest dostępny w środowisku.
4. Uruchom skrypt instalacyjny (`install.sh` na Linux/macOS lub `install.ps1`
   na Windows). Skrypt wywołuje `verify_fingerprint.py`, który:

   * wczytuje podpisany plik `config/fingerprint.expected.json`,
   * porównuje fingerprint z runtime (moduł `bot_core.security.fingerprint`),
   * zapisuje wynik w logu audytowym `logs/security_admin.log` (zdarzenie
     `installer_run`).

   W przypadku niezgodności fingerprintu instalacja zostaje przerwana, a log
   otrzymuje wpis ze statusem `failed` lub `error`.

5. Po poprawnej weryfikacji przenieś katalogi `daemon/` i `ui/` w docelowe
   lokalizacje oraz zaktualizuj konfigurację (`config/`, `secrets/`).

## Aktywacja licencji

1. Skopiuj pakiet licencyjny (`*.lic`) do `var/licenses/active/`.
2. Uruchom aplikację Qt i otwórz panel aktywacji. Weryfikacja odbywa się przez
   `bot_core.security.license_service.LicenseService`, który sprawdza podpis
   Ed25519, fingerprint urządzenia oraz zapisuje migawkę w
   `var/security/license_status.json`.
3. Każda próba aktywacji generuje wpis JSONL w `logs/security_admin.log`
   (`event = "license_snapshot"`). Dodatkowo log przechowuje identyfikator
   licencji, informacje o edycji, flagę maintenance i sumę SHA-256 payloadu.
4. Po aktywacji potwierdź w UI, że licencja ma status `Aktywna` oraz sprawdź,
   czy moduły wymagające licencji (np. `reporting_pro`) zostały udostępnione.

## Troubleshooting

* **Fingerprint mismatch** – sprawdź wpisy `installer_run` w
  `logs/security_admin.log`, aby zweryfikować oczekiwany fingerprint oraz
  identyfikator klucza (`key_id`). W razie potrzeby ponownie uruchom
  `verify_fingerprint.py --expected <ścieżka>` z ustawioną zmienną
  `OEM_FINGERPRINT`.
* **Brak wpisów audytowych** – upewnij się, że katalog `logs/` jest zapisywalny
  oraz że moduł `bot_core.security.fingerprint.append_fingerprint_audit` jest
  dostępny. Skrypt instalacyjny wypisze ostrzeżenie w przypadku problemów z
  logowaniem.
* **Niepoprawna licencja** – weryfikację szczegółową umożliwia narzędzie
  `python -m bot_core.security.ui_bridge --license-status`, które korzysta z
  tych samych ścieżek (`var/security/license_status.json`).
* **Konflikty presetów** – nowy kreator `StrategyPresetWizard` eksportuje
  podpisane presety (`preset.json` + sekcja `signature`) i automatycznie
  dokleja do każdej pozycji licencję, klasy ryzyka oraz wymagane dane na bazie
  katalogu strategii. Przed wdrożeniem sprawdź, czy podpis HMAC (`key_id`)
  odpowiada oczekiwanemu kluczowi.

### Szybka lista kontrolna operatora

1. `export OEM_BUNDLE_HMAC_KEY=...`
2. `bootstrap/install.sh` (log audytu `installer_run`).
3. Skopiuj licencję i potwierdź wpis `license_snapshot` w `logs/security_admin.log`.
4. Sprawdź w UI nowe dashboardy raportów (krzywa kapitału, heatmapa aktywów).
5. W razie problemów użyj wpisów audytowych i kreatora presetów do walidacji.
