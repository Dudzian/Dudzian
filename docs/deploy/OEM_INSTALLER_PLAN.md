# OEM Installer Fingerprint Assignment – Plan działania

Dokument opisuje rozszerzony proces przygotowania instalatorów OEM przypisywanych do
konkretnych fingerprintów urządzeń. Plan zakłada wykorzystanie istniejących narzędzi
licencyjnych oraz nowo dodanych komponentów marketplace i magazynu przydziałów.

## 1. Kolekcja fingerprintów docelowych

1. Zebrane fingerprinty (JSON lub podpisane pakiety HMAC) umieścić w repozytorium
   `var/licenses/registry.jsonl` wraz z metadanymi klienta.
2. Zweryfikować fingerprinty lokalnie poleceniem `python scripts/ui_marketplace_bridge.py list`
   z parametrem `--fingerprint`, aby upewnić się, że urządzenia posiadają aktywne licencje.
3. Dla nowych urządzeń wygenerować wpis w `seat_policy.assignments` odpowiedniej licencji,
   aby uniknąć ostrzeżeń o brakujących przydziałach.

## 2. Budowa paczek instalatora

1. Uruchomić pipeline pakowania (`python scripts/build_desktop_installer.py`) z parametrami:
   * `--fingerprint <DEVICE_ID>` – identyfikator urządzenia;
   * `--marketplace-assign <preset_id>:<portfolio_id>` – automatyczne przydzielenie presetów
     do portfeli korzystając z magazynu `.meta/assignments.json`.
2. W trakcie builda skrypt wykorzysta `PresetAssignmentStore`, aby przygotować konfigurację
   portfeli oraz listę wymaganych presetów marketplace.
3. Zastosować nowy moduł marketplace do wygenerowania planu instalacji (`MarketplaceIndex`
   + `plan_installation`) – logi planu dołączamy do artefaktów QA.

## 3. Walidacja podpisanych presetów

1. W środowisku CI lub na stacji QA uruchomić testy E2E `pytest tests/e2e/test_marketplace_signed_presets.py`
   w celu potwierdzenia poprawności podpisów oraz poprawnej synchronizacji katalogu strategii.
2. Przygotowane pliki `.json` presetów powinny zawierać metadane aktualizacji (`updates`) oraz
   zależności (`dependencies`), dzięki czemu UI zaprezentuje operatorowi wymagane kroki.

## 4. Generowanie instalatora z przypisaniem do fingerprintu

1. Po zbudowaniu artefaktu, w katalogu `dist/installers` powstaje paczka z nazwą
   `oem-installer-<fingerprint>-<timestamp>.zip`.
2. Do paczki dołączamy:
   * licencję OEM z sekcją `seat_policy` zawierającą bieżący fingerprint;
   * plik `assignments.json` generowany przez `PresetAssignmentStore`;
   * raport planu aktualizacji marketplace (`plan.json`).
3. Instalator zawiera prekonfigurowane ścieżki presetów i portfeli, dzięki czemu po uruchomieniu
   klient otrzymuje środowisko przypisane do własnego urządzenia.

## 5. Kontrola jakości i archiwizacja

1. QA uruchamia `ui_marketplace_bridge.py list` z parametrem `--presets-dir` wskazującym paczkę,
   aby potwierdzić, że przydziały portfeli zostały zapisane.
2. Logi z procesu (w tym telemetria licencji `_emit_license_telemetry`) trafiają do
   `logs/security/license_telemetry.jsonl` i są dołączane do zestawu audytowego OEM.
3. Artefakty (instalator, logi, plan marketplace) archiwizujemy w `var/oem/installers/<fingerprint>/`.

Plan umożliwia pełną automatyzację procesu przypisywania instalatorów do fingerprintów oraz
zapewnia dodatkowe dane diagnostyczne dla zespołów QA i compliance.
