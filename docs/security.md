# Walidacja TPM i licencji OEM

Dokument opisuje przepływ weryfikacji środowiska bezpieczeństwa w aplikacji desktopowej.

## Główne komponenty

1. **`bot_core.security.license.validate_license`** – analizuje pakiet licencyjny oraz opcjonalne artefakty (fingerprint, lista odwołań) i zwraca `LicenseValidationResult`.
2. **`bot_core.security.tpm.validate_attestation`** – weryfikuje dowód TPM/secure enclave i zwraca `TpmValidationResult`.
3. **`bot_core.security.ui_bridge`** – łączy wyniki walidacji i udostępnia je aplikacji Qt w formacie JSON (policzalne pola `warning_details`, `error_details`).
4. **`SecurityAdminController` (Qt)** – prezentuje status w UI, zapisuje zdarzenia audytowe i podnosi alerty.

## Przepływ walidacji licencji

1. Odczytanie pakietu licencyjnego (`license.json`) i opcjonalnego fingerprintu (`fingerprint.json`).
2. Normalizacja pól (profil, wystawca, schemat, wersja, identyfikator, znaczniki czasu).
3. Weryfikacja podpisów HMAC dla licencji, fingerprintu oraz listy odwołań (jeśli skonfigurowana).
4. Analiza listy odwołań – dekodowanie pozycji, sprawdzenie sygnatury, ocena wieku dokumentu.
5. Budowa `LicenseValidationResult`, w którym komunikaty (`warnings`/`errors`) są reprezentowane jako `ValidationMessage` (kod, tekst, wskazówka).

## Przepływ walidacji TPM

1. Załadowanie dowodu TPM oraz opcjonalnego zestawu kluczy publicznych (keyring).
2. Dekodowanie payloadu i podpisu, normalizacja znaczników czasu.
3. Weryfikacja podpisu Ed25519 (jeśli dostępny) lub dodanie ostrzeżeń o brakach.
4. Porównanie fingerprintu z licencją i datą wygaśnięcia dowodu.
5. Zwrócenie `TpmValidationResult` z listą komunikatów `ValidationMessage`.

## Integracja w UI

1. `ui_bridge.verify_tpm_evidence` i `ui_bridge.dump_state` serializują komunikaty jako:
   * `warnings` / `errors` – lista krótkich tekstów do szybkiego wyświetlenia,
   * `warning_details` / `error_details` – lista słowników `{code, message, hint}` do szczegółów i logów.
2. `SecurityAdminController::verifyTpmBinding` przekształca szczegóły na gotowe komunikaty (`warningMessages`, `errorMessages`) i uzupełnia audyt.
3. Zdarzenia są logowane oraz prezentowane jako alerty w panelu bezpieczeństwa.

## Kody komunikatów

| Kod                                   | Opis                                                                 | Wskazówka                                                                 |
|---------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------|
| `license.field.profile_missing`       | Brak pola `profile` w licencji.                                      | Użyj kompletnego pakietu licencyjnego.                                   |
| `license.revocation.list_stale`       | Lista odwołań jest starsza niż dopuszczony limit.                    | Odśwież dokument `revocations.json`.                                     |
| `license.time.expired`                | Licencja wygasła.                                                    | Odnów licencję.                                                          |
| `tpm.signature.missing`               | Dowód TPM nie zawiera podpisu.                                       | Wygeneruj atest z podpisem Ed25519.                                      |
| `tpm.fingerprint.mismatch`            | Dowód TPM nie odpowiada bieżącemu fingerprintowi licencji.           | Zweryfikuj przypisanie licencji do urządzenia.                           |
| `tpm.attestation.expired`             | Dowód TPM stracił ważność.                                           | Uruchom ponownie procedurę atestacji TPM.                                |

> **Uwaga:** Pełna lista kodów dostępna jest w modułach `bot_core.security.license` oraz `bot_core.security.tpm`.

## Rejestrowanie audytu

Każde uruchomienie walidacji zapisuje w logu:

- znacznik czasu,
- kategorię (`license`, `tpm`, `integrity`),
- wynik (`status`, `valid`, `errorMessages`, `warningMessages`),
- dodatkowe szczegóły (ścieżki plików, identyfikatory kluczy, fingerprinty).

Dzięki temu operator ma pełną historię prób walidacji oraz gotowe wskazówki naprawcze wynikające z `ValidationMessage.hint`.
