# Stage5 – walidacja raportów zgodności

## Cel
Zweryfikować kompletność i integralność raportów zgodności Stage5, w tym
artefakty OEM, rotację kluczy, logi szkoleń oraz raport TCO. Proces kończy się
podpisanym raportem walidacji.

## Przygotowanie
1. Zbierz raporty JSON z katalogu `var/audit/stage5/compliance` lub innej
   lokalizacji wskazanej w hypercare.
2. Zapewnij dostęp do klucza HMAC użytego przy generowaniu raportów
   (zmienna środowiskowa lub plik Base64).
3. Ustal katalog wyjściowy na raport walidacyjny.

## Kroki
1. Uruchom skrypt walidacyjny:
   ```bash
   python -m scripts.validate_compliance_reports \
     var/audit/stage5/compliance \
     --signing-key-env STAGE5_COMPLIANCE_HMAC \
     --require-signature
   ```
2. Przejrzyj wynik na standardowym wyjściu – brak wpisów w `errors` i `failures`
   oznacza pozytywną walidację.
3. Jeżeli raporty wymagają dodatkowych kontroli (np. specyficznych dla OEM),
   dodaj odpowiednie `--expected-control` zgodnie z polityką.
4. Zapisz wynik do pliku (opcjonalnie):
   ```bash
   python -m scripts.validate_compliance_reports \
     var/audit/stage5/compliance/*.json \
     --signing-key-file secrets/stage5_hmac.b64 \
     --output-json var/audit/stage5/compliance/validation.json
   ```

## Artefakty / Akceptacja
- Raport walidacyjny JSON (np. `var/audit/stage5/compliance/validation.json`).
- Lista raportów zgodności wraz z informacją o podpisach HMAC.
- Brak kontroli w statusie `fail` oraz brak błędów strukturalnych.
- Notatka w decision logu Stage5 z sygnaturą walidacji i datą wykonania.
