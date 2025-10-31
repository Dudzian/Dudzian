# Podręcznik operacyjny

Dokument opisuje codzienne czynności operacyjne dla zespołów utrzymaniowych. Zawiera
procedury zbierania logów, walidacji pakietów aktualizacji, diagnozowania środowiska oraz
odtwarzania usług po awarii. Wszystkie polecenia należy wykonywać w głównym katalogu
instalacji bota.

## 1. Zbiorczy eksport logów i alertów

1. Upewnij się, że zmienne `BOT_CORE_SECURITY_AUDIT_KEY` oraz `BOT_CORE_SECURITY_AUDIT_KEY_ID`
   zawierają klucz HMAC (np. `hex:<klucz>`).
2. Uruchom wiersz poleceń:

   ```bash
   python -m bot_core.security.ui_bridge export-security-bundle \
       --audit-path logs/security_admin.log \
       --alerts-path logs/security_alerts.log \
       --include-log logs/trading_controller.log \
       --output-dir exports/security
   ```

3. Wygenerowany plik JSON zawiera log audytowy, alerty oraz ogon najważniejszych logów
   diagnostycznych. Podpis HMAC umieszczony jest w sekcji `signature`.
4. W aplikacji desktopowej ta sama operacja dostępna jest w panelu bezpieczeństwa – przycisk
   „Eksportuj logi bezpieczeństwa” zapisuje pakiet w katalogu `exports/security`.

## 2. Walidacja pakietu aktualizacji offline

1. Rozpakuj paczkę do katalogu tymczasowego, np. `tmp/update_bundle`.
2. Zweryfikuj manifest i integralność artefaktów:

   ```bash
   python scripts/check_update_integrity.py tmp/update_bundle \
       --hmac-key hex:<klucz_lub_ścieżka_do_pliku>
   ```

3. Skrypt wypisze wynik w formacie JSON. Wartość `status: ok` oznacza, że podpis i sumy
   kontrolne są zgodne. W przypadku błędów otrzymasz szczegółową listę plików do naprawy.

## 3. Kontrola dowodu TPM/licencji

1. Przygotuj plik `evidence.json` wyeksportowany z modułu TPM.
2. (Opcjonalnie) wskaż aktywną licencję OEM w `var/licenses/active/license.json`.
3. Uruchom:

   ```bash
   python scripts/check_tpm_status.py evidence.json \
       --license-path var/licenses/active/license.json \
       --keyring secrets/oem_tpm_keys.json
   ```

4. Status `ok` potwierdza ważny dowód. Status `invalid` lub `missing` wymaga eskalacji do
   zespołu bezpieczeństwa.

## 4. Diagnostyka runtime'u tradingowego

1. Zapewnij dostęp do pliku `config/exchanges.json` z poprawnymi danymi API.
2. Uruchom podstawowy health-check:

   ```bash
   python scripts/check_runtime_health.py \
       --exchange binance_spot \
       --credentials-file config/exchanges.json \
       --output text
   ```

3. Parametr `--list-checks` wypisze dostępne testy bez ich uruchamiania. Dodatkowe parametry
   (np. `--mode`, `--environment-config`) są przekazywane do oryginalnego CLI `bot_core`.
4. Zwrócony kod wyjścia 0 oznacza, że health-check zakończył się powodzeniem. Kod różny od 0
   wymaga analizy – szczegóły znajdują się w logach oraz na standardowym wyjściu polecenia.

## 5. Procedura aktualizacji i rollbacku

1. Wykonaj eksport bezpieczeństwa (sekcja 1) przed każdą aktualizacją.
2. Zweryfikuj paczkę skryptem `check_update_integrity.py`.
3. W UI uruchom menedżer aktualizacji offline i wskaż zweryfikowany pakiet.
4. Po zakończeniu aktualizacji sprawdź zdrowie runtime'u (sekcja 4).
5. W razie niepowodzenia uruchom rollback z poziomu UI. Skrypt testowy
   `pytest tests/update/test_offline_update_flow.py::test_offline_update_rollback` przedstawia
   kompletny scenariusz odtwarzania.

## 6. Odzyskiwanie po awarii

1. Ustal ostatnie dostępne pakiety logów bezpieczeństwa i aktualizacji.
2. Przywróć środowisko z kopii zapasowej (obrazy dysków lub paczki release).
3. Zastosuj ostatnią zatwierdzoną aktualizację po weryfikacji integralności.
4. Zweryfikuj dowód TPM oraz licencję (sekcje 2–3).
5. Wykonaj health-check runtime'u i raportuj wyniki do zespołu SRE.

## 7. Checklisty operacyjne

- [ ] HMAC ustawiony w środowisku (`BOT_CORE_SECURITY_AUDIT_KEY`).
- [ ] Pakiet aktualizacji zweryfikowany (`check_update_integrity.py`).
- [ ] Dowód TPM potwierdzony (`check_tpm_status.py`).
- [ ] Health-check giełd przeszedł pomyślnie (`check_runtime_health.py`).
- [ ] Pakiet logów bezpieczeństwa zarchiwizowany w `exports/security`.
- [ ] Dokumentacja aktualizacji uzupełniona w systemie ticketowym.

Przestrzeganie powyższych kroków zapewnia zgodność z procesami audytowymi i minimalizuje
ryzyko utraty danych lub naruszenia licencji OEM.
