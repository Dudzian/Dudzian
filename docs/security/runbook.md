# Runbook reagowania na alerty bezpieczeństwa

Ten dokument opisuje standard operacyjny reagowania na alerty bezpieczeństwa dla zespołu Dudzian.

## 1. Klasyfikacja alertów

| Poziom | Kryteria | Wymagany czas reakcji |
| --- | --- | --- |
| **Krytyczny** | aktywne wykorzystanie podatności, wyciek danych, kompromitacja kluczy | do 30 minut |
| **Wysoki** | nowe podatności o wysokim CVSS, naruszenia kontroli dostępu | do 4 godzin |
| **Średni** | ostrzeżenia narzędzi SAST/DAST, nieudane próby logowania masowego | do 1 dnia roboczego |
| **Niski** | błędy konfiguracyjne, ostrzeżenia jakościowe | do 3 dni roboczych |

## 2. Kanały zgłoszeń

1. Powiadomienia automatyczne z pipeline'ów bezpieczeństwa (Slack `#sec-alerts`, mailing `security@dudzian.example`).
2. Zgłoszenia manualne od zespołu operacji lub supportu (`Jira/SOC`).
3. Eskalacje partnerów zewnętrznych (CERT, dostawcy chmurowi).

Wszystkie zgłoszenia muszą zostać zarejestrowane w narzędziu śledzenia incydentów wraz z priorytetem i opisem.

## 3. Proces reagowania

1. **Triaging**
   - Dyżurny inżynier bezpieczeństwa ocenia ważność alertu, potwierdza priorytet i przypisuje właściciela.
   - Weryfikuje dostępność dodatkowych danych (logi z `logs/security/`, raporty z CI, metryki).
2. **Zawężenie i potwierdzenie**
   - Reprodukuje problem (jeżeli możliwe) na środowisku izolowanym.
   - Potwierdza zakres wpływu: systemy, konta, dane.
3. **Działania natychmiastowe**
   - Izolacja zagrożonych zasobów (wyłączenie usług, rotacja kluczy, blokada kont).
   - Aktualizacja firewall/WAF oraz polityk dostępu.
4. **Remediacja trwała**
   - Przygotowanie i wdrożenie poprawek (kod, konfiguracja, aktualizacje zależności).
   - Uruchomienie testów regresyjnych i bezpieczeństwa po wdrożeniu.
5. **Komunikacja**
   - Aktualizacje co 30/60 minut dla alertów krytycznych/wysokich.
   - Informacje dla interesariuszy (produkt, operacje, compliance).
6. **Zamknięcie**
   - Potwierdzenie usunięcia zagrożenia i brak kolejnych alertów.
   - Uzupełnienie dokumentacji incydentu i oznaczenie jako zamknięty w narzędziu śledzącym.

## 4. Analiza po-incydentowa (Postmortem)

- Termin: maksymalnie 5 dni roboczych od zamknięcia alertu krytycznego/wysokiego.
- Wymagane elementy:
  - Oś czasu zdarzeń.
  - Analiza przyczyn źródłowych (RCA) z przypisaniem kontrolom zapobiegawczym.
  - Lekcje wyniesione i zadania zapobiegające ponownemu wystąpieniu.
  - Aktualizacje runbooków i automatyzacji.

## 5. Monitoring i metryki

- `MTTA` (średni czas reakcji) oraz `MTTR` (średni czas rozwiązania) liczone miesięcznie.
- Liczba otwartych alertów według poziomu ważności.
- Pokrycie pipeline'ów bezpieczeństwa (SAST, DAST, skanowanie sekretów, IaC).

## 6. Kontakt i dyżury

- Lista dyżurujących inżynierów przechowywana w `docs/operations/oncall.md`.
- Kanał eskalacyjny: `+48 123 456 789` (24/7 SOC) oraz `incident@dudzian.example`.
- W przypadku braku odpowiedzi w ciągu 15 minut – eskalacja do Security Lead oraz CTO.

## 7. Przegląd runbooka

- Runbook jest przeglądany co kwartał przez Security Lead i aktualizowany w repozytorium.
- Zmiany są zatwierdzane w ramach przeglądu technicznego (minimum 2 recenzentów z zespołu bezpieczeństwa).

## 8. Aktywacja licencji offline i odzyskiwanie

1. **Aktywacja**
   - Przekaż klientowi plik licencji (`*.json`) podpisany Ed25519.
   - Na maszynie docelowej uruchom proces weryfikacji (np. `python -m bot_core.security.license_service` poprzez narzędzia instalatora) wskazując plik licencji.
   - Po poprawnej weryfikacji powstaje plik `var/security/license_status.json` zawierający możliwości licencji oraz podpis HMAC (`HMAC-SHA384`) powiązany z lokalnym fingerprintem sprzętowym.
   - Lokalny sekret HMAC jest generowany automatycznie przy pierwszej aktywacji, zapisywany w natywnym keychainie (Keychain/DPAPI/libsecret), a dodatkowo utrwalany w zaszyfrowanym pliku `var/security/license_secret.key` powiązanym z fingerprintem urządzenia.
   - Pliki w formacie legacy (czysty base64) są blokowane błędem „Sekret licencji ma nieobsługiwany format 'legacy'”. Zastosuj narzędzie `python -m dudzian_migrate.license_secret` zgodnie z [docs/migrations/2024-legacy-storage-removal.md](../migrations/2024-legacy-storage-removal.md).

2. **Ochrona przed rollbackiem**
   - Każde kolejne uruchomienie licencji porównuje monotoniczny stan (`sequence`, `issued_at`, `effective_date`).
   - W przypadku próby wgrania starszej licencji system zgłosi `LicenseRollbackDetectedError` i przerwie start runtime'u.
   - Jeśli podpis statusu licencji został zmodyfikowany lub usunięty, zostanie zgłoszony `LicenseStateTamperedError`; wymagane jest ponowne wydanie licencji lub ręczna weryfikacja plików.

3. **Odzyskiwanie po awarii / wymiana sprzętu**
   - Zabezpiecz kopie `var/security/license_status.json`, zaszyfrowanego `var/security/license_secret.key` oraz dziennika `logs/security_admin.log` przed przeniesieniem systemu.
   - W razie wymiany sprzętu wystąp do zespołu licencyjnego o ponownie podpisany pakiet licencji z nowym fingerprintem.
   - Przy błędach podpisu usuń tylko plik statusu (pozostawiając `license_secret.key`), po czym wczytaj najnowszą licencję – system wygeneruje nowy snapshot i podpis.

## 9. Magazyn kluczy API i rotacja

- `KeyringSecretStorage` szyfruje każdy sekret algorytmem AES-GCM (klucz pochodny z lokalnego fingerprintu) i przechowuje go w natywnym keychainie. Indeks pomocniczy znajduje się w `var/security/secret_index.json`.
- W razie podejrzenia kompromitacji uruchom rotację klucza głównego:
  ```bash
  python - <<'PY'
  from bot_core.security.keyring_storage import KeyringSecretStorage

  storage = KeyringSecretStorage(service_name="dudzian.trading")
  storage.rotate_master_key()
  PY
  ```
- Po rotacji zweryfikuj dostępność krytycznych sekretów (Binance, Coinbase, alerty) i wykonaj audyt uprawnień.
- W środowiskach headless korzystaj z `EncryptedFileSecretStorage` i przechowuj passphrase w sejfie zgodnie z polityką SOC.

