# Recovery środowisk OEM

Instrukcja krok po kroku dla zespołów wsparcia terenowego. Celem jest szybkie
odtworzenie środowiska `demo`, `paper` lub `live` po awarii sprzętu bądź błędnej
aktualizacji.

## 1. Przygotowanie środowiska

1. Zainstaluj wymagane zależności (`pyinstaller`, `briefcase`) oraz upewnij się,
   że posiadasz klucze OEM (`secrets/oem_manifest.key`, klucz licencyjny).
2. Sklonuj repozytorium i odtwórz katalog `var/security/updates/` wraz z
   manifestami i podpisami.
3. Uruchom orchestratora, aby przygotować strukturę katalogów:
   ```bash
   python scripts/local_orchestrator.py prepare live
   ```
4. Jeżeli bundel będzie odbudowywany z metadanych pobieranych z sieci,
   przygotuj listę zaufanych odcisków certyfikatów TLS (`--metadata-url-cert-fingerprint`),
   oczekiwanych atrybutów tematu certyfikatu (`--metadata-url-cert-subject`, np.
   `commonName=updates.oem.example.com`), atrybutów wystawcy
   (`--metadata-url-cert-issuer`, np. `organizationName=Trusted CA`), wpisów
   subjectAltName (`--metadata-url-cert-san`, np. `DNS=updates.oem.example.com`), wymaganych
   rozszerzeń Extended Key Usage (`--metadata-url-cert-eku`, np. `serverAuth`),
   identyfikatorów certificatePolicies (`--metadata-url-cert-policy`, np. `anyPolicy`) oraz
   numerów seryjnych certyfikatów (`--metadata-url-cert-serial`), aby zapobiec podszywaniu się
   pod źródła aktualizacji.

## 2. Weryfikacja licencji i paczek aktualizacji

1. Zweryfikuj aktualną licencję OEM (profil licencji musi znajdować się na
   liście `allowed_profiles` zapisanej w manifeście):
   ```bash
   python scripts/local_orchestrator.py verify-update live \
       --manifest /media/usb/core-runtime-1.2.0-linux-x86_64/manifest.json \
       --bundle-dir /media/usb/core-runtime-1.2.0-linux-x86_64 \
       --signature /media/usb/core-runtime-1.2.0-linux-x86_64/manifest.sig \
       --signing-key secrets/oem_manifest.key
   ```
2. Jeśli weryfikacja zakończy się błędem, zablokuj wdrożenie i powiadom zespół
   bezpieczeństwa. W logach (`logs/update_verification.jsonl`) zapisz wynik wraz
   z fingerprintem urządzenia.

## 3. Instalacja bundla

1. Rozpakuj zweryfikowany bundla do katalogu środowiska, np.:
   ```bash
   unzip /media/usb/core-runtime-1.2.0-linux-x86_64.zip -d var/orchestrator/live/bundles
   ```
2. Zaktualizuj konfigurację `.env`/`config/`, jeśli wymaga tego sprzęt (np. nowe
   ścieżki do dysków, identyfikatory portów szeregowych).
3. Zaplanuj restart usług – orchestrator przechowuje ostatnią konfigurację w
   `var/orchestrator/state.json`, dzięki czemu można porównać wersje.

## 4. Uruchomienie i testy

1. Uruchom harmonogram w trybie smoke-test:
   ```bash
   python scripts/local_orchestrator.py launch live --run-once
   ```
   *Jeśli chcesz tylko zweryfikować parametry bez wykonania kodu, użyj opcji* `--dry-run`.
2. Przejrzyj logi (`var/orchestrator/live/logs/`) i metryki – brak błędów licencji
   oraz poprawne połączenia z giełdami są warunkiem przejścia dalej.
3. W trybie produkcyjnym uruchom orchestratora bez flagi `--run-once`.

## 5. Eskalacja i dokumentacja

* Zapisz wynik recovery w `audit/recovery/<data>.json` – użyj danych z
  `var/orchestrator/state.json` (ostatnia wersja bundla, timestamp).
* Zaktualizuj decision log o numer licencji i fingerprint urządzenia.
* Powiadom operatorów o zakończeniu procedury wraz z godziną wznowienia usług.

## Aneks: checklista skrócona

1. [ ] Zweryfikowano podpis aktualizacji (`verify-update`).
2. [ ] Potwierdzono ważność licencji OEM.
3. [ ] Odtworzono katalog bundla i konfigurację.
4. [ ] Smoke-test (`launch --run-once`) zakończony powodzeniem.
5. [ ] Aktualizacja dzienników operacyjnych.
