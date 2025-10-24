# Instalacja OEM: bundling, orkiestracja i aktualizacje

Dokument opisuje proces przygotowania kompletu artefaktów OEM dla środowisk
`demo`, `paper` i `live`. Skupiamy się na trzech filarach:

1. Budowa binariów Pythona (`bot_core`) oraz klienta Qt przy użyciu PyInstaller/
   Briefcase.
2. Integracja modułu aktualizacji z walidacją licencji OEM (`bot_core.security`).
3. Lokalne zarządzanie środowiskami za pomocą prostego orchestratora CLI.

## Wymagania wstępne

* Python 3.11+ wraz z pakietami `pyinstaller` i `briefcase` w aktywnym
  środowisku (`pip install pyinstaller briefcase`).
* Zbudowana w trybie `Release` aplikacja Qt (`ui/`) – katalog z binarką i
  zasobami QML (`build/ui/Release` lub analogiczny).
* Pakiet `KryptoLowca` zainstalowany w tym samym środowisku (np.
  `pip install -e ./KryptoLowca`), aby moduły `KryptoLowca.*` – w tym
  `KryptoLowca.ai_models` – były dostępne na ścieżce importu.
* Plik konfiguracyjny `config/core.yaml` oraz tajemnice potrzebne do walidacji
  licencji OEM.
* Klucz HMAC (BASE64 lub plaintext) używany do podpisywania manifestów.

## Budowa bundla przy użyciu PyInstaller

1. Przygotuj katalog roboczy, np. `var/orchestrator/demo`:
   ```bash
   python scripts/local_orchestrator.py prepare demo
   ```
2. Zbuduj bundla dla wybranej platformy:
   ```bash
   python scripts/local_orchestrator.py bundle demo \
       --platform linux-x86_64 \
       --version 1.2.0 \
       --qt-dist build/ui/Release \
       --hidden-import bot_core.runtime.bootstrap \
      --include config=config/core.yaml \
      --allowed-profile demo \
      --allowed-profile paper \
      --metadata channel=stable \
      --metadata-file configs/oem_metadata.json \
      --metadata-url https://oem.example.com/bundles/metadata.json \
      --metadata-url-header "Authorization=Bearer <token>" \
      --metadata-url-timeout 5 \
      --metadata-url-max-size 65536 \
      --metadata-url-allowed-host oem.example.com \
      --metadata-url-cert-fingerprint sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
      --metadata-url-cert-subject commonName=updates.oem.example.com \
      --metadata-url-cert-issuer organizationName=Trusted\\ CA \
      --metadata-url-cert-san DNS=updates.oem.example.com \
      --metadata-url-cert-eku serverAuth \
      --metadata-url-cert-serial 0x9F8A4B \
      --metadata-url-ca secrets/oem_ca.pem \
      --metadata-url-client-cert secrets/oem_client.pem \
      --metadata-url-client-key secrets/oem_client.key \
      --metadata-ini configs/oem_metadata.ini \
      --metadata-toml configs/oem_metadata.toml \
      --metadata-yaml configs/oem_metadata.yaml \
      --metadata-dotenv configs/oem_metadata.env \
      --metadata-env-prefix BUNDLE_META_ \
      --signing-key secrets/oem_manifest.key
  ```
  Kluczowe elementy procesu:
   * `deploy/packaging/build_pyinstaller_bundle.py` uruchamia PyInstaller dla
    `python scripts/run_multi_strategy_scheduler.py`, dołączając wymagane moduły.
   * Katalog Qt (`--qt-dist`) kopiowany jest do `ui/` w strukturze bundla.
   * Wygenerowany `manifest.json` zawiera sumy SHA-384 wszystkich artefaktów i
    może zostać podpisany przy użyciu HMAC (`manifest.sig`). Dodatkowo można
    określić listę dozwolonych profili OEM (`--allowed-profile`) oraz pola
    metadanych: pojedyncze wpisy (`--metadata klucz=wartość`, wartość parsowana
    jako JSON), całe słowniki ładowane z plików (`--metadata-file` z obiektem
    JSON) lub pobierane z zaufanych adresów URL (`--metadata-url`, oczekiwany
    jest obiekt JSON; domyślnie wymagane jest HTTPS oraz zgodność hosta z listą
    `--metadata-url-allowed-host`, można też przekazać nagłówki HTTP
    `--metadata-url-header`, nadpisać timeout `--metadata-url-timeout`, ograniczyć
    rozmiar odpowiedzi `--metadata-url-max-size`, wskazać dodatkowy łańcuch CA
    (`--metadata-url-ca` lub katalog `--metadata-url-capath`), odcisk certyfikatu
    serwera (`--metadata-url-cert-fingerprint` w formacie `algorytm:HEX`,
    umożliwia pinning TLS), wymagane atrybuty tematu certyfikatu TLS
    (`--metadata-url-cert-subject`, np. `commonName=updates.oem.example.com`), atrybuty
    wystawcy (`--metadata-url-cert-issuer`, np. `organizationName=Trusted CA`), wpisy
    subjectAltName (`--metadata-url-cert-san`, np. `DNS=updates.oem.example.com`), wymagane
    rozszerzenia Extended Key Usage (`--metadata-url-cert-eku`, np. `serverAuth` lub
    `1.3.6.1.5.5.7.3.2`), identyfikatory certificatePolicies (`--metadata-url-cert-policy`, np.
    `anyPolicy` lub konkretne OID-y), konkretne numery seryjne certyfikatu TLS
    (`--metadata-url-cert-serial`, akceptuje zapis dziesiętny, szesnastkowy lub z dwukropkami)
    oraz certyfikat klienta
    (`--metadata-url-client-cert` oraz opcjonalnie `--metadata-url-client-key`) i – w razie
    potrzeby testów lokalnych – dopuścić HTTP poprzez `--metadata-url-allow-http`), konfiguracje zapisane w INI (`--metadata-ini`, obsługa
    sekcji oraz kluczy z `__` jako separatora kropki), TOML (`--metadata-toml`,
    obsługa tabel i kluczy kropkowanych) oraz YAML (`--metadata-yaml`, wspiera
    klucze kropkowane i zagnieżdżone struktury), wpisy odczytywane z plików
    `.env` (`--metadata-dotenv` – obsługa komentarzy, wpisów `export KEY=...`
    oraz `__` jako separatora klucza) oraz wartości pobierane dynamicznie ze
    zmiennych środowiskowych (`--metadata-env-prefix PREFIX`, gdzie nazwy po
    prefiksie używają `__` jako separatora kropkowanych kluczy). Wszystkie wpisy trafiają
    do sekcji `metadata` manifestu z walidacją duplikatów kluczy. Obsługiwane są
    też klucze kropkowane, np. `--metadata release.branch=main`, które wstawią
    `{"release": {"branch": "main"}}`.
   * Finalne archiwum ZIP umieszczane jest w `var/orchestrator/<env>/dist/`.

3. (Opcjonalnie) Zamiast `--qt-dist` można wskazać projekt Briefcase
   (`--briefcase-project ui/briefcase_app`), który zostanie zbudowany przez
   `briefcase create/build/package` i spakowany do bundla.

## Moduł aktualizacji i licencja OEM

Nowy moduł `bot_core.security.update` umożliwia weryfikację paczek aktualizacji
na podstawie podpisanego manifestu oraz profilu licencji OEM. Kluczowe funkcje:

* `verify_update_bundle()` – oblicza sumy SHA-384 artefaktów, weryfikuje podpis
  HMAC i sprawdza, czy profil licencji (`security.license.profile`) znajduje się
  na liście `allowed_profiles` manifestu.
* Wynik (`UpdateVerificationResult`) udostępnia pola `signature_valid`,
  `license_ok` oraz listę błędów/warningów. W przypadku niepowodzenia weryfikacji
  należy zablokować aktualizację.

### Weryfikacja paczki aktualizacyjnej

```bash
python scripts/local_orchestrator.py verify-update demo \
    --manifest var/dist/core-runtime-1.2.0-linux-x86_64/manifest.json \
    --bundle-dir var/dist/core-runtime-1.2.0-linux-x86_64 \
    --signature var/dist/core-runtime-1.2.0-linux-x86_64/manifest.sig \
    --signing-key secrets/oem_manifest.key
```

* Orchestrator wczytuje `config/core.yaml`, wykonuje `validate_license_from_config`
  i przekazuje wynik do `verify_update_bundle`.
* Weryfikacja kończy się sukcesem tylko wtedy, gdy podpis i profil licencji są
  poprawne, a sumy SHA-384 odpowiadają rzeczywistości.

## Lokalne środowiska demo/paper/live

`python scripts/local_orchestrator.py` konsoliduje typowe operacje operatorskie:

* `prepare <env>` – tworzy katalogi robocze (`logs/`, `bundles/`, `dist/`).
* `bundle <env>` – buduje bundla PyInstaller/Briefcase i zapisuje metadane w
  `var/orchestrator/state.json`. Można przekazywać profile OEM, metadane z plików
  JSON/INI/TOML/YAML/dotenv, CLI oraz zmiennych środowiskowych
  (`--metadata-env-prefix`). Pobieranie z URL domyślnie wymaga HTTPS i zgodności z listą
  hostów (`--metadata-url-allowed-host`), wspiera nagłówki (`--metadata-url-header`),
  limit czasu (`--metadata-url-timeout`), limit rozmiaru (`--metadata-url-max-size`),
  dodatkowe źródła zaufanych certyfikatów (`--metadata-url-ca`, `--metadata-url-capath`),
  pinning certyfikatów TLS (`--metadata-url-cert-fingerprint`), weryfikację atrybutów tematu
  certyfikatu (`--metadata-url-cert-subject`), atrybutów wystawcy
  (`--metadata-url-cert-issuer`), wpisy SAN (`--metadata-url-cert-san`), wymagane EKU
  (`--metadata-url-cert-eku`), polityki certyfikatu (`--metadata-url-cert-policy`),
  konkretne numery seryjne (`--metadata-url-cert-serial`) oraz uwierzytelnianie klienta
  (`--metadata-url-client-cert`, `--metadata-url-client-key`) i
  opcjonalne dopuszczenie HTTP na potrzeby środowisk odciętych (`--metadata-url-allow-http`).
* `launch <env>` – uruchamia harmonogram multi-strategy (`run_multi_strategy_scheduler.py`)
  z odpowiednimi parametrami (`--environment`, `--scheduler`).
* `verify-update <env>` – deleguje proces weryfikacji paczek aktualizacyjnych.
* `status` – pokazuje ostatnie bundlowania/uruchomienia oraz ścieżki konfiguracji.

### Przykładowy scenariusz wdrożeniowy

1. `prepare` dla środowisk `demo` i `paper`.
2. `bundle` dla obu środowisk, podpisując manifesty tym samym kluczem OEM.
3. `verify-update` po stronie laboratorium QA – potwierdzenie zgodności profilu
   licencji z manifestem.
4. `launch --run-once` w trybie `demo` jako smoke-test, a następnie uruchomienie
   ciągłe `paper`.
5. Zapis metryk orchestratora (`var/orchestrator/state.json`) do repozytorium
   operacyjnego (np. `audit/deployments/2024-07-15.json`).

## Integracja z istniejącymi runbookami

* Checklisty OEM nadal obowiązują (`docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md`).
* Bundling PyInstaller można uruchomić w pipeline CI, korzystając z flag
  `--hidden-import` i `--include` dołączających wszystkie wymagane zasoby.
* W sekcji Recovery (poniżej) opisano odtwarzanie środowiska z podpisanych paczek.

## Zabezpieczenia i audyt

* Manifesty i podpisy przechowuj w repozytorium `var/security/updates/` wraz z
  logami weryfikacji (`logs/update_verification.jsonl`).
* Dla każdej paczki odnotuj fingerprint urządzenia (moduł
  `bot_core.security.fingerprint`) oraz identyfikator licencji w rejestrze
  decision log (`docs/runbooks/OEM_LICENSE_PROVISIONING.md`).
