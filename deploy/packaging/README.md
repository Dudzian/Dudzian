# Core OEM Packaging

Ten katalog zawiera narzędzia do budowy bundla "Core OEM" obejmującego demona `bot_core`, klienta Qt/QML oraz podpisane pliki konfiguracyjne.

## Struktura bundla
```
core-oem-<wersja>-<platforma>.{tar.gz|zip}/
├── manifest.json (+ manifest.sig)
├── daemon/
├── ui/
├── config/
│   ├── core.yaml (+ core.yaml.sig)
│   ├── .env (+ .env.sig)
│   └── fingerprint.expected.json
└── bootstrap/
    ├── verify_fingerprint.py
    ├── install.sh
    └── install.ps1
```

- Wszystkie pliki posiadają sumy SHA-384 w `manifest.json`.
- Pliki konfiguracyjne (`config/*.yaml`, `.env`) posiadają towarzyszące pliki `.sig` z kanonicznym ładunkiem (`payload`) i podpisem HMAC (`signature`).
- `fingerprint.expected.json` jest dokumentem JSON zawierającym `payload` (fingerprint + `generated_at`) oraz podpis HMAC; weryfikacja wymaga ustawienia zmiennej `OEM_BUNDLE_HMAC_KEY` (base64).

## Budowanie bundla
```
python deploy/packaging/build_core_bundle.py \
  --platform linux \
  --version 1.0.0 \
  --signing-key-path /secure/hmac.key \
  --daemon dist/linux/botd \
  --ui ui/dist/linux \
  --config core.yaml=config/core.yaml \
  --config env=.env.production \
  --resource scripts=scripts/runtime_setup.sh \
  --fingerprint-placeholder PLACEHOLDER-FP \
  --output-dir var/dist
```

Parametry `--daemon` i `--ui` można podać wielokrotnie (pliki lub katalogi). Każdy wpis `--config` ma postać `<nazwa>=<ścieżka>` i trafia do `config/<nazwa>`.

> **Bezpieczeństwo klucza podpisującego:** plik wskazany w `--signing-key-path` musi być zwykłym plikiem (nie symlinkiem), a w środowisku POSIX wymagane są uprawnienia ograniczające dostęp do właściciela (`chmod 600`). W przeciwnym razie budowanie bundla zakończy się błędem.

## Weryfikacja fingerprintu podczas instalacji
1. Instalator powinien dostarczyć klucz HMAC w postaci base64 poprzez `OEM_BUNDLE_HMAC_KEY` (np. z bezpiecznego tokena).
2. Fingerprint urządzenia może być przekazany jako `OEM_FINGERPRINT` lub dostarczony przez moduł `bot_core.security.fingerprint` (po wdrożeniu sprintu 2).
3. Skrypt `bootstrap/verify_fingerprint.py` porównuje fingerprint i podpis z bundla; w przypadku rozbieżności instalacja kończy się błędem.

## Pipeline rozszerzeń (notaryzacja, delty, walidacja fingerprintu)

Skrypt `deploy/packaging/build_core_bundle.py` potrafi uruchomić dodatkowe kroki
po zbudowaniu bundla. Konfiguracja przekazywana jest przez opcję
`--pipeline-config <plik.{json,yaml}>` i umożliwia:

* **notaryzację** archiwum przy pomocy `xcrun notarytool` (w trybie online lub
  `dry-run` z raportem),
* **generowanie aktualizacji delta** na podstawie wcześniej wydanych manifestów
  (`manifest.json` lub całych archiwów `.zip`/`.tar.gz`),
* **walidację hardware fingerprintu** w osadzonym dokumencie
  `config/fingerprint.expected.json` z opcjonalną weryfikacją podpisu HMAC oraz
  porównaniem z lokalnym fingerprintem odczytanym z modułu `bot_core.security.fingerprint`.

Przykładowa konfiguracja (YAML):

```yaml
notarization:
  bundle_id: com.example.core
  apple_id: operator@example.com
  password: env:APPLE_NOTARY_PASSWORD
  team_id: Z123456789
  dry_run: true
  log_path: var/dist/notary/core-macos.json
delta_updates:
  bases:
    - var/dist/core-oem-2024.05-macos
    - var/dist/releases/core-oem-2024.04-macos.tar.gz
  output_dir: var/dist/delta
  compression: zip
fingerprint_validation:
  expected: OEM-FP-PLACEHOLDER
  hmac_key: env:OEM_BUNDLE_HMAC_KEY
  verify_local: true
```

Ścieżki w konfiguracji są rozwiązywane względem katalogu z plikiem konfiguracyjnym.
Wartości `env:` i `file:` pozwalają wczytać sekrety z odpowiednio zmiennych
środowiskowych lub plików.

> **Raportowanie post-processingu:** przekazanie argumentu
> `--pipeline-report var/dist/core-oem-<wersja>-pipeline.json` powoduje zapisanie
> raportu JSON z wynikami weryfikacji fingerprintu, listą paczek delta oraz
> metadanymi notaryzacji. Raport umożliwia późniejsze audytowanie przebiegu
> pipeline'u na środowiskach offline.

## Integracja CI
W katalogu `deploy/ci/github_actions_oem_packaging.yml` znajduje się pipeline GitHub Actions uruchamiający proces bundlowania dla Linux/macOS/Windows na self-hosted runnerach offline. Pipeline publikuje artefakty z katalogu `var/dist`.

## Lista kontrolna "Installer Acceptance"
Pełna lista kontrolna wraz z wymaganymi artefaktami znajduje się w `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md`.

## Powiązane narzędzia licencyjne
- Generowanie fingerprintu oraz podpisanego dokumentu: `bot_core.security.fingerprint`.
- Provisioning licencji offline/QR/USB: `scripts/oem_provision_license.py` zapisujący do `var/licenses/registry.jsonl`.
- Runbook operacyjny: `docs/runbooks/OEM_LICENSE_PROVISIONING.md` (checklista artefaktów i walidacji).

## Pakiety mTLS
- Generowanie pakietu CA/server/client: `scripts/generate_mtls_bundle.py --output-dir secrets/mtls --bundle-name core-oem`.
- Metadane (`*-metadata.json`) dołączamy do decision logu i kontrolujemy za pomocą `bot_core/security/tls_audit.py`.
- Rejestr rotacji (`var/security/tls_rotation.json`) aktualizowany jest automatycznie – wpis wymagany przed startem live.

## Pakiet strategii Stage4
- Budowa paczki strategii i datasetów: `python deploy/packaging/build_strategy_bundle.py --version 2024.06 --signing-key-path secrets/stage4_strategy.key --output-dir var/dist/strategies`.
- Skrypt generuje archiwum `stage4-strategies-<wersja>.zip` wraz z kopiami `stage4-strategies-<wersja>.manifest.{json,sig}` podpisanymi HMAC (domyślnie SHA-384) oraz raportuje listę strategii (`mean_reversion`, `volatility_target`, `cross_exchange_arbitrage`) i datasetów (`data/backtests/normalized/*`).
- Pipeline `deploy/ci/github_actions_stage4_multi_strategy.yml` uruchamia bundler po smoke teście Stage4, deponuje artefakty w `var/stage4_smoke/strategy_bundle` i publikuje wersję w zmiennej `STAGE4_STRATEGY_BUNDLE_VERSION`.
- Publikacja release’u wraz z metadanymi i wpisem decision logu odbywa się przez `python scripts/publish_strategy_bundle.py --version <wersja> --signing-key-path <klucz> --release-dir var/releases/strategies --decision-log-*`, co tworzy katalog `var/releases/strategies/<wersja>` z kopiami archiwum, manifestu, podpisu i `metadata.json`.
