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

## Weryfikacja fingerprintu podczas instalacji
1. Instalator powinien dostarczyć klucz HMAC w postaci base64 poprzez `OEM_BUNDLE_HMAC_KEY` (np. z bezpiecznego tokena).
2. Fingerprint urządzenia może być przekazany jako `OEM_FINGERPRINT` lub dostarczony przez moduł `bot_core.security.fingerprint` (po wdrożeniu sprintu 2).
3. Skrypt `bootstrap/verify_fingerprint.py` porównuje fingerprint i podpis z bundla; w przypadku rozbieżności instalacja kończy się błędem.

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
