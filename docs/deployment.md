# Deployment Hardening & Offline Releases

## Twarde zabezpieczenia licencyjne
- `core/licensing/` udostępnia kontroler licencyjny, który pobiera fingerprint sprzętowy (CPU, płyta główna, TPM), porównuje go z licencją i magazynem zaszyfrowanym kluczem z hardware’u.
- Magazyn licencji (`bot_core.security.license_store`) używa AES-GCM z kluczem pochodzącym z fingerprintu. W razie zmiany sprzętu raportowany jest stan blokady.
- `deploy/packaging/build_pyinstaller_bundle.py` przyjmuje nowe flagi (`--license-json`, `--license-fingerprint`, `--license-hmac-key`) pozwalające dołączyć zaszyfrowany magazyn licencji i plik integralności podpisany HMAC.

## Offline’owe paczki aktualizacji
- `scripts/package_offline_release.py` buduje paczkę offline i zapisuje manifest.
- `scripts/offline_update.py` udostępnia polecenia:
  - `prepare-release` – pakuje modele/strategie z `data/models` oraz `data/strategies` do archiwum `.tar.gz`, buduje manifest i (opcjonalnie) podpisuje go HMAC.
  - `verify-release` – weryfikuje sumy kontrolne i podpis manifestu.
  - `install-release` – instaluje paczkę do lokalnych katalogów, tworząc kopie zapasowe w `var/offline_updates/backups`.
- Moduł `core.update.installer` obsługuje pakowanie, weryfikację i instalację offline.

## Szybki start
```bash
python deploy/packaging/build_pyinstaller_bundle.py \
  --platform linux --version 1.2.3 \
  --entrypoint scripts/run_multi_strategy_scheduler.py \
  --license-json licensing/provisioning/license.json \
  --license-fingerprint HWID-123456 \
  --license-hmac-key bundler=BASE64SECRET

python scripts/offline_update.py prepare-release \
  --version 2024.07 \
  --output var/releases/offline-2024.07.tar.gz \
  --signing-key release=BASE64SECRET

python scripts/offline_update.py verify-release \
  --archive var/releases/offline-2024.07.tar.gz \
  --signing-key release=BASE64SECRET

python scripts/offline_update.py install-release \
  --archive var/releases/offline-2024.07.tar.gz \
  --models-dir data/models --strategies-dir data/strategies
```
