# Instalator desktopowy KryptoŁowcy

Ten dokument opisuje proces budowania i walidacji pakietu instalatora desktopowego
z wykorzystaniem narzędzia `deploy/packaging/desktop_installer.py`.

## Budowa paczki

```bash
python deploy/packaging/desktop_installer.py \
  --version 1.2.3 \
  --platform linux \
  --profiles-dir deploy/packaging/profiles \
  --hook-source probe_keyring.py
```

* `--version` – numer wersji umieszczany w nazwie archiwum oraz metadanych.
* `--platform` – docelowa platforma (`linux`, `windows`, `macos` lub `all`).
* `--profiles-dir` – katalog z profilami TOML opisującymi layout instalatora.
* `--hook-source` – ścieżka do skryptu `probe_keyring.py`, który zostanie
  skopiowany do pakietu i wykorzystany do walidacji HWID.

Narzędzie korzysta z sekcji `[bundle]` profilu, kopiując wskazane katalogi do
tymczasowego stagingu, dodając dystrybucję Qt (jeśli zdefiniowana) oraz generując
plik `manifest.json` z sumami SHA-256 wszystkich artefaktów. Wynikowa paczka ZIP
jest zapisywana w katalogu `output_dir` z profilu, a skrócony manifest również w
`metadata_path`.

## Walidacja HWID podczas instalacji

W każdym archiwum znajdują się pliki:

* `hooks/probe_keyring.py` – oryginalny skrypt, rozszerzony o funkcję
  `install_hook_main`, pozwalającą odczytać i zweryfikować fingerprint sprzętu.
* `hooks/validate_hwid.py` – automatycznie wygenerowany wrapper, który:
  1. Odczytuje oczekiwany fingerprint z pliku `config/fingerprint.expected.json`
     (lub ścieżki wskazanej zmienną `KBOT_EXPECTED_HWID_FILE`).
  2. Wywołuje `probe_keyring.install_hook_main`, zgłaszając błąd przy braku
     fingerprintu lub rozbieżności.
  3. Opcjonalnie zapisuje log walidacji do `KBOT_INSTALL_LOG`.

Tym samym instalator może uruchomić `python hooks/validate_hwid.py` jako krok
preinstalacyjny – niepowodzenie kończy instalację błędem, sukces umożliwia
kontynuację.

## Integracja z lokalnym CI

Pipeline `deploy/local_ci.yml` posiada zadanie `desktop-installer`, które buduje
pakiet dla profilu `linux` i publikuje artefakty w `var/dist/installers`. Dzięki
temu każda kompilacja lokalna potwierdza kompletność konfiguracji profili oraz
integrację hooków HWID.

## Testy automatyczne

`tests/packaging/test_installer.py` buduje przykładową paczkę, po czym weryfikuje
obecność hooków HWID, manifestu oraz spójność metadanych. Test korzysta z
próbkowych zasobów w `deploy/packaging/samples`, co umożliwia szybkie, offline’owe
sprawdzenie konfiguracji.
