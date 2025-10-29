# Blokada sprzętowa i obsługa magazynu sekretów

Ten dokument opisuje proces instalacji blokady fingerprintu urządzenia, procedurę
weryfikacji podczas startu runtime oraz rozszerzone możliwości magazynu
sekretów wykorzystywanego przez CLI Stage6.

## Instalacja blokady fingerprintu

1. Uruchom na docelowej maszynie polecenie:

   ```bash
   python -m scripts.install_device_lock --include-factors --pretty
   ```

   Skrypt wykorzystuje `DeviceFingerprintGenerator`, zapisuje blokadę w
   `var/security/device_fingerprint.json` (lub ścieżce wskazanej flagą
   `--output`) oraz opcjonalnie drukuje czynniki użyte do wyliczenia odcisku.

2. Powstały plik powinien zostać zarchiwizowany wraz z pakietem instalacyjnym
   – bootstrap zatrzyma uruchomienie, jeżeli fingerprint urządzenia będzie się
   różnił od zainstalowanej blokady.

## Weryfikacja przy starcie runtime

Moduł `bot_core.runtime.bootstrap` weryfikuje blokadę podczas inicjalizacji
środowiska. W przypadku błędnego fingerprintu lub uszkodzonego pliku bootstrap
emituje alert `security.hardware` i kończy proces. Brak pliku blokady nie
powoduje zatrzymania – środowiska bez wymogu blokady działają jak dotychczas.

## Magazyn sekretów – rotacja i kopie zapasowe

`EncryptedFileSecretStorage` obsługuje:

* rotację hasła (metoda `rotate_passphrase`),
* eksport kopii zapasowej (`export_backup`),
* odtwarzanie z kopii (`recover_from_backup`).

CLI Stage6 (`bot_core/runtime/stage6_preset_cli.py`) zapewnia nowe flagi:

* `--secrets-rotate-passphrase[*]` – rotacja hasła oraz opcjonalna zmiana
  liczby iteracji PBKDF2 (`--secrets-rotate-iterations`),
* `--secrets-backup` / `--secrets-backup-stdout` – zapis/wyświetlenie kopii,
* `--secrets-recover-from` – odtworzenie magazynu z kopii zapasowej.

Wszystkie operacje obsługują tryb `--dry-run`, drukując komunikaty bez zapisu.

## Automatyzacja bundlowania aplikacji

Skrypt `scripts/packaging/build_app_bundles.py` uruchamia:

* PyInstaller (`--pyinstaller-entry`),
* Briefcase (`--briefcase-app`).

Wyniki trafiają domyślnie do `var/dist/pyinstaller` i `var/dist/briefcase`.
Skrypt można zintegrować z pipeline CI, aby zapewnić spójne pakiety z UI i
zależnościami.

Workflow GitHub Actions `deploy/ci/github_actions_desktop_packaging.yml`
uruchamia bundler w macierzy `linux`/`macos`/`windows`, korzystając z
konfiguracji przekazanej jako parametry wywołania (manualne `workflow_dispatch`)
lub zmienne repozytorium `DESKTOP_*`. Pipeline przygotowuje katalogi wynikowe,
uruchamia `python -m scripts.packaging.build_app_bundles` i publikuje artefakty
(`var/dist/pyinstaller`, `var/dist/briefcase`) wraz z manifestem
`desktop_packaging_<platforma>.json`, który zawiera sumy SHA-256 wygenerowanych
plików. Dzięki temu ten sam mechanizm może być wywoływany lokalnie i w CI bez
modyfikacji skryptów.【F:deploy/ci/github_actions_desktop_packaging.yml†L1-L261】

## Testy akceptacyjne

* `tests/test_runtime_bootstrap_hardware.py` sprawdza blokadę fingerprintu.
* `tests/test_security_manager.py` uzupełniono o scenariusze rotacji i
  odzyskiwania magazynu sekretów.
* `tests/scripts/test_build_app_bundles.py` weryfikuje automatyzację
  bundlowania PyInstaller/Briefcase (w tym normalizację platform).

