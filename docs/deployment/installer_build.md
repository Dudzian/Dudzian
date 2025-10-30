# Budowa instalatora desktopowego

Dokument opisuje wykorzystanie nowych profili PyInstaller/Briefcase oraz pipeline
CI do przygotowania bundla desktopowego na systemach Linux, macOS i Windows.

## Profile bundlujące

W katalogu `deploy/packaging/profiles/` znajdują się pliki TOML:

* `linux.toml`
* `macos.toml`
* `windows.toml`

Każdy profil definiuje:

* entrypoint PyInstaller (`pyinstaller.entrypoint`), nazwę binarki i ukryte importy,
* katalogi robocze/wynikowe PyInstaller oraz Briefcase,
* zasoby dołączane do bundla (`bundle.include`),
* lokalizację metadanych (`bundle.metadata_path`).

Profile można nadpisać, ustawiając zmienną środowiskową `PROFILE` lub przekazując
inną ścieżkę do skryptów budujących.

## Skrypty budujące

W repozytorium dostępne są skrypty:

```bash
./scripts/build_installer_linux.sh --version 1.0.0
./scripts/build_installer_macos.sh --version 1.0.0
pwsh -File scripts/build_installer_windows.ps1 -Version 1.0.0
```

Skrypty opakowują `scripts/build_installer_from_profile.py`, który:

1. Odczytuje profil (PyInstaller/Briefcase/bundle).
2. Uruchamia PyInstaller oraz (opcjonalnie) Briefcase.
3. Woła `deploy/packaging/build_pyinstaller_bundle.py`, aby zbudować bundla.
4. Zapisuje metadane w `var/dist/installers/<platforma>/installer_metadata.json`.

Polecenia akceptują dodatkowe argumenty przekazywane bezpośrednio do
`build_installer_from_profile.py` (np. `--metadata-out` lub `--skip-briefcase`).

## Pipeline CI

Workflow [`deploy/ci/github_actions_cross_installer.yml`](../../deploy/ci/github_actions_cross_installer.yml)
uruchamia proces budowy na macierzy runnerów (Linux/macOS/Windows). Kroki obejmują:

1. Instalację zależności z `deploy/packaging/requirements-desktop.txt`.
2. Uruchomienie odpowiedniego skryptu (`build_installer_{platform}.sh` lub `.ps1`).
3. Odczyt metadanych (`installer_metadata.json`).
4. Test dymny polegający na uruchomieniu bundla z parametrem `--help`.
5. Publikację artefaktów (zip + metadane) jako artefaktów Actions.

Domyślna wersja bundla to `<run_number>.<run_attempt>`, ale można ją nadpisać
parametrem `version` w `workflow_dispatch` lub `workflow_call`.

## Troubleshooting

* **Brak natywnego backendu keyring** – upewnij się, że system posiada
  odpowiedni backend (`SecretService`, `macOS Keychain`, `Windows Credential Manager`).
* **PyInstaller zgłasza brakujące moduły Qt** – zainstaluj zależności z
  `deploy/packaging/requirements-desktop.txt` i wskaż katalog UI w profilu
  (`bundle.qt_dist`).
* **Pipeline nie znajduje `installer_metadata.json`** – sprawdź, czy profil
  wskazuje katalog zapisywalny oraz czy runner posiada uprawnienia do `var/dist`.

Szczegółowe informacje o starej ścieżce manualnej znajdują się w
[`docs/deployment/desktop_installer.md`](desktop_installer.md).
