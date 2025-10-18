# Budowa pakietu instalacyjnego OEM

Nowy panel administracyjny oraz kreator pierwszego uruchomienia wymagają gotowej paczki
zawierającej aplikację Qt, zasoby QML oraz mostki Pythona wykorzystywane przez UI.
Poniższa procedura wykorzystuje `PyInstaller` do zbudowania podpisanego updatera i
spakowania artefaktów w katalogu `var/dist/desktop`.

## Wymagania

* Zbudowana wersja `Release` aplikacji (`bot_trading_shell`) z katalogu `ui/`.
* Zainstalowany `pyinstaller` (np. `pip install pyinstaller`).
* Klucz HMAC używany do podpisania binarki updatera.

## Kroki

1. Zbuduj aplikację Qt w trybie `Release`, np.:
   ```bash
   cmake -S ui -B build/ui -DCMAKE_BUILD_TYPE=Release
   cmake --build build/ui --target bot_trading_shell -j
   ```
2. Uruchom skrypt budujący pakiet:
   ```bash
   python scripts/build_desktop_installer.py \
       --build-dir build/ui \
       --platform linux \
       --signing-key $(cat secrets/updater_hmac.key)
   ```
   Dostępne opcje:
   * `--reports` – katalog z raportami, które mają zostać dołączone do paczki (domyślnie `var/reports`).
   * `--updater-script` – alternatywny entrypoint updatera (domyślnie `scripts/desktop_updater.py`).
   * `--output` – katalog wynikowy (domyślnie `var/dist/desktop`).

3. Skrypt tworzy strukturę `bot_trading_shell/` zawierającą:
   * binarkę Qt,
   * katalogi `qml/`, `config/` i `bot_core/`,
   * zbudowanego przez PyInstaller updatera (`desktop_updater`),
   * plik podpisu `desktop_updater.sig`,
   * manifest `INSTALL_MANIFEST.json` z metadanymi pakietu.

4. Finalny ZIP (`bot_trading_shell_bundle.zip`) można dystrybuować operatorom.

## Aktualizacje w terenie

`desktop_updater.py` weryfikuje podpis HMAC (`--key`) i wypakowuje archiwum aktualizacji
w docelowym katalogu. Manifest aktualizacji jest kopiowany do `update_manifest.json`,
co umożliwia szybkie potwierdzenie wersji u klienta.

## Dokumentacja użytkownika

Operatorzy znajdują w panelu administracyjnym zakładki:

* **Strategia** – edycja instrumentu i parametrów performance guard,
* **Monitorowanie** – podgląd raportów z `bot_core.reporting` i eksport plików,
* **Licencje i profile** – zarządzanie licencją OEM oraz profilami RBAC.

Przy pierwszym uruchomieniu kreator prowadzi użytkownika przez zapis fingerprintu,
walidację licencji i blokuje uruchomienie na innym sprzęcie, dopóki licencja nie
zostanie aktywowana.

## Parametry uruchomieniowe UI

`bot_trading_shell` udostępnia dodatkowe opcje CLI dla mostka raportów:

* `--reports-directory` – katalog bazowy z raportami (domyślnie `var/reports`).
* `--reporting-python` – ścieżka do interpretera Pythona uruchamiającego bridge.

Te same ustawienia można nadpisać zmiennymi środowiskowymi
`BOT_CORE_UI_REPORTS_DIR` oraz `BOT_CORE_UI_REPORTS_PYTHON`, co ułatwia
konfigurację na stanowiskach operatorskich bez zmiany skryptów startowych.
