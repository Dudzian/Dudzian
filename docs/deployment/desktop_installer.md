# Budowa pakietu instalacyjnego OEM

Nowy panel administracyjny oraz kreator pierwszego uruchomienia wymagają gotowej paczki
zawierającej aplikację Qt, zasoby QML oraz mostki Pythona wykorzystywane przez UI.
Poniższa procedura wykorzystuje `PyInstaller` do zbudowania podpisanego updatera i
spakowania artefaktów w katalogu `var/dist/desktop`.

## Wymagania

* Zbudowana wersja `Release` aplikacji (`bot_trading_shell`) z katalogu `ui/`.
* Środowisko Pythona zainstalowane wg `deploy/packaging/requirements-desktop.txt`
  (obejmuje `numpy`, `pandas`, `joblib`, `pyinstaller`, `briefcase`).
* Zablokowany zestaw zależności z `deploy/packaging/requirements-desktop.lock`
  dla stanowisk bez dostępu do internetu.
* Klucz HMAC używany do podpisania binarki updatera.

Przed uruchomieniem procesu budowania wykonaj:

```bash
python -m venv .venv-desktop
source .venv-desktop/bin/activate
pip install --upgrade pip
pip install -r deploy/packaging/requirements-desktop.txt
# aby odwzorować dokładny zestaw wersji, zaimportuj lockfile:
# pip install -r deploy/packaging/requirements-desktop.lock
# w środowiskach offline wskaż katalog z lokalnymi kołami, np.:
# pip install --no-index --find-links dist/ -r deploy/packaging/requirements-desktop.lock
# a następnie doinstaluj bieżący projekt oraz zależności desktopowe z lokalnego koła
pip install --no-index --find-links dist/ 'dudzian-bot[desktop]'
```

> **Uwaga:** przed dystrybucją artefaktów offline zbuduj pakiet kołowy poleceniem
> `pip wheel . -w dist/`, aby `pip` mógł rozwiązać zależność `dudzian-bot` bez
> dostępu do internetu.

## Walidacja środowiska zależności

Po odtworzeniu środowiska z lockfile zalecamy potwierdzić, że najważniejsze
moduły AI oraz strategie startują bez brakujących importów:

```bash
pytest \
  tests/test_ai_manager_multimodel.py \
  tests/test_cross_exchange_arbitrage_strategy.py \
  tests/test_build_desktop_installer.py \
  tests/test_desktop_environment_imports.py \
  tests/test_pyproject_desktop_dependencies.py \
  tests/test_requirements_desktop_txt.py \
  tests/test_requirements_desktop_lock.py
```

Powyższe testy uruchamiają wielomodelowego menedżera AI, sprawdzają strategię
arbitrażu międzygiełdowego oraz scenariusz budowy instalatora desktopowego, co
pozwala szybko wykryć brakujące biblioteki numeryczne.
Dodatkowo `tests/test_desktop_environment_imports.py` upewnia się, że moduły
UI oraz podstawowe biblioteki (`numpy`, `pandas`, `joblib`) są dostępne w
środowisku.
Dodatkowo test `tests/test_pyproject_desktop_dependencies.py` pilnuje, aby
deklaracje zależności w `pyproject.toml` zawsze obejmowały stos numeryczny
oraz pakiety bundlujące wymagane do budowy instalatora.
Dodatkowo test `tests/test_requirements_desktop_txt.py` weryfikuje, że
`requirements-desktop.txt` pozostaje zsynchronizowany z deklaracjami w
`pyproject.toml`, a przypięte wersje spełniają zadeklarowane ograniczenia.
Dodatkowo test `tests/test_requirements_desktop_lock.py` kontroluje spójność
lockfile względem listy pakietów, dzięki czemu bundle nie powstanie na
niezsynchronizowanym zestawie zależności.

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

## Dialogi administracyjne raportów

Panel administratora udostępnia dwa główne dialogi operacyjne:

* **Usuń** – deleguje komendę `python -m bot_core.reporting.ui_bridge delete`,
  usuwając wskazany raport lub katalog eksportów. Podgląd dialogu uruchamia się
  z zaznaczeniem „Podgląd” (odpowiednik `--dry-run`), dzięki czemu operator
  widzi liczbę plików i rozmiar przed potwierdzeniem trwałej operacji.
* **Archiwizuj** – korzysta z `python -m bot_core.reporting.ui_bridge archive`,
  kopiując raporty do katalogu archiwum (`--destination`) w jednym z formatów:
  `directory`, `zip`, `tar`. Opcja „Nadpisz istniejące” mapuje się na flagę
  `--overwrite`. W trybie podglądu UI prezentuje wynik `--dry-run`.

Konfiguracja dialogów wymaga poprawnych ścieżek w ustawieniach UI:

1. **Katalog raportów** – `var/reports` lub lokalizacja wskazana przez
   `BOT_CORE_UI_REPORTS_DIR` / `--reports-directory`. Ścieżka musi być dostępna
   do zapisu dla konta operatorskiego.
2. **Interpreter Pythona** – pełna ścieżka do środowiska zawierającego moduł
   `bot_core.reporting.ui_bridge` (np. `/opt/dudzian/venv/bin/python`).
3. **Katalog docelowy archiwów** – UI wykorzystuje wartości z konfiguracji
   profilu (sekcja `reporting.archive_destination`). Ścieżka powinna wskazywać
   zasób poza katalogiem raportów – domyślnie `var/reports_archives`.

### Scenariusz: Usunięcie raportu z poziomu UI

1. W widoku Raportów zaznacz katalog lub plik wymagający usunięcia.
2. Wybierz „Usuń” → „Podgląd” i zweryfikuj statystyki (`removed_files`,
   `removed_size`). Jeśli wynik budzi wątpliwości, zrezygnuj z operacji i
   skonsultuj filtr w CLI (`overview`).
3. Utwórz kopię zapasową (np. `rsync -av var/reports/<ścieżka>
   audit/backups/<ścieżka>_$(date +%Y%m%d)/`).
4. Potwierdź operację w UI. Wynik (`status=deleted`) zapisz w notatce
   operacyjnej (np. `audit/maintenance_logs/`).

### Scenariusz: Archiwizacja dziennych raportów z UI

1. Zaznacz raporty dzienne (możliwe zaznaczenie wielu elementów).
2. Wybierz „Archiwizuj” i ustaw format `ZIP`. Włącz „Podgląd”, aby upewnić się,
   że katalog docelowy posiada wystarczającą ilość miejsca.
3. Po akceptacji przełożonego wyłącz podgląd i uruchom archiwizację. UI
   zapisze raport JSON w `logs/report_center.log`, który należy dołączyć do
   dziennika audytowego.
4. Zweryfikuj powstałe archiwa (sprawdź zawartość `summary.json`, `ledger.csv`).

### Wskazówki bezpieczeństwa

* Wymuś w konfiguracji systemowej limit uprawnień – konto operatora powinno
  mieć dostęp wyłącznie do katalogów raportów/archiwów, bez możliwości
  modyfikacji pozostałych zasobów.
* Przed potwierdzeniem dialogu „Usuń” lub „Archiwizuj” wykonaj eksport
  wyników `--dry-run` i przechowuj go w repozytorium audytowym co najmniej 24
  miesiące (wymóg zgodności). Podgląd jest dostępny w panelu w formie JSON.
* Więcej procedur i checklist znajduje się w runbooku
  [`docs/runbooks/report_maintenance.md`](../runbooks/report_maintenance.md).

## Pakiet wsparcia

Zakładka **Wsparcie** w panelu administratora wywołuje skrypt
`scripts/export_support_bundle.py`, który pakuje katalogi `logs/`,
`var/reports`, `var/licenses`, `var/metrics` oraz opcjonalnie `var/audit` do
archiwum `tar.gz` lub `zip`. Operator może wskazać dodatkowe zasoby (`--support-bundle-include label=/ścieżka`) oraz wyłączyć
domyślne elementy (`--support-bundle-disable logs`). Zmienna
`BOT_CORE_UI_SUPPORT_OUTPUT_DIR` pozwala skierować pakiety np. na dysk USB,
co przyspiesza przekazanie danych zespołowi wsparcia.
