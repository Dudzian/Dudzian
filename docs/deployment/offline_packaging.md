# Pakietowanie offline powłoki OEM

## Cel

Pakiet offline łączy binarkę Qt, konfigurację, dane referencyjne i dokumentację
operacyjną w pojedynczym archiwum, które można dostarczyć na stanowiska bez
połączenia z siecią. Proces bazuje na skryptach `python scripts/build_desktop_installer.py`
(kompilacja i podpis updatera) oraz `python scripts/deploy/offline_packager.py` (konsolidacja
artefaktów).

## Wymagania

* Zbudowana aplikacja Qt (`cmake --build ui/build --config Release`).
* Zainstalowany PyInstaller w aktywnym środowisku (do zbudowania updatera).
* Dostęp do katalogów `config/`, `docs/`, `data/` oraz logów OEM (opcjonalnie).
* Klucz HMAC do podpisu updatera (opcjonalnie, przekazywany do `--signing-key`).

## Budowanie pakietu offline

1. Zbuduj aplikację desktopową:

   ```bash
   cmake -S ui -B ui/build -GNinja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH="/ścieżka/do/Qt/6.5.0/gcc_64"
   cmake --build ui/build --target bot_trading_shell
   ```

2. W katalogu głównym repozytorium uruchom pakowacz offline. Najczęstszy scenariusz
   kopiuje predefiniowane datasety, konfigurację i dokumentację OEM:

   ```bash
   python scripts/deploy/offline_packager.py \
     --ui-build ui/build \
     --platform linux \
     --signing-key "$UPDATER_HMAC" \
     --extra docs/licensing/activation_checklist.md \
     --extra var/licenses/expected_fingerprint.json
   ```

   Kluczowe parametry:

   * `--ui-build` – katalog z plikiem `bot_trading_shell` lub `bot_trading_shell.exe`.
   * `--platform` – platforma docelowa (`linux`, `windows`, `mac`).
   * `--datasets` – lista katalogów z danymi do spakowania (domyślnie `data/trading_stub/datasets`).
   * `--docs` – dodatkowa dokumentacja operatorska (domyślnie `docs/deployment/*`).
 * `--extra` – dowolne dodatkowe pliki (np. manifest licencyjny, notatki L2).
  * `--signing-key` – HMAC (hex/tekst), którym podpisywany jest updater.

3. W katalogu `var/dist/offline/` powstanie struktura:

   ```
   var/dist/offline/
     ├─ installer/                # artefakty pośrednie (pakiet Qt + updater)
     ├─ bundle/
     │   ├─ bot_trading_shell_bundle.zip
     │   ├─ config/
     │   ├─ datasets/
     │   ├─ docs/
     │   ├─ extras/
     │   └─ MANIFEST.json
     └─ offline_bundle.tar.gz     # finalny pakiet do dystrybucji offline
   ```

  Plik `MANIFEST.json` opisuje zawartość archiwum i ułatwia weryfikację integralności
  podczas odbioru pakietu na stanowisku docelowym.

### Dołączanie artefaktów modeli AI

* Modele Decision Engine muszą zostać zbudowane przed pakietowaniem i umieszczone
  w katalogu `ai_models/packaged/` (dystrybucja barebone) lub `bot_core/ai_models_packaged/`
  (wariant OEM). Katalog musi zawierać `manifest.json` z listą wersji oraz
  podkatalogi z plikami `*.json`, `*.metadata.json`, `checksums.sha256` i opcjonalnym
  podpisem `*.sig` wygenerowanymi przez `bot_core.ai.models.generate_model_artifact_bundle`.
* Integralność pakietu jest weryfikowana przez nowy krok testów
  `pytest tests/ai/test_model_artifact_bundle.py`. Testy muszą przechodzić w pipeline,
  dzięki czemu brak artefaktów lub rozbieżne sumy kontrolne blokują wydanie.
* Po skopiowaniu artefaktów warto lokalnie uruchomić
  `pytest tests/ai/test_ai_manager_degradation.py::test_ai_manager_detects_invalid_packaged_repository`
  – test potwierdza, że pakiet zawiera kompletne modele i menedżer AI nie przełącza
  się w tryb degradacji.

### Walidacja pakietu

* Zespół CI/CD może okresowo uruchamiać `pytest tests/scripts/test_offline_packager.py`
  aby upewnić się, że skrypt kopiuje konfigurację, datasety i dokumentację do
  manifestu w oczekiwanej strukturze. Testy symulują artefakty instalatora i
  sprawdzają, że domyślne pliki `docs/deployment/*` są dołączane, gdy nie podano
  własnych materiałów operatora.
* Po zbudowaniu pakietu sprawdź zawartość `bundle/MANIFEST.json`, czy wymienione
  są wszystkie katalogi (`config`, `datasets`, `docs`, `extras`) oraz poprawna nazwa
  archiwum instalatora. To ułatwia odbiór jakościowy bez rozpakowywania całości.

## Aktualizacje i dystrybucja

* Pakiet można przenosić na nośnikach USB. Po rozpakowaniu `offline_bundle.tar.gz`
  na stacji docelowej uruchom instalator (`bot_trading_shell_bundle.zip`) i postępuj
  zgodnie z instrukcją `docs/deployment/oem_installation.md`.
* W przypadku aktualizacji wystarczy ponownie uruchomić skrypt pakujący z nowym buildem
  oraz danymi – podpis updatera gwarantuje spójność binarek.
* `docs/deployment/offline_packaging.md` oraz `docs/licensing/*` należy aktualizować
  razem z pakietem, aby operatorzy mieli dostęp do bieżących procedur aktywacji i
  odzyskiwania licencji.
