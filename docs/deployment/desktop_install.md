# Instalacja i aktualizacje desktopowego bota

Ten runbook opisuje, jak przygotować, zainstalować oraz zaktualizować aplikację desktopową w trybie całkowicie offline. Procedura zakłada dostęp do repozytorium projektu oraz pakietów licencyjnych przypisanych do konkretnej maszyny (fingerprint sprzętowy).

## 1. Budowa paczki instalacyjnej

1. Przygotuj katalogi z gotowym runtime (binarki/venv), zasobami UI oraz dodatkowymi plikami konfiguracyjnymi.
2. Wygeneruj/pozyskaj plik licencji JSON przypisany do konkretnego fingerprintu (np. `secrets/licenses/desktop.json`).
3. Uruchom skrypt:
   ```bash
   python scripts/build/desktop_distribution.py \
       --version 1.2.0 \
       --platform linux \
       --runtime-dir build/runtime \
       --ui-dir ui/dist \
       --include resources=deploy/samples/resources \
       --license-json secrets/licenses/desktop.json \
       --license-fingerprint HW-ABC-12345 \
       --signing-key release=hex:001122...
   ```
   Skrypt tworzy archiwum `tar.gz` w `var/dist/installers/` (lub katalogu podanym w `--output-dir`), szyfruje licencję oraz generuje manifest wraz z opcjonalnym podpisem HMAC.

## 2. Instalacja na hostcie docelowym

1. Przenieś wygenerowane archiwum na komputer docelowy (np. przez dysk USB).
2. Wypakuj archiwum w docelowej lokalizacji, np.:
   ```bash
   tar -xzf bot-desktop-1.2.0-linux.tar.gz -C /opt/bot
   ```
3. W katalogu `resources/license/` znajduje się zaszyfrowany magazyn licencji (`license_store.json`) oraz raport integralności. Nie wymaga on dodatkowej konfiguracji – licencja jest powiązana z fingerprintem wskazanym przy budowie pakietu.
4. Uruchom aplikację (`runtime/.../start.sh` lub odpowiedni binarny wrapper) i zweryfikuj, że interfejs UI poprawnie odczytuje stan licencji w zakładce „Licencja”.

## 3. Budowa pakietu aktualizacji

1. Przygotuj katalogi zawierające poprzednią oraz nową wersję instalacji (po wypakowaniu archiwów z sekcji 1).
2. Uruchom skrypt generujący aktualizację:
   ```bash
   python scripts/build/prepare_update_package.py \
       --base-dir /opt/bot/bot-desktop-1.1.0-linux \
       --target-dir /opt/bot/bot-desktop-1.2.0-linux \
       --package-id bot-suite \
       --version 1.2.0 \
       --platform linux \
       --runtime desktop \
       --base-id bot-suite-1.1.0 \
       --signing-key update=hex:8899aabb \
       --metadata changelog="\"UI redesign\""
   ```
   Skrypt tworzy katalog z manifestem aktualizacji, pełnym archiwum nowej wersji oraz opcjonalną łatką różnicową. Manifest jest automatycznie weryfikowany funkcją `verify_update_bundle`.

## 4. Aktualizacja z użyciem pakietu offline

1. Skopiuj wygenerowany katalog pakietu na hosta docelowego.
2. Uruchom istniejący updater (np. `scripts/desktop_updater.py`) wskazując manifest `manifest.json` z pakietu.
3. Updater pobierze informacje o usuwanych plikach (`deleted_files.json`), wgra nowe artefakty i zaktualizuje manifest integralności.
4. Po aktualizacji uruchom aplikację w trybie paper trading i wykonaj szybki test (np. dostępny w UI lub przez CLI `run_paper_smoke_ci.py`), aby potwierdzić działanie licencji i modułów giełdowych.

## 5. Najczęstsze problemy

| Problem | Rozwiązanie |
| --- | --- |
| Błąd `DesktopBuildError: Katalog runtime ... nie istnieje` | Zweryfikuj ścieżkę przekazaną w `--runtime-dir`; skrypt wymaga gotowego katalogu z binarkami. |
| Weryfikacja pakietu aktualizacji nie powiodła się (`verify_update_bundle`) | Sprawdź, czy używasz tego samego klucza HMAC, którym podpisano manifest (`--signing-key`). |
| Licencja oznaczona jako nieaktywna po instalacji | Upewnij się, że fingerprint hosta odpowiada wartości użytej podczas budowy pakietu (`--license-fingerprint`). W razie zmiany sprzętu należy wygenerować nowy magazyn licencyjny. |

Dokument można rozszerzać o specyficzne procedury OEM, np. dystrybucję paczek przez nośniki szyfrowane lub integrację z wewnętrznym systemem helpdesk.
