# Odzyskiwanie po błędzie importu aktualizacji offline

## Zakres i wynik

Import pojedynczego pakietu `.dudzianpkg` w
`core/update/offline_updater.py` przygotowuje teraz całą zawartość w ukrytym
katalogu na docelowym systemie plików. Istniejący katalog pakietu pozostaje
dostępny podczas zapisywania manifestu i kopiowania artefaktów. Dopiero po
zakończeniu przygotowania importer przenosi dotychczasowy katalog do kopii
roboczej i publikuje przygotowany katalog przez `rename`.

Obsłużony błąd publikacji powoduje próbę przywrócenia poprzedniego katalogu.
Jeżeli również przywrócenie zakończy się błędem, importer zachowuje kopię
i zgłasza jej ścieżkę w `OfflinePackageError`, aby umożliwić odzyskiwanie
operatorskie. Przygotowywanie i weryfikacja usuwają katalogi tymczasowe
także podczas propagowanego `KeyboardInterrupt`. Zwracane ścieżki artefaktów
wskazują opublikowany katalog.

## Dowody i walidacja

Testy regresji obejmują błąd częściowego kopiowania dla nowego i istniejącego
pakietu, przerwanie po przeniesieniu poprzedniego katalogu, błąd publikacji,
nieudaną próbę przywrócenia, sprzątanie po przerwaniu ekstrakcji i weryfikacji,
błąd katalogu docelowego oraz poprawne zastąpienie istniejącego pakietu.
Negatywne przypadki używają syntetycznych pakietów i katalogów tymczasowych.

Walidacja lokalna:

- Cztery moduły testów: `tests/update/test_offline_updater.py`,
  `tests/update/test_offline_update_flow.py`,
  `tests/update/test_update_package_cli.py` i `tests/test_offline_updates.py`:
  26 zaliczonych testów.
- Pełny zakres Ruff z `scripts/quality/run_fast.ps1`: PASS.
- Dokładny hook `python -m pre_commit run ruff-format --all-files`: PASS,
  bez zmian po przebiegu.
- Skonfigurowany mypy dla `linux` i `win32`: PASS, po 424 pliki.
- Dodatkowy mypy zmienionego modułu importera: PASS.
- Betterleaks dla zmienionych plików: PASS, bez wykrytych sekretów.
- `git diff --check`: PASS.

Mypy działał w izolowanym środowisku hooka z zależnościami repozytorium.
Sprawdzenie `--platform win32` stanowi analizę statyczną; nie jest wykonaniem
na Windows. Kontrole lokalne nie zastępują wymaganych kontroli CI.

## Ograniczenia i wpływ

Zmiana dotyczy obsłużonych błędów i przerwań pojedynczego importu w Pythonie.
Nie ustanawia odporności na wymuszone zakończenie procesu, utratę zasilania
ani równoległe importy. Pomiędzy dwiema operacjami `rename` istnieje krótki
okres bez katalogu docelowego. Zachowana po nieudanym przywróceniu kopia
wymaga działania operatora; importer nie wykonuje automatycznego odzyskiwania
po ponownym uruchomieniu.

Osobny instalator CLI `core/update/installer.py`, instalacja wielu komponentów
oraz ścieżka instalacji C++ pozostają poza zakresem tej poprawki. Zasady
weryfikacji podpisów, integralności, ścieżek i zawartości archiwów nie są
w tym raporcie kwalifikowane. Zielone testy importera nie są potwierdzeniem
bezpieczeństwa całego procesu aktualizacji.

Nie zmieniono konfiguracji CI, progów pokrycia, wyłączeń skanerów ani
bramek sprzętowych i manualnych. Stage 9, Stage 10 i dotychczasowy status
produkcyjny NO-GO pozostają bez zmian.
