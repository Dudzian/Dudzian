# Proces wydawniczy

Dokument opisuje kroki przygotowania wydań bota w kanałach **dev**, **rc** oraz
stabilnym.  Pipeline `deploy/local_ci.yml` agreguje większość kroków i może być
uruchamiany lokalnie przed dystrybucją paczek do klientów.

## Kanał RC

Kanał release candidate jest przeznaczony do walidacji paczek przed publikacją
stabilną.  Workflow `release-candidate` w `deploy/local_ci.yml` wykonuje
następujące czynności:

1. Buduje instalator desktopowy dla bieżącej platformy (zadanie `desktop-installer`).
2. Uruchamia zestaw testów smoke poprzez `scripts/run_smoke_tests.py`, który
   generuje artefakty w `reports/smoke/` (JSON + Markdown) oraz zwraca kod
   wyjścia pytest.
3. Tworzy tag RC w repozytorium w formacie `rc-<wersja>-<timestamp>`, zapisuje
   manifest `var/dist/rc_manifest.json` i plik `var/dist/rc_tag.txt`.
4. Kopiuje pakiety instalatora do `var/dist/rc/` jako paczki pre-release.

Raporty smoke opisane są w `reports/smoke/README.md`.  Aby je zinterpretować:

- `exit_status` różny od zera oznacza, że smoke testy nie przeszły i build nie
  powinien zostać promowany.
- Wysoka liczba `skipped` sygnalizuje brak zależności środowiskowych (np.
  PySide6) i wymaga ręcznej oceny.

## Kryteria promocji RC → GA

Wydanie RC może zostać oznaczone jako stabilne (GA), jeśli spełnione są
następujące warunki:

- Wszystkie testy smoke zakończą się powodzeniem (`exit_status == 0`, `failed == 0`).
- Nie wykryto naruszeń guardrail’i w raportach `reports/guardrails/`.
- Audyt zgodności (`reports/compliance/`) nie zawiera alertów o priorytecie
  krytycznym.
- Instalator desktopowy został zweryfikowany na docelowych platformach.

## Checklist przed publikacją stabilną

1. Zaktualizuj changelog i numer wersji w `pyproject.toml`.
2. Upewnij się, że ostatni tag RC jest zsynchronizowany z repozytorium
   (`git push --tags`).
3. Opublikuj paczki instalatora oraz raporty smoke/compliance do systemu
   dystrybucyjnego.
4. Po potwierdzeniu jakości, utwórz tag stabilny i zamknij sprint release.
