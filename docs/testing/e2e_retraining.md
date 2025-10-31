# Scenariusz E2E: walidacja retreningu Decision Engine

Ten scenariusz odwzorowuje lokalne uruchomienie retreningu wraz z walidacją danych,
fallbackiem backendu ML oraz generowaniem artefaktów raportowych wykorzystywanych w
regresji.

## Kroki uruchomienia

1. Przygotuj środowisko wirtualne (np. `poetry shell`) i upewnij się, że repozytorium
   posiada katalogi na artefakty: `reports/e2e/retraining/` oraz `logs/e2e/retraining/`.
2. Uruchom skrypt retreningu korzystając z przygotowanej konfiguracji E2E i katalogów
   docelowych:

   ```bash
   poetry run python scripts/run_retraining_cycle.py \
       --config config/e2e/retraining_validation.yml \
       --preferred-backend lightgbm \
       --report-dir reports/e2e/retraining \
       --kpi-snapshot-dir reports/e2e/retraining \
       --e2e-log-dir logs/e2e/retraining \
       --fallback-log-dir logs/e2e/retraining/fallback \
       --validation-log-dir logs/e2e/retraining/validation
   ```

   Brak zainstalowanego `lightgbm` spowoduje kontrolowane przełączenie na backend
   referencyjny. Walidacja datasetu zapisze raport JSON w katalogu podanym flagą
   `--validation-log-dir`.

3. Po zakończeniu skrypt wypisze na STDOUT raport JSON oraz utworzy następujące
   artefakty:

   - Markdown i JSON w `reports/e2e/retraining/` (raport retreningu),
   - Snapshot KPI w `reports/e2e/retraining/kpi_*.json`,
   - Log przebiegu scenariusza w `logs/e2e/retraining/retraining_run_*.json`,
   - Log fallbacku backendów w `logs/e2e/retraining/fallback/fallback_*.json`,
   - Raport walidacji danych w `logs/e2e/retraining/validation/dataset_validation_*.json`.

## Walidacja automatyczna

Test `pytest -m e2e_retraining` uruchamia powyższy skrypt na syntetycznym zbiorze danych i
weryfikuje obecność wszystkich artefaktów, łańcucha fallbacku oraz poprawność raportu
KPI. Test używa katalogów tymczasowych, dzięki czemu nie wpływa na pliki w repozytorium.
