# Walidacja migracji warstwy async

Ten dokument opisuje procedurę porównania wyników strategii przed i po migracji do asynchronicznej kolejki I/O.

## 1. Przygotowanie snapshotów legacy

1. Upewnij się, że środowisko działa w wersji _przed_ migracją async.
2. Uruchom scenariusz `scripts/run_local_bot.py --mode demo --report-dir logs/e2e --report-markdown-dir reports/e2e`.
3. Po zakończeniu scenariusza skopiuj wygenerowane logi i raporty KPI do katalogu `data/snapshots/legacy/` (domyślne pliki `summary.json`).

## 2. Wykonanie migracji i zebranie snapshotów async

1. Zaktualizuj repozytorium do wersji z warstwą async oraz nowymi guardrails.
2. Powtórz scenariusz `scripts/run_local_bot.py --mode demo ...` korzystając z tych samych presetów strategii.
3. Wyniki zapisz w katalogu `data/snapshots/async/`.

## 3. Porównanie wyników

1. Uruchom skrypt `python scripts/analysis/compare_snapshots.py`.
2. Opcjonalnie dostosuj parametry `--relative-tolerance` oraz `--absolute-tolerance`, aby dopasować dopuszczalne odchylenia KPI.
3. Status w konsoli wskaże brakujące strategie, metryki lub KPI przekraczające tolerancję.

## 4. Interpretacja rezultatów

- **Sukces** – brak komunikatów o brakujących danych i brak metryk przekraczających tolerancję.
- **Ostrzeżenia** – lista strategii/metryk wymagających ponownej kalibracji lub analizy regresji.
- Dokumentuj wynik porównania jako załącznik do raportu guardrails (`reports/guardrails/`).

## 5. Automatyzacja w CI

- Dodaj wywołanie skryptu do pipeline'u regresyjnego, aby automatycznie porównywać snapshoty po każdej migracji.
- Artefakty JSON/Markdown przechowuj w `logs/e2e/` oraz `reports/e2e/` w celu audytu QA.
