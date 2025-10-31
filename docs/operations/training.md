# Operacje treningowe Decision Engine

Ten dokument opisuje uproszczony pipeline treningowy dostępny lokalnie oraz
procedury troubleshootingu dla środowisk, w których brakuje części zależności
uczenia maszynowego.

## Pipeline z fallbackiem backendów

Moduł `core.ml.training_pipeline.TrainingPipeline` pozwala uruchomić trening
z listą preferowanych backendów. Pipeline próbuje kolejno wskazane backendy,
rejestruje niepowodzenia (np. brak modułu `lightgbm`) i automatycznie
przełącza się na następny kandydat, domyślnie `reference`.

* Fallbacki są logowane do katalogu `logs/ml/fallback/` w formacie JSON.
* W pliku logu znajdują się informacje o priorytecie backendów, wybranym
  fallbacku oraz sugerowane polecenia instalacyjne.

## Uruchamianie treningu z CLI

```
python scripts/train_model.py \
  --preferred-backend lightgbm \
  --preferred-backend reference \
  --config config/ml/backends.yml
```

Parametry:

* `--dataset` – opcjonalny plik JSON z polami `features` i `targets`. Jeżeli
  nie zostanie podany, skrypt użyje syntetycznego zbioru treningowego.
* `--preferred-backend` – można wskazać wiele razy, aby ustalić kolejność
  prób. Domyślna kolejność to `lightgbm`, `reference`.
* `--fallback-log-dir` – katalog zapisu logów fallbacku (domyślnie
  `logs/ml/fallback`).

Skrypt wypisuje na stdout JSON zawierający nazwę wykorzystanego backendu oraz
ścieżkę do logu fallbacku (jeżeli wystąpił).

## Troubleshooting backendów

1. **Brak modułu LightGBM lub XGBoost** – sprawdź log w `logs/ml/fallback/…`
   i wykonaj polecenie instalacyjne z pola `install_hint`
   (np. `pip install lightgbm`).
2. **Nieprawidłowy format datasetu** – upewnij się, że plik JSON zawiera
   dwie listy (`features`, `targets`) o tej samej długości i że każdy wpis w
   `features` jest słownikiem z wartościami liczbowymi.
3. **Brak logu fallbacku** – pipeline tworzy log tylko wtedy, gdy faktycznie
   nastąpiło przełączenie backendu. Jeżeli brakuje logu, oznacza to, że
   pierwszy dostępny backend zakończył trening poprawnie.

W razie dalszych problemów włącz logowanie na poziomie `DEBUG` za pomocą
parametru `--log-level DEBUG`, aby uzyskać pełne komunikaty diagnostyczne.

