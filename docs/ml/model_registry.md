# Rejestr modeli ML

Rejestr modeli umożliwia śledzenie artefaktów ML wykorzystywanych przez lokalny
runtime oraz oznaczanie aktywnej wersji modelu. Mechanizm jest w pełni
lokalny – metadane zapisywane są w katalogu `var/models` (domyślnie plik
`registry.json`).

## Metadane modelu

Każdy zarejestrowany model posiada następujące atrybuty:

| Pole              | Opis                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| `model_id`        | Unikalny identyfikator w postaci `<timestamp>_<hash>`                |
| `backend`         | Nazwa backendu ML użytego do trenowania                              |
| `artifact_path`   | Ścieżka do pliku modelu na dysku                                     |
| `sha256`          | Suma kontrolna artefaktu                                             |
| `created_at`      | Znacznik czasu publikacji w formacie ISO-8601                        |
| `dataset_metadata`| Dowolne metadane opisujące zestaw danych (np. liczba próbek, źródło) |

Metadane są automatycznie synchronizowane z `RuntimeStateManager`, dzięki czemu
ostatni aktywny model jest widoczny w metadanych checkpointu (`metadata.active_model`).

## Operacje CLI

Do obsługi rejestru służy skrypt `scripts/manage_models.py`. Dostępne są trzy
polecenia:

```bash
python -m scripts.manage_models publish --artifact path/to/model.bin --backend reference \
    --metadata rows=1024 --state-dir var/runtime --registry-dir var/models

python -m scripts.manage_models list --registry-dir var/models --output json

python -m scripts.manage_models rollback --model-id 20250101103000_abcd1234 \
    --state-dir var/runtime --registry-dir var/models
```

- `publish` – dodaje nowy model do rejestru, wylicza jego hash i ustawia jako
  aktywny. Metadane datasetu można przekazać w formie par `KEY=VALUE`.
- `list` – wypisuje wszystkie zarejestrowane modele (format tabeli lub JSON).
- `rollback` – oznacza wcześniej opublikowany model jako aktywny.

W przypadku nieistniejącego pliku artefaktu lub błędnego formatu metadanych
skrypt zwraca kod wyjścia `2` wraz z komunikatem w logach.

## Integracja z pipeline'em treningowym

Po zakończeniu trenowania modelu można wykorzystać rejestr do publikacji
artefaktu oraz utrwalenia informacji o wybranym backendzie. Dzięki temu podczas
kolejnych uruchomień lokalny runtime zna ostatnio aktywną wersję i może
wykorzystać ją do weryfikacji konfiguracji lub procedur rollbacku.

