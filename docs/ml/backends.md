# Backendowe modele ML

Pakiet `core.ml` udostępnia mechanizm wyboru backendów uczenia maszynowego
wykorzystywany przez pipeline Decision Engine. Konfiguracja priorytetów
znajduje się w pliku `config/ml/backends.yml` i pozwala określić kolejność
prób uruchomienia konkretnych implementacji.

## Referencyjny regresor liniowy

Backend `reference` to lekki model regresji liniowej działający w czystym
Pythonie. Implementacja bazuje na równaniach normalnych rozwiązywanych
eliminacją Gaussa, dzięki czemu nie wymaga dodatkowych bibliotek takich jak
NumPy czy LightGBM. Model obsługuje następujące operacje:

- `fit(samples, targets)` – trening na zbiorze prób złożonym z map cech,
- `predict(sample)` oraz `batch_predict(samples)` – inferencja dla pojedynczej
  próbki lub całej sekwencji,
- `save(path)` i `load(path)` – serializacja do formatu JSON w celu przenoszenia
  artefaktów między środowiskami.

Model pełni rolę fallbacku w sytuacji, gdy preferowane backendy (np. LightGBM)
nie są dostępne w środowisku użytkownika. Fabryka backendów (`core/ml/factory.py`)
wykorzystuje konfigurację priorytetów do wyboru najlepszego dostępnego
kandydata, zachowując spójny interfejs dla pozostałych modułów systemu.
