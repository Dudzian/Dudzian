# Archiwalny pipeline modeli handlowych

Dawny moduł modeli w katalogu `archive/` zapewniał zunifikowaną fabrykę modeli oraz
narzędzia backtestowe dla warstwy AI botów. Moduł łączył klasyczne algorytmy ML
(LightGBM, XGBoost, SVR, RandomForest) z sekwencyjnymi sieciami neuronowymi (LSTM,
GRU, MLP, Transformer, TCN, N-BEATS, TFT, Mamba, Autoformer, DeepAR) i udostępniał
podstawowe komponenty:

- `ModelFactory` – abstrakcję do tworzenia modeli na podstawie identyfikatora
typu, z rozróżnieniem backendów PyTorch i scikit-learn.
- `BacktestEngine` – wspólny silnik walidacji z integracją `vectorbt`.
- Narzędzia przetwarzania sekwencji (`windowize_df_robust`) i skalery odporne
na wartości odstające (`CustomRobustScaler`).
- Mechanizmy serializacji (`joblib`/`pickle`) i ładowania backendów przez
`require_backend`.

Kod nie jest już utrzymywany ani używany przez bieżący runtime. Został usunięty z
katalogu `archive/`, aby uniknąć przypadkowego importowania oraz aby CI mogło
łatwiej wychwycić zalegające moduły. Pełną treść można w razie potrzeby
odtworzyć z historii repozytorium (commit sprzed usunięcia pliku) lub z backupu
archiwalnego `archive/`.

> **Status:** wyłącznie referencja historyczna na potrzeby migracji i audytów.
> Wszystkie aktywne modele znajdują się w przestrzeni `bot_core.ai_models`.
