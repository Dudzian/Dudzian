# AI Decision Pipeline

## Przepływ danych i scoringu

1. **Źródła OHLCV** – moduł `bot_core.data.ohlcv` dostarcza zunifikowane świece, które są normalizowane przez `CachedOHLCVSource` i udostępniane pipeline'owi AI.
2. **Feature engineering** – `FeatureEngineer` oblicza cechy momentum, zmienności, relacje wolumenu oraz rozpiętości cen dla okien kroczących. Zestaw rozszerzają wskaźniki trendowe (luka do EMA szybkiej/wolnej, DEMA/TEMA, MACD + histogram i luka do linii sygnałowej, Average Directional Index z DI+/DI−, chmura Ichimoku z pozycją ceny, kanały Donchiana i Keltnera, luka względem Parabolic SAR z kierunkiem trendu, TRIX, Vortex oraz wskaźniki Aroon up/down/oscillator, Elder Ray bull/bear power), oscylatory (RSI, Bollinger position/width, Stochastic %K/%D, Stochastic RSI, Williams %R, Money Flow Index, Ultimate Oscillator, Chande Momentum Oscillator, Detrended Price Oscillator, Relative Vigor Index + sygnał, True Strength Index z linią sygnałową i luką, Schaff Trend Cycle, Fisher Transform, Percentage Price Oscillator), a także miary momentum (`price_rate_of_change`) i przepływu kapitału (standaryzowany wolumen, trend zmienności, ATR w relacji do ceny, CCI, znormalizowany OBV i Price Volume Trend, pozytywny/negatywny indeks wolumenowy, znormalizowany Force Index, Chaikin Money Flow, Ease of Movement, linia akumulacji/dystrybucji z oscylatorem Chaikina, luka do VWAP, Balance of Power, Connors RSI, Intraday Intensity wraz z komponentem wolumenowym). Kontekst ryzyka uzupełniają metryki typu Ulcer Index oraz efficiency ratio, adaptacyjne KAMA (luka i nachylenie), FRAMA (luka i nachylenie), krzywa Coppocka, indeks choppiness, mass index oraz wskaźnik Klingera (oscylator + luka do sygnału), a także Qstick, które mierzą stabilność trendu, głębokość obsunięć i relację korpusu świec. Uzupełnieniem są świece Heikin Ashi (trend, relacja cieni) oraz klasyczne pivoty (P/R1/S1) i luki do ostatnich fraktali, pomagające ocenić strukturę świec i położenie ceny względem poziomów wsparcia/oporu. Targetem jest przyszły zwrot w punktach bazowych, dodatkowo monitorowany przez wskaźniki momentum takie jak price rate of change. Każdy `FeatureDataset` zapisuje metadane `feature_names` i `feature_stats` (min/max/średnia/odchylenie), co pozwala audytować stabilność dystrybucji cech.
3. **Trenowanie** – `ModelTrainer` standaryzuje cechy na podstawie statystyk z porcji treningowej i wykorzystuje `SimpleGradientBoostingModel` do budowy artefaktu (`ModelArtifact`). Parametry `validation_split` i `test_split` pozwalają wydzielić walidację oraz zestaw testowy z końca próbki, generując osobne metryki MAE/RMSE (train + validation + test), liczbę wierszy w każdej porcji oraz mapę `feature_scalers` (średnia, odchylenie) umożliwiającą odtworzenie normalizacji podczas inference.
4. **Repozytorium modeli** – `ModelRepository` utrzymuje wersjonowane artefakty (JSON). Retrain scheduler odkłada nowe artefakty w wybranym katalogu, a inference ładuje je on-line.
5. **Inference** – `DecisionModelInference` mapuje cechy kandydata na oczekiwany zwrot i prawdopodobieństwo sukcesu. Przed scoringiem uzupełnia brakujące cechy średnimi z `feature_scalers` i standaryzuje wartości identycznie jak w treningu. Wynik wstrzykiwany jest do `DecisionOrchestrator`, który łączy scoring z limitami kosztów i ryzyka.
6. **Decision loop** – `AIDecisionLoop` w `bot_core.runtime.controller` buduje dataset w locie, tworzy kandydatów z metadanymi cech i odpytuje orchestrator wraz z snapshotem ryzyka.

## Walidacja walk-forward

- `WalkForwardValidator` dzieli zbiór cech na okna treningowe/testowe, dla każdego trenuje model i liczy średnie MAE oraz directional accuracy.
- Wyniki walidacji należy archiwizować razem z metadanymi treningu i wskaźnikiem `target_scale` – pozwala to audytować kalibrację prawdopodobieństwa inference.
- Testy jednostkowe w `tests/decision/` pokrywają retraining scheduler oraz pipeline walidacji.

## Checklisty compliance

- **AI Artefacts** – każda wersja modelu musi mieć: (a) podpisany artefakt (`ModelArtifact`), (b) raport metryk z walidacji walk-forward, (c) wpis w decision journal z informacją o aktywacji modelu, (d) zweryfikowaną zgodność `feature_scalers` z danymi treningowymi.
- **On-line scoring** – przed wdrożeniem zweryfikuj, że `DecisionOrchestrator` ładuje aktualne wagi (`DecisionModelInference.is_ready == True`) oraz że `AIDecisionLoop` loguje odchylenia kosztów w alertach.
- **Retraining** – scheduler powinien mieć dowód ostatniego wykonania (`RetrainingScheduler.last_run`) i plan kolejnego (`next_run`). Brak retreningu > interwał wymaga otwarcia incydentu risk/compliance.
- **Dane wejściowe** – `FeatureEngineer` wymaga kompletności kolumn OHLCV. Monitoring musi raportować luki oraz nieciągłości w `bot_core.data.ohlcv` przed uruchomieniem inference.

## Specyfikacja `ModelArtifact`

| Pole | Typ | Opis | Źródło |
| --- | --- | --- | --- |
| `model_version` | string | Semantyczna wersja artefaktu (`major.minor.patch`). | `ModelRepository` |
| `created_at` | ISO datetime | Znacznik czasu wygenerowania artefaktu. | `ModelTrainer` |
| `target_scale` | number | Odchylenie standardowe targetu (bps), wykorzystywane do kalibracji prawdopodobieństwa. | Dane treningowe |
| `feature_scalers` | dict[str, {mean: float, stdev: float}] | Parametry normalizacji cech z porcji treningowej. | `ModelTrainer` |
| `training_rows` | int | Liczba rekordów wykorzystanych w treningu. | `ModelTrainer` |
| `validation_rows` | int | Liczba rekordów użytych do walidacji. | `WalkForwardValidator` / `ModelTrainer` |
| `test_rows` | int | Liczba rekordów zestawu testowego (jeśli dotyczy). | `ModelTrainer` |
| `metrics.train` | dict[str, float] | Zestaw metryk MAE/RMSE/directional accuracy dla zbioru treningowego. | `ModelTrainer` |
| `metrics.validation` | dict[str, float] | Metryki walidacyjne (puste, jeśli split=0). | `ModelTrainer` |
| `metrics.test` | dict[str, float] | Metryki dla hold-out testu (puste, jeśli split=0). | `ModelTrainer` |
| `metrics.summary` | dict[str, float] | Płaska reprezentacja metryk treningowych wzbogacona o prefiksowane wartości walidacji/testu (np. `validation_mae`, `test_directional_accuracy`) dla zachowania kompatybilności z konsumentami oczekującymi płaskiej mapy. | `ModelTrainer` |
| `decision_journal_entry_id` | string | Identyfikator wpisu w decision journalu powiązany z artefaktem. | Decision journal |

Artefakt powinien być podpisany kryptograficznie oraz przechowywany wraz z `checksums.sha256`, aby umożliwić audyt integralności. Schemat JSON przechowujemy w `docs/schemas/model_artifact.schema.json`, a każde odchylenie wymaga aktualizacji dokumentacji oraz zatwierdzenia compliance.
W CI uruchamiamy walidację `tests/decision/test_model_artifact_schema.py`, która potwierdza zgodność artefaktu z tym schematem (w tym obecność pól `target_scale`, `training_rows` i ustrukturyzowanych metryk).

## Monitoring danych wejściowych

Monitoring skupia się na wykrywaniu anomalii w danych zasilających inference:

- **Kompletność świec OHLCV** – `bot_core.ai.monitoring.DataCompletenessWatcher` monitoruje luki czasowe (`missing_bars`) i raportuje podsumowanie (`missing_ratio`, `ok_ratio`, `status`) zapisywane w `audit/ai_decision/data_quality/<date>.json` poprzez `AIManager.record_data_quality_issues`.
- **Dryf statystyczny cech** – `bot_core.ai.monitoring.FeatureDriftAnalyzer` porównuje rozkłady cech z wykorzystaniem wskaźników PSI oraz testu KS. Wyniki (`metrics.features`, `distribution_summary.max_psi`, `triggered_features`) trafiają do raportów `audit/ai_decision/drift/<date>.json` generowanych przez `_persist_drift_report`, a `metrics.feature_drift.psi` odzwierciedla maksymalne odchylenie.
- **Automatyczne kontrole pipeline'u** – `AIManager.register_data_quality_check` pozwala zarejestrować kontrole jakości danych (`DataQualityCheck`) wykonywane przy każdym `run_pipeline`. Raporty są zapisywane jako `audit/ai_decision/data_quality/<date>.json` z nazwą `pipeline:<symbol>:<kontrola>`, co scala monitoring z audytem bez dodatkowego kodu.
- **Ścieżki audytu** – `AIManager.record_data_quality_issues` konsoliduje wyniki monitoringu jakości danych (również typu `DataQualityAssessment`) w katalogu `audit/ai_decision/data_quality/`, a `AIManager.run_pipeline` automatycznie tworzy rozszerzone raporty dryfu (`baseline_window`, `production_window`, `metrics.features`, `distribution_summary`) w `audit/ai_decision/drift/` po każdej selekcji modeli. Najnowsze artefakty można pobierać przez helpery `load_latest_*` lub przeglądać listę dostępnych plików dzięki `list_audit_reports`. Każdy zapis raportu generuje dodatkowo wpis w `TradingDecisionJournal` (`event` = `ai_data_quality_report` lub `ai_drift_report`) z metadanymi `report_path`, `status/triggered`, `threshold` oraz kontekstem środowiska, co zapewnia spójność audytu danych z decyzjami operacyjnymi.
- **Walidacja zakresów** – `bot_core.ai.monitoring.FeatureBoundsValidator` weryfikuje, że wartości inference mieszczą się w przedziałach wyznaczonych przez `feature_scalers` ± `σ * multiplier`. Przekroczenia zapisują alert `feature_out_of_bounds` i blokują scoring do momentu podpisu Risk.

Każdy komponent monitoringu ma dedykowany runbook (`docs/runbooks/ai_data_monitoring.md`) opisujący kroki reakcji, eskalację oraz wymagane podpisy compliance.
