# Przepływ trenowania modeli AI

Ten dokument podsumowuje, w jaki sposób komponenty `bot_core.ai.manager` oraz
`bot_core.ai.pipeline` współpracują w procesie trenowania, walidacji i
rejestrowania modeli Decision Engine. Opis jest punktem odniesienia przy
rozszerzaniu orkiestracji modeli oraz wdrażaniu automatycznego retrainingu.

## Warstwa zarządzająca (`bot_core.ai.manager.AIManager`)

1. **Repozytoria artefaktów** – `AIManager` utrzymuje katalog modeli decyzyjnych
   (`configure_decision_repository`) i cache inference per symbol/model. Dzięki
   temu pipeline może bezpośrednio publikować nowe artefakty, a runtime ma
   natychmiastowy dostęp do najnowszych wag.
2. **Harmonogram treningów** – poprzez `register_training_job` menedżer
   integruje `RetrainingScheduler`, walidatory walk-forward oraz dziennik
   decyzji. Każde zadanie zapisuje historię uruchomień, raporty audytowe i
   metadane jakościowe.
3. **Rejestr zespołów modeli** – funkcje `register_ensemble`,
   `snapshot_ensembles` oraz `diff_ensemble_snapshots` pozwalają budować
   wielomodelowe strategie (mean, weighted, max/min) i monitorować zmiany w
   czasie.
4. **Integracja z journalingiem** – zarówno sukcesy, jak i degradacje modeli są
   logowane poprzez `TradingDecisionJournal`, co zapewnia ślad audytowy dla
   decyzji AI.

## Pipeline treningowy (`bot_core.ai.pipeline`)

1. **Przygotowanie danych** – funkcje pomocnicze `_split_training_frame` i
   `_extract_learning_set` dzielą ramkę danych na zbiory train/validation/test
   oraz budują listy cech wykorzystywane przez modele.
2. **Trenowanie modeli bazowych** – `train_gradient_boosting_model` tworzy
   artefakt JSON zawierający metryki (MAE, RMSE, directional accuracy), dane o
   podziale zbiorów, kalibracji predykcji oraz parametry monitora dryfu.
3. **Rejestracja w Decision Orchestrator** – `register_model_artifact` ładuje
   artefakt do repozytorium modeli, aktualizuje metryki w orchestratorze oraz
   przygotowuje inference gotowe do pracy online.
4. **Monitoring jakości danych** – `score_with_data_monitoring` i integracja z
   `DataQualityCheck` umożliwiają rejestrowanie raportów jakości oraz driftu,
   które następnie konsumuje `AIManager` i dziennik decyzji.

## Współpraca komponentów

1. Pipeline zapisuje artefakt treningowy i – opcjonalnie – rejestruje go w
   `AIManagerze`, co aktualizuje aktywne inference.
2. `AIManager` śledzi historię pipeline’u (`PipelineExecutionRecord`), magazynuje
   metryki oraz raporty audytowe i udostępnia je modułom runtime.
3. Harmonogramy retrainingu (`RetrainingScheduler`) są konfigurowane przez
   menedżera i korzystają z pipeline’u do wytworzenia nowych modeli. Po każdym
   uruchomieniu dziennik otrzymuje wpis z kluczowymi metrykami i ew. przyczyną
   degradacji.

Ta architektura umożliwia rozszerzenie pipeline’u o orkiestrację wielu modeli,
konfigurację zespołów/ensembles oraz automatyczny retraining z walidacją
jakości, zachowując spójność z istniejącym journalingiem i audytem.
