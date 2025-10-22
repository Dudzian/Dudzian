# Phase 2 – Stream AI Plan

## Kontekst

Faza druga rozwoju strumienia decyzyjnego koncentruje się na przygotowaniu i operacjonalizacji pipeline'u AI przed przekazaniem go autotraderowi. Plan uwzględnia wymagania architektoniczne z `docs/architecture/ai_decision_pipeline.md`, procesy compliance oraz integrację z DecisionOrchestrator.

## Epik 2.1 – Artefakty i walidacja pipeline'u AI

### Cele
- Zapewnienie pełnej transparentności i powtarzalności procesu trenowania modeli.
- Dostarczenie artefaktów modeli wraz z metadanymi umożliwiającymi audyt i reprodukcję.

### Zadania
- [ ] Przygotować `ModelArtifact` dla każdej iteracji retreningu z uzupełnionymi metadanymi `target_scale`, `feature_scalers`, `training_rows`, `validation_rows` oraz metrykami MAE/RMSE (train/validation/test).
- [ ] Zaimplementować i udokumentować proces walidacji walk-forward (`WalkForwardValidator`) wraz z archiwizacją raportów w `audit/ai_decision/` oraz podpisem w decision journalu.
- [ ] Zbudować monitoring danych wejściowych (`bot_core.data.ohlcv`, `FeatureEngineer`) wykrywający braki, luki i dryf dystrybucji cech przed inferencją.
- [ ] Zintegrować checklistę compliance dla artefaktów AI z repozytorium dokumentacyjnym (podpisy Risk/Compliance, referencje do raportów walidacyjnych).
- [ ] Dostarczyć schemat JSON `docs/schemas/model_artifact.schema.json` opisujący pola artefaktu oraz zautomatyzowaną walidację w pipeline CI.

## Epik 2.2 – Operacjonalizacja inference i retreningu

### Cele
- Zapewnienie ciągłości działania inference z zachowaniem kontroli ryzyka.
- Przygotowanie infrastruktury do cyklicznego retreningu i publikacji modeli.

### Zadania
- [ ] Skonfigurować scheduler retreningu (`RetrainingScheduler`) z dowodem ostatniego i kolejnego uruchomienia (`last_run`, `next_run`) oraz eksportem do `audit/ai_decision/scheduler.json`.
- [ ] Włączyć integrację inference (`DecisionModelInference`) z `DecisionOrchestrator`, gwarantując `is_ready == True` przed przekazaniem decyzji do autotradera.
- [ ] Opracować runbook walidacyjny potwierdzający, że `AIDecisionLoop` loguje monitoring kosztów oraz podpisuje decyzje w decision journalu.
- [ ] Zabezpieczyć harmonogram audytów dryfu/danych i checklist compliance, tak aby zatwierdzone podpisy były wymagane przed aktywacją modeli w autotraderze.
- [ ] Przygotować procedurę rollbacku modelu wraz z checklistą akceptacji dla incydentów dryfu danych.

## Harmonogram sprintów

| Sprint | Zakres | Kamień | Uwaga |
| --- | --- | --- | --- |
| 1 | Setup repozytorium modeli + generowanie `ModelArtifact` | Artefakty z kompletnymi metadanymi + podpis w decision journalu | Dostarcza zadanie Epiku 2.1 dot. metadanych artefaktów. |
| 2 | Walidacja walk-forward + monitoring danych | Raporty walidacyjne i alerty dryfu danych w `audit/ai_decision/` | Raporty wymagane przed wejściem w epik 2.2. |
| 3 | Scheduler retreningu + integracja inference ↔️ DecisionOrchestrator | `RetrainingScheduler` z `last_run/next_run` oraz potwierdzony `is_ready` w DecisionOrchestrator | Obejmuje przygotowanie runbooków operacyjnych. |
| 4 | Audyty dryfu/danych + podpisy compliance + go-live autotradera | Zatwierdzone checklisty Risk/Compliance oraz raport audytu dryfu przed przekazaniem modeli autotraderowi | Modele mogą zostać aktywowane dopiero po uzyskaniu podpisów compliance i potwierdzeniu braku anomalii danych. |

## Artefakty wyjściowe
- Repozytorium modeli zawierające wersjonowane `ModelArtifact` wraz z metadanymi i podpisami.
- Raporty z walidacji walk-forward i monitoringu danych zapisane w `audit/ai_decision/`.
- Dokumentacja integracji inference ↔️ DecisionOrchestrator oraz runbook retreningu.
- Checklisty Risk/Compliance podpisane przed aktywacją modeli w autotraderze.
- Schemat `model_artifact.schema.json` z testami CI blokującymi merge przy niezgodnych metadanych.

## Kryteria akceptacji sprintów

| Sprint | Kryterium akceptacji | Powiązane artefakty |
| --- | --- | --- |
| 1 | `ModelArtifact` generowany automatycznie z pełnymi metadanymi, podpisany w decision journalu, zwalidowany schematem JSON w CI. | `models/<version>/artifact.json`, `docs/schemas/model_artifact.schema.json`, wpis w `decision_journal.md` |
| 2 | Raporty walk-forward i monitoring danych posiadają podpis Risk/Compliance, alerty dryfu testowane na danych historycznych. | `audit/ai_decision/walk_forward/<date>.json`, `audit/ai_decision/data_quality/<date>.json`, notatka compliance |
| 3 | Scheduler retreningu i integracja z DecisionOrchestrator potwierdzają gotowość inference oraz runbook reakcji na incydenty. | `audit/ai_decision/scheduler.json`, `docs/runbooks/ai_decision_loop.md` |
| 4 | Audyty dryfu i checklisty compliance zatwierdzone przed go-live autotradera, procedura rollbacku podpisana przez Risk. | `audit/ai_decision/drift/<date>.json`, `docs/compliance/ai_pipeline_signoff.md` |

## Audyty i compliance

- Audyt dryfu danych i walidacja monitoringów muszą zakończyć się co najmniej tydzień przed planowanym przekazaniem modeli do autotradera.
- Podpisy Risk/Compliance są zbierane w `docs/compliance/ai_pipeline_signoff.md` i blokują aktywację modeli do czasu spełnienia checklisty 1–5.
- Decyzja o go-live wymaga review `DecisionOrchestrator` z udziałem zespołów Risk, Platform i Data – protokół przechowywany w `audit/ai_decision/go_live_minutes.md`.
- Każde odchylenie (np. brak raportu dryfu) musi skutkować cofnięciem statusu sprintu 4 oraz eskalacją zgodnie z runbookiem `docs/runbooks/ai_data_monitoring.md`.
