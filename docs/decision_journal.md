# Decision Journal

## 2024-10-05 – Standaryzacja metadanych artefaktów AI

- **Decyzja**: rozszerzamy `ModelArtifact` o pola `target_scale`, `training_rows`, `validation_rows`, `test_rows`, `feature_scalers` oraz `decision_journal_entry_id`; metryki train/validation/test przechowujemy w zagnieżdżonych mapach (`metrics.summary/train/validation/test`).
- **Powód**: ułatwienie audytu modeli AI, jednoznaczne rozdzielenie metadanych oraz przygotowanie artefaktów do walidacji JSON Schema w CI.
- **Implementacja**: schemat (`docs/schemas/model_artifact.schema.json`) wraz z testem `tests/decision/test_model_artifact_schema.py`; aktualizacja pipeline'ów i inferencji do korzystania z nowych pól.
- **Ryzyko**: konieczność aktualizacji integracji konsumujących stare klucze (`train_mae`, `validation_mae` itd.). Zapewniono kompatybilność dzięki mapowaniu na `metrics.summary` oraz fallbackom podczas deserializacji.

## 2024-11-12 – Automatyczne walidacje artefaktów i podpisów compliance

- **Decyzja**: włączamy walidację schematu JSON dla każdego `ModelArtifact` generowanego przez harmonogram retreningu oraz blokujemy aktywację inference bez kompletnych podpisów Risk/Compliance na ostatnich raportach `data_quality` i `drift`.
- **Powód**: zamknięcie luk audytowych z roadmapy fazy 2 (`docs/roadmap/phase2_stream_plan.md`) i zapewnienie, że publikowane modele mają zatwierdzony komplet dokumentacji.
- **Implementacja**: moduł `bot_core.ai.validation.validate_model_artifact_schema` wymuszający zgodność z `docs/schemas/model_artifact.schema.json` oraz bramka `ensure_compliance_sign_offs` wywoływana przez `AIManager` przed podłączeniem inference do DecisionOrchestratora.
- **Ryzyko**: niezatwierdzone raporty blokują aktywację modeli – wymaga to ścisłej koordynacji z zespołami Risk/Compliance i aktualizacji runbooków (`docs/runbooks/ai_decision_loop.md`, `docs/runbooks/ai_data_monitoring.md`).

## 2024-11-18 – Raporty walk-forward z podpisami i podsumowaniami

- **Decyzja**: rozszerzamy audyt retreningu o podpisy Risk/Compliance na raportach walk-forward oraz udostępniamy agregaty jakości (`summarize_walk_forward_reports`) w celu szybkiej oceny gotowości modeli.
- **Powód**: roadmapa fazy 2 wymaga dokumentowania walidacji walk-forward i egzekwowania podpisów przed promocją modeli do DecisionOrchestratora.
- **Implementacja**: `bot_core.ai.audit.save_walk_forward_report` zapisuje blok `sign_off`, `load_recent_walk_forward_reports` zwraca najnowsze raporty z metadanymi, a `summarize_walk_forward_reports` oblicza średnie MAE/kierunkowość oraz wskazuje brakujące podpisy. Runbook `docs/runbooks/ai_decision_loop.md` został zaktualizowany o nowe kroki kontrolne.
- **Ryzyko**: brak aktualnych podpisów blokuje pipeline – operatorzy muszą korzystać z `bot_core.ai.update_sign_off`, aby synchronizować wpisy z checklistą `docs/compliance/ai_pipeline_signoff.md`.
## 2024-11-21 – Procedura rollbacku modeli po incydentach dryfu

- **Decyzja**: dokumentujemy operacyjny rollback modeli AI przy użyciu
  runbooka [`docs/runbooks/ai_model_rollback.md`](runbooks/ai_model_rollback.md)
  oraz integrujemy go z checklistą compliance i runbookami monitoringu.
- **Powód**: roadmapa fazy 2 wymaga kompletnej ścieżki wycofania modeli,
  aby alerty dryfu danych mogły być zamknięte z podpisami Risk/Compliance.
- **Implementacja**: dodano runbook rollbacku, rozszerzono `docs/runbooks/ai_data_monitoring.md`
  i `docs/runbooks/ai_decision_loop.md` o odwołania do procedury oraz
  checklistę `docs/compliance/ai_pipeline_signoff.md` o krok potwierdzający
  wykonanie rollbacku.
- **Ryzyko**: ręczne kroki wymagają dyscypliny operacyjnej; brak logów
  `ai_model_rollback_*` w decision journalu blokuje zakończenie incydentu.
