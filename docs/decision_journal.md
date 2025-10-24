# Decision Journal

## 2024-10-05 – Standaryzacja metadanych artefaktów AI

- **Decyzja**: rozszerzamy `ModelArtifact` o pola `target_scale`, `training_rows`, `validation_rows`, `test_rows`, `feature_scalers` oraz `decision_journal_entry_id`; metryki train/validation/test przechowujemy w zagnieżdżonych mapach (`metrics.summary/train/validation/test`).
- **Powód**: ułatwienie audytu modeli AI, jednoznaczne rozdzielenie metadanych oraz przygotowanie artefaktów do walidacji JSON Schema w CI.
- **Implementacja**: schemat (`docs/schemas/model_artifact.schema.json`) wraz z testem `tests/decision/test_model_artifact_schema.py`; aktualizacja pipeline'ów i inferencji do korzystania z nowych pól.
- **Ryzyko**: konieczność aktualizacji integracji konsumujących stare klucze (`train_mae`, `validation_mae` itd.). Zapewniono kompatybilność dzięki mapowaniu na `metrics.summary` oraz fallbackom podczas deserializacji.

## 2024-11-12 – Automatyczne walidacje artefaktów i podpisów compliance

- **Decyzja**: włączamy walidację schematu JSON dla każdego `ModelArtifact` generowanego przez harmonogram retreningu oraz blokujemy aktywację inference bez kompletnych podpisów Risk/Compliance na ostatnich raportach `data_quality` i `drift`.
- **Powód**: zamknięcie luk audytowych z roadmapy fazy 2 (`docs/roadmap/phase2_stream_plan.md`) i zapewnienie, że publikowane modele mają zatwierdzony komplet dokumentacji.
- **Implementacja**: moduł `bot_core.ai.validation.validate_model_artifact_schema` wymuszający zgodność z `docs/schemas/model_artifact.schema.json`, konfigurowalna lista ról (`AIManager.set_compliance_sign_off_roles`) oraz bramka `ensure_compliance_sign_offs` (aktywna po `AIManager.set_compliance_sign_off_requirement(True)`, z walidacją ról i ostrzeżeniami o brakach podpisów) wywoływana przez `AIManager` przed podłączeniem inference do DecisionOrchestratora, uzupełniona o `collect_pending_compliance_sign_offs` i `AIManager.get_pending_compliance_sign_offs()` do bieżącego raportowania braków podpisów oraz o automatyczne ładowanie raportów audytu po stronie `ensure_compliance_sign_offs`.
- **Ryzyko**: niezatwierdzone raporty blokują aktywację modeli – wymaga to ścisłej koordynacji z zespołami Risk/Compliance i aktualizacji runbooków (`docs/runbooks/ai_decision_loop.md`, `docs/runbooks/ai_data_monitoring.md`).
