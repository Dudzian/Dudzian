# Decision Journal

## 2024-10-05 – Standaryzacja metadanych artefaktów AI

- **Decyzja**: rozszerzamy `ModelArtifact` o pola `target_scale`, `training_rows`, `validation_rows`, `test_rows`, `feature_scalers` oraz `decision_journal_entry_id`; metryki train/validation/test przechowujemy w zagnieżdżonych mapach (`metrics.summary/train/validation/test`).
- **Powód**: ułatwienie audytu modeli AI, jednoznaczne rozdzielenie metadanych oraz przygotowanie artefaktów do walidacji JSON Schema w CI.
- **Implementacja**: schemat (`docs/schemas/model_artifact.schema.json`) wraz z testem `tests/decision/test_model_artifact_schema.py`; aktualizacja pipeline'ów i inferencji do korzystania z nowych pól.
- **Ryzyko**: konieczność aktualizacji integracji konsumujących stare klucze (`train_mae`, `validation_mae` itd.). Zapewniono kompatybilność dzięki mapowaniu na `metrics.summary` oraz fallbackom podczas deserializacji.
