# Runbook: Rollback modelu AI po incydencie dryfu danych

## Cel
Zapewnić bezpieczne i udokumentowane wycofanie modelu AI Decision Engine
po wykryciu dryfu danych lub alertu jakości wymagającego eskalacji.
Runbook zakłada wykorzystanie helperów `bot_core.ai` oraz istniejących
kontroli compliance opisanych w `docs/compliance/ai_pipeline_signoff.md`.

## Warunki uruchomienia
- `bot_core.ai.summarize_drift_reports(...)` lub
  `bot_core.ai.summarize_data_quality_reports(...)` wskazuje wpisy
  `pending_sign_off` dla ról Risk/Compliance albo alert `status == "critical"`.
- Decision journal zawiera zdarzenie `ai_drift_report`/`ai_data_quality_report`
  z tym samym `report_path`, które wymaga działań naprawczych.
- Ryzyko (Risk) zatwierdziło decyzję o rollbacku w checklistcie
  `docs/compliance/ai_pipeline_signoff.md` (sekcja *Post-incident review*).

## Kroki operacyjne
1. **Zabezpiecz bieżący stan**
   - Wyeksportuj aktualny scheduler:
     ```bash
     python - <<'PY'
     from bot_core.ai.audit import load_scheduler_state

     print(load_scheduler_state())
     PY
     ```
   - Zapisz najnowsze raporty audytu:
     ```bash
     python - <<'PY'
     from bot_core.ai import load_recent_drift_reports, load_recent_data_quality_reports

     print(load_recent_drift_reports(limit=1))
     print(load_recent_data_quality_reports(limit=1))
     PY
     ```
   - Dodaj wpis `rollback_requested` w decision journalu:
     ```bash
     python - <<'PY'
     from datetime import datetime, timezone
     from bot_core.runtime.journal import TradingDecisionEvent, JsonlTradingDecisionJournal

     journal = JsonlTradingDecisionJournal('logs/decision_journal/rollback.log')
     event = TradingDecisionEvent(
         event_type='ai_model_rollback_requested',
         timestamp=datetime.now(timezone.utc),
         environment='ai-monitoring',
         portfolio='ai-manager',
         risk_profile='ai-monitoring',
         metadata={'reason': 'data_drift_alert'}
     )
     journal.record(event)
     PY
     ```

2. **Wstrzymaj aktywną wersję**
   - Jeśli inference działa w DecisionOrchestratorze, odłącz ją:
     ```bash
     python - <<'PY'
     from bot_core.ai import AIManager

     manager = AIManager(model_dir='var/models')
     manager.detach_decision_orchestrator()
     PY
     ```
   - W przypadku autotradera ustaw flagę maintenance w konfiguracji
     (`config/auto_trader.yaml`) zgodnie z runbookiem operacyjnym.

3. **Wybierz wersję referencyjną**
   - W repozytorium modeli znajdź ostatni artefakt z podpisami:
     ```bash
     python - <<'PY'
     from pathlib import Path
     from bot_core.ai import ModelRepository

     repo = ModelRepository(Path('models/decision_engine'))
     artifacts = sorted(repo.base_path.glob('*.json'))
     rollback_target = artifacts[-2]  # ostatni stabilny model
     print('Rollback target:', rollback_target)
     PY
     ```
   - Zweryfikuj schemat i podpisy:
     ```bash
     python - <<'PY'
     from pathlib import Path
     from bot_core.ai import (
         load_recent_drift_reports,
         load_recent_data_quality_reports,
         ensure_compliance_sign_offs,
     )
     from bot_core.ai.validation import validate_model_artifact_schema
     from bot_core.ai.inference import ModelRepository

     repo = ModelRepository(Path('models/decision_engine'))
     artifact = repo.load('models/decision_engine/rollback_target.json')
     validate_model_artifact_schema(artifact)
     ensure_compliance_sign_offs(
         data_quality_reports=load_recent_data_quality_reports(limit=5),
         drift_reports=load_recent_drift_reports(limit=5),
     )
     PY
     ```

4. **Przywróć artefakt**
   - Wczytaj model i podłącz inference:
     ```bash
     python - <<'PY'
     from pathlib import Path
     from bot_core.ai import AIManager

     manager = AIManager(model_dir=Path('var/models'))
     manager.load_decision_artifact(
         name='rollback',
         artifact='models/decision_engine/rollback_target.json',
         set_default=True,
     )
     PY
     ```
   - Jeśli korzystasz z DecisionOrchestratora, ponownie go podepnij i potwierdź
     `DecisionModelInference.is_ready`:
     ```bash
     python - <<'PY'
     from bot_core.ai import AIManager
     from bot_core.decision.orchestrator import DecisionOrchestrator
     from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds

     thresholds = DecisionOrchestratorThresholds(
         max_cost_bps=20.0,
         min_net_edge_bps=1.5,
         max_daily_loss_pct=0.05,
         max_drawdown_pct=0.2,
         max_position_ratio=0.8,
         max_open_positions=8,
         max_latency_ms=500.0,
         max_trade_notional=50_000.0,
     )
     orchestrator = DecisionOrchestrator(DecisionEngineConfig(
         orchestrator=thresholds,
         profile_overrides={},
         stress_tests=None,
         min_probability=0.4,
         require_cost_data=False,
         penalty_cost_bps=0.0,
     ))
     manager = AIManager(model_dir='var/models')
     manager.attach_decision_orchestrator(orchestrator, default_model='rollback')
     inference = manager.score_decision_features(
         {'momentum': 0.0, 'volume_ratio': 1.0},
         model_name='rollback',
     )
     print('Rollback inference ok:', inference)
     PY
     ```

5. **Aktualizuj audyt i checklisty**
   - Zapisz raport incydentu w `audit/ai_decision/incident_journal.md` z
     odniesieniami do raportów data-quality i drift.
   - Uzupełnij `docs/compliance/ai_pipeline_signoff.md` (sekcja *Post-incident review*).
   - Dodaj wpis `ai_model_rollback_completed` do decision journalu z linkiem do
     przywróconego artefaktu oraz podpisami Risk/Compliance.

6. **Przywróć scheduler**
   - Ustaw najbliższy retrening po rollbacku:
     ```bash
     python - <<'PY'
     from datetime import datetime, timedelta, timezone
     from bot_core.ai.scheduler import RetrainingScheduler

     scheduler = RetrainingScheduler(interval=timedelta(hours=6))
     scheduler.mark_executed(datetime.now(timezone.utc))
     print('Scheduler next_run:', scheduler.next_run())
     PY
     ```
   - Przeprowadź smoke test pętli decyzyjnej (`scripts/run_decision_engine_smoke.py --mode live --risk-snapshot <ścieżka> --candidates <ścieżka> --tco-report <ścieżka> --output <ścieżka>`; w razie potrzeby testów kontrolnych dostępny jest tryb `paper` z danymi w `data/decision_engine/paper/`).

## Eskalacja
- Jeżeli rollback nie usuwa alertu dryfu, eskaluj do Platform Ops z logami
  `audit/ai_decision/drift/*.json` oraz wpisem decision journalu.
- W przypadku niepowodzenia wczytania artefaktu (błąd walidacji schematu)
  zgłoś incydent do zespołu ML Engineering i rozważ zatrzymanie strategii
  (runbook `docs/runbooks/rollback_multi_strategy.md`).

## Artefakty
- `audit/ai_decision/drift/<timestamp>.json` – raport wywołujący rollback.
- `models/decision_engine/<model>.json` – przywrócony artefakt.
- `logs/decision_journal/rollback.log` – wpisy `ai_model_rollback_requested`
  i `ai_model_rollback_completed`.
- `docs/compliance/ai_pipeline_signoff.md` – podpisy zespołów Risk/Compliance.
