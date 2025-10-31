# Runbook: Failover AI → strategie rules-based

Ten runbook opisuje procedurę przełączenia kontrolera handlowego na strategie rules-based, gdy backend AI przechodzi w tryb degradacji.

## Symptomy
- W dzienniku decyzji (`TradingDecisionJournal`) pojawia się zdarzenie `ai_failover` ze statusem `activated`.
- W logach kontrolera widoczny jest komunikat `Pomijam sygnał ... w trybie failover AI`.
- W alertach strategii widoczne są wpisy `signal_skipped` z metadanymi `reason=ai_failover_active`.

## Kroki natychmiastowe
1. **Zweryfikuj kondycję AI**
   - Uruchom polecenie:
     ```bash
     python -m bot_core.cli.ai status
     ```
   - Sprawdź ostatnie metryki w `var/log/ai/health.json`. Pole `failing_models` wskaże modele z degradacją jakości.
2. **Potwierdź przełączenie kontrolera**
   - W dzienniku decyzji upewnij się, że pojawiły się zlecenia z metadanymi `mode=rules`.
   - Jeżeli brak transakcji rule-based, wymuś emisję sygnałów rules (np. przez strategię `mean_reversion_rules`).
3. **Stabilizuj backend AI**
   - Sprawdź logi `logs/ai_manager.log` pod kątem błędów importu modeli (`fallback_ai_models`).
   - Jeżeli problem dotyczy jakości modeli, zaplanuj retraining:
     ```bash
     python scripts/ai_retrain.py --profile default
     ```

## Przywrócenie trybu AI
1. Po ustabilizowaniu modeli uruchom testy regresyjne:
   ```bash
   pytest tests/ai/test_failover.py tests/runtime/test_controller_failover.py
   ```
2. Jeśli wszystkie testy zakończą się powodzeniem, w runtime wykonaj polecenie odświeżenia modeli:
   ```bash
   python scripts/ai_reload_models.py
   ```
3. Obserwuj dziennik decyzji – powinno pojawić się zdarzenie `ai_failover` ze statusem `cleared` oraz sygnały z metadanymi `mode=ai`.

## Checklist po incydencie
- [ ] Zarchiwizowano logi `ai_manager.log` i `runtime_controller.log`.
- [ ] Upewniono się, że scheduler retrainingu jest aktywny (`bot_core/runtime/schedulers.py`).
- [ ] Uaktualniono raport post-mortem w `audit/ai_failover/`.
- [ ] Zaktualizowano wskaźniki obserwowalności (dashboard Stage6) i powiadomiono interesariuszy.
