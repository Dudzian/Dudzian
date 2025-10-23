# Runbook: AI Decision Loop Harmonogram Retrainingu

## Weryfikacja harmonogramu retreningu
1. Zaloguj się na hosta, na którym działa pętla decyzyjna AI (konto serwisowe `ai-runner`).
2. Przejdź do katalogu repozytorium (`/opt/decision-engine`).
3. Odczytaj aktualny stan harmonogramu:
   ```bash
   cat audit/ai_decision/scheduler.json | jq
   ```
4. Sprawdź pola:
   - `version` – wersja schematu pliku (powinna wynosić `5`).
   - `last_run` – ostatnie zakończone retreningi (UTC).
   - `next_run` – planowana kolejna aktualizacja modeli.
   - `interval` – długość interwału (w sekundach).
   - `updated_at` – znacznik czasu ostatniej aktualizacji stanu (UTC, zwykle równy `last_run`).
   - `last_failure` – czas ostatniej nieudanej próby retreningu (UTC, `null` jeśli brak).
   - `failure_streak` – liczba kolejnych niepowodzeń od ostatniego sukcesu (0 oznacza brak problemów).
   - `last_failure_reason` – skrócony opis błędu zwrócony przez pipeline treningowy.
   - `cooldown_until` – czas blokady kolejnych prób po awarii (UTC). Jeśli ustawiony, scheduler nie podejmie treningu przed tą datą.
   - `paused_until` – koniec ręcznej pauzy (UTC). Do tej chwili harmonogram nie rozpocznie nowego retreningu.
   - `paused_reason` – opis powodu pauzy ustawionej przez SRE/ML Ops.
5. Jeżeli `last_run` jest nieobecne, oblicz je jako `next_run - interval` i zapisz obserwację w dzienniku operacyjnym.
6. Jeżeli wartości są starsze niż oczekiwane SLA lub `cooldown_until` jest w przyszłości, wykonaj dodatkową walidację:
   ```bash
   python - <<'PY'
   import json
   from datetime import datetime, timezone

   with open("audit/ai_decision/scheduler.json", "r", encoding="utf-8") as handle:
       payload = json.load(handle)
   next_run = datetime.fromisoformat(payload["next_run"]).astimezone(timezone.utc)
   print("Następny retrening UTC:", next_run.isoformat())
   PY
   ```

## Procedura reakcji na opóźnienia
1. **Potwierdź opóźnienie:**
   - Zweryfikuj logi `logs/ai_training/*.log` pod kątem błędów w trakcie ostatniego uruchomienia.
   - Sprawdź wpisy w decision journalu (`logs/decision_journal`) dla zdarzeń `ai_retraining` oraz `ai_retraining_failed` z ostatnich 24h.
   - Oceń `failure_streak` i `cooldown_until` w `audit/ai_decision/scheduler.json`; wartości `failure_streak >= 2` lub przyszłe `cooldown_until` wymagają eskalacji do Platform Ops.
2. **Diagnoza:**
   - Uruchom `pytest tests/decision/test_scheduler.py -k retrain` w środowisku CI, aby upewnić się, że logika harmonogramu działa poprawnie.
   - Sprawdź dostępność źródeł danych (`scripts/check_feature_feeds.py`).
3. **Działania naprawcze:**
   - Jeśli problem wynika z braku danych, skoordynuj się z zespołem DataOps i ponów retrening ręcznie:
     ```bash
     PYTHONPATH=. python scripts/run_retraining_job.py --job btc-retrain
     ```
   - W przypadku błędów infrastrukturalnych (GPU, storage), eskaluj do zespołu Platform Ops z logami i metrykami (`/var/log/system.log`).
   - Jeżeli `cooldown_until` blokuje pilny retrening, po usunięciu przyczyny awarii rozważ ręczne wyzerowanie stanu poprzez polecenie serwisowe:
     ```bash
     python - <<'PY'
     from datetime import timedelta
     from bot_core.ai.scheduler import RetrainingScheduler

     scheduler = RetrainingScheduler(interval=timedelta(minutes=30))
     scheduler.mark_executed()
     print("Resetowano failure_streak, cooldown_until:", scheduler.export_state()["cooldown_until"])
     PY
     ```
   - Jeśli konieczna jest tymczasowa zmiana interwału, zaktualizuj go poprzez narzędzie serwisowe:
     ```bash
     python - <<'PY'
     from datetime import timedelta
     from bot_core.ai.scheduler import RetrainingScheduler

     scheduler = RetrainingScheduler(interval=timedelta(minutes=30))
     scheduler.update_interval(timedelta(minutes=45))
     print("Nowy interwał zapisany:", scheduler.export_state()["interval"], "s")
     PY
     ```
   - W przypadku planowanej przerwy technicznej wstrzymaj harmonogram i potwierdź zapis pauzy:
     ```bash
     python - <<'PY'
     from datetime import timedelta
     from bot_core.ai.scheduler import RetrainingScheduler

     scheduler = RetrainingScheduler(interval=timedelta(minutes=30))
     scheduler.pause(duration=timedelta(hours=2), reason="maintenance window")
     print("Pauza aktywna do:", scheduler.export_state()["paused_until"])
     PY
     ```
   - Po zakończeniu okna serwisowego wznowienie harmonogramu:
     ```bash
     python - <<'PY'
     from datetime import timedelta
     from bot_core.ai.scheduler import RetrainingScheduler

     scheduler = RetrainingScheduler(interval=timedelta(minutes=30))
     scheduler.resume()
     print("Pauza usunięta, next_run:", scheduler.export_state()["next_run"])
     PY
     ```
4. **Rejestracja incydentu:**
   - Zanotuj opis zdarzenia oraz wykonane czynności w `docs/incident_log.md`.
   - Upewnij się, że po ręcznej interwencji plik `audit/ai_decision/scheduler.json` został zaktualizowany, `failure_streak` wrócił do `0`, `cooldown_until` jest `null`, a w decision journalu pojawiły się wpisy `ai_retraining_failed` (dla incydentu) i `ai_retraining` (dla skutecznego retreningu).
5. **Powrót do trybu automatycznego:**
   - Po udanym retreningu potwierdź, że `next_run` wskazuje oczekiwaną przyszłą datę.
   - Zamknij incydent w systemie ticketowym (tag `AI-DECISION`).
