# Checklist: AI Pipeline Risk & Compliance Sign-off

## Cel
Zapewnienie, że pipeline danych i inference modeli decyzyjnych spełnia wymagania regulacyjne przed wdrożeniem oraz po każdym incydencie danych.

## Wymagania wstępne
- Aktualny model posiada raport walidacji (`audit/ai_decision/walk_forward/<data>.json`).
- Monitoring danych generuje raporty w `audit/ai_decision/data_quality/` oraz `audit/ai_decision/drift/`.
- Wykonano przegląd runbooka [`docs/runbooks/ai_data_monitoring.md`](../runbooks/ai_data_monitoring.md).

## Checklist przed wdrożeniem
- [ ] `DecisionModelInference.is_ready == True` dla aktywowanego modelu.
- [ ] Ostatni raport `data_quality` ma status `ok` lub podpis Risk/Compliance z planem remediacji.
- [ ] Wynik `bot_core.ai.load_recent_data_quality_reports(limit=1)` wskazuje brak aktywnych alertów wymagających blokady.
- [ ] `bot_core.ai.summarize_data_quality_reports(...)` nie zawiera zaległych podpisów (`pending_sign_off`).
- [ ] Wszystkie alerty oznaczone `policy.enforce == False` posiadają formalną decyzję (ticket Risk/Compliance).
- [ ] Nie ma otwartych alertów dryfu (`drift_score <= threshold` lub zaakceptowane przez Risk).
- [ ] Dane wejściowe spełniają kryteria kompletności (`missing_features == []`).
- [ ] Przeprowadzono testy regresyjne (`pytest tests/decision/`).
- [ ] Zatwierdzenie Risk: ____________________  Data: __________
- [ ] Zatwierdzenie Compliance: ______________ Data: __________

## Post-incident review
- [ ] Dodano wpis do `audit/ai_decision/incident_journal.md`.
- [ ] Zaktualizowano raporty `data_quality`/`drift` podpisami Risk/Compliance.
- [ ] Podpisy zarejestrowano przy użyciu `bot_core.ai.update_sign_off`, aby raporty JSON zawierały status i datę zatwierdzenia.
- [ ] Zweryfikowano w `bot_core.ai.load_recent_drift_reports()` brak nowych alertów dryfu po wdrożeniu działań naprawczych.
- [ ] `bot_core.ai.summarize_drift_reports(...)` potwierdza brak zaległych podpisów dla alertów przekraczających próg.
- [ ] Zweryfikowano root-cause i wdrożono poprawki w pipeline danych.
- [ ] Potwierdzono stabilność po rollbacku / retreningu.

## Dokumentacja dodatkowa
- Raport monitoringu danych (`audit/ai_decision/data_quality/<data>.json`).
- Raport dryfu (`audit/ai_decision/drift/<data>.json`).
- Potwierdzenie testów (logi `pytest`).

## Historia zmian
- 2024-04-05 – utworzenie checklisty.
- 2024-04-08 – dodano weryfikację wyjątków polityki danych.
- 2024-04-10 – dodano checklistę podsumowań podpisów dla alertów jakości danych i dryfu.
