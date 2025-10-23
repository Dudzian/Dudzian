# Runbook: AI Decision Engine Data Monitoring

## Cel
Zapewnienie ciągłości i zgodności monitoringu jakości danych wejściowych modeli decyzyjnych. Runbook opisuje kroki reagowania na alerty `DataCompletenessWatcher`, `FeatureBoundsValidator` oraz dryf statystyczny raportowany przez `_FeatureDriftMonitor`.

## Zakres
- Dane wejściowe do pipeline'u `DecisionModelInference` (cechy kandydatów, wartości OHLCV przetworzone przez `FeatureEngineer`).
- Alerty zapisane w `audit/ai_decision/data_quality/*.json` oraz `audit/ai_decision/drift/*.json`.

## Procedura operacyjna
1. **Codzienny przegląd raportów**
   - Lokalizacja: `audit/ai_decision/data_quality/` (kompletność i zakresy) oraz `audit/ai_decision/drift/` (dryf).
   - Dla szybszego przeglądu użyj helperów `bot_core.ai.load_recent_data_quality_reports()`
     i `bot_core.ai.load_recent_drift_reports()` – zwracają najnowsze wpisy (najnowszy
     raport jako pierwszy) wraz z pełną ścieżką pliku JSON. Następnie zastosuj
     `bot_core.ai.summarize_data_quality_reports(...)` oraz
     `bot_core.ai.summarize_drift_reports(...)`, aby szybko zidentyfikować alerty
     wymagające podpisów Risk/Compliance (sekcja `pending_sign_off`).
   - Zweryfikuj metadane `context.model`, `status` oraz listę naruszeń.
   - Sprawdź sekcję `policy.enforce`. Jeśli `False`, alert jest rejestrowany jako ostrzeżenie konfiguracyjne (np. kategoria wyłączona w `metadata.data_quality.categories`).
   - Potwierdź podpisy Risk/Compliance – jeśli `status == "alert"` i `policy.enforce == True`, obie sekcje muszą zostać zaktualizowane w ciągu 4 godzin.
2. **Diagnoza braków danych**
   - Sprawdź, które cechy znajdują się w `missing_features` lub `null_features`.
   - Odtwórz pipeline danych (skrypt `scripts/backfill.py` lub zadanie ETL) dla konkretnego symbolu/interwału.
   - Jeśli błąd wynika z dostawcy, eskaluj do zespołu Data Platform (kanał `#data-incidents`).
3. **Naruszenie zakresów cech**
   - Odczytaj `violations` i porównaj z oczekiwanymi limitami (metadane modelu, `feature_stats`).
   - Wykonaj backtest na ostatnich 30 dniach, aby zweryfikować stabilność.
   - Jeśli naruszenie dotyczy wielu cech i `policy.enforce == True`, rozważ wstrzymanie modelu (ustaw `DecisionOrchestrator.disable_model(name)`).
   - Jeżeli kategoria została oznaczona jako `policy.enforce == False`, odnotuj w JIRA uzasadnienie konfiguracji (link do decyzji Risk/Compliance).
4. **Dryf cech**
   - Raporty dryfu zawierają `drift_score` i `threshold`. Jeżeli `drift_score > threshold * 1.5`, natychmiast powiadom właściciela modelu.
   - Przygotuj plan roll-back (np. aktywuj poprzednią wersję modelu przez `ModelRepository.set_active_version`).
   - Zbierz przykładowe próbki (ostatnie 50 obserwacji) i dołącz do zgłoszenia w JIRA (`AI-DRIFT`).
5. **Eskalacja**
   - Eskalacja krytyczna (`status == "alert"` i brak danych > 30 min):
     1. Powiadom Incident Commander.
     2. Umieść wpis w `docs/compliance/ai_pipeline_signoff.md` (sekcja *Post-incident review*).
     3. Rozważ przełączenie na strategię awaryjną (`MarketRegime.MEAN_REVERSION`).
6. **Powrót do normalnej pracy**
   - Po rozwiązaniu problemu zaktualizuj `sign_off` w raporcie (Risk/Compliance podpisują się z datą i komentarzem).
   - Użyj helpera `bot_core.ai.update_sign_off(report, role=..., status=...)`, aby zsynchronizować podpisy w pamięci i pliku JSON.
   - Zweryfikuj, że `load_recent_data_quality_reports(category=...)` nie zwraca otwartych
     alertów dla kategorii objętych remediacją.
   - Dodaj streszczenie do `audit/ai_decision/incident_journal.md` (jeśli incydent został otwarty).
   - Zweryfikuj, czy tymczasowe wyłączenia kategorii (`policy.enforce == False`) zostały cofnięte lub formalnie zatwierdzone.

## Checklista on-call
- [ ] Raporty `data_quality` przejrzane do godziny 09:00 UTC.
- [ ] Raporty `drift` przejrzane do godziny 12:00 UTC.
- [ ] Wszystkie alerty posiadają wpis w dzienniku incydentów lub uzasadnione zamknięcie.
- [ ] Potwierdzono poprawność podpisów Risk/Compliance.
- [ ] Wyniki `summarize_data_quality_reports` oraz `summarize_drift_reports` nie zawierają
      zaległych podpisów starszych niż 24h.

## Kontakty
- **Owner modelu:** ml-decisions@firma.example
- **Zespół Data Platform:** data-platform@firma.example
- **Risk:** risk-office@firma.example
- **Compliance:** compliance@firma.example

## Historia zmian
- 2024-04-05 – utworzenie runbooka.
- 2024-04-08 – dodano obsługę polityk egzekwowania na poziomie kategorii monitoringu.
- 2024-04-10 – uzupełniono runbook o podsumowania alertów i walidację podpisów.
