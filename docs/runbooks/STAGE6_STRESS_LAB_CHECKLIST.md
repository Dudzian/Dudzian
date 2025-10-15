# Stage6 Stress Lab – lista kontrolna

## Cel
Zweryfikowanie odporności portfela wielostrate-gicznego przy użyciu modułu `bot_core/risk/stress_lab.py` oraz potwierdzenie, że raporty stresowe są generowane i podpisywane zgodnie z wymogami OEM.

## Wymagane artefakty
- Raport JSON: `var/audit/stage6/stress_lab/stress_lab_report.json`
- Podpis HMAC raportu: `var/audit/stage6/stress_lab/stress_lab_report.json.sig`
- Log wykonania CI lub lokalny log operatorski
- Zrzut konfiguracji `stress_lab` z `config/core.yaml`

## Lista kontrolna
1. **Przygotowanie danych**
   - [ ] Zweryfikowano dostępność plików metryk w `data/stage6/metrics/` (BTCUSDT, ETHUSDT lub inne wymagane rynki).
   - [ ] Uruchomiono `python scripts/build_market_intel_metrics.py --config config/core.yaml` lub workflow CI publikujący metryki
         Market Intelligence (weryfikacja manifestu `var/audit/stage6/market_intel/manifest.json`).
   - [ ] W razie braku danych – potwierdzono w runbooku dopuszczalność trybu syntetycznego i zapisano wpis w decision logu.
2. **Uruchomienie Stress Lab**
   - [ ] Ustawiono zmienną środowiskową `STRESS_LAB_SIGNING_KEY` (lub przekazano ścieżkę klucza przez CLI).
   - [ ] Uruchomiono `python scripts/run_stress_lab.py --config config/core.yaml` lub pipeline CI `Stage6 stress lab`.
   - [ ] Raport został wygenerowany w katalogu wskazanym w `config.core.yaml:stress_lab.report_directory`.
3. **Weryfikacja wyników**
   - [ ] Raport JSON zawiera wszystkie scenariusze z konfiguracji i brak błędów deserializacji.
   - [ ] Brak wpisów w polu `failures` dla scenariuszy lub – w razie naruszeń – przeprowadzono procedurę eskalacji.
   - [ ] Plik podpisu `.sig` zawiera algorytm `HMAC-SHA256` i identyfikator klucza.
4. **Integracja z pipeline’em demo→paper→live**
   - [ ] Dołączono raport i podpis do artefaktów release’u (CI lub ręczne archiwum `var/audit/acceptance/<TS>`).
   - [ ] Zaktualizowano decision log o wynik Stress Lab (status, timestamp, operator).
5. **Akceptacja**
   - [ ] Operator L2 zatwierdził raport i potwierdził brak blokujących naruszeń progów Stage6.
   - [ ] Checklistę podpisano i dołączono do pakietu release’owego.

## Odniesienia
- `bot_core/risk/stress_lab.py`
- `scripts/run_stress_lab.py`
- `deploy/ci/github_actions_stage6_stress_lab.yml` *(po wdrożeniu pipeline’u)*
