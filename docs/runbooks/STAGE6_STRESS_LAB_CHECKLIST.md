# Stage6 – Stress Lab & Kalibracja progów

Lista kontrolna dla operatorów hypercare obejmująca uruchomienie Stress Lab,
kalibrację progów płynności/latencji oraz archiwizację podpisanych artefaktów.

> **Uwaga:** Skrypty Stage6 uruchamiamy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie `./scripts/...` nie są wspierane, aby zachować właściwe zależności i konfigurację środowiska.

## Przygotowanie
- [ ] Upewnij się, że raport Paper Labs (`risk_simulation_report.json`) został
      zaktualizowany dla bieżącego tygodnia.
- [ ] Wygeneruj świeży raport Market Intelligence dla aktywów governora:
      `python scripts/build_market_intel_metrics.py --environment <env> --governor <gov> --interval 1h --lookback-bars 168 --output var/audit/stage6/market_intel.json`.
- [ ] Zweryfikuj dostępność klucza HMAC w magazynie `secrets/hypercare/stage6_hmac.key`.

## Kalibracja progów
- [ ] Uruchom kalibrację progów Stress Lab:
      `python scripts/calibrate_stress_lab_thresholds.py --market-intel var/audit/stage6/market_intel.json --risk-report var/audit/stage6/risk_simulation_report.json --config config/core.yaml --governor <gov> --output-json var/audit/stage6/stress_lab_calibration.json --output-csv var/audit/stage6/stress_lab_calibration.csv --signing-key secrets/hypercare/stage6_hmac.key --signing-key-id stage6`.
- [ ] Jeżeli governorem zarządzamy szeroki koszyk aktywów, można skorzystać z
      automatycznej segmentacji wolumenowej:
      `python scripts/calibrate_stress_lab_thresholds.py --market-intel ... --volume-buckets 3 --volume-min-symbols 2 --volume-name-prefix liq --output-json ...`.
- [ ] Zweryfikuj komunikat końcowy skryptu i potwierdź obecność plików JSON, CSV
      oraz `.sig`.
- [ ] Przejrzyj wartości progów (szczególnie segmenty o niskiej płynności) i
      w razie potrzeby zasięgnij opinii zespołu tradingowego.
- [ ] Potwierdź, że w `config/core.yaml:stress_lab.thresholds` obowiązują wartości
      z warsztatu 2024-06-07: płynność 0.55, spready 45 bps, wolatylność 0.8,
      sentyment 0.50, funding 25 bps, latency 150 ms, blackout 40 min,
      dyspersja 50 bps.

## Uruchomienie Stress Lab
- [ ] `python scripts/run_stress_lab.py --risk-report var/audit/stage6/risk_simulation_report.json --config config/core.yaml --governor <gov> --output-json var/audit/stage6/stress_lab_report.json --output-csv var/audit/stage6/stress_lab_insights.csv --overrides-csv var/audit/stage6/stress_lab_overrides.csv --signing-key secrets/hypercare/stage6_hmac.key --signing-key-id stage6` (subkomenda `evaluate` jest opcjonalna – skrypt doda ją automatycznie).
- [ ] Zweryfikuj liczbę insightów i overridów w komunikacie końcowym.
- [ ] Jeśli pojawiły się rekomendacje `critical`, powiadom właściciela portfela i
      zaplanuj dodatkowe obserwacje.

## Artefakty / Akceptacja
- [ ] `var/audit/stage6/stress_lab_calibration.json`
- [ ] `var/audit/stage6/stress_lab_calibration.csv`
- [ ] `var/audit/stage6/stress_lab_calibration.json.sig`
- [ ] `var/audit/stage6/stress_lab_report.json`
- [ ] `var/audit/stage6/stress_lab_insights.csv`
- [ ] `var/audit/stage6/stress_lab_overrides.csv`
- [ ] `var/audit/stage6/stress_lab_report.json.sig`
- [ ] Potwierdzenie review progów (notatka w decision logu Stage6).
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
   - [ ] Zasilono `market_metrics.sqlite` używając buildera Stage6:
         `python scripts/build_market_intel_metrics.py --config config/core.yaml --sqlite-path var/stage6/market_intel/market_metrics.sqlite --output-dir var/audit/stage6/market_intel --manifest var/audit/stage6/market_intel/manifest.json --populate-sqlite --sqlite-provider <module:callable>` (provider korzysta z adapterów `bot_core.exchanges/*`).
         Po wykonaniu polecenia sprawdzono komunikat o weryfikacji sum kontrolnych.
   - [ ] Uruchomiono `python scripts/build_market_intel_metrics.py --config config/core.yaml` lub workflow CI publikujący metryki
         Market Intelligence (weryfikacja manifestu `var/audit/stage6/market_intel/manifest.json`).
   - [ ] Brak danych traktowany jest jako błąd blokujący – potwierdzono, że wszystkie wymagane JSON-y Stage6 istnieją. Tryb syntetyczny dostępny jest wyłącznie w środowiskach testowych po ustawieniu zmiennej `STRESS_LAB_ALLOW_SYNTHETIC_FALLBACK`; każdorazowe użycie zostało odnotowane w decision logu.
2. **Uruchomienie Stress Lab**
   - [ ] Ustawiono zmienną środowiskową `STRESS_LAB_SIGNING_KEY` (lub przekazano ścieżkę klucza przez CLI).
   - [ ] Uruchomiono `python scripts/run_stress_lab.py --config config/core.yaml` (subkomendę `run` można pominąć) lub pipeline CI `Stage6 stress lab`.
   - [ ] Raport został wygenerowany w katalogu wskazanym w `config.core.yaml:stress_lab.report_directory`.
3. **Weryfikacja wyników**
   - [ ] Raport JSON zawiera wszystkie scenariusze z konfiguracji i brak błędów deserializacji.
   - [ ] Brak wpisów w polu `failures` dla scenariuszy lub – w razie naruszeń – przeprowadzono procedurę eskalacji.
   - [ ] Plik podpisu `.sig` zawiera algorytm `HMAC-SHA256` i identyfikator klucza.
   - [ ] Porównano sekcję `thresholds` oraz overrides scenariusza `exchange_blackout_and_latency`
         (latency 180 ms, blackout 55 min) z wynikami raportu i zanotowano status w decision logu.
4. **Integracja z pipeline’em demo→paper→live**
   - [ ] Dołączono raport i podpis do artefaktów release’u (CI lub ręczne archiwum `var/audit/acceptance/<TS>`).
   - [ ] Zaktualizowano decision log o wynik Stress Lab (status, timestamp, operator).
5. **Akceptacja**
   - [ ] Operator L2 zatwierdził raport i potwierdził brak blokujących naruszeń progów Stage6.
   - [ ] Checklistę podpisano i dołączono do pakietu release’owego.

## Automatyczna weryfikacja progów (dodatek)
- [ ] Użyj wspólnego skryptu audytowego, aby potwierdzić konfigurację bez
      ręcznego pisania fragmentów Python oraz zarchiwizować wynik audytu:
  ```bash
  python scripts/verify_stage6_thresholds.py \
    --config config/core.yaml \
    --json-report var/audit/stage6/stress_lab/stage6_thresholds_audit.json
  ```
- [ ] Dołącz output skryptu wraz z raportem JSON do
      `var/audit/stage6/stress_lab/stress_lab_report.json`
      jako komentarz audytowy i referencję w decision logu.

## Odniesienia
- `bot_core/risk/stress_lab.py`
- `python scripts/run_stress_lab.py`
- `deploy/ci/github_actions_stage6_stress_lab.yml` *(po wdrożeniu pipeline’u)*
