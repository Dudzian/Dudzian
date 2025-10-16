# Stage6 – Stress Lab & Kalibracja progów

Lista kontrolna dla operatorów hypercare obejmująca uruchomienie Stress Lab,
kalibrację progów płynności/latencji oraz archiwizację podpisanych artefaktów.

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

## Uruchomienie Stress Lab
- [ ] `python scripts/run_stress_lab.py --risk-report var/audit/stage6/risk_simulation_report.json --config config/core.yaml --governor <gov> --output-json var/audit/stage6/stress_lab_report.json --output-csv var/audit/stage6/stress_lab_insights.csv --overrides-csv var/audit/stage6/stress_lab_overrides.csv --signing-key secrets/hypercare/stage6_hmac.key --signing-key-id stage6`.
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
