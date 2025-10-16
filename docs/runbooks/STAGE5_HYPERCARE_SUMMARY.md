# STAGE5 – Checklist: Raport zbiorczy hypercare

## Cel
Zebrać wyniki rutyn Stage5 (TCO, rotacje kluczy, szkolenia, raporty zgodności,
SLO, akceptacja OEM) w jednym podpisanym raporcie JSON, który można
zarchiwizować w `var/audit/stage5/hypercare/` oraz dołączyć do decision logu.

## Prerekwizyty
- Wykonane raporty częściowe: `run_tco_analysis.py`, `rotate_keys.py`,
  `log_stage5_training.py`, `validate_compliance_reports.py`,
  `slo_monitor.py` (jeśli wymagane) oraz `run_oem_acceptance.py`.
- Dostęp do skryptu weryfikującego `verify_stage5_hypercare_summary.py` oraz
  klucza HMAC pozwalającego potwierdzić podpis raportu zbiorczego.
- Podpisane artefakty binarne znajdują się w katalogach `var/audit/...`.
- Klucze HMAC do weryfikacji raportów cząstkowych oraz do podpisu podsumowania
  hypercare (np. `secrets/hmac/stage5_hypercare.key`).

## Procedura
1. Zidentyfikuj najnowsze artefakty z bieżącego cyklu hypercare Stage5 i
   zanotuj ich ścieżki.
2. Uruchom skrypt agregujący i podpisujący raport zbiorczy (przykład):
   ```bash
   python scripts/run_stage5_hypercare_cycle.py \
       --tco-summary var/audit/tco/20240101T120000Z/tco_summary.json \
       --tco-signature var/audit/tco/20240101T120000Z/tco_summary.signature.json \
       --tco-signing-key-file secrets/hmac/tco.key \
       --rotation-summary var/audit/stage5/key_rotation/key_rotation_paper_20240101T140000Z.json \
       --rotation-signing-key-file secrets/hmac/key_rotation.key \
       --compliance-report var/audit/stage5/compliance/report_20240101.json \
       --compliance-signing-key-file secrets/hmac/compliance.key \
       --training-log var/audit/stage5/training/training_20240101.json \
       --training-signing-key-file secrets/hmac/training.key \
       --slo-report var/audit/stage6/slo/report.json \
       --slo-signature var/audit/stage6/slo/report.sig \
       --slo-signing-key-file secrets/hmac/slo.key \
       --oem-summary var/audit/acceptance/20240101/summary.json \
       --oem-signature var/audit/acceptance/20240101/summary.signature.json \
       --oem-signing-key-file secrets/hmac/oem_acceptance.key \
       --oem-require-signature \
       --output var/audit/stage5/hypercare/20240101T150000Z/summary.json \
       --signature var/audit/stage5/hypercare/20240101T150000Z/summary.signature.json \
       --signing-key-file secrets/hmac/stage5_hypercare.key \
       --signing-key-id stage5-hypercare
   ```
   W zależności od potrzeb możesz pominąć opcjonalne artefakty (np. SLO lub OEM
   dla konkretnych cykli), jeśli nie są wymagane.
3. Zweryfikuj podpisany raport, aby upewnić się, że artefakty zostały poprawnie
   zarchiwizowane, np.:
   ```bash
   python scripts/verify_stage5_hypercare_summary.py \
       var/audit/stage5/hypercare/20240101T150000Z/summary.json \
       --hmac-key-file secrets/hmac/stage5_hypercare.key \
       --require-signature
   ```
   W przypadku ostrzeżeń/błędów skrypt wskaże brakujące podpisy lub błędy
   struktury raportu.
4. Sprawdź wynik na stdout (`status` powinien wynosić `ok` lub `warn`) oraz
   potwierdź, że raport i podpis zostały zapisane w katalogu docelowym.
5. Dodaj wpis do decision logu z odwołaniem do ścieżki `var/audit/stage5/hypercare/<TS>/`.
6. Po potwierdzeniu raportu Stage5 dołącz go do pełnego przeglądu hypercare
   zgodnie z runbookiem `FULL_HYPERCARE_CHECKLIST.md`.

## Artefakty/Akceptacja
| Artefakt | Lokalizacja | Kryteria akceptacji |
| --- | --- | --- |
| Raport zbiorczy hypercare | `var/audit/stage5/hypercare/<TS>/summary.json` | `overall_status` ≠ `fail`, sekcja `artifacts` zawiera wszystkie wymagane moduły, raport zweryfikowany skryptem `verify_stage5_hypercare_summary.py` |
| Podpis raportu | `var/audit/stage5/hypercare/<TS>/summary.signature.json` | HMAC zweryfikowany lokalnie kluczem `stage5_hypercare` |
| Podpis akceptacji OEM | `var/audit/acceptance/<TS>/summary.signature.json` | Weryfikacja `oem_acceptance.details.signature.verified` = `True` |
| Log CLI | stdout/`logs/hypercare_stage5.log` (opcjonalnie) | Zawiera wynik JSON ze statusem skryptu |
