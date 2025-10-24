# STAGE6 – Checklist: Resilience & Failover Drill

## Cel
Zapewnić, że paczki odpornościowe, plan failover i raporty audytu Stage6 są
aktualne, podpisane HMAC oraz gotowe do użycia w scenariuszu przełączenia.

## Prerekwizyty
- Aktualne klucze HMAC w `secrets/hmac/` zarejestrowane w hypercare.
- Paczki i manifesty w `var/resilience/` przygotowane przez
  `python scripts/export_resilience_bundle.py`.
- Zdefiniowany plan w `data/stage6/resilience/failover_plan.json`.

> **Uwaga:** Wszystkie polecenia CLI zakładają uruchamianie skryptów Stage6 poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie wywołania `./scripts/...` pomijają konfigurację środowiska i nie są wspierane.

## Procedura
1. Zweryfikuj konfigurację i integralność paczek (opcjonalne, przed cyklem):
   ```bash
   python scripts/verify_resilience_bundle.py --bundle var/resilience/bundle.zip \
       --manifest var/resilience/manifest.json --hmac-key secrets/hmac/resilience.key
   ```
2. Uruchom zautomatyzowany cykl hypercare Stage6 generujący paczkę, audyt,
   podsumowanie failover oraz raport self-healing w jednym przebiegu:
   ```bash
   python scripts/run_stage6_resilience_cycle.py --source var/audit/resilience \
       --plan data/stage6/resilience/failover_plan.json \
       --bundle-output-dir var/resilience --audit-json var/audit/resilience/audit_summary.json \
       --failover-json var/audit/resilience/failover_summary.json \
       --self-heal-config configs/resilience_self_heal.json \
       --self-heal-output var/audit/resilience/self_healing_report.json \
       --signing-key-path secrets/hmac/resilience.key --signing-key-id stage6
   ```
   Polecenie zapisze również raporty CSV/podpisy, jeśli wskażesz opcjonalne
   parametry (`--audit-csv`, `--failover-csv`, `--self-heal-signature`).
3. W razie potrzeby użyj pojedynczych narzędzi (`python scripts/audit_resilience_bundles.py`,
   `python scripts/failover_drill.py`) dla dodatkowych scenariuszy DR lub analizy ręcznej.
4. Zweryfikuj statusy `resilience` w PortfolioGovernor i odnotuj ewentualne
   alerty krytyczne.
5. Zaktualizuj decision log Stage6 wpisem referencyjnym do wygenerowanych
   artefaktów.
6. (Opcjonalnie) Użyj `python scripts/run_stage6_hypercare_cycle.py`, aby połączyć
   wyniki Resilience z Observability i Portfolio w jednym raporcie zbiorczym –
   szczegóły w checklistcie Stage6 Hypercare.

## Artefakty/Akceptacja
- `var/audit/resilience/audit_summary.json` (+ podpis HMAC `.sig`).
- `var/audit/resilience/failover_summary.json` oraz `failover_summary.csv`
  (podpisy `.sig`).
- `var/audit/resilience/self_healing_report.json` (+ podpis HMAC `.sig`).
- Zapis w decision logu z linkiem do raportów audytu.
- Potwierdzenie w PortfolioGovernor, że brak blokujących alertów po drillu.
