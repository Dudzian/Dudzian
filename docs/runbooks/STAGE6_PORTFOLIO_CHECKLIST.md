# Stage6 – Checklista PortfolioGovernor

## Cel
Zapewnić udokumentowane logowanie decyzji portfelowych Stage6 wraz z
podpisami HMAC oraz integracją z obserwowalnością i Market Intelligence.

## Krok po kroku[^noexec]

1. Uruchom agregację Market Intel dla governora (`python scripts/build_market_intel_metrics.py`).
2. (Opcjonalnie) wykonaj polecenie
   `python scripts/run_multi_strategy_scheduler.py --run-once`, aby koordynator
   runtime zapisał świeżą decyzję PortfolioGovernora wraz z
   metadanymi scheduler-a. Koordynator automatycznie wczyta raporty SLO oraz
   Stress Lab ze ścieżek zdefiniowanych w
   `runtime.multi_strategy_schedulers.*.portfolio_inputs`.
3. Wygeneruj bieżący raport SLO (`python scripts/slo_monitor.py`) z podpisem HMAC.
4. Przygotuj plik alokacji (np. `var/audit/allocations/<data>.json`) z wagami
   aktywów przekazanymi z runtime lub Stress Lab oraz raport Stress Lab
   (`var/audit/stress_lab/<data>.json`) jeśli generował overridy.
5. Uruchom automatyczny cykl hypercare portfela (zapisując podpis do osobnego
   pliku):
   ```bash
   python scripts/run_stage6_portfolio_cycle.py \
     --config config/core.yaml \
     --environment <środowisko> \
     --governor <nazwa_governora> \
     --allocations <plik z wagami> \
     --portfolio-value <wartość_portfela> \
     --market-intel <raport_market_intel.json> \
     --slo-report <raport_slo.json> \
     --stress-report <raport_stress_lab.json> \
     --summary var/audit/portfolio/summary.json \
     --summary-csv var/audit/portfolio/summary.csv \
     --summary-signature var/audit/portfolio/summary.sig \
     --signing-key-path secrets/hmac/stage6_portfolio.key \
     --signing-key-id stage6-portfolio
   ```
   Komenda wygeneruje podsumowanie JSON/CSV, podpis HMAC oraz wpis w decision
   logu korzystając z konfiguracji `runtime.portfolio_decision_log`.
6. W scenariuszu ręcznym (np. dla dodatkowych eksperymentów) wykonaj nadal
   polecenie:
   ```bash
   python scripts/log_portfolio_decision.py \
     --signing-key-path secrets/hmac/stage6_portfolio.key \
     --signing-key-id stage6-portfolio
   ```
   Pozwoli to zarejestrować pojedynczą decyzję wraz z podpisem HMAC i spójnym
   identyfikatorem klucza.
7. Zweryfikuj w logu stdout komunikaty o podpisach HMAC i lokalizacji
   artefaktów.
8. Jeśli konieczny jest natychmiastowy rebalance, przekaż wygenerowane wpisy do
   modułów egzekucji zgodnie z polityką Stage6.
9. (Opcjonalnie) uruchom `python scripts/run_stage6_hypercare_cycle.py`, aby w
   jednym kroku połączyć wyniki Portfolio, Observability i Resilience –
   konfigurację znajdziesz w checklistcie Stage6 Hypercare.

[^noexec]: Repozytorium nie dostarcza plików z ustawionym bitem wykonywalnym, dlatego uruchamiamy je przez interpreter Pythona.

## Artefakty / Akceptacja
- `var/audit/decision_log/*.jsonl` – wpisy PortfolioGovernora z polem
  `signature` (HMAC).
- `var/audit/decision_log/*.jsonl.sig` – opcjonalne podpisy zbiorcze, jeśli
  stosowane.
- Raporty wejściowe użyte w kroku 4 (allocations, market intel, SLO,
  stress_overrides).
- Adnotacja w decision logu (lub decision journal) o przeglądzie decyzji przez
  operatora Stage6.
