# Stage6 – Checklista PortfolioGovernor

## Cel
Zapewnić udokumentowane logowanie decyzji portfelowych Stage6 wraz z
podpisami HMAC oraz integracją z obserwowalnością i Market Intelligence.

> **Uwaga:** Skrypty Stage6 wykonujemy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie uruchamianie `./scripts/...` może pominąć konfigurację środowiska i nie jest wspierane.

## Krok po kroku
1. Uruchom agregację Market Intel dla governora, wskazując docelowy plik
   hypercare:
  ```bash
  python scripts/build_market_intel_metrics.py \
    --environment <środowisko> \
    --governor <nazwa_governora> \
    --output "var/market_intel/market_intel_stage6_core_$(date -u +%Y%m%dT%H%M%SZ).json"
  ```
   Zastąp `--environment`/`--governor` właściwymi wartościami; parametr
   `--output` zachowaj, aby raport trafił do lokalizacji oczekiwanej przez
   Hypercare (domyślny wzorzec nazwy to `market_intel_{governor}_{timestamp}.json`).
   Jeśli konfiguracja Hypercare nie zawiera pola `portfolio.inputs.market_intel`,
   orchestrator sam wybierze najświeższy raport z katalogu `var/market_intel`
   zgodny z wzorcem `market_intel_<governor>_<timestamp>.json`.
2. (Opcjonalnie) wykonaj `PYTHONPATH=. python scripts/run_multi_strategy_scheduler.py --run-once`,
   aby koordynator runtime zapisał świeżą decyzję PortfolioGovernora wraz z
   metadanymi scheduler-a. Koordynator automatycznie wczyta raporty SLO oraz
 Stress Lab ze ścieżek zdefiniowanych w `runtime.multi_strategy_schedulers.*.
  portfolio_inputs`.
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
     --market-intel <raport_market_intel.json_lub_katalog> \
     --slo-report <raport_slo.json> \
     --stress-report <raport_stress_lab.json> \
     --summary var/audit/portfolio/summary.json \
     --summary-csv var/audit/portfolio/summary.csv \
     --summary-signature var/audit/portfolio/summary.sig \
     --signing-key-path secrets/hmac/stage6_portfolio.key \
     --signing-key-id stage6-portfolio
  ```
   Komenda wygeneruje podsumowanie JSON/CSV, podpis HMAC oraz wpis w decision
   logu korzystając z konfiguracji `runtime.portfolio_decision_log`. Parametr
   `--market-intel` może wskazywać katalog lub wzorzec z symbolem
   `<timestamp>`; skrypt automatycznie wybierze najświeższy raport Market Intel
   zgodny z nazwą governora, korzystając także z katalogów przekazanych przez
   `--fallback-dir` (np. `--fallback-dir var/audit/portfolio`).
6. W scenariuszu ręcznym (np. dla dodatkowych eksperymentów) wykonaj nadal
   `python scripts/log_portfolio_decision.py`, aby zarejestrować pojedynczą decyzję
   wraz z własnym podpisem HMAC.
7. Zweryfikuj w logu stdout komunikaty o podpisach HMAC i lokalizacji
   artefaktów.
8. Jeśli konieczny jest natychmiastowy rebalance, przekaż wygenerowane wpisy do
   modułów egzekucji zgodnie z polityką Stage6.
9. (Opcjonalnie) Uruchom `python scripts/run_stage6_hypercare_cycle.py`, aby w jednym
   kroku połączyć wyniki Portfolio, Observability i Resilience – konfigurację
   znajdziesz w checklistcie Stage6 Hypercare.

## Kontrola progów po warsztacie 2024-06-07
10. Zweryfikuj progi Stage6, korzystając z przygotowanego skryptu audytowego i
    od razu zarchiwizuj raport JSON:
    ```bash
    python scripts/verify_stage6_thresholds.py \
      --config config/core.yaml \
      --json-report var/audit/stage6/stage6_thresholds_portfolio.json
    ```
    Skrypt potwierdza wartości Market Intel, Portfolio Governora oraz Stress Lab
    względem warsztatowych uzgodnień (w tym limity strategii i mnożniki sygnałów)
    i zapisuje wynik audytu do pliku, który dołącz do decision journala.
11. Zarchiwizuj output i raport JSON z kroku 10 w decision journalu hypercare.

## Artefakty / Akceptacja
- `var/audit/decision_log/*.jsonl` – wpisy PortfolioGovernora z polem
  `signature` (HMAC).
- `var/audit/decision_log/*.jsonl.sig` – opcjonalne podpisy zbiorcze, jeśli
  stosowane.
- Raporty wejściowe użyte w kroku 4 (allocations, market intel, SLO,
  stress_overrides).
- Adnotacja w decision logu (lub decision journal) o przeglądzie decyzji przez
  operatora Stage6.
