# Sandbox proof pack (Binance / Kraken / OKX)

> Cel: dać powtarzalny, audytowalny dowód jakości execution flow bez udawania wyników produkcyjnych.

## 1) Runbook uruchomień sandboxowych

### 1.1 Założenia i kontrakty
- Pack opiera się o istniejące kontrakty adapterów i managera:
  - `Environment.TESTNET` oraz model `OrderResult(filled_quantity, status, avg_price)`.
  - `ExchangeManager.configure_failover(...)` i przełączanie backendu native ↔ CCXT.
  - monitor limitów/retry (`exchange_rate_limit_monitor_events_total`, `exchange_retry_monitor_events_total`, alert counters).
  - reporter jakości sygnałów (`exchange_signal_fill_ratio`, `exchange_signal_latency_seconds`, statusy `filled|partial|failed`).
- Zakres adapterów sandbox: `binance`, `kraken`, `okx`.

### 1.2 Jawna klasyfikacja coverage/verdict
W tym packu każdy wynik musi mieć jedną z klas:
- `PASS` – test/scenariusz przeszedł,
- `FAIL` – test/scenariusz wykonał się i nie przeszedł,
- `NOT EXECUTED` – nie uruchomiono,
- `SKIPPED` – uruchomiono, ale framework oznaczył jako skip,
- `BLOCKED` – nie da się uruchomić (np. brak sekretów/sieci),
- `ENVIRONMENTAL LIMITATION` – ograniczenie środowiska runnera (np. brak zależności, brak uprawnień).

### 1.3 Evidence base (zweryfikowane node IDs)
Poniższe node IDs istnieją w repo i są bazą dowodową packa:

**Rate-limit / retry (adaptery):**
- `tests/integration/exchanges/test_binance.py::test_binance_spot_rate_limit_and_retry`
- `tests/integration/exchanges/test_kraken.py::test_kraken_spot_rate_limit_and_retry`
- `tests/integration/exchanges/test_okx.py::test_okx_spot_rate_limit`

**Failover (manager):**
- `tests/integration/test_exchange_manager_failover.py::test_exchange_manager_failover_to_ccxt_backend`
- `tests/integration/test_exchange_manager_failover.py::test_exchange_manager_stays_on_ccxt_after_rate_limit`

**Private backend contracts / partial fills:**
- `tests/exchanges/test_ccxt_private_backend_contracts.py::test_create_order_maps_response`
- `tests/exchanges/test_ccxt_private_backend_contracts.py::test_fetch_open_orders_maps_fields`

**Runbook governance sanity:**
- `tests/integration/test_exchange_runbooks.py`

### 1.4 Pre-flight (przed każdym runem)
1. Zweryfikuj lokalne zależności:
   ```bash
   python -m pytest --version
   ```
2. Ustal katalog artefaktów dowodowych:
   ```bash
   export SANDBOX_PROOF_DIR="audit/sandbox/$(date -u +%Y%m%dT%H%M%SZ)"
   mkdir -p "$SANDBOX_PROOF_DIR"
   ```
3. Ustal tryb testów sieciowych:
   - **offline/mock only**: nie ustawiaj `RUN_NETWORK_TESTS`.
   - **real sandbox**: `export RUN_NETWORK_TESTS=1` i dostarcz testnetowe sekrety poza repo.

### 1.5 Uruchomienia bazowe (dowód kontraktowy lokalny)
1. **Kontrakt packa (dokument + template):**
   ```bash
   pytest tests/integration/test_sandbox_proof_pack.py -q
   ```
2. **Runbooks integracyjne:**
   ```bash
   pytest tests/integration/test_exchange_runbooks.py -q
   ```
3. **Adapter evidence (rate-limit/retry):**
   ```bash
   pytest tests/integration/exchanges/test_binance.py::test_binance_spot_rate_limit_and_retry -q
   pytest tests/integration/exchanges/test_kraken.py::test_kraken_spot_rate_limit_and_retry -q
   pytest tests/integration/exchanges/test_okx.py::test_okx_spot_rate_limit -q
   ```
4. **Failover evidence (manager):**
   ```bash
   pytest tests/integration/test_exchange_manager_failover.py::test_exchange_manager_failover_to_ccxt_backend -q
   pytest tests/integration/test_exchange_manager_failover.py::test_exchange_manager_stays_on_ccxt_after_rate_limit -q
   ```
5. **Partial fills / mapping contracts:**
   ```bash
   pytest tests/exchanges/test_ccxt_private_backend_contracts.py::test_create_order_maps_response -q
   pytest tests/exchanges/test_ccxt_private_backend_contracts.py::test_fetch_open_orders_maps_fields -q
   ```

### 1.6 Obecny stan coverage (uczciwy snapshot, aktualizuj per run)
| Area | Binance | Kraken | OKX | Failover manager |
|---|---|---|---|---|
| Stability | PASS/SKIPPED (zależnie od markerów sieciowych) | PASS/SKIPPED (zależnie od markerów sieciowych) | **FAIL (znany w ostatnim runie evidence subset)** | **ENVIRONMENTAL LIMITATION** (brak `pandas` przy uruchomieniu failover testu) |
| Rate limits | PASS/SKIPPED | PASS/SKIPPED | FAIL (patrz wyżej) | n/a |
| Failover | n/a | n/a | n/a | ENVIRONMENTAL LIMITATION |
| Partial fills | dowód pośredni przez kontrakty CCXT | dowód pośredni przez kontrakty CCXT | dowód pośredni przez kontrakty CCXT | dowód pośredni przez kontrakty CCXT |
| Recovery | NOT EXECUTED (wymaga pełniejszych runów i artefaktów watchdog) | NOT EXECUTED | NOT EXECUTED | BLOCKED/ENVIRONMENTAL LIMITATION |

> Ten snapshot **nie jest uniwersalnym wynikiem projektu**, tylko punktem startowym packa i ma być nadpisywany danymi z Twojego konkretnego runu.

### 1.7 (Opcjonalnie) realny sandbox load profile
Uruchamiaj tylko jeśli masz bezpieczne testnet API keys:
```bash
python -m scripts.exchange_load_test BTC/USDT --exchange binance --mode spot --operation ticker --duration 60 --concurrency 4 --testnet
python -m scripts.exchange_load_test BTC/USDT --exchange kraken --mode spot --operation ohlcv --duration 60 --concurrency 2 --testnet
python -m scripts.exchange_load_test BTC/USDT --exchange okx --mode spot --operation order_book --duration 60 --concurrency 2 --testnet
```
Jeśli brak kluczy/seci: oznacz `BLOCKED` lub `ENVIRONMENTAL LIMITATION` (nie `PASS`).

### 1.8 Pakietowanie dowodów
Do `SANDBOX_PROOF_DIR` zapisz:
- stdout/stderr z pytest i skryptów,
- snapshot metryk Prometheus (jeśli dostępny endpoint),
- wyeksportowane raporty jakości sygnałów (`reports/exchanges/signal_quality/*.json`),
- wypełniony raport wg `reports/templates/sandbox_proof_report_template.md`.

---

## 2) Checklist scenariuszy do odpalenia

### A. Stability
- [ ] 3x powtórzenie tego samego zestawu testów bez flakiness.
- [ ] Brak niekontrolowanych wyjątków poza oczekiwanymi scenariuszami fault-injection.
- [ ] Każdy adapter ma jawny status (`PASS|FAIL|NOT EXECUTED|SKIPPED|BLOCKED|ENVIRONMENTAL LIMITATION`).

### B. Rate limits
- [ ] Potwierdzona obsługa 429 / rate limit z retry i backoff.
- [ ] Zarejestrowane eventy monitora limitów (`exchange_rate_limit_monitor_events_total`).
- [ ] Brak lawinowego retry bez ograniczeń.

### C. Failover
- [ ] Native backend fail -> przełączenie na CCXT fallback.
- [ ] Podczas cooldown kolejne zlecenia idą przez fallback.
- [ ] Jeśli test nieprzechodni przez środowisko, oznacz `ENVIRONMENTAL LIMITATION` i podaj blokadę.

### D. Partial fills
- [ ] Częściowe wykonanie aktualizuje `filled_quantity` i `remaining_quantity`.
- [ ] Fill ratio < 1.0 oznaczane jako status `partial` w quality reporterze.
- [ ] Raport zawiera wpływ partial fills na final execution quality.

### E. Recovery
- [ ] Watchdog rejestruje degradację i recovery.
- [ ] Po incydencie system wraca do stanu zdolnego do przyjmowania zleceń.
- [ ] Brak dowodu = `NOT EXECUTED` / `BLOCKED` (nie domniemuj PASS).

---

## 3) Raport/template wyników
- Główny template: `reports/templates/sandbox_proof_report_template.md`.
- Wymagane pola:
  - verdict per adapter × area,
  - test/node reference i artefakt dla każdego verdictu,
  - rationale (1–3 zdania),
  - jawne statusy `PASS|FAIL|NOT EXECUTED|SKIPPED|BLOCKED|ENVIRONMENTAL LIMITATION`.

---

## 4) Lekkie integration scaffolding
- W repo dodany jest test dokumentacyjno-kontraktowy:
  ```bash
  pytest tests/integration/test_sandbox_proof_pack.py -q
  ```
- Test pilnuje, że proof pack wskazuje realne node IDs i wymusza rozróżnienie coverage/limitations.

---

## 5) Definicja metryk sukcesu

### Stability
- **Metric:** pass rate dla zdefiniowanego subsetu testów.
- **Target:** 100% pass dla testów deterministycznych (bez network flakes).

### Rate limits
- **Metrics:**
  - `exchange_rate_limit_monitor_events_total`,
  - `exchange_rate_limit_alerts_total`,
  - `exchange_retry_monitor_events_total`,
  - `exchange_retry_alerts_total`.
- **Target:** retry kontrolowane, brak eskalacji alertów ERROR poza scenariuszem fault-injection.

### Failover
- **Metric:** czas i skuteczność przełączenia backendu.
- **Target:** pierwsza awaria natywnego backendu skutkuje fallbackiem bez utraty zlecenia testowego.

### Partial fills
- **Metrics:**
  - `exchange_signal_fill_ratio`,
  - `exchange_signal_status_total{status="partial"}`,
  - `filled_quantity` vs requested quantity w payloadach orderów.
- **Target:** poprawna klasyfikacja `partial` i poprawna arytmetyka fill ratio.

### Recovery
- **Metrics:**
  - `exchange_watchdog_events_total`,
  - `exchange_watchdog_status`,
  - `exchange_watchdog_degradation_total`.
- **Target:** po ustaniu błędu watchdog wraca do stanu healthy, a recovery time mieści się w uzgodnionym SLO.

## Ograniczenia i uczciwość dowodowa
- Ten pack **nie udaje wyników realnego sandboxu**, jeśli środowisko nie ma połączeń/sekretów.
- Jeśli dowód jest niepełny, wpisz `FAIL`, `NOT EXECUTED`, `SKIPPED`, `BLOCKED` lub `ENVIRONMENTAL LIMITATION`.
- Nie wpisuj `PASS` bez artefaktu i node reference.
