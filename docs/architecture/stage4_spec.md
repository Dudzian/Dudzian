# Specyfikacja Etapu 4 – Rozszerzona biblioteka strategii

## 1. Kontekst
Etap 4 programu rozwoju `bot_core` koncentruje się na budowie zdywersyfikowanej biblioteki strategii oraz wielowarstwowego harmonogramu wykonawczego obsługującego sekwencję demo → paper → live. Wymagane jest zachowanie modularnej architektury (adaptery giełdowe, dane, strategie, ryzyko, egzekucja, alerty oraz runtime) oraz pełna zgodność z ograniczeniami bezpieczeństwa (RBAC, mTLS, pinning certyfikatów, decyzje podpisywane HMAC) przy braku zależności chmurowych i wsparciu dla Windows/macOS/Linux.

## 2. Zakres
- Implementacja trzech nowych strategii: mean reversion, volatility targeting, cross-exchange arbitrage.
- Wprowadzenie scheduler-a wielostrate-gicznego z telemetrią i integracją z pipeline’ami demo/paper/live.
- Aktualizacja konfiguracji rdzenia (`core.yaml`) o nowe parametry profili ryzyka, koszyki instrumentów oraz presety strategii.
- Rozbudowa dokumentacji (strategii, runbooki, plan testów) wraz z checklistami audytowymi.
- Rozszerzenie harnessu testowego silnika ryzyka o scenariusze obejmujące nowe strategie i profile.
- Przygotowanie testów jednostkowych, integracyjnych i regresyjnych umożliwiających utrzymanie pipeline’u demo → paper → live.
- Uzupełnienie biblioteki danych backtestowych o zestawy dla multi-asset/multi-exchange oraz wprowadzenie procedur walidacji jakości (spójność, braki, outliery) skorelowanych z profilami ryzyka, w tym manifest `manifest.yaml`, próbki CSV dla każdej strategii i narzędzie CLI `validate_backtest_datasets.py`.
- Rozbudowa warstwy obserwowalności: nowe metryki scheduler-a i strategii w `telemetry_risk_profiles.py` (progi dla `avg_abs_zscore`, `allocation_error_pct`, `spread_capture_bps`, `secondary_delay_ms`), dashboard Grafany z panelami latencji i skuteczności sygnałów, reguły Alertmanagera dla PnL/ryzyka/opóźnień oraz integracja logów decyzyjnych multi-strategy z centralnym decision logiem.
- Automatyzacja CI: włączenie rozszerzonych testów (pytest, backtesty, smoke CLI `smoke_demo_strategies.py`) do pipeline’u, dostarczenie stubów/fixtures danych i raportów pokrycia z gatingiem jakości dla modułów strategii i scheduler-a.
- Wzmocnienie bezpieczeństwa i compliance: przegląd RBAC/mTLS, aktualizacja schematu decision log JSONL o pola multi-strategy, mini-audyt podpisów HMAC i rotacji kluczy.
- Wsparcie operacyjne i stabilność: playbook L1/L2, szkolenie operatorów, testy obciążeniowe scheduler-a/strategii, monitoring budżetów zasobów oraz procedury awaryjnego wyłączenia/rollbacku.
- Testy wydajności i monitoring zasobów: moduł `scheduler_load_test` z CLI `load_test_scheduler.py`, monitor budżetów `resource_monitor` oraz limity `runtime.resource_limits` konsumowane w audycie bezpieczeństwa.

## 3. Zależności
- Strategia arbitrażowa wymaga danych z co najmniej dwóch adapterów giełdowych oraz interfejsów wykonawczych obsługujących ograniczenia RBAC.
- Harmonogram multi-strategy wykorzystuje `DailyTrendController`, `ExecutionService` oraz `ThresholdRiskEngine`; konieczne są stuby w warstwie testowej.
- Profil ryzyka musi dostarczać dane jakościowe (`data_quality`) zgodne z rozszerzonym `telemetry_risk_profiles.yaml`.
- Decision log podpisywany kluczem HMAC musi być obsługiwany przez `verify_decision_log.py` (zależność testowa).

## 4. Definicje ukończenia (Definition of Done)
- **Kod**: strategie, scheduler i integracje przechodzą linters/testy jednostkowe/integracyjne; brak regresji w istniejącym zestawie; moduły objęte raportem pokrycia z progami gatingu.
- **Konfiguracja**: `core.yaml` zawiera komplet parametrów nowych strategii oraz profile ryzyka zaktualizowane o limity specyficzne dla mean reversion / arbitrage, a biblioteka danych backtestowych jest znormalizowana i powiązana z presetami profili wraz z raportem walidacyjnym `DataQualityValidator`.
- **Dokumentacja**: dostępne są opisy strategii, plan testów regresyjnych, runbook paper tradingu, playbook wsparcia L1/L2 oraz procedury rollbacku.
- **Testy**: istnieją testy jednostkowe, integracyjne, smoke CLI (`smoke_demo_strategies.py`), backtesty i testy obciążeniowe pokrywające główne scenariusze (sygnały wejścia/wyjścia, scheduler, risk harness, latencja, jitter) oraz objęte progami coverage.
- **Operacje i bezpieczeństwo**: checklisty demo/paper/live zawierają kryteria smoke testów paper, audytów decyzji, weryfikację RBAC/mTLS (również dla scheduler-a) oraz monitoring budżetów zasobów (`resource_monitor`, `runtime.resource_limits`); alerty i dashboardy Prometheus/OTEL są zaktualizowane.

## 5. Kamienie milowe
1. Specyfikacja i konfiguracja (bieżący dokument, aktualizacja `core.yaml`, plan danych i obserwowalności).
2. Implementacja i testy mean reversion + volatility targeting wraz z pierwszą wersją zestawów danych.
3. Implementacja i testy cross-exchange arbitrage + scheduler wielostrate-giczny; rozszerzenie harnessu ryzyka i telemetryki.
4. Integracja z pipeline’em i aktualizacja runbooków/test planów; wdrożenie alertów i decision log.
5. Automatyzacja CI/CD, szkolenie operatorów oraz testy obciążeniowe z raportem wydajności.

## 6. Ryzyka i mitgacje
- **Brak danych testnetowych**: wykorzystanie stubów danych OHLCV i symulacji rozbieżności cen (testy integracyjne) oraz znormalizowane snapshoty w repozytorium danych.
- **Niespójna telemetria**: scheduler eksportuje metryki w standardzie `MetricsService` z przypisaniem do profili ryzyka, a dashboardy Prometheus/OTEL są walidowane w smoke teście CLI.
- **Ograniczenia RBAC/mTLS**: konfiguracja scheduler-a dopuszcza jedynie kanały gRPC/HTTP2 oraz weryfikację tokenów z `ServiceTokenConfig`; przeprowadzany jest mini-audyt rotacji kluczy.
- **Degradacja wydajności**: testy obciążeniowe i monitoring budżetu zasobów identyfikują regresje; procedury rollbacku minimalizują MTTR.
- **Alert fatigue**: playbook L1/L2 i parametry alertów PnL/ryzyko/latencja są kalibrowane pod profile ryzyka, z eskalacją tylko w przypadku przekroczenia progów krytycznych.

## 7. Harmonogram wysokiego poziomu
- Tydzień 1: Dostarczenie kodu i testów mean reversion + volatility targeting, przygotowanie schematu danych backtestowych.
- Tydzień 2: Arbitraż i scheduler, rozszerzenie harnessu ryzyka, metryki telemetryczne.
- Tydzień 3: Integracja pipeline’u, runbooki, audyty decision log + RBAC/mTLS.
- Tydzień 4: Automatyzacja CI (smoke CLI, backtesty, coverage) oraz testy obciążeniowe i monitoring budżetów.
- Tydzień 5: Szkolenie operatorów, playbook L1/L2, procedury rollbacku i finalny mini-audyt zgodności.

