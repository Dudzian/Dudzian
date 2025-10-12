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

## 3. Zależności
- Strategia arbitrażowa wymaga danych z co najmniej dwóch adapterów giełdowych oraz interfejsów wykonawczych obsługujących ograniczenia RBAC.
- Harmonogram multi-strategy wykorzystuje `DailyTrendController`, `ExecutionService` oraz `ThresholdRiskEngine`; konieczne są stuby w warstwie testowej.
- Profil ryzyka musi dostarczać dane jakościowe (`data_quality`) zgodne z rozszerzonym `telemetry_risk_profiles.yaml`.
- Decision log podpisywany kluczem HMAC musi być obsługiwany przez `verify_decision_log.py` (zależność testowa).

## 4. Definicje ukończenia (Definition of Done)
- **Kod**: strategie, scheduler i integracje przechodzą linters/testy jednostkowe/integracyjne; brak regresji w istniejącym zestawie.
- **Konfiguracja**: `core.yaml` zawiera komplet parametrów nowych strategii oraz profile ryzyka zaktualizowane o limity specyficzne dla mean reversion / arbitrage.
- **Dokumentacja**: dostępne są opisy strategii, plan testów regresyjnych oraz zaktualizowany runbook paper tradingu.
- **Testy**: istnieją testy jednostkowe i integracyjne pokrywające główne scenariusze (sygnały wejścia/wyjścia, scheduler, risk harness).
- **Operacje**: checklisty demo/paper/live zawierają kryteria smoke testów paper i audytów decyzji.

## 5. Kamienie milowe
1. Specyfikacja i konfiguracja (bieżący dokument, aktualizacja `core.yaml`).
2. Implementacja i testy mean reversion + volatility targeting.
3. Implementacja i testy cross-exchange arbitrage + scheduler wielostrate-giczny.
4. Integracja z pipeline’em i aktualizacja runbooków/test planów.

## 6. Ryzyka i mitgacje
- **Brak danych testnetowych**: wykorzystanie stubów danych OHLCV i symulacji rozbieżności cen (testy integracyjne).
- **Niespójna telemetria**: scheduler eksportuje metryki w standardzie `MetricsService` z przypisaniem do profili ryzyka.
- **Ograniczenia RBAC/mTLS**: konfiguracja scheduler-a dopuszcza jedynie kanały gRPC/HTTP2 oraz weryfikację tokenów z `ServiceTokenConfig`.

## 7. Harmonogram wysokiego poziomu
- Tydzień 1: Dostarczenie kodu i testów mean reversion + volatility targeting.
- Tydzień 2: Arbitraż i scheduler, rozszerzenie harnessu ryzyka.
- Tydzień 3: Integracja pipeline’u, runbooki, audyty, finalne regresje.

