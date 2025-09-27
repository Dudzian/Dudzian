# Przegląd kodu bota handlowego – maj 2024

> Cel dokumentu: zebrać aktualny stan repozytorium, zidentyfikować najważniejsze luki względem wymagań
> produktowych i bezpieczeństwa oraz zaproponować etapowy plan rozwoju prowadzący do funkcjonalności
> porównywalnej z komercyjnymi botami (np. CryptoHopper). Poniższe wnioski powstały na bazie pełnej
> analizy struktury `bot_core` oraz historycznego monolitu `KryptoLowca`.

## 1. Ocena struktury i jakości modułów

### 1.1. Warstwa `bot_core`

| Obszar | Pliki | Ocena i rekomendacje |
| --- | --- | --- |
| Adaptery giełdowe | `bot_core/exchanges/*` | Interfejs `ExchangeAdapter` dobrze oddziela środowiska (`live`/`paper`/`testnet`). Implementacje Binance/Kraken/Zonda są kompletne, ale wymagają dalszego hardeningu (np. circuit breaker, cache uprawnień). Należy utrzymywać rozdział kluczy *read-only* i *trading* oraz rozszerzyć testy integracyjne o symulację rate limitów. |
| Warstwa danych | `bot_core/data/base.py`, `bot_core/data/ohlcv/*` | Proces backfillu i cache SQLite spełnia założenia bezkosztowego pozyskania OHLCV. Brakuje jeszcze eksportu do Parquet oraz schedulerów aktualizacji intraday. Zalecane dodanie walidacji integralności danych i metryk (liczba nowych świec). |
| Strategie | `bot_core/strategies/*` | Strategia `DailyTrendMomentumStrategy` odwzorowuje trend-following + momentum na D1 i ma testy jednostkowe. Przygotowany interfejs pod przyszłe klasy (mean reversion, arbitraż). Kolejny krok to implementacja modułu walk-forward i parametryzacji per instrument. |
| Ryzyko | `bot_core/risk/*` | `ThresholdRiskEngine` egzekwuje limity dziennych strat, maksymalną liczbę pozycji oraz poziom dźwigni dla profili konserwatywny/zbalansowany/agresywny. Wymagane dopracowanie modułu raportowania naruszeń i archiwizacji stanu (logi append-only + backupy). |
| Egzekucja | `bot_core/execution/*` | `PaperTradingExecutionService` zapewnia realistyczną symulację z poślizgiem i prowizjami. Brakuje jeszcze modułu `LiveExecutionService` z mechanizmami retry/backoff oraz monitorowania SLO. |
| Alerty | `bot_core/alerts/*` | Router i kanały (Telegram, e-mail, SMS, Signal, WhatsApp, Messenger) są gotowe, posiadają audyt i health-check. Do wdrożenia pozostaje integracja z realnymi API (np. webhook Telegrama) oraz obsługa ról odbiorców w przyszłości. |
| Bezpieczeństwo | `bot_core/security/*` | `SecretManager` obsługuje keyring (Windows/macOS/Linux) i zaszyfrowany magazyn plikowy. Wymagane są testy manualne na docelowych OS oraz procedury rotacji co 90 dni z przypomnieniami alertowymi. |
| Runtime | `bot_core/runtime/*` | `bootstrap_environment` i `TradingSession` spinają strategię, ryzyko, egzekucję i alerty. Konieczne jest dodanie integracji z raportami oraz scheduler do codziennego uruchamiania strategii D1. |
| Observability | `bot_core/observability/*` | Prometheus-kompatybilne metryki dostępne, ale trzeba zbudować eksportera/endpoint oraz dashboard (docelowo w desktop UI). |

### 1.2. Historyczny monolit `KryptoLowca`

| Plik | Problemy | Zalecenia |
| --- | --- | --- |
| `trading_gui.py` | Silne sprzężenie GUI z logiką, brak IPC. | Pozostawić jako referencję, dalszy rozwój przenieść do nowej architektury. |
| `exchange_manager.py`, `core/trading_engine.py` | Monolityczne klasy, brak separacji środowisk, zależność od CCXT. | Migracja do `bot_core`, stopniowa deprecjacja modułów. |
| `risk_management.py` | Heurystyki różnią się od nowych profili i limitów. | Zachować do analizy, ale wdrożyć nowy silnik ryzyka w produkcyjnym kodzie. |
| `managers/security_manager.py` | Brak integracji z natywnymi keychainami. | Zastąpić implementacją `bot_core/security`. |
| `data_preprocessor.py` | Przetwarza dane in-memory, brak cache/backfill. | Utrzymywać jedynie do kompatybilności; nowe procesy w `bot_core/data`. |

## 2. Luki względem wymagań biznesowych

1. **Scheduler i orkiestracja** – brak automatycznego harmonogramu uruchamiania strategii D1, raportów dziennych/tygodniowych i rotacji kluczy.
2. **Raportowanie i compliance** – konieczne generowanie CSV/PDF oraz archiwów JSON/Parquet z logami decyzji i audytem (retencja 24 miesiące).
3. **Observability produkcyjne** – trzeba dostarczyć endpoint metryk, agregację logów i definicję SLO (latencja zleceń, fill rate, error rate).
4. **Paper trading poza Binance** – brak symulatorów nakładanych na real-time feed Kraken/Zonda (na wypadek braku pełnego sandboxa).
5. **Testy integracyjne** – potrzebne scenariusze end-to-end (backfill → strategia → risk → paper execution) oraz testy regresyjne przy rotacji konfiguracji.

## 3. Etapowy plan rozwoju

### Etap 0 – Stabilizacja i dokumentacja (1 tydzień)
- Zatwierdzenie architektury (`ADR-001`), uporządkowanie konfiguracji i checklist bezpieczeństwa.
- Przygotowanie skryptów bootstrap/test (`scripts/run_paper_session.py`).
- Testy: `python -m compileall bot_core`, `pytest tests/test_config_loader.py`, manualna weryfikacja keyringów.

### Etap 1 – Integracja Binance (4–5 tygodni)
1. **Adaptery**: finalizacja obsługi błędów sieciowych, rate limitów i retry (spot + futures, testnet/live).
2. **Dane**: rozszerzenie backfillu o eksport Parquet, walidację integralności i harmonogram aktualizacji.
3. **Strategie**: wdrożenie walk-forward + parametryzacja per instrument w YAML.
4. **Ryzyko**: raporty naruszeń, reset dziennych limitów, integracja z alertami.
5. **Egzekucja**: `LiveExecutionService` z politykami retry/backoff oraz monitorowaniem SLO.
6. **Alerty**: połączenie z realnymi API (Telegram webhook, SMTP, SMS provider) i rejestr audytowy.
7. **Paper trading**: smoke test na Binance Testnet + dzienny raport P&L.

### Etap 2 – Kraken (3 tygodnie)
- Adaptery spot/futures z obsługą specyficznych podpisów, mapowania par i IP allowlist.
- Symulator paper tradingu na feedzie REST/websocket, odwzorowanie prowizji i limitów.
- Rozszerzenie konfiguracji o status KYC/AML kont i przypomnienia rotacji kluczy.

### Etap 3 – Zonda (3 tygodnie)
- Adapter spot z podpisem HMAC-SHA512, wsparcie par PLN/EUR oraz mapowania symboli.
- Paper trading na real-time danych, raport porównujący z Binance/Kraken.
- Integracja alertów i risk engine ze specyficznymi limitami fiat.

### Etap 4 – Rozszerzenia strategiczne i raportowanie (6+ tygodni)
- Biblioteka strategii (mean reversion, volatility breakout na 1h/15m), marketplace presetów.
- Moduł raportowania (CSV + PDF + archiwa JSON/Parquet) z retencją 24–60 miesięcy.
- Dashboard w przyszłym desktop UI (metryki, alerty, dzienniki decyzji).

## 4. Pierwszy krok rozwoju (do wykonania po zatwierdzeniu)
1. Utworzyć skrypt `scripts/run_paper_session.py`, który:
   - ładuje `config/core.yaml`,
   - inicjalizuje `bootstrap_environment` dla środowiska paper Binance,
   - uruchamia jedną iterację strategii `DailyTrendMomentumStrategy` na danych z cache,
   - loguje wynik do audytu i konsoli.
2. Przygotować zestaw testów jednostkowych dla nowego skryptu (mock adaptera + risk engine).
3. Uruchomić dry-run na Binance Testnet (papierowy) przed wprowadzeniem jakichkolwiek kont live.

## 5. Rekomendacje bezpieczeństwa i compliance
- Rotacja kluczy API co 90 dni, osobne klucze `read-only` i `trading`, brak uprawnień do wypłat.
- IP allowlist na dwa–trzy adresy (stacja robocza, VPN, host CI) – oddzielnie dla live i paper.
- Maskowanie sekretów w logach, audyt append-only z timestampami UTC, podpisy archiwów.
- Backup konfiguracji i cache na zaszyfrowanych nośnikach, testy odtwarzania środowiska.
- Zgodność z RODO: minimalizacja danych, możliwość usunięcia wpisów dotyczących operatora.

## 6. Wskazówki testowe po każdej iteracji
- ✅ `python -m compileall bot_core`
- ✅ `pytest tests/<zależne moduły>`
- ⚠️ Manualny backtest/paper trading (wyłącznie środowiska demo/testnet)
- ⚠️ Dry-run alertów (mock webhooków/API)

> Uwaga: aż do uzyskania spójności P&L pomiędzy backtestem a paper tradingiem **wszystkie testy
> wykonujemy na środowiskach demo/testnet**. Dopiero po pozytywnym raporcie z etapu 1 rozważamy
> ograniczone wdrożenie live na niewielkiej alokacji kapitału.
