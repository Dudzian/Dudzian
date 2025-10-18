# Plan refaktoryzacji GUI i warstwy backendowej

## Cele
- Rozdzielenie logiki biznesowej od prezentacji, aby uprościć testowanie i wdrożenia 24/7.
- Zapewnienie interfejsu API (REST/WebSocket) możliwego do konsumpcji przez desktop, aplikację webową oraz automatyzację.
- Przygotowanie pod wielo użytkowników, autoryzację i audyt (KYC/AML, ACL).

## Etapy

### Etap 1 – Audyt i przygotowanie kodu
1. **Inwentaryzacja zależności GUI**: spisać funkcje, które dziś wywołują menedżery (AI, Exchange, Risk, Database).
2. **Wydzielenie konfiguracji**: przenieść ustawienia Tkinter (`tk.Variable`) do klas DTO/Pydantic, aby backend mógł je przyjmować w API.
3. **Centralne logowanie i alerty**: wszystkie moduły korzystają z `bot_core.alerts`, co upraszcza przekierowanie zdarzeń do backendu.

### Etap 2 – Backend FastAPI
1. **Serwis `trading-service` (FastAPI)**
   - endpointy `/auth`, `/presets`, `/strategies`, `/orders`, `/alerts`.
   - integracja z `DatabaseManager` i `TradingEngine` jako usługami singleton.
   - middleware uwierzytelniający (token API, w przyszłości OAuth/Keycloak).
2. **WebSocket / SSE** do strumieniowania alertów i ticków strategii.
3. **Zewnętrzny scheduler** (APScheduler/Celery) do zadań okresowych: retraining AI, rollowanie logów, backup.

### Etap 3 – Lekki klient
1. **Desktop**: Tkinter lub Qt ograniczone do prezentacji, komunikacja przez API.
2. **Przeglądarkowy dashboard** (opcjonalnie React + vite) wykorzystujący te same endpointy.
3. **Testy E2E**: scenariusze Playwright/Selenium na sandboxie paper tradingu.

## Zależności i ryzyka
- **Bezpieczeństwo kluczy**: backend przejmuje odpowiedzialność za odszyfrowywanie, dlatego należy stosować `SecurityManager` i separację sieciową.
- **Migracje DB**: przed startem backendu wdrożyć Alembica, aby zmiany schematu nie blokowały deployu.
- **Skalowanie**: backend przygotować pod konteneryzację (Docker Compose/Kubernetes) z osobnym workerem do backtestów.

# Plan pipeline'u backtestu i schedulera strategii

## Wymagania biznesowe
- Obsługa wielu strategii (rule-based, AI, DCA) z rankingiem wyników.
- Backtest typu walk-forward na danych historycznych zapisanych w bazie lub plikach Parquet.
- Automatyczne publikowanie wyników do dashboardu i alertów.

## Architektura docelowa
1. **`BacktestService`** – moduł orchestrujący runy: pobiera dane z `DatabaseManager`, generuje pipeline cech (`data_preprocessor`), uruchamia strategie i risk manager.
2. **`StrategyScheduler`** – kolejka zadań (APScheduler/Celery) planująca: live tick, backtest, retraining, rebalancing.
3. **Repo wyników** – tabele `backtest_runs`, `strategy_results`, `walkforward_segments` + eksport raportów (HTML/PDF).
4. **Integracja alertów** – niepowodzenia schedulera/backtestu trafiają do `alerts` z kategorią `backtest`.

## Etapy wdrożenia
1. **Minimalny pipeline**
   - API do uruchomienia pojedynczego backtestu (symbol, zakres dat, strategia).
   - Zapis wyników do DB + raport CSV.
2. **Walk-forward & optymalizacja**
   - segmentacja danych (train/test) + metryki (Sharpe, Sortino, hit-rate, max DD).
   - automatyczne wybieranie najlepszych hiperparametrów (grid/Optuna).
3. **Scheduler strategii**
   - definicje crontab (`run_every`, `run_on_signal`), integracja z danymi live.
   - możliwość pauzowania/awaryjnego stopu przez GUI/web.
4. **Dashboard i alerty**
   - widoki rankingów, porównanie equity curve vs benchmark.
   - webhooki/SMS/e-mail przy przekroczeniu progów (np. drawdown > 10%).

## Testy i walidacja
- **Testy jednostkowe**: symulacje VaR, sizingu, poprawność schedulerów (mock czasu).
- **Testy integracyjne**: backtest na danych z Binance testnet (np. 1 tydzień BTCUSDT) – weryfikacja PnL, VaR.
- **Testy obciążeniowe**: uruchamianie wielu strategii równolegle, monitoring zużycia CPU/RAM.
- **Tryb demo**: wszystkie nowe funkcje domyślnie w trybie paper/testnet, dopiero po walidacji można odblokować live.

## Następne kroki operacyjne
1. Zebranie wymagań użytkownika (preferowane giełdy, rynki, limity ryzyka) – wpływa na adaptery i scheduler.
2. Przygotowanie backlogu technicznego (Jira/Linear) z zadaniami opisanymi powyżej.
3. Uruchomienie środowiska staging z oddzielną bazą, aby testować backend bez wpływu na produkcję.
