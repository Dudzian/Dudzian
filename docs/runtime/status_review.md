# Przegląd stanu bota handlowego – warstwy runtime i UI

## Co już działa
- **Strumieniowanie long-poll** – adaptery giełdowe korzystają z neutralnego interfejsu `MarketStreamHandle` i klas bazowych REST, dzięki czemu eliminują zależności po WebSocketach i zachowują zgodność z menedżerem giełd.【F:KryptoLowca/exchanges/interfaces.py†L1-L115】【F:KryptoLowca/exchanges/streaming.py†L1-L120】
- **Autonomiczny AutoTrader** – pętla `_auto_trade_loop` integruje klasyfikator reżimów, orkiestrator decyzji, dziennik audytu i usługę egzekucji, a nowe testy E2E pokrywają scenariusze paper/live z udaną i nieudaną egzekucją.【F:bot_core/auto_trader/app.py†L1-L120】【F:tests/e2e/test_autotrader_autonomy.py†L1-L140】【F:tests/e2e/fixtures/execution.py†L1-L80】
- **UI runtime** – `RuntimeService` udostępnia QML-owi znormalizowane wpisy `TradingDecisionJournal`, a dashboard QML zawiera kartę „Decyzje AI” z odświeżaniem i obsługą błędów.【F:ui/backend/runtime_service.py†L1-L200】【F:ui/qml/dashboard/RuntimeOverview.qml†L1-L120】

## Najważniejsze luki
- **Brak integracji runtime_service ↔ gRPC** – `RuntimeService` ładuje dane bezpośrednio z lokalnego dziennika i nie korzysta z transportu gRPC, więc w środowiskach produkcyjnych UI wciąż nie zobaczy decyzji z procesu backendowego.【F:ui/backend/runtime_service.py†L200-L260】
- **Testy UI zależne od PySide6** – regresyjne testy QML są domyślnie pomijane, bo środowisko nie zapewnia PySide6, co utrudnia automatyczną weryfikację nowych kart dashboardu.【F:tests/ui/test_runtime_overview.py†L1-L40】
- **AutoTrader nadal bazuje na prywatnych metodach** – scenariusze E2E wywołują `_auto_trade_loop` zamiast publicznych API, a wykonywanie decyzji wymaga dalszego czyszczenia zależności (np. `RiskService`, `ExecutionContext`).【F:tests/e2e/test_autotrader_autonomy.py†L60-L140】
- **Monitorowanie strumieni** – nowy `LongPollSubscription` nie raportuje metryk ani statystyk błędów, przez co brak widoczności kondycji połączeń w telemetryce runtime.【F:KryptoLowca/exchanges/streaming.py†L80-L120】

## Priorytetowe poprawki
1. **Integracja decyzji przez gRPC** – należy dodać serwis backendowy, który serializuje wpisy dziennika po gRPC i wykorzystać go w `RuntimeService`, aby UI odczytywało dane z procesu runtime zamiast lokalnych stubów.【F:ui/backend/runtime_service.py†L200-L260】
2. **Stabilizacja testów UI** – zapewnić binaria PySide6 (np. przez wheel w repo lub kontener testowy) i zautomatyzować uruchamianie `tests/ui/test_runtime_overview.py`, aby nowe widoki były realnie testowane.【F:tests/ui/test_runtime_overview.py†L1-L40】
3. **Uporządkowanie API AutoTradera** – udostępnić publiczną metodę wyzwalającą pojedynczy cykl decyzyjny oraz zredukować zależności od atrybutów `_execution_context`/`_schedule_mode`, co uprości dalszą automatyzację i testowanie.【F:bot_core/auto_trader/app.py†L1-L120】

## Proponowane sprinty
### Sprint A – Integracja decyzji przez gRPC
- Dodać endpoint w `bot_core/api/server` zwracający wpisy `TradingDecisionJournal` (limit, filtrowanie).【F:bot_core/api/server.py†L1-L120】
- Rozszerzyć `RuntimeService`, aby pobierał dane z gRPC i przełączał się na lokalny dziennik tylko w trybie offline.【F:ui/backend/runtime_service.py†L200-L260】
- Uzupełnić testy integracyjne o scenariusz `grpc_decision_feed`, pokrywający serializację i obsługę błędów połączenia.【F:tests/integration/test_grpc_transport.py†L1-L120】

### Sprint B – Automatyzacja testów UI
- Przygotować zależność PySide6 w środowisku CI (np. paczka wheel lub kontener z Qt).【F:tests/ui/test_runtime_overview.py†L1-L40】
- Zaktualizować testy QML tak, aby korzystały z mocku `RuntimeService` i weryfikowały przełączanie kart oraz komunikaty błędów.【F:tests/ui/test_runtime_overview.py†L40-L160】
- Dodać raport z uruchomień UI do pipeline’u (artefakt zrzutów ekranów / logów QML).【F:ui/qml/dashboard/RuntimeOverview.qml†L1-L200】

### Sprint C – Uprawnienie API AutoTradera
- Zapewnić publiczne API `run_decision_cycle()` delegujące do dotychczasowej logiki `_auto_trade_loop` bez przecieków atrybutów prywatnych.【F:bot_core/auto_trader/app.py†L1-L120】
- Zrefaktoryzować `_build_trader` i testy E2E, aby używały nowego API oraz jawnego `ExecutionContext`.【F:tests/e2e/test_autotrader_autonomy.py†L60-L200】
- Podłączyć rejestrowanie metryk strumieni i decyzji do `MetricsRegistry`, aby dashboard mógł raportować błędy z long-pollingu i wykonania zleceń.【F:bot_core/auto_trader/app.py†L1-L120】【F:KryptoLowca/exchanges/streaming.py†L80-L120】
