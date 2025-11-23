# Enterprise'owy bot handlowy – blueprint architektury

> Dokument opisuje docelową architekturę i standardy implementacyjne dla w pełni lokalnego bota
> handlowego klasy enterprise z opcjonalnym modułem serwerowym/chmurowym ukrytym za podpisaną
> flagą. Blueprint jest zgodny z aktywnym kodem Stage6 (PySide6/QML) i ma służyć jako referencja
> dla zespołów produktowych, QA oraz bezpieczeństwa przy dalszej rozbudowie.

## Podstawowe założenia
- **100% lokalny core**: trading, AI/ML, cache/baza (SQLite/PostgreSQL) i UI działają na maszynie
  użytkownika. Domyślny proces to `scripts/run_local_bot.py` + klient PySide6/QML.
- **Cloud/serwer – za flagą**: komponent gRPC (`bot_core.cloud.CloudRuntimeService`) startuje
  wyłącznie po spełnieniu podpisanej flagi (`cloud.enabled_signed` w `config/runtime.yaml`).
  Flaga musi być podpisana przeze mnie (developerem) kluczem HMAC/Ed25519; bez niej aplikacja
  odrzuca uruchomienie cloud zarówno z CLI, jak i z UI.
- **Komercyjna dystrybucja**: licencje przypisane do fingerprintu HWID, klucze API szyfrowane
  lokalnie, krytyczne moduły przygotowane do obfuskacji/Cython. Kod, logi i decyzje transakcyjne
  są audytowalne.

## Warstwy systemu
1. **Core tradingowy (`bot_core.trading`, `bot_core.execution`)**
   - Zarządza cyklem decyzji (`DecisionOrchestrator`), realizacją zleceń i synchronizacją pozycji.
   - Obsługuje multi-account i multi-exchange (spot/futures/margin) poprzez abstrakcję
     `ExchangeClient` z retry, rate-limit i backoff.
   - Każda decyzja (sygnał, hedge, grid, DCA) jest logowana i oznaczona kontekstem strategii.

2. **Warstwa danych (`bot_core.data`, `bot_core.market_intel`, `bot_core.database`)**
   - Cache OHLCV/orderbook/tick, lokalny storage (SQLite/PostgreSQL), walidacja jakości feedu i SLA.
   - Samo-naprawiające się źródła: synchronizacja brakujących świec, fallback na alternatywne węzły.

3. **AI/ML (`bot_core.ai`, `bot_core.market_intel.regime`, `bot_core.optimization`)**
   - Pipeline treningowy walk-forward, klasyfikacja reżimu rynku (trend/konsolidacja/volatility),
     propozycje parametrów strategii i explainability w logach.
   - Automatyczna adaptacja presetów i harmonogram retrainingu; modele i metryki trzymane lokalnie.

4. **Strategie – pluginy (`bot_core.strategies.*`)**
   - Abstrakcja `Strategy` + rejestr pluginów (`catalog`, `marketplace`).
   - Pokrycie: trend-following (MA cross, MACD, Ichimoku, Supertrend), mean-reversion (RSI,
     Bollinger), grid/DCA/martingale z limitami bezpieczeństwa, scalping/market-making,
     breakout/volatility (ATR, range), ML-driven i arbitraż (cross-exchange/triangular).
   - Konfiguracja przez UI (QML formularze) i pliki YAML/JSON; przypisywanie per para/giełda,
     strategie kompozytowe i personalizacja użytkownika.

5. **Risk & portfolio (`bot_core.risk`, `bot_core.portfolio`)**
   - Limity ekspozycji (per pozycja, giełda, aktywo), dzienny/tygodniowy max drawdown,
     stop-loss/trailing/TP/time exits, blacklisty i cooldowny po serii strat.
   - Profile portfela z docelowymi alokacjami i rebalancingiem, uwzględniające koszty maker/taker.

6. **Backtest/symulacje (`bot_core.backtest`, `bot_core.simulation`, `bot_core.testing`)**
   - Multi-symbol/multi-timeframe backtesting z symulacją prowizji i poślizgu.
   - Forward/walk-forward, paper trading na żywym rynku, metryki: equity curve, MDD, Sharpe,
     winrate, PF, liczba transakcji.

7. **UI desktop (PySide6/QML – `ui/`)**
   - Nowoczesne QML z efektami blur, motywy dark/light, FontAwesome/SVG, responsywne layouty.
   - Zarządzanie giełdami, strategiami, profilami ryzyka, backtestami, licencją, logami i monitorem
     stanu bota. Podgląd PnL/pozycji/zleceń oraz status feedu/połączeń.

8. **Bezpieczeństwo i licencje (`bot_core.security`, `bot_core.compliance`)**
   - HW fingerprint, licencje przypisane do urządzenia z tolerancją na drobne zmiany.
   - Szyfrowanie kluczy API w storage lokalnym, audyt operacji, opcjonalna obfuskacja Cythonem.

9. **Cloud/serwer (opcjonalny, `bot_core.cloud`)**
   - gRPC API do monitoringu, synchronizacji konfiguracji i ewentualnego zdalnego zarządzania.
   - Domyślnie nieaktywny; uruchomienie wymaga podpisanej flagi i jest możliwe tylko po stronie
     developera. TLS, autoryzacja HMAC/Ed25519, whitelisty HWID/licencji w manifestach.

## Automatyzacja i orkiestracja
- **Tryb automatyczny**: AI wybiera strategię i parametry, przełącza reżimy (scalping/hedge/grid),
  wymusza protective mode po przekroczeniu progów ryzyka.
- **Tryb półautomatyczny**: bot proponuje decyzje i parametry, użytkownik zatwierdza w UI.
- **Explainability**: każda zmiana trybu/parametru zapisuje rationale (features, metryki,
  klasyfikacja reżimu) w dzienniku decyzji.

## Dystrybucja i operacje
- Instalatory Windows/macOS/Linux, aktualizacje offline, podpisywanie releasów.
- Runbooki SLA feedu i HyperCare; alerting do kanałów zewnętrznych (Telegram/Signal/e-mail).
- Testy jednostkowe i integracyjne (`pytest`, `mypy`, `ruff/flake8`, `black`, `isort`),
  observability (telemetria UI, health probes, readiness pliki/STDOUT dla cloud).

## Minimalne wymagania implementacyjne (checklista dewelopera)
- [ ] Utrzymać pełną separację core ↔ UI ↔ cloud; cloud startuje tylko po pozytywnej walidacji
      `validate_runtime_cloud_flag`.
- [ ] Stosować typowanie PEP 484 + walidację danych (pydantic/dataclasses) w nowych modułach.
- [ ] Każdy plugin strategii musi rejestrować się w katalogu i emitować audyt decyzji.
- [ ] Nowe integracje giełdowe implementować przez `ExchangeClient` z polityką retry/rate-limit.
- [ ] Backtesty i symulacje muszą raportować koszty i poślizg oraz zapisywać equity curve.
- [ ] UI QML utrzymywać w zgodzie z design systemem (blur + motywy + ikony SVG/FA).
- [ ] Każda funkcja włączająca cloud ma być za flagą developer-only oraz logować próbę użycia.

## Włączenie modułu chmurowego (tylko dla developera)
1. W `config/runtime.yaml` ustaw sekcję `cloud.enabled_signed` wskazującą na
   `var/runtime/cloud_flag.json` i `var/runtime/cloud_flag.sig`.
2. Podpisz payload flagi HMAC/Ed25519 prywatnym kluczem developerskim (patrz README – przykłady
   generowania podpisu). Dystrybucje produkcyjne nie zawierają klucza, dlatego użytkownicy końcowi
   nie mogą aktywować chmury.
3. Uruchom `python scripts/run_cloud_service.py --config config/cloud/server.yaml --emit-stdout`
   lub `scripts/run_local_bot.py --enable-cloud-runtime --cloud-client-config config/cloud/client.yaml`.
4. Logi bezpieczeństwa (odrzucone/zaakceptowane próby) zapisują się w `logs/security_admin.log`.

## Rozszerzalność
- **Marketplace presetów**: publikowane jako manifesty YAML z metadanymi, podpisami i metrykami;
  UI umożliwia import/eksport i inspekcję bezpieczeństwa.
- **Integracje zewnętrzne**: webhooki, gRPC streams i pluginy Python (entrypoints) – wszystko
  sandboxowane przez polityki bezpieczeństwa i konfigurację licencyjną.
- **Monitoring offline**: eksport metryk i decision journal do plików, integracja z notebookami
  bez potrzeby łączenia z chmurą.

## Minimalny plan wdrożeniowy (MVP → GA)
1. **MVP lokalne**: kluczowe strategie spot/futures, risk engine, paper trading, podstawowy UI.
2. **Beta**: AI/ML regime classifier + automatyczna adaptacja presetów, backtesting multi-symbol,
   dashboard PnL i alerty SLA feedu.
3. **RC/GA**: pełna lista strategii, marketplace presetów, harmonogram retrainingu, compliance
   (licencje/HWID), opcjonalny cloud z ręczną akceptacją developera.
