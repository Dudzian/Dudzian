# Architektura KryptoŁowcy – stan po Etapie 0

## Przegląd modułów

- **core/trading_engine.py** – serce aplikacji. Orkiestruje dane rynkowe, prognozy AI, zarządzanie ryzykiem i tworzy plany transakcyjne. W Etapie 0 doprowadziliśmy go do spójnego interfejsu (walidacja danych, bezpieczne limity, stop-loss/take-profit).
- **managers/** – adaptery na zewnątrz:
  - `exchange_manager.py` (fasada CCXT/paper),
  - `ai_manager.py` (ładowanie i trening modeli),
  - `risk_manager_adapter.py` (most do rozbudowanego modułu `risk_management.py`),
  - `database_manager.py` (ORM + SQLAlchemy dla logów i pozycji),
  - `security_manager.py` (szyfrowanie kluczy API).
- **trading_strategies.py** – masywny moduł strategii i backtestów (zachowany bez zmian; Etap 1 obejmie jego modularizację).
- **risk_management.py** – kalkulacje limitów ryzyka, zwraca strukturę `PositionSizing` używaną przez adapter.
- **gui/trading_gui.py** (nie modyfikowany w tym kroku) – interfejs desktopowy.

## Najważniejsze decyzje Etapu 0

1. **Stabilizacja TradingEngine** – usunęliśmy wywołania nieistniejących metod (`validate()`), dodaliśmy defensywną obsługę błędów i logiczny przepływ sygnałów AI → limit pozycji → plan z SL/TP. Dzięki temu bot nie generuje już pustych planów ani wyjątków przy braku `ai_threshold_bps`.
2. **Spójne zależności** – `requirements.txt` odzwierciedla teraz realne importy (AI, raporty, integracje). Instalacja w czystym środowisku nie powinna kończyć się brakującymi pakietami.
3. **Testy regresyjne** – zaktualizowane `tests/test_trading_engine.py` pokrywają najważniejsze scenariusze (brak sygnału, brak shortów, limit pozycji, brak konfiguracji). To baza do dalszej refaktoryzacji.
4. **Dokumentacja architektoniczna** – niniejszy plik opisuje moduły i decyzje; kolejne etapy będą rozszerzać sekcję o pipeline danych, licencjonowanie i integracje giełdowe.

## Plan rozbudowy – Etap 1

### Cele nadrzędne

1. **Strategie i sygnały** – uporządkować moduł `trading_strategies.py` tak, aby można było niezależnie rozwijać strategie trend-following, mean reversion i momentum oraz łatwo je testować.
2. **Zarządzanie ryzykiem MVP** – dostarczyć prosty, ale kompletny zestaw zasad (pozycja %, dzienny limit straty, stop-loss/take-profit), który można później rozszerzać.
3. **Symulacja/paper trading** – umożliwić pełny cykl działania bota bez prawdziwych środków dzięki backendowi paper oraz spójnym logom.
4. **Fundamenty licencjonowania** – przygotować bezpieczny proces aktywacji na pojedynczej maszynie (fingerprint + walidacja klucza).

### Strumienie prac i zadania

#### 1. Modularizacja strategii
- **1.1. Podział pliku** – wydzielić katalog `strategies/` z podmodułami: `base.py` (interfejs strategii), `indicators.py` (EMA/RSI/ATR), `trend_following.py`, `mean_reversion.py`, `momentum.py`.
- **1.2. Warstwa konfiguracji** – zdefiniować klasę `StrategyConfig`, która pozwoli ładować parametry z presetów GUI.
- **1.3. Generator sygnałów** – utworzyć moduł `signal_pipeline.py`, który łączy dane rynkowe, wskaźniki i wynik strategii.
- **1.4. Testy** – przygotować testy jednostkowe dla wskaźników oraz test integracyjny sprawdzający poprawność wygenerowanego sygnału dla znanego zestawu OHLCV.
- **1.5. Dokumentacja** – opis w `docs/` jak dodawać nowe strategie (szablon klasy, wymagane metody).

#### 2. Risk Management w wersji MVP
- **2.1. Refaktor API** – uprościć `risk_management.py`, tak by `calculate_position_size` zwracała float (liczba jednostek), a struktury pomocnicze trafiły do `legacy/`.
- **2.2. Limity globalne** – dodać funkcje `enforce_daily_loss_limit`, `enforce_max_exposure`, `apply_stop_levels` używane przez `TradingEngine`.
- **2.3. Konfiguracja GUI** – przygotować preset domyślnych limitów (1.5% na pozycję, 5% dziennie, 25% ekspozycji) z możliwością edycji.
- **2.4. Testy** – jednostkowe dla każdej zasady oraz test integracyjny z `TradingEngine`, który symuluje serię strat i oczekuje blokady handlu.

#### 3. Backend paper i integracja giełdowa
- **3.1. PaperAccount** – dodać w `exchange_manager.py` klasę obsługującą paper trading (saldo startowe, księgowanie transakcji, fee, historia).
- **3.2. Tryb demo** – dostosować `TradingEngine`, aby wykonywał zlecenia przez adapter paper lub CCXT zależnie od konfiguracji.
- **3.3. Logging & audyt** – wprowadzić jednolity format logów transakcyjnych (JSONL) zapisywany w `logs/trading/`.
- **3.4. Testy** – scenariusz e2e: dane mockowane → strategia → risk manager → zlecenie w paper backendzie; dodatkowo smoke test CCXT na Binance Testnet (GET balance, GET ticker).

#### 4. Licencjonowanie i bezpieczeństwo
- **4.1. Fingerprint** – wykorzystać `SecurityManager` do zbudowania odcisku (CPU, MAC, dysk) i zapisać go w zaszyfrowanej formie.
- **4.2. Walidacja klucza** – utworzyć moduł `licensing.py` z funkcją `validate_license(fingerprint, key)` oraz CLI do aktywacji.
- **4.3. Integracja z GUI** – podczas startu aplikacji GUI ma wymagać klucza (z możliwością trybu demo).
- **4.4. Testy** – jednostkowe dla generatora fingerprintów, test integracyjny aktywacji (klucz poprawny/niepoprawny), test manualny odświeżenia licencji.

### Kamienie milowe

1. **Milestone A – Strategie + testy** (tydzień 1): zakończone zadania 1.1–1.4, testy przechodzą lokalnie.
2. **Milestone B – Risk Manager** (tydzień 2): wdrożone zasady, integracja z TradingEngine, dokumentacja limitów.
3. **Milestone C – Paper backend** (tydzień 3): pełna pętla paper tradingu, logi transakcyjne, smoke test CCXT.
4. **Milestone D – Licencjonowanie** (tydzień 4): aktywacja na pojedynczej maszynie, opis procedury wsparcia użytkownika.

### Testy i walidacja po Etapie 1

- `python -m pytest tests/strategies tests/risk_management tests/test_trading_engine.py` – regresja jednostkowa po refaktorze.
- `python scripts/run_backtest.py --strategy trend_following --pair BTC/USDT --interval 1h --mode paper` – krótki backtest kontrolny (tylko na danych historycznych/sandboxie).
- `python run_trading_gui_paper.py` – manualna sesja paper tradingu (kontrola logów i reakcji risk managera).
- Test manualny licencji: aktywacja + ponowne uruchomienie aplikacji na tej samej maszynie, a następnie próba na innej (powinna się nie powieść).

### Wymagane ustalenia od właściciela produktu

- Docelowe pary/rynki do obsługi w module strategii (np. BTC/USDT, ETH/USDT) – potrzebne do przygotowania presetów.
- Preferowany poziom agresywności limitów ryzyka (czy utrzymać proponowane domyślne, czy je zaostrzyć/rozluźnić).
- Czy licencja ma działać offline po aktywacji (jeśli tak – ile dni między walidacjami online).

**Bezpieczeństwo:** wszystkie testy integracyjne wykonujemy na sandboxach lub kontach paper. W kodzie dodajemy obsługę wyjątków i bogaty logging, by błąd nie skutkował stratą finansową ani utratą danych.
