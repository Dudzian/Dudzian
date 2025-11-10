# Wstępna ocena kodu (wrzesień 2025)

Ten dokument podsumowuje stan obecnego repozytorium bota handlowego w momencie rozpoczęcia
prac modernizacyjnych. Analiza obejmuje strukturę kodu, jakość implementacji, luki
funkcjonalne oraz ryzyka operacyjne. Wnioski służą jako punkt odniesienia przy migracji do
nowej architektury `bot_core` oraz przy planowaniu kolejnych iteracji rozwoju.

## 1. Struktura i modularność

- **Warstwa `bot_core`** agreguje wspólne elementy runtime: loader konfiguracji
  (`runtime/metadata.py`), modele ryzyka (`risk/`), adaptery paper tradingu i
  bootstrap środowisk. Dzięki temu logika niezależna od UI nie jest już
  powielana w launcherach.
- **Pakiet `bot_core.ui.trading`** udostępnia modularny interfejs graficzny
  (podział na `app.py`, `controller.py`, `state.py`, `view.py`). GUI renderuje
  baner profilu ryzyka i kontrolki frakcji wprost z runtime settings i potrafi
  przeładowywać `core.yaml` z poziomu przycisku.
- **Launchery AutoTradera** zostały wydzielone do pakietu
  `bot_core.auto_trader.app`; archiwalny moduł `KryptoLowca.auto_trader` został usunięty
  zgodnościowy (`run_autotrade_paper.py`, `paper_auto_trade_app.py`). Moduły
  headless i GUI korzystają ze wspólnego loadera profilu ryzyka i rejestrują
  nasłuchy przeładowań (GUI, SIGHUP, watcher pliku).
- W katalogu głównym nadal znajdują się aliasowe pliki z epoki monolitu, jednak
  stopniowo delegują one działanie do nowych pakietów.

## 2. Funkcjonalność względem wymagań produktowych

- **Warstwa giełdowa** opiera się na pojedynczej konfiguracji CCXT bez separacji środowisk
  (testnet/paper/live) oraz bez rozdziału kluczy read-only/trade.
- **Dane rynkowe**: brak procesu „backfill + cache” opartego na publicznych API i lokalnym
  magazynie Parquet/SQLite; `data_preprocessor.py` zakłada gotowe dane w pamięci.
- **Strategie**: istnieją pojedyncze implementacje, ale brak systemu walk-forward i
  parametryzacji przez pliki konfiguracyjne.
- **Zarządzanie ryzykiem** nie obejmuje wymaganych profili (konserwatywny, zbalansowany,
  agresywny, ręczny) ani egzekucji limitów dziennych, ATR i hard-stop drawdown.
- **Powiadomienia** ograniczają się do logów – brak kanałów Telegram, e-mail, SMS czy
  komunikatorów, a także audytu i health-checków.
- **Bezpieczeństwo sekretów** bazuje na lokalnym szyfrowaniu; nie ma integracji z natywnymi
  keychainami (Windows Credential Manager, macOS Keychain, GNOME Keyring/age) ani polityki
  rotacji i allowlist IP.

## 3. Jakość kodu i testy

- Testy jednostkowe pokrywają wybrane moduły, ale są ściśle powiązane z monolitem –
  ich utrzymanie przy migracji będzie trudne.
- Wiele metod ma szeroki zakres odpowiedzialności (np. `TradingEngine.execute()`), co
  utrudnia dodawanie logiki compliance lub rozbudowanej telemetrii.
- Brak spójnego logowania strukturalnego oraz standardu obsługi wyjątków; krytyczne błędy
  mogą pozostać niezauważone podczas handlu live.

## 4. Ryzyka operacyjne i compliance

- **Brak audytu** działań (append-only) z retencją 24–60 miesięcy i podpisem kryptograficznym.
- **Polityka KYC/AML** nie jest odnotowywana w logach ani konfiguracji środowisk.
- **Brak obserwowalności**: brak metryk latencji zleceń, fill rate, error rate i brak SLO.
- **Paper trading** dzieli stan z kontem produkcyjnym; nie ma izolacji pozwalającej na
  realistyczne testy bez ryzyka dla kapitału.

## 5. Rekomendowane kierunki działania

1. **Migracja do modułowej architektury `bot_core`** – oddzielenie adapterów giełdowych,
   warstwy danych, strategii, ryzyka, egzekucji, alertów i bezpieczeństwa.
2. **Nowy system konfiguracji** (`config/core.yaml`) z definicją środowisk, profili ryzyka,
   koszyków instrumentów oraz kanałów alertów.
3. **Proces „backfill + cache”** korzystający z publicznych endpointów REST (Binance, Kraken,
   Zonda) z zapisem do SQLite/Parquet i inkrementalnymi aktualizacjami.
4. **Silnik strategii D1** (trend-following + momentum breakout) z parametryzacją,
   walk-forward i spójnym modelem kosztów/poślizgu.
5. **Risk engine** egzekwujący profile konserwatywny/zbalansowany/agresywny/ręczny z twardymi
   limitami oraz natychmiastowym zamykaniem pozycji po przekroczeniu dziennej straty.
6. **Alerty wielokanałowe** (Telegram, e-mail, SMS, komunikatory) z audytem i health-checkami.
7. **Secret manager** oparty o natywne keychainy i rotację kluczy co 90 dni, z wymuszaniem
   modelu najmniejszych uprawnień i allowlist IP.
8. **Observability**: metryki, logi strukturalne, eksport raportów CSV/PDF, retention 24 mies.

## 6. Pierwsze kroki wdrożeniowe

- Utrzymać monolit w stanie „frozen” – bez dalszych modyfikacji poza poprawkami krytycznych błędów.
- Rozpocząć prace w pakiecie `bot_core`, począwszy od interfejsów i konfiguracji (już obecne w repo),
  a następnie iteracyjnie przenosić funkcjonalność z monolitu.
- Wprowadzić standard dokumentowania decyzji architektonicznych (ADRs) oraz checklistę testów dla
  każdego PR (backtest → paper → live, logi audytu, alerty).

## 7. Testy zalecane po modernizacji

- **Backtesty** na danych D1 i 1h (BTC/USDT, ETH/USDT, SOL/USDT) z podziałem in/out-of-sample.
- **Paper trading** na Binance Testnet oraz symulatorze dla giełd bez sandboxa.
- **Testy integracyjne alertów** z kanałami Telegram/E-mail/SMS (mock API + sandbox).
- **Testy bezpieczeństwa**: rotacja kluczy, odtwarzanie środowiska z backupu, walidacja allowlist IP.

> **Bezpieczeństwo**: wszystkie nowe moduły należy testować w środowisku demo/testnet, zanim
> cokolwiek trafi na konto live. Każde odstępstwo musi być zatwierdzone i odnotowane w audycie.
