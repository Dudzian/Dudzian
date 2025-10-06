# Architektura rdzenia – etap 1

Dokument opisuje modularną architekturę przygotowującą bota do integracji z Binance (spot + futures),
Krakenem i Zondą. W pierwszym etapie budujemy szkielet umożliwiający bezpieczny rozwój bez licencji
zewnętrznych, z jasnym podziałem na warstwy i środowiska.

> **Zasada etapu 1:** wszystkie nowe moduły aplikacji korzystają wyłącznie z
> przestrzeni nazw `bot_core`. Kod znajdujący się w `legacy_bridge/` jest
> wyłącznie mostkiem zgodności do historycznego pakietu `KryptoLowca` i ma
> charakter read-only – nie wolno go rozszerzać ani modyfikować przy pracach nad
> fundamentem architektury.

## Podział na moduły

| Moduł | Odpowiedzialność | Kluczowe klasy/interfejsy |
| --- | --- | --- |

| `bot_core/exchanges` | Adaptery giełdowe z rozdzieleniem środowisk i uprawnień | `ExchangeAdapter`, `ExchangeCredentials`, `BinanceSpotAdapter`, `BinanceFuturesAdapter`, `KrakenSpotAdapter`, `KrakenFuturesAdapter`, `ZondaSpotAdapter` |
| `bot_core/data` | Pobieranie, normalizacja i cache danych OHLCV | `DataSource`, `CachedOHLCVSource`, `PublicAPIDataSource` |
| `bot_core/strategies` | Silnik strategii i walk-forward | `StrategyEngine`, `MarketSnapshot`, `StrategySignal`, `WalkForwardOptimizer` |
| `bot_core/risk` | Profile i enforcement limitów ryzyka | `RiskProfile`, `RiskEngine`, `RiskCheckResult`, profile konserwatywny/zbalansowany/agresywny/ręczny |
| `bot_core/execution` | Warstwa składania zleceń z retry/backoff | `ExecutionService`, `ExecutionContext`, `RetryPolicy` |
| `bot_core/alerts` | Wspólne API alertów i kanały powiadomień | `AlertChannel`, `AlertRouter`, kanały `TelegramChannel`, `EmailChannel`, `SMSChannel`, `SignalChannel`, `WhatsAppChannel`, `MessengerChannel` |
| `bot_core/config` | Ładowanie konfiguracji i mapowanie na dataclasses | `CoreConfig`, `EnvironmentConfig`, `RiskProfileConfig`, `load_core_config` |
| `bot_core/security` | Bezpieczne przechowywanie kluczy API i integracja z keychainami | `SecretManager`, `KeyringSecretStorage` |
| `bot_core/reporting` | Generowanie dziennych pakietów audytowych | `generate_daily_paper_report`, `PaperReportArtifacts` |

Adaptery `BinanceSpotAdapter` oraz `BinanceFuturesAdapter` obsługują dane publiczne (lista symboli,
świece OHLCV) oraz podpisane wywołania konta i składania/anulowania zleceń. Wspierają separację
środowisk (`live`/`paper`/`testnet`), podpis HMAC SHA-256 oraz kontrolę uprawnień (`read`/`trade`).
Wersja futures korzysta z endpointów USD-M, wymaga jawnego podania symbolu przy anulowaniu i
odczytuje metryki marginesu z `totalMarginBalance`/`totalAvailableBalance`. Testy jednostkowe
potwierdzają poprawność podpisów i mapowania odpowiedzi.

Dodany `KrakenSpotAdapter` implementuje podpis HMAC-SHA512 z wymaganym nonce, obsługuje publiczne
endpointy (`AssetPairs`, `OHLC`) oraz prywatne wywołania (`Balance`, `TradeBalance`, `AddOrder`,
`CancelOrder`). Adapter egzekwuje podział uprawnień (`read`/`trade`), zapewnia monotonistyczny nonce,
przetwarza dane konta na `AccountSnapshot` i integruje się z bootstrapem środowiska. Testy
jednostkowe weryfikują poprawność podpisu oraz serializację zleceń. `KrakenFuturesAdapter`
korzysta z endpointów `derivatives/api/v3`, generuje podpis HMAC-SHA256 wymagany przez Kraken
Futures (nagłówki `APIKey`, `Authent`, `Nonce`), mapuje wartości marginesu oraz bilansów na
`AccountSnapshot`, wspiera składanie zleceń `mkt`/`lmt` i anulowanie przez `DELETE /orders/{id}`.
Wzbogacone testy jednostkowe potwierdzają budowę podpisu i serializację ciała żądania.
`ZondaSpotAdapter` wykorzystuje REST API Zondy do pobierania świec (endpoint `trading/candle/history`) oraz danych konta (`trading/balance`). Implementacja korzysta z podpisu HMAC-SHA512 zgodnie z nagłówkiem `API-Hash`, obsługuje mapowanie statusów zleceń, zabezpiecza się przed brakiem uprawnień (`read`/`trade`) oraz przelicza wszystkie salda na walutę referencyjną zdefiniowaną w `adapter_settings.valuation_asset` (np. PLN lub EUR) z wykorzystaniem kursów `trading/ticker` i triangulacji przez aktywa pomocnicze (`secondary_valuation_assets`). Adapter został dodany do domyślnych fabryk bootstrapa, dzięki czemu środowiska paper/live mogą korzystać z Zondy bez modyfikacji logiki strategii czy ryzyka.

## Warstwa konfiguracji

Plik `config/core.yaml` przechowuje:

- parametry profili ryzyka (zgodne z wymaganiami: dzienne limity strat, ATR, dźwignia, liczba pozycji,
  hard-stop drawdown),
- definicje środowisk (paper/live/testnet) wraz z kluczami do menedżera sekretów, nazwą wpisu w
  keychainie (`credential_purpose`), ścieżkami cache i listą kanałów alertów,
- zbiory instrumentów (`instrument_universes`) opisujące koszyk rynków przypisany do środowisk,
  w tym aliasy symboli na poszczególnych giełdach oraz rekomendowane zakresy backfillu dla interwałów
  (np. 10 lat D1 dla BTC/ETH, 5 lat dla głównych altów, 1h dla walidacji ryzyka),
- ustawienia raportowania (czas dziennych/tygodniowych raportów, retencja logów).

Loader `load_core_config` mapuje YAML na dataclasses i zapewnia konwersję pól liczbowych oraz
walidację środowiska poprzez `Environment` enum. Dzięki temu logika aplikacji otrzymuje w pełni
ustrukturyzowany obiekt konfiguracyjny wraz z kompletną definicją uniwersum instrumentów, co
upraszcza backfill danych oraz konfigurację strategii.

### Bootstrap środowiska

Nowy moduł `bot_core/runtime/bootstrap.py` dostarcza funkcję `bootstrap_environment`, która w oparciu
o `core.yaml` i `SecretManager` buduje kompletny kontekst uruchomieniowy (`BootstrapContext`). W jego
skład wchodzą: zainicjalizowany adapter giełdowy (z fabryki dopasowanej do `exchange`), zarejestrowany
profil ryzyka w `ThresholdRiskEngine`, skonfigurowany router alertów z kanałami Telegram/E-mail/SMS/
Signal/WhatsApp/Messenger
oraz dziennik audytu. Dzięki temu pojedyncze środowisko (paper/live/testnet) można uruchomić w sposób
deterministyczny, a desktopowy interfejs w kolejnych etapach będzie mógł komunikować się z core przez
prostą warstwę IPC, korzystając z gotowego kontekstu.

## Środowiska i separacja uprawnień

Każdy adapter giełdowy otrzymuje `ExchangeCredentials` z informacją o środowisku (live/paper/testnet)
oraz zakresem uprawnień. Struktura przygotowuje grunt pod rozdzielenie kluczy `read-only` i `trading`
w Windows Credential Manager/macOS Keychain/Linux keyring (`keychain_key` w konfiguracji). Metoda
`configure_network` umożliwi egzekwowanie IP allowlist.

Sekcja środowisk w `core.yaml` została rozszerzona o pole `adapter_settings`, dzięki któremu możemy
przekazywać parametry specyficzne dla danej giełdy bez łamania ogólnego kontraktu adapterów. Przykład:
`kraken_live` korzysta z `valuation_asset: ZEUR`, co powoduje, że `KrakenSpotAdapter` raportuje kapitał
w walucie referencyjnej EUR zgodnie z wymaganiami risk engine'u i raportowania P&L.

Konfiguracja każdego środowiska zawiera także sekcje `required_permissions` i `forbidden_permissions`.
`SecretManager.load_exchange_credentials` odczytuje te listy i blokuje start, jeżeli klucz API nie ma
kompletu minimalnych uprawnień (np. `trade`) lub posiada zabronione możliwości (`withdraw`). Dzięki
temu wymuszamy model najmniejszych uprawnień zanim kontroler runtime połączy się z adapterem.

## Dane rynkowe

`PublicAPIDataSource` zostanie połączony z adapterami do pobierania danych OHLCV z publicznych API.
`CachedOHLCVSource` w połączeniu z `OHLCVBackfillService` obsługuje proces „backfill + cache” z
podziałem na okna czasowe oraz deduplikacją zapisów. Domyślna konfiguracja wykorzystuje
`DualCacheStorage`, które łączy `ParquetCacheStorage` (partycjonowane katalogi
`exchange/symbol/granularity/year=YYYY/month=MM/`) z lekkim manifestem w `SQLiteCacheStorage`
(`ohlcv_manifest.sqlite`). Dzięki temu Parquet jest „źródłem prawdy” dla świeczek, a manifest
przechowuje metadane (ostatni timestamp, liczba rekordów) bez konieczności otwierania wszystkich
plików. Zarówno nowy skrypt `scripts/backfill.py`, jak i uproszczony `scripts/backfill_ohlcv.py`
wykorzystują tę samą warstwę storage, dzięki czemu backtesty i runtime paper/live czytają identyczne dane.
`OHLCVBackfillService` domyślnie wprowadza throttling zapytań (konfigurowalny interwał, jitter i limit
retry z wykładniczym backoffem), co pozwala realizować długie zasypy historii w duchu „good citizen” wobec
publicznych API giełd – bez przekraczania limitów i bez konieczności manualnego wstawiania pauz w skryptach.

## Strategie i walk-forward

`StrategyEngine` definiuje kontrakt odbierania snapshotów rynkowych oraz generowania sygnałów.
Pierwszą ukończoną implementacją jest `DailyTrendMomentumStrategy`, która łączy średnie kroczące,
wybicia Donchiana i trailing stop na bazie ATR, aby realizować trend-following na interwale D1.
Parametry strategii (okna średnich, próg momentum, wielkość ATR) znajdują się w sekcji `strategies`
pliku `config/core.yaml` i są mapowane na dataclass `DailyTrendMomentumStrategyConfig`. Interfejsy
pozostają przygotowane pod kolejne moduły (mean reversion, arbitraż, optymalizacja walk-forward)
korzystające z tego samego kontraktu `StrategyEngine` i `WalkForwardOptimizer`.

## Zarządzanie ryzykiem

`RiskProfile` i `ThresholdRiskEngine` egzekwują dzienny limit straty, liczbę równoległych pozycji,
dźwignię, ekspozycję per instrument oraz hard-stop drawdown. Dostępne są cztery gotowe profile
(konserwatywny, zbalansowany, agresywny, manualny) odwzorowujące wymagane parametry. Silnik
utrzymuje stan w repozytorium (`RiskRepository`), kontroluje reset doby w UTC, wymusza tryb
awaryjny po przekroczeniu limitów i wskazuje dopuszczalną maksymalną wielkość zlecenia. Podczas
oceny nowego zlecenia rynekowy dystans stop-loss może być większy niż minimalna wielokrotność ATR
zdefiniowana w profilu – w takim przypadku limit pozycji wyznaczany jest na podstawie faktycznego
stopu, natomiast zbyt ciasne ustawienie (poniżej minimum) prowadzi do natychmiastowego odrzucenia
zlecenia.

## Bezpieczeństwo i przechowywanie sekretów

Warstwa `bot_core/security` dostarcza `SecretManager`, który serializuje poświadczenia giełdowe i
umieszcza je w natywnym keychainie systemu operacyjnego poprzez `KeyringSecretStorage`. Dzięki
temu klucze API są składowane poza repozytorium kodu, z rozdzieleniem środowisk i uprawnień.
Manager pilnuje zgodności środowiska (paper/live/testnet) oraz pozwala na wielokrotne
zapisanie/rotację kluczy bez ręcznych zmian w konfiguracji YAML. W środowiskach headless można
docelowo podmienić implementację `SecretStorage` na wariant oparty o zaszyfrowany plik (np. `age`).

Uzupełniająco moduł `security.rotation` utrzymuje rejestr dat wymiany kluczy w pliku
`security/rotation_log.json` (per środowisko) i udostępnia API do obliczania, ile czasu pozostało do
kolejnej rotacji. Skrypt `scripts/check_key_rotation.py` korzysta z `core.yaml`, generuje raport dla
wszystkich środowisk oraz – opcjonalnie – zapisuje nową datę rotacji po zakończeniu procedury
„bez-downtime”. Dzięki temu proces wymiany co 90 dni posiada mierzalne wsparcie operacyjne i łatwo
go audytować.

## Egzekucja

`ExecutionService` definiuje pełny cykl życia zlecenia z kontekstem (`ExecutionContext`) zawierającym
informacje o profilu ryzyka i środowisku. Interfejs `RetryPolicy` pozwoli na polityki zależne od
błędów API (np. exponential backoff, circuit breaker). Pierwszą implementacją jest
`PaperTradingExecutionService`, który symuluje egzekucję z natychmiastowym fill'em,
uwzględnia prowizje maker/taker, poślizg (w punktach bazowych), walidację wielkości zlecenia oraz
prowadzi dziennik audytowy transakcji – zarówno w pamięci na potrzeby bieżącej sesji, jak i
w trwałych plikach JSONL rotowanych zgodnie z polityką retencji.

## Raportowanie i audyt

`bot_core/reporting` dostarcza funkcję `generate_daily_paper_report`, która tworzy dzienne archiwum ZIP z
blotterem (`ledger.csv`), zdarzeniami decyzyjnymi (`decisions.jsonl`) oraz zwięzłym podsumowaniem (`summary.json`).
Raport filtruje wpisy według strefy czasowej środowiska, wspiera retencję 24 miesięcy i przygotowuje pakiety do
podpisu kryptograficznego oraz szyfrowania w kolejnych etapach. Pakiet stanowi bazę do dziennych raportów P&L oraz
audytów KYC/AML. Operacyjny proces paper tradingu opisuje runbook `docs/runbooks/paper_trading.md`, a
append-only log audytowy prowadzimy w `docs/audit/paper_trading_log.md`.

## Integracja z desktopową powłoką Qt/QML

Zgodnie z rozszerzonym zakresem produktowym planujemy pełną aplikację desktopową z ciężkim demonem C++ i
powłoką Qt Quick/QML. Dedykowany dokument [`desktop_shell_plan.md`](desktop_shell_plan.md) opisuje strukturę
repozytorium (`/core`, `/ui`, `/proto`, `/packaging`, `/ops`), kontrakty Protobuf oraz wymagania wydajnościowe
(60/120 Hz, KPI event→frame <150 ms). W praktyce oznacza to, że `bot_core` stanie się biblioteką logiki
domainowej konsumowaną przez demon gRPC, a UI pozostanie cienkim klientem renderującym dane bez bezpośrednich
połączeń z giełdami. Integracja będzie przebiegała etapami: zamrożenie kontraktów Protobuf, implementacja demona
gRPC (MarketData/Risk/Orders/Metrics), stworzenie powłoki Qt z dedykowanym komponentem wykresu (QQuickItem) oraz
pipeline’u packagingowego (MSI/DMG/AppImage) z auto-update i podpisami. Plan zakłada brak WebSocketów,
wykorzystanie mTLS, RBAC oraz obserwowalność OpenTelemetry/Prometheus, a także pełną obsługę animacji 60/120 Hz
przy adaptacji „reduce motion” dla środowisk obciążonych.

## Alerty i obserwowalność

`AlertChannel` i `AlertRouter` zapewniają jednolite API dla powiadomień (Telegram, e-mail, SMS, a w
przyszłości Signal/WhatsApp/Messenger). Implementacja `DefaultAlertRouter` obsługuje audyt (`InMemoryAlertAuditLog`),
kontynuuje wysyłkę pomimo błędów pojedynczych kanałów i zwraca migawkę `health_check` do monitoringu SLO.
Adaptery kanałów posiadają zabezpieczenia (timeouty, logowanie błędów, walidację odpowiedzi API) oraz
formatowanie wiadomości z kontekstem ryzyka i znacznikami czasu UTC.

Nowy `FileAlertAuditLog` pozwala utrzymywać dziennik zdarzeń w formacie JSONL z rotacją dobową i
polityką retencji zgodną z wymaganiami (domyślnie 24 miesiące). Ścieżka oraz wzorzec nazw plików
są definiowane w konfiguracji środowiska (`alert_audit`), a bootstrap automatycznie wybiera backend
plikowy lub pamięciowy zależnie od ustawień. Dzięki temu logi alertów spełniają wymogi audytu i
mogą być w prosty sposób archiwizowane lub agregowane do raportów.

Moduł `runtime.journal` dodaje `TradingDecisionJournal`, który zapisuje w formacie JSONL pełną
historię decyzji (przyjęte/odrzucone sygnały, korekty ryzyka, egzekucje, błędy) wraz z metadanymi
środowiska i portfela. Domyślna konfiguracja `decision_journal` w `core.yaml` wskazuje katalog
`audit/decisions` z retencją 24 miesięcy i opcją `fsync` dla środowisk produkcyjnych. Dziennik jest
wykorzystywany przy raportach compliance, ponieważ pozwala odtworzyć dokładny powód każdej decyzji
silnika ryzyka lub modułu egzekucji.

Nowy mechanizm throttlingu pozwala dodatkowo ograniczyć powtarzalne alerty informacyjne: dla każdego
środowiska w `core.yaml` można zdefiniować długość okna, wykluczone kategorie lub poziomy `severity`
oraz limit bufora. Router zapisuje wstrzymane komunikaty w audycie (`channel="__suppressed__"`),
dzięki czemu zachowujemy pełną ścieżkę zgodności i możemy analizować historię incydentów bez zalewania
powiadomień produkcyjnych.

## Kolejne kroki implementacyjne

1. Rozszerzyć testy integracyjne kanałów komunikatorów o scenariusze awarii i fallback do alternatywnych
   dostawców oraz dodać mechanizm throttle/acknowledgement dla krytycznych incydentów.
2. Rozszerzyć system alertów o kolejne kanały komunikatorów (Signal/WhatsApp/Messenger) oraz mechanizm
   throttle/acknowledgement dla krytycznych incydentów.
3. Podłączyć metryki Prometheus (latencja wysyłek, error rate) i dashboard zgodny z wymaganiami SLO.
4. Ustandaryzować eksport audytu do formatu Parquet/CSV z podpisem kryptograficznym oraz przygotować
   rotację logów zgodną z polityką 24–60 miesięcy.
5. Zintegrować alerty z planowanym modułem raportowania dziennego/tygodniowego.

Dokument będzie aktualizowany wraz z postępem implementacji kolejnych modułów.
