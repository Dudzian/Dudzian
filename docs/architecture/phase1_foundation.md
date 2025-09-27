# Architektura rdzenia – etap 1

Dokument opisuje modularną architekturę przygotowującą bota do integracji z Binance (spot + futures),
Krakenem i Zondą. W pierwszym etapie budujemy szkielet umożliwiający bezpieczny rozwój bez licencji
zewnętrznych, z jasnym podziałem na warstwy i środowiska.

## Podział na moduły

| Moduł | Odpowiedzialność | Kluczowe klasy/interfejsy |
| --- | --- | --- |

| `bot_core/exchanges` | Adaptery giełdowe z rozdzieleniem środowisk i uprawnień | `ExchangeAdapter`, `ExchangeCredentials`, `BinanceSpotAdapter`, `BinanceFuturesAdapter`, `KrakenSpotAdapter` |
| `bot_core/data` | Pobieranie, normalizacja i cache danych OHLCV | `DataSource`, `CachedOHLCVSource`, `PublicAPIDataSource` |
| `bot_core/strategies` | Silnik strategii i walk-forward | `StrategyEngine`, `MarketSnapshot`, `StrategySignal`, `WalkForwardOptimizer` |
| `bot_core/risk` | Profile i enforcement limitów ryzyka | `RiskProfile`, `RiskEngine`, `RiskCheckResult`, profile konserwatywny/zbalansowany/agresywny/ręczny |
| `bot_core/execution` | Warstwa składania zleceń z retry/backoff | `ExecutionService`, `ExecutionContext`, `RetryPolicy` |
| `bot_core/alerts` | Wspólne API alertów i kanały powiadomień | `AlertChannel`, `AlertRouter`, kanały `TelegramChannel`, `EmailChannel`, `SMSChannel` |
| `bot_core/config` | Ładowanie konfiguracji i mapowanie na dataclasses | `CoreConfig`, `EnvironmentConfig`, `RiskProfileConfig`, `load_core_config` |
| `bot_core/security` | Bezpieczne przechowywanie kluczy API i integracja z keychainami | `SecretManager`, `KeyringSecretStorage` |

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
jednostkowe weryfikują poprawność podpisu oraz serializację zleceń.

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
ustrukturyzowany obiekt konfiguracyjny.

### Bootstrap środowiska

Nowy moduł `bot_core/runtime/bootstrap.py` dostarcza funkcję `bootstrap_environment`, która w oparciu
o `core.yaml` i `SecretManager` buduje kompletny kontekst uruchomieniowy (`BootstrapContext`). W jego
skład wchodzą: zainicjalizowany adapter giełdowy (z fabryki dopasowanej do `exchange`), zarejestrowany
profil ryzyka w `ThresholdRiskEngine`, skonfigurowany router alertów z kanałami Telegram/E-mail/SMS
oraz dziennik audytu. Dzięki temu pojedyncze środowisko (paper/live/testnet) można uruchomić w sposób
deterministyczny, a desktopowy interfejs w kolejnych etapach będzie mógł komunikować się z core przez
prostą warstwę IPC, korzystając z gotowego kontekstu.

## Środowiska i separacja uprawnień

Każdy adapter giełdowy otrzymuje `ExchangeCredentials` z informacją o środowisku (live/paper/testnet)
oraz zakresem uprawnień. Struktura przygotowuje grunt pod rozdzielenie kluczy `read-only` i `trading`
w Windows Credential Manager/macOS Keychain/Linux keyring (`keychain_key` w konfiguracji). Metoda
`configure_network` umożliwi egzekwowanie IP allowlist.

## Dane rynkowe

`PublicAPIDataSource` zostanie połączony z adapterami do pobierania danych OHLCV z publicznych API.
`CachedOHLCVSource` w połączeniu z `OHLCVBackfillService` obsługuje proces „backfill + cache” z
podziałem na okna czasowe oraz deduplikacją zapisów. Domyślny backend `SQLiteCacheStorage`
przechowuje dane w pliku `ohlcv.sqlite` (tryb WAL) i udostępnia metadane do audytu. Dla użytkownika
końcowego przygotowano skrypt `scripts/backfill_ohlcv.py`, który na podstawie `config/core.yaml`
pobiera świece z Binance i aktualizuje lokalny cache w trybie bezkosztowym.

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
awaryjny po przekroczeniu limitów i wskazuje dopuszczalną maksymalną wielkość zlecenia.

## Bezpieczeństwo i przechowywanie sekretów

Warstwa `bot_core/security` dostarcza `SecretManager`, który serializuje poświadczenia giełdowe i
umieszcza je w natywnym keychainie systemu operacyjnego poprzez `KeyringSecretStorage`. Dzięki
temu klucze API są składowane poza repozytorium kodu, z rozdzieleniem środowisk i uprawnień.
Manager pilnuje zgodności środowiska (paper/live/testnet) oraz pozwala na wielokrotne
zapisanie/rotację kluczy bez ręcznych zmian w konfiguracji YAML. W środowiskach headless można
docelowo podmienić implementację `SecretStorage` na wariant oparty o zaszyfrowany plik (np. `age`).

## Egzekucja

`ExecutionService` definiuje pełny cykl życia zlecenia z kontekstem (`ExecutionContext`) zawierającym
informacje o profilu ryzyka i środowisku. Interfejs `RetryPolicy` pozwoli na polityki zależne od
błędów API (np. exponential backoff, circuit breaker). Pierwszą implementacją jest
`PaperTradingExecutionService`, który symuluje egzekucję z natychmiastowym fill'em,
uwzględnia prowizje maker/taker, poślizg (w punktach bazowych), walidację wielkości zlecenia oraz
prowadzi dziennik audytowy transakcji.

## Alerty i obserwowalność

`AlertChannel` i `AlertRouter` zapewniają jednolite API dla powiadomień (Telegram, e-mail, SMS, a w
przyszłości Signal/WhatsApp/Messenger). Implementacja `DefaultAlertRouter` obsługuje audyt (`InMemoryAlertAuditLog`),
kontynuuje wysyłkę pomimo błędów pojedynczych kanałów i zwraca migawkę `health_check` do monitoringu SLO.
Adaptery kanałów posiadają zabezpieczenia (timeouty, logowanie błędów, walidację odpowiedzi API) oraz
formatowanie wiadomości z kontekstem ryzyka i znacznikami czasu UTC.

## Kolejne kroki implementacyjne

1. Rozszerzyć system alertów o kolejne kanały komunikatorów (Signal/WhatsApp/Messenger) oraz mechanizm
   throttle/acknowledgement dla krytycznych incydentów.
2. Podłączyć metryki Prometheus (latencja wysyłek, error rate) i dashboard zgodny z wymaganiami SLO.
3. Ustandaryzować eksport audytu do formatu Parquet/CSV z podpisem kryptograficznym oraz przygotować
   rotację logów zgodną z polityką 24–60 miesięcy.
4. Zintegrować alerty z planowanym modułem raportowania dziennego/tygodniowego.

Dokument będzie aktualizowany wraz z postępem implementacji kolejnych modułów.
