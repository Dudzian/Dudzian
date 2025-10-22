# Konfiguracja kont i giełd dla dystrybucji desktopowej

Dystrybucja desktopowa dostarcza uproszczony interfejs do zarządzania kluczami API
oraz trybami handlu (spot, margin, futures). Poniżej znajduje się lista kroków,
które należy wykonać przed uruchomieniem bota w środowisku lokalnym.

## Wymagane poświadczenia

1. **Binance** – klucz API z aktywnymi uprawnieniami `read` i `trade`.
   - Dla trybu margin należy aktywować margin w profilu Binance i przypisać
     limit transferów do subkonta używanego przez bota.
   - Dla kontraktów futures (USD-M) wymagane są osobne uprawnienia API.
2. **Kraken** – klucz API z uprawnieniami `Query Funds`, `Query Open Orders & Trades`
   oraz `Create & Modify Orders`.
   - Handel margin wymaga dodatkowo zaznaczenia flagi **Allow Margin Trading**.
3. **Zonda** – klucz publiczny i prywatny z włączonymi uprawnieniami do
   pobierania stanu konta oraz składania/anulowania zleceń.
   - Margin należy aktywować w panelu klienta i nadać limit kredytowy.
4. **nowa_gielda (spot)** – para kluczy REST (`key`, `secret`) z dostępem do danych
   publicznych i prywatnych zleceń.
   - Adapter `nowa_gielda_spot` korzysta z osobnych baz URL dla środowisk `live`,
     `paper` i `testnet`, dlatego generując klucze testowe należy użyć portalu
     odpowiadającego wybranemu środowisku.
   - API udostępnia endpoint `GET /private/account`, z którego pobierane są pola
     `balances`, `totalEquity`, `availableMargin` oraz `maintenanceMargin`. Upewnij
     się, że konto posiada uprawnienia do odczytu stanu rachunku.
   - Historia transakcji dostępna jest pod `GET /private/trades`. Zapytanie wymaga
     podpisu HMAC i akceptuje parametry `symbol`, `start`, `end` oraz `limit`.
     Adapter filtruje wyniki po symbolu oraz waliduje typy liczbowe (`price`,
     `quantity`, `fee`, `timestamp`). Endpoint objęty jest limitem 5 żądań na
     sekundę, dlatego długie zakresy czasowe należy dzielić na mniejsze porcje.
   - Otwarte zlecenia pobierane są z `GET /private/orders`. Endpoint wspiera
     parametry `symbol` oraz `limit` i zwraca pola `orderId`, `status`, `side`,
     `type`, `price`, `quantity`, `filledQuantity`, `avgPrice` oraz `timestamp`.
     Limit wagowy pozwala na maksymalnie 5 zapytań w 1 sekundzie.
   - Historia zamkniętych zleceń dostępna jest przez `GET /private/orders/history`.
     Zapytanie można filtrować po `symbol`, `start`, `end` i `limit`. Adapter
     waliduje pola `orderId`, `executedQuantity`, `closedAt` oraz poprawność
     tłumaczenia symbolu, a limit zapytań wynosi 5 na sekundę.
   - Historia depozytów obsługiwana jest przez `GET /private/deposits`, który
     akceptuje parametry `symbol`, `status`, `start`, `end` i `limit`. Odpowiedź
     zawiera pola `depositId`, `amount`, `fee`, `network`, `txId`, `timestamp`
     oraz `completedAt`. Adapter konwertuje symbol do formatu wewnętrznego,
     waliduje liczby i podpisuje żądanie HMAC. Limit zapytań wynosi 5 wywołań w
     sekundę.
   - Historia wypłat dostępna jest pod `GET /private/withdrawals`. Zapytanie
     przyjmuje parametry `symbol`, `status`, `start`, `end` i `limit`, a w
     odpowiedzi spodziewane są pola `withdrawalId`, `amount`, `fee`, `network`,
     `address`, `tag`, `txId`, `timestamp` oraz `completedAt`. Adapter odrzuca
     wpisy z błędnymi typami i weryfikuje zgodność symboli, a limit wynosi 5
     zapytań na sekundę.
   - Transfery wewnętrzne (np. między kontami spot i margin) obsługiwane są
     przez endpoint `GET /private/transfers`. Wspiera on parametry `symbol`,
     `direction`, `from`, `to`, `status`, `start`, `end` i `limit`. Adapter
     podpisuje zapytania, tłumaczy symbole z/do formatu wewnętrznego, pilnuje
     typów pól (`amount`, `timestamp`, `completedAt`) i egzekwuje limit 5 wywołań
     na sekundę przy wadze 2 jednostek.
   - Stawki prowizyjne (`maker`/`taker`) można pobrać z `GET /private/fees`.
     Endpoint akceptuje opcjonalny parametr `symbol` i zwraca listę obiektów z
     polami `symbol`, `maker`, `taker` oraz `thirtyDayVolume`. Adapter konwertuje
     symbol do formatu wewnętrznego, waliduje typy liczbowe i podpisuje zapytanie
     HMAC. Limit wynosi 5 wywołań na sekundę przy wadze 2 jednostek.
   - Zwroty prowizyjne dostępne są przez endpoint `GET /private/rebates`, który
     obsługuje parametry `symbol`, `type`, `start`, `end` i `limit`. Odpowiedź
     zawiera pola `rebateId`, `amount`, `rate`, `type`, `orderId`, `timestamp`
     oraz `settledAt`. Adapter tłumaczy symbol, waliduje wartości liczbowe,
     podpisuje zapytanie HMAC i egzekwuje limit 5 wywołań na sekundę (waga 2).
   - Naliczanie odsetek od pożyczonych środków dostępne jest przez `GET /private/interest`.
     Endpoint przyjmuje parametry `symbol`, `start`, `end` i `limit`, a w odpowiedzi
     spodziewane są pola `interestId`, `amount`, `rate`, `type`, `orderId`, `timestamp`
     oraz `accrualTimestamp`. Adapter tłumaczy symbole, waliduje liczby (np. `amount`,
     `rate`), zapewnia obecność identyfikatora i podpisuje zapytanie HMAC. Limit wynosi
     5 wywołań na sekundę przy wadze 2 jednostek.
   - Dane historyczne świec pobierane są z `GET /public/ohlcv`, który wymaga
     parametrów `symbol` (np. `BTC-USDT`) oraz `interval` (`1m`, `1h`, `1d`).
     Opcjonalnie można przekazać `start`, `end` i `limit`, aby zawęzić zakres
     czasowy. Endpoint ten objęty jest limitem 10 zapytań w oknie 1 sekundy.

Poświadczenia przechowujemy w zaszyfrowanym magazynie `secrets/desktop.toml`.
Każdy wpis zawiera identyfikator giełdy (`binance`, `kraken`, `zonda`) oraz
pola `key` i `secret`. Opcjonalnie można dodać `passphrase` (np. Coinbase).

```toml
[binance]
key = "BINANCE_KEY"
secret = "BINANCE_SECRET"
mode = "margin"  # spot | margin | futures

[kraken]
key = "KRAKEN_KEY"
secret = "KRAKEN_SECRET"
mode = "futures"
leverage = "3"

[zonda]
key = "ZONDA_KEY"
secret = "ZONDA_SECRET"
mode = "margin"
valuation_currency = "PLN"

[nowa_gielda_spot]
key = "NOWA_GIELDA_KEY"
secret = "NOWA_GIELDA_SECRET"
environment = "paper"  # live | paper | testnet
```

## Parametry środowiskowe

W pliku `env.desktop` można określić domyślne środowisko dla każdej giełdy:

```
EXCHANGE_ENVIRONMENT=live        # live | testnet | paper
BINANCE_MARGIN_TYPE=isolated     # cross | isolated
KRAKEN_ENVIRONMENT=testnet
ZONDA_ENVIRONMENT=live
NOWA_GIELDA_ENVIRONMENT=paper
```

Zmienne są odczytywane przez warstwę konfiguracyjną i przekazywane do adapterów.
Jeżeli desktop uruchamiany jest w trybie testnetowym, należy upewnić się, że
klucze API zostały wygenerowane na odpowiedniej platformie.

W przypadku `nowa_gielda_spot` dodatkowo można zdefiniować allowlistę IP używając
metody `configure_network()` adaptera. Jeśli operator korzysta z filtrów sieciowych,
wprowadź adresy IP proxy w konfiguracji startowej bota, aby uniknąć odrzuconych
połączeń prywatnych endpointów.

## Health-check i watchdog

Moduł `bot_core.exchanges.health` udostępnia `HealthMonitor`, `Watchdog` oraz
`CircuitBreaker`, które są automatycznie wykorzystywane przez menedżera giełd
w dystrybucji desktopowej. W przypadku problemów sieciowych (np. limity API)
watchdog podejmuje ponowne próby, a po przekroczeniu progu błędów otwiera
wyłącznik, aby ochronić konto przed eskalacją problemu. `ExchangeManager`
posiada metody `set_watchdog()` i `configure_watchdog()`, dzięki czemu można
dostosować liczbę ponowień, próg wyłącznika i listę wyjątków, które mają
inicjować retry. Tę samą instancję strażnika można wykorzystać w monitoringu
zdrowia: metoda `create_health_monitor()` zwraca `HealthMonitor` dzielący
watchdog-a z natywnymi adapterami, co pozwala raportować spójny stan w panelu
desktopowym.

```python
from bot_core.exchanges.manager import ExchangeManager
from bot_core.exchanges.health import HealthCheck

manager = ExchangeManager(exchange_id="binance")
manager.set_mode(margin=True)
manager.configure_watchdog(
    retry_policy={"max_attempts": 5, "base_delay": 0.5, "max_delay": 2.0},
    circuit_breaker={"failure_threshold": 3, "recovery_timeout": 60.0},
)

health_monitor = manager.create_health_monitor([
    HealthCheck(name="private_api", check=lambda: manager.fetch_balance()),
])
```

Aby ręcznie uruchomić diagnostykę, użyj polecenia:

```
python -m bot_core.cli health-check --exchange binance

# opcjonalnie można pominąć prywatny check (np. gdy testujemy tylko publiczne API)
python -m bot_core.cli health-check --exchange binance --skip-private
```

Polecenie wypisze statusy `healthy`, `degraded` lub `unavailable` dla
kanałów publicznych i prywatnych wraz z czasem odpowiedzi.

## Szybki start

1. Skopiuj `env.example` do `env.desktop` i uzupełnij zmienne.
2. Dodaj poświadczenia do `secrets/desktop.toml`.
3. Uruchom `python -m bot_core.desktop --profile live` aby zweryfikować połączenia.
4. W dashboardzie desktopowym wybierz giełdę oraz tryb (spot/margin/futures).
5. Aktywuj strategię – monitoruj sekcję „Watchdog” w panelu statusów.

Po zakończeniu sesji zaleca się dezaktywację margin/futures w interfejsie
platformy, a klucze API przechowywać w sejfie haseł.
