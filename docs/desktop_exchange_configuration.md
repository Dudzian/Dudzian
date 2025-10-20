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
```

## Parametry środowiskowe

W pliku `env.desktop` można określić domyślne środowisko dla każdej giełdy:

```
EXCHANGE_ENVIRONMENT=live        # live | testnet | paper
BINANCE_MARGIN_TYPE=isolated     # cross | isolated
KRAKEN_ENVIRONMENT=testnet
ZONDA_ENVIRONMENT=live
```

Zmienne są odczytywane przez warstwę konfiguracyjną i przekazywane do adapterów.
Jeżeli desktop uruchamiany jest w trybie testnetowym, należy upewnić się, że
klucze API zostały wygenerowane na odpowiedniej platformie.

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
