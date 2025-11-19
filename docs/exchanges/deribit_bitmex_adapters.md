# Adaptery Deribit i BitMEX (REST + long-poll)

Ten dokument opisuje sposób korzystania z natywnych adapterów Deribit i BitMEX
w trybach spot/futures z pełnym wsparciem REST oraz long-pollowych kanałów dla
order booków, tickerów i fills.

## Rejestracja adapterów

Adaptery są dostępne z poziomu `ExchangeManager` dzięki wpisom w `config/core.yaml`:

- `bot_core.exchanges.deribit.futures:DeribitFuturesAdapter`
- `bot_core.exchanges.deribit.spot:DeribitSpotAdapter`
- `bot_core.exchanges.bitmex.futures:BitmexFuturesAdapter`
- `bot_core.exchanges.bitmex.spot:BitmexSpotAdapter`

Każdy wpis posiada domyślne reguły rate limitów, politykę retry oraz ustawienia
strumieni long-pollowych (endpointy live/testnet/paper).

## Profile środowiskowe (paper/testnet/live)

Pliki `config/exchanges/deribit.yaml` oraz `config/exchanges/bitmex.yaml`
zawierają trzy profile wymagane przez `ExchangeManager`:

- **paper** – symulator z limitem dźwigni i fundingiem (perpetual), bez kluczy API.
- **testnet** – środowisko testowe futures/spot z własnymi kluczami API i streamem
  `https://stream.sandbox.dudzian.ai/exchanges`.
- **live** – środowisko produkcyjne z polityką watchdog/circuit breaker oraz streamem
  `https://stream.hyperion.dudzian.ai/exchanges`.

## Kanały long-pollowe

Każdy adapter udostępnia pomocnicze metody budujące kanały long-pollowe:

- `stream_order_book(symbol, depth=50)` → kanał `order_book:<symbol>:<depth>`
- `stream_ticker(symbol)` → kanał `ticker:<symbol>`
- `stream_fills(symbol)` → kanał prywatny `fills:<symbol>`

Metody działają zarówno dla spot, jak i futures, dzięki czemu w dashboardzie
można porównywać metryki long-polla gRPC vs REST w jednym zestawie etykiet
(adapter/scope/environment).

## Snapshoty REST

Adaptery udostępniają skróty do najczęstszych operacji REST:

- `fetch_order_book(symbol, limit=None, params=None)`
- `fetch_ticker(symbol, params=None)`
- `fetch_my_trades(symbol, since=None, limit=None, params=None)`

Wywołania przechodzą przez `Watchdog` i przestrzegają skonfigurowanych limitów
ratelimit/retry. Przykład użycia:

```python
creds = ExchangeCredentials(key_id="…", secret="…")
adapter = BitmexFuturesAdapter(creds, environment=Environment.LIVE)
adapter.configure_network(ip_allowlist=())
book = adapter.fetch_order_book("BTC/USDT", limit=50)
```

## Diagnostyka i metryki

Long-pollowe streamy publikują metryki latencji, reconnectów i downtime do
Prometheusa. Eksport snapshotów (`export_metrics_snapshot`) jest wykorzystywany
przez `scripts/list_exchange_adapters.py` do raportów HyperCare i benchmarków
porównujących gRPC vs long-poll.
