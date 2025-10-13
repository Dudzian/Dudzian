# Strategia Cross-Exchange Arbitrage

## Opis
- Monitoruje różnice bid/ask pomiędzy giełdami `primary_exchange` i `secondary_exchange`.
- Wejście przy `spread_entry` (wyrażonym jako procent ceny mid).
- Zamknięcie przy spadku spreadu poniżej `spread_exit` lub po `max_open_seconds`.
- Limit pozycji `max_notional` zabezpiecza przed przeciążeniem płynności.

## Wymagania operacyjne
- Źródła danych: orderbooki 30s/1m (`cross_exchange_core`).
- Kanał wykonawczy: gRPC/HTTP2 bez WebSocketów; synchronizacja zegarów z NTP.
- Logowanie decyzji z HMAC (współdzielone z `verify_decision_log.py`).
- Wymóg RBAC: token `scheduler-local` z zakresem `runtime.schedule.write`.

## Ryzyka
- Opóźnienia sekundowe między giełdami → telemetry `secondary_delay_ms`.
- Brak par GBP/CHF: raportować w runbooku jako gap.
- Brak VPN/testnet: fallback do `run_trading_stub_server.py` w trybie demo.

## Testy
- Jednostkowe: `tests/test_cross_exchange_arbitrage_strategy.py` (wejście, wyjście, time-stop).
- Integracyjne (plan): symulacja przez `trading_stub_server` z asymetrycznymi cenami.

