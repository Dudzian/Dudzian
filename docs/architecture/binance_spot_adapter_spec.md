# Specyfikacja adaptera Binance Spot

**Stan dokumentu:** v0.1 (05-10-2025, 13:30 UTC)

## 1. Cel i zakres
Adapter `BinanceSpotAdapter` zapewnia interfejs REST dla rynku spot giełdy Binance w ramach platformy `bot_core`. Celem adaptera jest obsługa:

- odczytu danych prywatnych (saldo konta, otwarte zlecenia, historia transakcji),
- składania zleceń symulowanych (paper) oraz rzeczywistych (po zakończeniu walidacji),
- agregacji danych publicznych OHLCV i tickerów na potrzeby wyceny portfela oraz warstwy danych.

Adapter musi wspierać profile środowiskowe `LIVE`, `PAPER`, `TESTNET` oraz uwzględniać pipeline wdrożeniowy demo → paper → live.

## 2. Mapowanie symboli i instrumentów
- Podstawowe pary: `BTC/USDT`, `ETH/USDT`.
- Rozszerzone pary: `SOL/USDT`, `BNB/USDT`, `XRP/USDT`, `ADA/USDT`, `LTC/USDT`, `MATIC/USDT`, `BTC/EUR`, `ETH/EUR`, `BTC/PLN`, `ETH/PLN`.
- Normalizacja nazw:
  - Adapter zwraca symbole w formacie `BASE/QUOTE` (np. `BTC/USDT`).
  - Konwersja z notacji Binance (`BTCUSDT`) do formatu domenowego następuje w warstwie danych i strategii poprzez funkcję `normalize_symbol` (do implementacji) wykorzystującą mapę instrumentów.
- Reguły fallback:
  - Jeśli giełda nie wspiera bezpośredniego kwotowania w walucie docelowej (np. `BTC/PLN`), adapter używa triangulacji przez waluty pośrednie (`USDT`, `BUSD`, `USDC`, `EUR`).
  - Lista pośredników jest konfigurowalna i przechowywana w ustawieniach adaptera (`secondary_valuation_assets`).

## 3. Endpoints i limity
- Publiczne dane rynkowe:
  - `/api/v3/klines` – OHLCV; limit 1200 świec na żądanie; domyślny interwał `1d`, walidacja interwałów `1h`, `15m`.
  - `/api/v3/ticker/price` i `/api/v3/ticker/bookTicker` – wycena instrumentów; limit 1200 requestów/min.
- Dane prywatne (wymagany podpis HMAC SHA256):
  - `/api/v3/account`, `/api/v3/myTrades`, `/api/v3/openOrders`, `/api/v3/order` (POST/DELETE).
- Limity przepustowości:
  - `X-MBX-USED-WEIGHT-1M` oraz `X-MBX-USED-WEIGHT-1S` monitorowane i logowane.
  - Retry/backoff po kodach 418/429, z eksponencjalnym opóźnieniem startowym 0.4 s, współczynnikiem 2 i jitterem 50–350 ms.
  - Po przekroczeniu `_MAX_RETRIES` adapter zgłasza `ExchangeThrottlingError` (do zaimplementowania) i eskaluje alert do kanału `telegram.alerts.binance`.

## 4. Architektura i moduły
```
bot_core/exchanges/binance/
├── __init__.py
├── spot.py              # logika podpisu, retry, przeliczania sald
├── symbols.py           # (planowane) mapa instrumentów i normalizacja
└── tests/
```

- `spot.py` odpowiada za wywołania HTTP oraz konwersję odpowiedzi Binance do struktur domenowych (`AccountSnapshot`, `OrderResult`).
- `symbols.py` (do dodania) będzie przechowywać mapę instrumentów oraz funkcje walidacji i normalizacji symboli.
- Moduł testowy będzie zawierał zestaw testów jednostkowych i integracyjnych z wykorzystaniem fixture’ów JSON.

## 5. Bezpieczeństwo i sieć
- Klucze API przechowywane w natywnym magazynie sekretów (Keyring) zgodnie z polityką bezpieczeństwa; brak zapisywania w plikach konfiguracyjnych.
- Oddzielne klucze i allowlisty IP dla środowisk `PAPER` i `LIVE`. Testnet korzysta z dedykowanej domeny `testnet.binance.vision` (wymaga tunelu VPN).
- Wymuszone nagłówki bezpieczeństwa: `X-MBX-APIKEY`, `Content-Type: application/json`.
- W logach usuwane są dane wrażliwe (`signature`, `recvWindow`, `timestamp`).
- Rotacja kluczy co 90 dni; adapter powinien potrafić pobrać aktualne dane z menedżera sekretów bez restartu procesu (watcher TBA).

## 6. Obsługa błędów i alerty
- Kategorie błędów:
  - `HTTPError` 4xx/5xx – mapowane na `ExchangeAPIError` z kodem i payloadem.
  - `URLError`, timeouts – mapowane na `ExchangeNetworkError` z logowaniem metryk latencji.
  - Błędy podpisu/uwierzytelnienia – `ExchangeAuthError`, natychmiastowa eskalacja.
- Alertowanie:
  - Po ≥5 błędach 429 w 10 minut otwierany incydent (Telegram + e-mail), monitorowany przez warstwę alertów.
  - Po 15 minutach nierozwiązania incydentu wysyłany SMS zgodnie z polityką.

## 7. Telemetria i logowanie
- Metryki Prometheus (do wdrożenia):
  - `binance_spot_http_latency_seconds{endpoint,method}` – histogram.
  - `binance_spot_retries_total{endpoint,reason}` – licznik.
  - `binance_spot_signed_requests_total` – licznik requestów podpisanych.
- Logi strukturalne JSON z polami: `exchange`, `environment`, `endpoint`, `status`, `latency_ms`, `retry_count`.
- Health check runtime’u sprawdza:
  - możliwość pobrania tickerów (`/api/v3/ticker/price`),
  - status allowlisty IP (sprawdzenie `403 Forbidden`).

## 8. Wymagania testowe
| Etap | Zakres | Narzędzia |
| --- | --- | --- |
| Testy jednostkowe | Retry, podpis, parsowanie sald, triangulacja kursów | `pytest`, fixture’y lokalne |
| Backtest OOS | Walidacja danych OHLCV z cache Parquet (`BTC/USDT`, `ETH/USDT`) | `python scripts/validate_parquet_cache.py` |
| Testnet | Symulacja zleceń limit/market na `testnet.binance.vision` z kontrolą limitów | VPN + allowlist, konto testnet |
| Paper trading | Integracja z lokalnym symulatorem prowizji, porównanie fill rate z testnetem | `paper_trading_engine` |
| Live (po audycie) | Checklista compliance, testy w okienkach niskiej płynności | Runbook operacyjny |

## 9. Roadmap wdrożeniowy
1. **Implementacja** (aktywny etap) – dokończenie obsługi OHLCV, walidacji limitów, zleceń IOC/limit/market.
2. **Testy jednostkowe** – rozszerzenie obecnych testów o przypadki błędów sieciowych i konwersję wielowalutową.
3. **Backtest OOS** – wykorzystanie danych Parquet do walidacji integralności i pokrycia instrumentów.
4. **Testnet** – konfiguracja VPN, rejestracja kluczy, odtworzenie scenariuszy z checklisty bezpieczeństwa.
5. **Paper trading** – porównanie metryk z testnetem, analiza poślizgów.
6. **Przegląd bezpieczeństwa i compliance** – audyt kodu, rotacja kluczy, dokumentacja incydentów.
7. **Dokumentacja końcowa** – aktualizacja runbooków, checklist, wpięcie do CI.

## 10. Otwarte zadania i zależności
- Rozszerzenie mapy symboli o waluty drugiego rzędu (np. GBP, CHF) po walidacji strategii.
- Integracja z magazynem sekretów oraz polityką rotacji kluczy.
- Przygotowanie fixture’ów OHLCV do testów jednostkowych (mocki plików Parquet).
- Skonfigurowanie metryk Prometheus i eksportera dla adaptera.
- Koordynacja z zespołem sieciowym w sprawie VPN i allowlist.

## 11. Decyzje projektowe
- Retry z eksponencjalnym backoffem i jitterem jest realizowany lokalnie (bez zewnętrznej biblioteki), aby ograniczyć zależności i umożliwić łatwiejszą kontrolę w testach.
- Publiczne endpointy w środowisku PAPER korzystają z produkcyjnych danych, co upraszcza pipeline backfillu przy zachowaniu separacji kluczy prywatnych.
- Triangulacja kursów wykorzystuje listę walut pośrednich z konfiguracji, co umożliwia rozszerzenie na pary PLN/EUR bez wprowadzania stałych zależności.

## 12. Ryzyka
1. **Brak danych historycznych na testnecie** – konieczność używania produkcyjnego endpointu dla danych publicznych; wymaga dodatkowych zabezpieczeń przed wyciekami kluczy.
2. **Zmienność limitów API** – Binance okresowo aktualizuje limity; adapter musi monitorować nagłówki `Retry-After` i `X-MBX-USED-WEIGHT` (planowana implementacja).
3. **Zależność od infrastruktury VPN** – opóźnienia lub brak dostępności VPN może blokować testy i wdrożenia paper/live.

