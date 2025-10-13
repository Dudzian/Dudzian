# Strategia Mean Reversion

## Założenia
- Horyzont: intraday (1–5 min) na parach o wysokiej płynności (BTC/ETH do USDT i EUR).
- Sygnał: z-score ceny zamknięcia względem okna `lookback`, filtrowany progiem zmienności (`volatility_cap`).
- Kierunek: kupno przy ujemnym odchyleniu (`entry_zscore`), sprzedaż/short przy dodatnim.
- Wyjście: domknięcie przy powrocie do pasma (`exit_zscore`), osiągnięciu limitu czasu (`max_holding_period`) lub wzroście zmienności powyżej 1.5× `volatility_cap`.
- Wolumen minimalny (`min_volume_usd`) chroni przed sygnałami na płytkich rynkach.

## Parametry konfiguracyjne
| Parametr | Opis | Wartość domyślna |
| --- | --- | --- |
| `lookback` | liczba barów w obliczeniu średniej/odchylenia | 96 |
| `entry_zscore` | próg otwarcia pozycji | 1.8 |
| `exit_zscore` | próg zamknięcia pozycji | 0.4 |
| `max_holding_period` | maks. liczba barów w pozycji | 12 |
| `volatility_cap` | dopuszczalna zrealizowana zmienność | 0.04 |
| `min_volume_usd` | minimalny obrót świecy | 100 000 |

## Zależność od profili ryzyka
- `conservative`: waga 20 % w `strategy_allocations`; aktywowany tylko na parze BTC/EUR z surowszym monitorowaniem płynności.
- `balanced`: 30 % udział w ekspozycji, praca na koszyku `stat_arb_core`.
- `aggressive`: 25 % udział i dopuszczalny short na USDT cross.

## Wymagania danych
- Dane OHLCV 1m/5m z obu giełd (`stat_arb_core`).
- Telemetria: liczba sygnałów, średni z-score, dryf schedulera.
- Audyt: każda decyzja logowana w `TradingDecisionJournal` z HMAC.

