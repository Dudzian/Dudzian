# Strategia Volatility Targeting

## Cele
- Stabilizacja zmienności portfela wokół `target_volatility`.
- Adaptacyjna wielkość pozycji ograniczona przedziałem [`min_allocation`, `max_allocation`].
- Rebalans wykonywany, gdy odchylenie od celu przekracza `rebalance_threshold`.

## Parametry
| Parametr | Znaczenie | Domyślnie |
| --- | --- | --- |
| `target_volatility` | docelowa zmienność roczna (w uproszczeniu dzienna) | 0.10 |
| `lookback` | liczba barów w estymacji zmienności | 60 |
| `rebalance_threshold` | relatywne odchylenie triggerujące rebalans | 0.10 |
| `min_allocation` | minimalna waga ekspozycji | 0.15 |
| `max_allocation` | maksymalna waga ekspozycji | 0.95 |
| `floor_volatility` | minimalna zmienność przy estymacji | 0.015 |

## Integracja z pipeline
- Scheduler `volatility_target_daily` pracuje na świecach D1 dla par BTC/ETH.
- Wyniki trafiają do `TradingDecisionJournal` oraz do telemetrii (`latency_ms`, `signals`).
- Risk profile: `conservative` (25 %) oraz `balanced` (20 %) – brak lewarowania.

## Testy
- Jednostkowe: `tests/test_volatility_target_strategy.py`.
- Regresyjne: plan w `stage4_test_plan.md` (sekcja volatility) – obejmuje symulacje stresowe i porównanie z benchmarkiem HV.

