# Bot modes (uczciwe profile produktowe)

Poniższe profile są ograniczone wyłącznie do tego, co repo faktycznie już obsługuje.
Wbudowane profile są ładowane jako zasoby pakietu (`bot_core.product.profiles`), więc działają niezależnie od bieżącego katalogu roboczego.

## 1) `signal_grid`
- Strategia: `grid_trading` (plugin obecny w runtime strategii).
- Egzekucja: `paper`.
- Zastosowanie: sygnałowe wejścia/wyjścia na siatce wokół kotwicy SMA z pasmem ATR.

## 2) `paper_monitoring`
- Strategia: dowolna (profil nie narzuca silnika).
- Egzekucja: tylko `paper`.
- Zastosowanie: monitorowany dry-run bez live order routing.

## 3) `rule_auto_router`
- Strategia: dowolna.
- Egzekucja: `auto`.
- Zastosowanie: regułowy router `live/paper` oparty o środowisko i ustawienia execution policy.

## Jak uruchomić sanity checks
```bash
pytest -q tests/product/test_bot_modes_profiles.py tests/execution/test_mode_policy.py tests/test_execution_paper.py
```
