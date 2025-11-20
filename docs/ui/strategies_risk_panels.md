# Panele strategii i risk controls (PySide6/QML)

Nowe panele **Strategies** i **RiskControls** są dostępne w układzie głównym Stage6 (PySide6/QML) i służą do szybkiej edycji parametrów strategii oraz limitów ryzyka. Funkcje są dostępne również w trybie demo/offline dzięki wbudowanym mockom danych w `RuntimeService`.

## Jak korzystać

1. Otwórz aplikację `python -m ui.pyside_app` i w menu **Panele** włącz **Strategie** lub **Risk Controls**.
2. Panel **Strategie** prezentuje listę strategii wraz z parametrami (exchange, symbol, TP/SL, wielkość zleceń). Po edycji pól kliknij **Zapisz zmiany**, aby wywołać `RuntimeService.saveStrategyConfig`. Dane są zapisywane w `config/strategies.yaml` (jeśli katalog istnieje) lub w pamięci dla trybu demo.
3. Panel **Risk Controls** udostępnia podstawowe limity: take profit, stop loss, maksymalną liczbę pozycji, limit pozycji w USD, maks. slippage oraz przełącznik kill-switch. Po aktualizacji wybierz **Zapisz limity**, co wywoła `RuntimeService.saveRiskControls` i zapisze wartości do `config/risk_controls.yaml`.
4. Przyciski **Odśwież** w obu panelach wymuszają ponowne wczytanie konfiguracji (przydatne przy ręcznej edycji plików YAML).

## Połączenie z backendem

- `RuntimeService.strategyConfigs` i `RuntimeService.riskControls` dostarczają dane do paneli QML. W trybie demo są wypełniane wartościami przykładowymi (grid/DCA oraz limity SL/TP/slippage).
- `saveStrategyConfig` i `saveRiskControls` walidują payloady, zapisują je do plików YAML (jeśli dostępne) oraz emitują sygnały `strategyConfigsChanged` i `riskControlsChanged`, dzięki czemu panele natychmiast aktualizują widok.
- W środowiskach produkcyjnych te metody można podpiąć pod realne API runtime lub usługę chmurową, zachowując ten sam interfejs QML.

## Dane demo i pliki konfiguracyjne

- Domyślne strategie: `grid_usdt` (BTC/USDT, 7 siatek, TP 1.2%, SL 2.5%) oraz `dca_eth` (ETH/USD, zamówienia bazowe/safety i limity TP/SL).
- Domyślne limity ryzyka: TP 1.5%, SL 2.0%, 5 pozycji, limit pozycji 5000 USD, maks. slippage 0.6%, kill-switch domyślnie wyłączony.
- Jeśli w katalogu `config/` istnieją pliki `strategies.yaml` lub `risk_controls.yaml`, RuntimeService wczyta je zamiast domyślnych mocków.
