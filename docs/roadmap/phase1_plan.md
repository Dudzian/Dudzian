# Roadmapa etapu 1 – Integracja Binance i fundamenty core

Dokument opisuje harmonogram i zakres zadań, które doprowadzą bota do stabilnego trybu paper
trading zintegrowanego z Binance (spot + futures). Plan bazuje na wymaganiach biznesowych,
bezpieczeństwa i compliance przedstawionych w specyfikacji. Każdy etap kończy się zestawem
konkretnych testów (backtest → paper), zanim przejdziemy do kolejnych prac.

## Założenia główne

- Architektura modułowa w pakiecie `bot_core` – interfejsy adapterów, warstwa danych,
  strategie, risk engine, execution i alerty.
- Brak zmian w monolicie `KryptoLowca` poza krytycznymi poprawkami; nowy rozwój to `bot_core`.
- Sekrety przechowywane w natywnych keychainach (Windows Credential Manager, macOS Keychain,
  GNOME Keyring/age dla środowisk headless).
- Testy przeprowadzamy w kolejności: **backtest** → **paper trading** na Binance Testnet →
  **ograniczony live** (po zakończeniu etapu 1).

## Harmonogram sześciotygodniowy

| Tydzień | Zakres | Artefakty | Testy końcowe |
| --- | --- | --- | --- |
| 1 | Finalizacja interfejsów (`exchanges`, `data`, `risk`, `execution`, `alerts`), struktura `config/core.yaml`, rejestry instrumentów i profili. | ADR #001 (architektura modułowa), zaktualizowany `config/core.yaml`. | ✅ `python -m compileall bot_core`<br>✅ `pytest tests/test_config_loader.py` |
| 2 | Implementacja `BinanceSpotAdapter` i `BinanceFuturesAdapter` (public + signed REST, testnet/live, rozdział uprawnień). | Adaptery + testy podpisów (`tests/test_binance_spot_adapter.py`, `tests/test_binance_futures_adapter.py`). | ✅ `pytest tests/test_binance_spot_adapter.py tests/test_binance_futures_adapter.py` |
| 3 | Warstwa danych: `OHLCVBackfillService`, `CachedOHLCVSource`, magazyn SQLite/Parquet, konfiguracja uniwersów instrumentów. | Skrypt `python scripts/backfill_ohlcv.py`, aktualizacja dokumentacji danych. | ✅ `pytest tests/test_ohlcv_backfill.py`<br>✅ manualny backfill BTC/USDT D1 (dry-run) |
| 4 | Risk engine (`ThresholdRiskEngine`) z profilami konserwatywny/zbalansowany/agresywny/ręczny, integracja z konfiguracją. | Testy `tests/test_risk_engine.py`, raport mapowania limitów. | ✅ `pytest tests/test_risk_engine.py` |
| 5 | `PaperTradingExecutionService`, router alertów z kanałami Telegram/E-mail/SMS, audyt alertów i health-check. | Testy `tests/test_paper_execution.py`, `tests/test_alerts.py`, dokument procedury alertów. | ✅ `pytest tests/test_paper_execution.py tests/test_alerts.py`<br>⚠️ Manualny dry-run alertów (mock API) |
| 6 | Integracja: `bootstrap_environment`, scenariusz end-to-end (backtest → paper), dokumentacja użytkownika i checklisty bezpieczeństwa. | `docs/runbooks/paper_trading.md`, log audytu papierowego dnia. | ✅ `pytest tests/test_runtime_bootstrap.py`<br>✅ symulacja paper tradingu BTC/USDT (24h na testnecie) |

> ⚠️ Wszystkie testy manualne (backfill, alerty, paper trading) wykonujemy wyłącznie na kontach
demo/testnet i z kluczami bez uprawnień do wypłat. W logach nie zapisujemy pełnych sekretów –
maskujemy klucze i stosujemy audyt append-only.

## Kryteria „Definition of Done” etapu 1

1. **Strategia D1** (trend-following + momentum breakout) działa w trybie paper trading,
   korzystając z danych backfillowanych lokalnie i profilu ryzyka zbalansowanego.
2. **Risk engine** egzekwuje dzienne limity strat, ekspozycję na instrument, ATR stop oraz
   hard-stop drawdown dla wszystkich profili.
3. **Alerty**: sygnały wejścia/wyjścia, przekroczenia progów ryzyka, błędy adapterów/API i
   health-check są wysyłane kanałami Telegram/E-mail/SMS z wpisem w audycie.
4. **Observability**: metryki podstawowe (latencja zleceń, fill rate, error rate) dostępne w
   warstwie telemetrycznej (np. Prometheus endpoint / plik JSON).
5. **Bezpieczeństwo**: klucze API zapisane w keychainie, rotacja i allowlist IP udokumentowane,
   a wszystkie środowiska korzystają z modelu najmniejszych uprawnień.

## Następne etapy (wysoki poziom)

- **Etap 2 – Kraken**: adaptery spot/futures, specyficzne opłaty i wymogi regulacyjne,
  symulator paper trading przy braku pełnego sandboxa.
- **Etap 3 – Zonda**: adapter spot, mapowanie fiatów (PLN/EUR), integracja z risk engine i
  raportowaniem.
- **Etap 4 – Rozszerzenia strategii**: biblioteka strategii (mean reversion, volatility targeting,
  arbitraż), scheduler, marketplace presetów.
- **Etap 5 – Observability i compliance+**: pełne raporty PDF/CSV, podpisane archiwa, monitoring SLO,
  moduł rotacji kluczy z przypomnieniami.

## Checklisty testowe dla kolejnych PR-ów

Każda iteracja powinna zawierać w opisie PR sekcję testów z następującego szablonu:

- ✅ `python -m compileall bot_core`
- ✅/⚠️ `pytest <wybrane moduły>`
- ⚠️ Manualny backtest/paper trading (opis zakresu, środowisko demo/testnet)
- ⚠️ Dry-run alertów (jeśli zmiany dotyczą powiadomień)

Utrzymanie tej checklisty zapewni spójność jakościową i ułatwi przejście do audytowalnego
procesu CI/CD w późniejszym czasie.
