# Panele zarządzania ryzykiem

Nowy widok **RiskControls** dostępny w _Centrum operacyjnym_ łączy dane z silnika ryzyka, pozwalając na:

- przegląd i edycję kluczowych limitów profilu (liczba pozycji, dźwignia, limity strat, docelowa zmienność),
- monitorowanie kosztów i agregatów ekspozycji (dzienne PnL, wartości brutto pozycji, koszty w punktach bazowych),
- śledzenie stanu kill-switcha oraz jego ręczne przełączanie.

## Modele danych

`Application` udostępnia trzy nowe modele QML:

| Właściwość QML | Typ | Opis |
| -------------- | --- | ---- |
| `riskLimitsModel` | `RiskLimitsModel` | edytowalna lista limitów z wartościami minimalnymi/maksymalnymi i formatowaniem procentowym. |
| `riskCostModel` | `RiskCostModel` | zagregowane metryki kosztów/ekspozycji do wyświetlania w tabelach i wykresach. |
| `riskKillSwitchEngaged` | `bool` | flaga kill-switcha synchronizowana z aktualnym stanem silnika ryzyka. |

Wszystkie modele są aktualizowane przy odbiorze `RiskSnapshotData`. Dodatkowe pola (`limits`, `statistics`, `costBreakdown`, `killSwitchEngaged`) zostały dodane do struktur risk snapshot.

## Integracja z silnikiem

- `TradingClient::convertRiskState` mapuje ekspozycje gRPC na konkretne limity (`max_positions`, `daily_loss_limit`, itd.),
  uzupełnia metryki w `statistics`, dekoduje pola `stat:*`/`cost:*` emitowane przez serwis ryzyka oraz wykrywa stan
  `force_liquidation` jako kill-switch.
- `OfflineRuntimeService::buildRiskSnapshot` dostarcza wartości demonstracyjne dla trybu offline (limity, statystyki, koszty oraz stan kill-switcha).
- `RiskLimitsModel` i `RiskCostModel` bazują na `RiskSnapshotData`, zapewniając spójny interfejs QML.
- `ThresholdRiskEngine.snapshot_state` wzbogaca snapshot o sekcje `statistics` i `cost_breakdown`, dzięki czemu testy i UI korzystają z rzeczywistych danych rdzenia.

## Testy

Test `tests/ui/qml/test_risk_panels.py` ładuje komponent QML z wykorzystaniem rzeczywistego `ThresholdRiskEngine`, aby weryfikować:

- renderowanie list limitów i metryk kosztowych,
- synchronizację kill-switcha z kontrolerem,
- możliwość modyfikacji limitów przez model danych.

Testy wymagają środowiska z `PySide6` i ustawiają platformę Qt na `offscreen`.

## Lokalizacja w UI

Panel `RiskControls` został dodany pod dashboard portfela w widoku `OperationsCenter.qml`. Dzięki temu użytkownik ma szybki dostęp do limitów i kosztów bez opuszczania głównego widoku operacyjnego.

