# Powłoka desktopowa Qt/QML

## Cel

Powłoka Qt Quick 6 zapewnia lekkie UI do komunikacji z demonem tradingowym (lub stubem gRPC) bezpośrednio przez `botcore.trading.v1`. Interfejs renderuje strumień OHLCV w 60/120 Hz, respektuje parametry `performance_guard` oraz umożliwia szybkie iteracje nad wyglądem i animacjami.

## Wymagania

* Qt 6.5+ (`qtbase`, `qtdeclarative`, `qtquickcontrols2`, `qtcharts`).
* Kompilator C++20, CMake ≥ 3.21.
* gRPC + Protobuf (`libgrpc++`, `libprotobuf`).
* Wygenerowane stuby C++ z `proto/trading.proto` (CMake generuje je automatycznie przy pierwszym buildzie).
* Opcjonalnie Pythonowy stub serwera (`python scripts/run_trading_stub_server.py`).

## Budowanie

```bash
cmake -S ui -B ui/build -GNinja \
  -DCMAKE_PREFIX_PATH="/ścieżka/do/Qt/6.5.0/gcc_64"
cmake --build ui/build
```

Artefakt `bot_trading_shell` znajduje się w `ui/build/bot_trading_shell`.

## Uruchomienie ze stubem gRPC

W pierwszym terminalu uruchom stub z wieloassetowym datasetem i pętlą strumieniową:

```bash
python scripts/run_trading_stub_server.py \
  --dataset data/trading_stub/datasets/multi_asset_performance.yaml \
  --stream-repeat --stream-interval 0.25
```

W drugim terminalu wystartuj powłokę:

```bash
ui/build/bot_trading_shell \
  --endpoint 127.0.0.1:50061 \
  --symbol BTC/USDT \
  --venue-symbol BTCUSDT \
  --exchange BINANCE \
  --granularity PT1M \
  --fps-target 120 \
  --jank-threshold-ms 12.0
```

Domyślne parametry są zgodne z plikiem `ui/config/example.yaml`. Wartości `--max-samples` oraz `--history-limit` pozwalają kontrolować rozmiar buforów i wpływają na wymagania pamięciowe.

## Architektura komponentów

* `src/grpc/TradingClient.*` – cienki klient gRPC pobierający historię i strumień OHLCV.
* `src/models/OhlcvListModel.*` – ring-buffer świec udostępniony QML jako `ListModel` z metodami `candleAt()` i `latestClose()`.
* `src/app/Application.*` – warstwa klejąca CLI ↔ gRPC ↔ QML. Udostępnia `appController` w kontekście QML.
* `qml/components/CandlestickChartView.qml` – widok wykresu z krzyżem celowniczym, autoprzeskalowaniem oraz mechanizmem sample-at-x.
* `qml/components/SidePanel.qml` – wizualizacja parametrów performance guard i statusu połączenia.

## Kolejne kroki

* Podpięcie realnego demona C++ (`/core`) przez TLS i RBAC.
* Dodanie warstwy animacji (Transitions/States) oraz adaptacji „reduce motion” na podstawie metryk gRPC.
* Benchmark QML Profiler 60/120 Hz + automatyczne raportowanie do `MetricsService`.
