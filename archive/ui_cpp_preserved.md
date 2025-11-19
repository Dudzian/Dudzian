# Legacy Qt/C++ shell (archived)

> **Status:** nieutrzymywany. Plik służy wyłącznie jako referencja OEM dla dawnych
> buildów `bot_trading_shell`. Aktualny interfejs użytkownika to PySide6/PyQt6 +
> Qt Quick – patrz `ui/README.md`.

## Wymagania

* Qt 6.7+ (`qtbase`, `qtdeclarative`, `qtquickcontrols2`, `qtcharts`).
* Kompilator C++20, CMake ≥ 3.21.
* gRPC + Protobuf (`libgrpc++`, `libprotobuf`).
* Wygenerowane stuby C++ z `proto/trading.proto` (CMake generuje je automatycznie przy pierwszym buildzie).
* Opcjonalnie Pythonowy stub serwera (`python scripts/run_trading_stub_server.py`).

## Budowanie wariantu C++ (legacy)

```bash
cmake -S ui -B ui/build -GNinja \
  -DCMAKE_PREFIX_PATH="/ścieżka/do/Qt/6.7.0/gcc_64"
cmake --build ui/build
```

Artefakt `bot_trading_shell` znajduje się w `ui/build/bot_trading_shell` i służy jako
referencja dla OEM korzystających z natywnego builda.
