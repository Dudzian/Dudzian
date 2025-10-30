# Lokalny runtime bota

Ten przewodnik opisuje uruchomienie kompletnego backendu bota handlowego (AutoTrader + serwer gRPC) na komputerze
programisty. Serwer jest przeznaczony do współpracy z desktopowym UI (Qt/QML) i działa całkowicie offline.

## Wymagania wstępne

* W repozytorium muszą znajdować się wygenerowane stuby gRPC (`bot_core/generated/trading_pb2*.py`).
* Zainstalowane zależności z `pyproject.toml` (moduły `grpcio`, `pandas`, `pyyaml`, `qt` itp.).
* Pliki konfiguracyjne `config/runtime.yaml` oraz `config/core.yaml` zgodne z aktualną strukturą projektu.
* Dane paper trading (cache OHLCV) dostępne lokalnie – pipeline korzysta z danych w katalogu `data/`.

## Uruchomienie

```bash
python scripts/run_local_bot.py --config config/runtime.yaml --entrypoint trading_gui
```

Parametry skryptu:

| Flaga | Opis |
| --- | --- |
| `--config` | Ścieżka do pliku `runtime.yaml` (domyślnie `config/runtime.yaml`). |
| `--entrypoint` | Nazwa punktu wejścia z sekcji `trading.entrypoints` (jeśli brak – używany jest wpis domyślny). |
| `--host`/`--port` | Adres i port nasłuchiwania serwera gRPC (domyślnie `127.0.0.1:0`, czyli port przydzielony dynamicznie). |
| `--ready-file` | Jeśli podano, skrypt zapisze plik JSON z adresem serwera (przydatne dla launcherów). |
| `--no-ready-stdout` | Nie wypisuje komunikatu `ready` na stdout (domyślnie komunikat jest drukowany). |
| `--manual-confirm` | Nie aktywuje automatycznie auto-tradingu – wymaga ręcznego potwierdzenia w UI. |
| `--log-level` | Poziom logowania (`INFO`, `DEBUG`, `WARNING`, …). |

Po uruchomieniu skrypt drukuje komunikat w formacie JSON, np.:

```json
{"event": "ready", "address": "127.0.0.1:9550", "metrics_url": "http://127.0.0.1:9177/metrics", "pid": 12345}
```

Adres należy skonfigurować w aplikacji desktopowej (sekcja ustawień gRPC). Pole `metrics_url` wskazuje lokalny
eksporter Prometheusa wykorzystywany przez zakładkę **Monitoring** – więcej informacji znajdziesz w
[`docs/monitoring_offline.md`](monitoring_offline.md). Skrypt reaguje na `SIGINT`/`SIGTERM` i zamknie w sposób
kontrolowany komponenty AutoTradera, serwer gRPC oraz wątki publikujące stan ryzyka.

## Struktura uruchamianych komponentów

* `build_local_runtime_context` (moduł `bot_core.api.server`) tworzy pipeline paper tradingu w oparciu o konfigurację.
* AutoTrader korzysta z lokalnych danych OHLCV, risk engine’u oraz execution service w trybie paper.
* Serwer gRPC udostępnia usługi: `MarketDataService`, `OrderService`, `RiskService`, `MetricsService` oraz `HealthService`.
* Risk snapshots są publikowane cyklicznie do pamięci pierścieniowej, co umożliwia podgląd w UI.

## Tryb developerski

Wszystkie poświadczenia są przechowywane w pamięci procesów (brak zapisu do keyring/TPM). Do testów można
modyfikować konfigurację `runtime.yaml`, np. zmieniając entrypointy, profile ryzyka czy strategie.

## Zatrzymanie

Wciśnięcie `Ctrl+C` zatrzymuje serwer i kończy pracę AutoTradera. W środowiskach zewnętrznych (np. podczas uruchomienia
z QProcess) rekomendowane jest przesłanie sygnału `SIGTERM`.

