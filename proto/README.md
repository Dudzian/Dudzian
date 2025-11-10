# Kontrakty gRPC (`proto/`)

Ten katalog zawiera definicję Protobuf v1 dla komunikacji pomiędzy powłoką desktopową Qt/QML a rdzeniem
`bot_core`. Zgodnie z architekturą shell↔daemon wszystkie operacje odbywają się wyłącznie przez gRPC
(HTTP/2) – brak WebSocketów i brak bezpośrednich połączeń HTTP z giełdami po stronie UI.

## Pliki

- `trading.proto` – pakiet `botcore.trading.v1` z usługami `MarketDataService`, `OrderService`,
  `RiskService`, `MetricsService` oraz `HealthService`. Zawiera kontrakty `GetOhlcvHistory`,
  `StreamOhlcv`, `SubmitOrder`, `CancelOrder`, `RiskState`, `StreamRiskState`, a także telemetryczne
  `MetricsSnapshot`.

## Generowanie stubów

Stosujemy jeden źródłowy plik `.proto` i generujemy klienta/serwer w różnych językach. Docelowo buildy
rdzenia (C++) i narzędzi Pythonowych będą korzystać z tych samych artefaktów.

### Generowanie stubów Python + gRPC

```bash
poetry run python scripts/generate_trading_stubs.py
```

Skrypt domyślnie generuje artefakty Pythona i C++. W razie potrzeby można ograniczyć zakres
(`--skip-python` / `--skip-cpp`) lub wskazać niestandardowe katalogi wyjściowe.

### Ręczne wywołanie protoc

Jeżeli potrzebna jest niestandardowa konfiguracja, można uruchomić `protoc` bezpośrednio.
Przykład generowania stubów C++ (dla demona) w katalogu `core/generated`:

```bash
protoc \
  --proto_path=proto \
  --cpp_out=core/generated \
  --grpc_out=core/generated \
  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
  proto/trading.proto
```

> **Wskazówka:** W katalogu `deploy/ci/github_actions_proto_stubs.yml` znajdziesz przykładowy
> workflow GitHub Actions, który instaluje `protoc`, uruchamia `buf lint`/`buf breaking`,
> odpala skrypt `generate_trading_stubs.py` oraz publikuje wygenerowane artefakty jako plik job artefact.

### Walidacja kontraktu przy pomocy Buf

W katalogu `proto/` znajduje się plik `buf.yaml` definiujący reguły lint/breaking. Lokalne sprawdzenie:

```bash
buf lint proto
buf breaking proto --against '.git#branch=main,subdir=proto'
```

Buf wykorzystujemy również w CI, dlatego przed push warto uruchomić te polecenia lokalnie (wymaga
zainstalowanego `buf`, patrz [instrukcje](https://buf.build/docs/installation)).

## Zasady utrzymania kontraktu

- Plik `trading.proto` jest traktowany jako kontrakt `v1`. Breaking changes są zabronione bez podniesienia
  wersji pakietu (np. `botcore.trading.v2`).
- Przed dodaniem nowych pól należy przygotować ADR i aktualizację testów golden.
- Każda zmiana przechodzi przez `buf lint` / `buf breaking` w pipeline CI; rekomendowane jest także
  lokalne uruchomienie powyższych poleceń przed wysyłką PR.
- Klient QML korzysta tylko z gRPC – UI nie łączy się bezpośrednio z giełdami.

## Migracja z REST/poprzedniej generacji

Poprzednia warstwa REST została usunięta. Wszystkie integracje powinny korzystać z Stage6 oraz
bieżących usług gRPC (`MarketDataService`, `OrderService`, `RiskService`, `MetricsService`,
`HealthService`). Narzędzia deweloperskie (m.in. `scripts/generate_trading_stubs.py`, feed Stage6)
obsługują wyłącznie aktualny protokół. Odwołania do dawnych endpointów REST należy usunąć podczas
migracji – próba ich użycia zakończy się błędem po stronie klienta lub serwera.
