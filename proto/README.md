# Kontrakty gRPC (`proto/`)

Ten katalog zawiera definicję Protobuf v1 dla komunikacji pomiędzy powłoką desktopową Qt/QML a rdzeniem
`bot_core`. Zgodnie z architekturą shell↔daemon wszystkie operacje odbywają się przez gRPC (HTTP/2) –
brak WebSocketów i brak bezpośrednich połączeń HTTP z giełdami po stronie UI.

## Pliki

- `trading.proto` – pakiet `botcore.trading.v1` z usługami `MarketDataService`, `OrderService`,
  `RiskService`, `MetricsService` oraz `HealthService`. Zawiera kontrakty `GetOhlcvHistory`,
  `StreamOhlcv`, `SubmitOrder`, `CancelOrder`, `RiskState`, `StreamRiskState`, a także telemetryczne
  `MetricsSnapshot`.

## Generowanie stubów

Stosujemy jeden źródłowy plik `.proto` i generujemy klienta/serwer w różnych językach. Docelowo buildy
rdzenia (C++) i narzędzi Pythonowych będą korzystać z tych samych artefaktów.

Przykładowe generowanie stubów Python + gRPC:

```bash
poetry run python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=bot_core/generated \
  --grpc_python_out=bot_core/generated \
  proto/trading.proto
```

Generowanie stubów C++ (dla demona) w katalogu `core/generated`:

```bash
protoc \
  --proto_path=proto \
  --cpp_out=core/generated \
  --grpc_out=core/generated \
  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
  proto/trading.proto
```

> **Wskazówka:** W pipeline CI przygotujemy dedykowany krok budujący stuby i publikujący je jako artefakt.
> Komendy powyżej służą jako punkt startowy do dalszej automatyzacji.

## Zasady utrzymania kontraktu

- Plik `trading.proto` jest traktowany jako kontrakt `v1`. Breaking changes są zabronione bez podniesienia
  wersji pakietu (np. `botcore.trading.v2`).
- Przed dodaniem nowych pól należy przygotować ADR i aktualizację testów golden.
- Każda zmiana powinna przejść przez `buf lint` / `buf breaking` (zaplanowane do wdrożenia w pipeline CI).
- Klient QML korzysta tylko z gRPC – UI nie łączy się bezpośrednio z giełdami.
