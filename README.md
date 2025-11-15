# Lokalny bot handlowy

## Opis projektu
Repozytorium zawiera kompletny stack aplikacji desktopowej do automatycznego handlu kryptowalutami. Projekt obejmuje modułowy backend (`bot_core/**`), interfejs Qt/QML (`ui/**`), narzędzia do AI, strategii, licencjonowania oraz procesy dystrybucji offline.

## Kluczowe funkcjonalności
- Integracja z najważniejszymi giełdami (Binance, Coinbase, Kraken, OKX, Bitget, Bybit) w trybie live i papierowym.
- Zaawansowany pipeline AI: trening walk-forward, walidacja jakości, automatyczne retrainingi.
- Bogaty katalog strategii (spot, futures, opcje, hedging cross-exchange) oraz marketplace presetów.
- Monitoring offline, alerty i dashboard portfela.
- Instalatory na Windows/macOS/Linux wraz z natywnym keyringiem.

## Szybki start
1. Skonfiguruj środowisko (Python 3.11, Poetry) i zainstaluj zależności: `poetry install` (polecenie
   zainstaluje również `keyring`; na Linuksie wymagany jest pakiet `secretstorage`).
2. Zainstaluj pakiety rdzeniowe (`bot_core`, `core`) w trybie deweloperskim, aby moduły były dostępne bez ręcznych modyfikacji `sys.path`: `python -m pip install -e .[compression]` (extras `compression` doinstaluje `brotli` lub `brotlicffi` oraz `zstandard`, zapewniając obsługę strumieni kompresowanych brotli i zstd).
3. Przygotuj konfigurację runtime: `python scripts/migrate_runtime_config.py --output config/runtime.yaml`.
4. Uruchom pipeline papierowy: `poetry run python scripts/run_local_bot.py --paper`.
5. Z aplikacji desktopowej (Qt) przeprowadź konfigurację w kreatorze.

Szczegółowe instrukcje znajdują się w dokumentacji:
- [Przewodnik użytkownika](docs/user_manual/index.md)
- [Troubleshooting](docs/user_manual/troubleshooting.md)
- [Plan wsparcia](docs/support/plan.md)
- [Instalacja i budowa instalatorów](docs/deployment/installer_build.md)
- [Monitorowanie offline](docs/monitoring_offline.md)
- [Benchmark Stage6 vs CryptoHopper](docs/benchmark/cryptohopper_comparison.md)

## Tryb cloud/serwerowy (behind flag)
- Moduł `bot_core.cloud` udostępnia serwer gRPC, który można uruchomić osobnym procesem: `python scripts/run_cloud_service.py --config config/cloud/server.yaml --emit-stdout`.
- Konfiguracja znajduje się w `config/cloud/server.yaml` i oprócz hosta/portu, entrypointu runtime i whitelisty usług zawiera sekcję `security.allowed_clients`. Każdy wpis określa parę `license_id`/`fingerprint` oraz źródło klucza HMAC (inline, plik lub zmienna ENV). Próby autoryzacji są logowane w `logs/security_admin.log`.
- `CloudAuthService.AuthorizeClient` realizuje obowiązkowy handshake HWID/licencji. Klient podpisuje payload `{license_id, fingerprint, nonce}` przy pomocy `sign_license_payload` i wysyła go do RPC – w odpowiedzi otrzymuje token `CloudSession`, który należy przekazywać w nagłówku `Authorization: CloudSession <token>` do wszystkich wywołań (`RuntimeService`, `MarketplaceService`, `MarketDataService` itd.). Tokeny wygasają po czasie ustawionym w `session_ttl_seconds`.
- Serwer inicjuje harmonogramy AI i synchronizację marketplace’u, a po starcie zapisuje payload `{ "event": "ready", "address": "host:port" }` (stdout lub plik wskazany flagą `--ready-file`).
- Sekcja `cloud` w `config/runtime.yaml` wraz z `config/cloud/client.yaml` opisuje profile zdalne – włączenie procesu `scripts/run_local_bot.py --enable-cloud-runtime` publikuje payload `ready` z sekcją `cloud` i nie uruchamia lokalnego kontekstu.
- Tryb cloud nie jest aktywowany automatycznie – dopiero ustawienie flagi (CLI/UI) przełącza dystrybucję desktopową z lokalnego `LocalRuntimeServer` na wskazany backend chmurowy.

> **Szybki skrót benchmarku:** Stage6 jest na parytecie strategii z przewagą automatyzacji i compliance; największą luką pozostaje integracja UI (feed gRPC „Decyzje AI”) oraz skalowanie marketplace’u presetów.
>
> **Nowości:** tablica wyników i harmonogram działań korygujących w benchmarku są aktualizowane miesięcznie na podstawie `docs/runtime/status_review.md` i checklisty wsparcia. Dzięki temu zespoły produktowe widzą, kto odpowiada za domykanie luk i jakie są cele metryczne na kolejne kwartały.
>
> **Kronika benchmarku:** `docs/benchmark/cryptohopper_comparison.md` prowadzi historię aktualizacji oraz opisuje procedurę zbierania metryk (hypercare, adaptery giełdowe, marketplace, UI, compliance). To miejsce referencyjne przy audytach releasowych i syncach produktowych.

## Aktualizacje
- Nowe wersje dystrybuujemy w formie podpisanych instalatorów. Procedura: [docs/deployment/oem_installation.md](docs/deployment/oem_installation.md).
- Zanim zainstalujesz aktualizację, wykonaj kopię `config/`, `secrets/` oraz katalogu danych użytkownika (`~/.dudzian/`).
- Komponenty runtime, dokumentacja i runbooki korzystają wyłącznie z przestrzeni `bot_core.*`; archiwalny pakiet `KryptoLowca` został usunięty. Mapowanie najczęściej używanych modułów znajdziesz w [docs/migrations/kryptolowca_namespace_mapping.md](docs/migrations/kryptolowca_namespace_mapping.md).
- Dawny katalog z archiwalnym botem został zlikwidowany – w repozytorium nie ma już shimów ani kodu wykonywalnego z poprzedniej warstwy.
- W `archive/` pozostawiamy wyłącznie materiały historyczne (np. [docs/archive/trading_model_pipeline.md](docs/archive/trading_model_pipeline.md)),
  aby nie rozpraszać zespołu nieużywaną implementacją. Aktywny kod żyje w `bot_core/**`.

## Kontrybucje
Zobacz [CONTRIBUTING.md](CONTRIBUTING.md) w celu poznania zasad współpracy. Przed zgłoszeniem zmian uruchom testy jednostkowe i integracyjne.

## Licencja
Wszystkie prawa zastrzeżone. Dystrybucja odbywa się na podstawie indywidualnych umów OEM.
