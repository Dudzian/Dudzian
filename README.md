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

## Aktualizacje
- Nowe wersje dystrybuujemy w formie podpisanych instalatorów. Procedura: [docs/deployment/oem_installation.md](docs/deployment/oem_installation.md).
- Zanim zainstalujesz aktualizację, wykonaj kopię `config/`, `secrets/` i `var/`.

## Kontrybucje
Zobacz [CONTRIBUTING.md](CONTRIBUTING.md) w celu poznania zasad współpracy. Przed zgłoszeniem zmian uruchom testy jednostkowe i integracyjne.

## Licencja
Wszystkie prawa zastrzeżone. Dystrybucja odbywa się na podstawie indywidualnych umów OEM.
