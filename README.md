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
5. Uruchom nowy klient PySide6 (`python -m ui.pyside_app --config ui/config/example.yaml`) i przeprowadź konfigurację w kreatorze.

`AutoTraderAIGovernorRunner` udostępnia oficjalne metody `run_cycle()` (manualny krok decyzji)
i `run_until()` (tryb launch-and-forget). `scripts/run_local_bot.py` po zbudowaniu pipeline
wykonuje kilka cykli runnera na rzeczywistym `DecisionOrchestratorze` i publikuje snapshot
AI Governora w komunikacie `ready`, więc UI natychmiast otrzymuje historię trybów
scalping/hedge/grid nawet przed nawiązaniem feedu gRPC.

Szczegółowe instrukcje znajdują się w dokumentacji:
- [Przewodnik użytkownika](docs/user_manual/index.md)
- [Troubleshooting](docs/user_manual/troubleshooting.md)
- [Plan wsparcia](docs/support/plan.md)
- [Instalacja i budowa instalatorów](docs/deployment/installer_build.md)
- [Monitorowanie offline](docs/monitoring_offline.md)
- [Raport katalogu presetów (2025-01-15)](reports/strategy/presets_2025-01-15.md)
- Walidacja katalogu: `python scripts/validate_marketplace_presets.py --presets config/marketplace/presets --min-count 15`
- [Benchmark Stage6 vs CryptoHopper](docs/benchmark/cryptohopper_comparison.md)

## Alerty SLA feedu i HyperCare
- Progi SLA (`latencja p95`, reconnecty, downtime) konfigurujesz w `observability.feed_sla` w `config/runtime.yaml`. Domyślne wartości (2.5/5 s, 3/6 reconnectów, 30/120 s downtime) są zgodne z runbookiem HyperCare i mogą być nadpisane zmiennymi środowiskowymi `BOT_CORE_UI_FEED_*`.
- Desktopowy `RuntimeService` korzysta z `UiTelemetryAlertSink`, aby każdą zmianę stanu (`warning`, `critical`, `recovered`) wysłać do kanałów HyperCare (Telegram/Signal/e-mail) oraz zapisać w `logs/ui_telemetry_alerts.jsonl`.
- Jeśli włączysz flagę `--enable-cloud-runtime` w `scripts/run_local_bot.py` (lub odpowiednik w UI), proces nie startuje lokalnego runtime, tylko publikuje payload `ready` opisujący klienta z `config/cloud/client.yaml`. Ten sam plik jest używany w UI (`runtimeService.cloudRuntimeStatus`) – można więc jednym kliknięciem przełączać się lokalnie ↔ cloud oraz widzieć status handshake’u HWID/licencji.
- Aktywacja flagi cloudowej wymaga podpisanej deklaracji `cloud.enabled_signed` w `config/runtime.yaml`. Domyślny wpis wskazuje na `var/runtime/cloud_flag.json` i podpis `var/runtime/cloud_flag.sig`, które muszą zostać potwierdzone HMAC (`CLOUD_RUNTIME_FLAG_SECRET`) albo Ed25519 zanim `bot_core/runtime/local_gateway.py` przyjmie `--enable-cloud-runtime` lub `scripts/run_cloud_service.py` wystartuje backend.
- Przykładowy workflow HMAC: właściciel eksportuje sekret (`export CLOUD_RUNTIME_FLAG_SECRET=base64:...`), tworzy payload `{"enabled": true, "issued_by": "SecOps", "expires_at": "2025-01-31T23:59:59Z"}` i podpisuje go helperem `bot_core.security.signing.build_hmac_signature`:

```bash
python - <<'PY'
import json, os, base64
from pathlib import Path
from bot_core.security.signing import build_hmac_signature

payload = {
    "enabled": True,
    "issued_by": "SecOps",
    "expires_at": "2025-01-31T23:59:59Z",
    "reason": "Go-live window"
}
flag_path = Path("var/runtime/cloud_flag.json")
sig_path = Path("var/runtime/cloud_flag.sig")
secret_value = os.environ["CLOUD_RUNTIME_FLAG_SECRET"]
if secret_value.startswith("base64:"):
    key = base64.b64decode(secret_value[7:])
elif secret_value.startswith("hex:"):
    key = bytes.fromhex(secret_value[4:])
else:
    key = secret_value.encode("utf-8")
flag_path.parent.mkdir(parents=True, exist_ok=True)
flag_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
signature = build_hmac_signature(payload, key=key)
sig_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
PY
```

Wariant Ed25519 polega na podpisaniu `bot_core.security.signing.canonical_json_bytes(payload)` prywatnym kluczem i zapisaniu wartości base64 w `cloud_flag.sig`. Dzięki temu zarówno klient UI, jak i serwer wymuszą, by tylko właściciel mógł aktywować tryb cloud.

## Tryb cloud/serwerowy (behind flag)
- Moduł `bot_core.cloud` udostępnia serwer gRPC, który można uruchomić osobnym procesem: `python scripts/run_cloud_service.py --config config/cloud/server.yaml --emit-stdout`.
- Konfiguracja znajduje się w `config/cloud/server.yaml` i oprócz hosta/portu, entrypointu runtime i whitelisty usług zawiera sekcję `security.allowed_clients`. Każdy wpis określa parę `license_id`/`fingerprint` oraz źródło klucza HMAC (inline, plik lub zmienna ENV). Próby autoryzacji są logowane w `logs/security_admin.log`.
- `CloudAuthService.AuthorizeClient` realizuje obowiązkowy handshake HWID/licencji. Klient podpisuje payload `{license_id, fingerprint, nonce}` przy pomocy `sign_license_payload` i wysyła go do RPC – w odpowiedzi otrzymuje token `CloudSession`, który należy przekazywać w nagłówku `Authorization: CloudSession <token>` do wszystkich wywołań (`RuntimeService`, `MarketplaceService`, `MarketDataService` itd.). Tokeny wygasają po czasie ustawionym w `session_ttl_seconds`.
- Serwer inicjuje harmonogramy AI i synchronizację marketplace’u, a po starcie zapisuje payload `{ "event": "ready", "address": "host:port" }` (stdout lub plik wskazany flagą `--ready-file`).
- Tryb cloud korzysta z `config/cloud/client.yaml` – flagi CLI (`--enable-cloud-runtime --cloud-client-config ...`) oraz UI ładują ten sam manifest i wykonują handshake `CloudAuthService`. Panel statusu pokazuje fingerprint, licencję i stan tokenu, a wszystkie wywołania gRPC otrzymują nagłówek `Authorization: CloudSession <token>`.
- Serwer można wystartować poleceniem `python scripts/run_cloud_service.py --config config/cloud/server.yaml --health-file var/runtime/cloud_health.json --emit-stdout`. `CloudRuntimeService` publikuje event `ready` (stdout/pliki), a dodatkowo aktualizuje JSON z health-checkiem – idealny jako `healthcheck` w docker-compose lub `ConditionPathExists` w unitach systemd.

> **Szybki skrót benchmarku:** Stage6 jest na parytecie strategii z przewagą automatyzacji i compliance; największą luką pozostaje integracja UI (feed gRPC „Decyzje AI”) oraz skalowanie marketplace’u presetów.
>
> **Nowości:** tablica wyników i harmonogram działań korygujących w benchmarku są aktualizowane miesięcznie na podstawie `docs/runtime/status_review.md` i checklisty wsparcia. Dzięki temu zespoły produktowe widzą, kto odpowiada za domykanie luk i jakie są cele metryczne na kolejne kwartały.
>
> **Kronika benchmarku:** `docs/benchmark/cryptohopper_comparison.md` prowadzi historię aktualizacji oraz opisuje procedurę zbierania metryk (hypercare, adaptery giełdowe, marketplace, UI, compliance). To miejsce referencyjne przy audytach releasowych i syncach produktowych.

## Aktualizacje
- Nowe wersje dystrybuujemy w formie podpisanych instalatorów. Procedura: [docs/deployment/oem_installation.md](docs/deployment/oem_installation.md).
- Zanim zainstalujesz aktualizację, wykonaj kopię `config/`, `secrets/` oraz katalogu danych użytkownika (`~/.dudzian/`).
- Komponenty runtime, dokumentacja i runbooki korzystają z przestrzeni `bot_core.*`; pozostałe odniesienia do historycznych artefaktów są sukcesywnie izolowane w `archive/hypercare_stage5` i nie są ładowane przez aktywny kod Stage6.
- Zachowane artefakty hypercare z poprzedniego etapu (raporty, checklisty, skrypty CLI) opisujemy jako materiał historyczny – nie są ładowane przez runtime, ale pozostają dostępne w `archive/hypercare_stage5` do celów audytowych.
- Desktopowe UI dostarczamy tylko w wariancie PySide6/PyQt6 + Qt Quick (blur + FontAwesome). Wszelkie instrukcje dotyczące dawnych C++ shelli przenieśliśmy do `archive/ui_cpp_preserved.md` i nie utrzymujemy dla nich shimów.
- Wszystkie checklisty i przewodniki w `docs/ui/**` opisują wyłącznie aktualny klient PySide6/QML. Jeśli potrzebujesz historycznych materiałów C++ lub screenshotów poprzedniego UI, znajdziesz je w `archive/`.
- Dawny katalog z archiwalnym botem jest utrzymywany wyłącznie jako materiał referencyjny w `archive/`; aktywne repozytorium nie zawiera shimów ani kodu wykonywalnego poprzedniej warstwy.
- W `archive/` pozostawiamy wyłącznie materiały historyczne (np. [docs/archive/trading_model_pipeline.md](docs/archive/trading_model_pipeline.md)),
  aby nie rozpraszać zespołu nieużywaną implementacją. Aktywny kod żyje w `bot_core/**`.

## Kontrybucje
Zobacz [CONTRIBUTING.md](CONTRIBUTING.md) w celu poznania zasad współpracy. Przed zgłoszeniem zmian uruchom testy jednostkowe i integracyjne.

## Licencja
Wszystkie prawa zastrzeżone. Dystrybucja odbywa się na podstawie indywidualnych umów OEM.
