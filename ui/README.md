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

## Moduły UI i pluginy

Powłoka udostępnia menedżer modułów (`UiModuleManager`), który ładuje pluginy QML/C++
z katalogów wskazanych flagą `--ui-module-dir` (można przekazać wiele ścieżek) lub
zmienną środowiskową `BOT_CORE_UI_MODULE_DIRS` (separator zgodny z `QDir::listSeparator`).
Domyślnie skanowane są katalogi `modules/` w folderze binarnym oraz repozytoryjne
`ui/modules`. Każdy moduł może rejestrować widoki QML (identyfikator, etykieta,
ścieżka `QUrl`) oraz serwisy dostępne z QML-a. Menedżer jest wystawiony do kontekstu
QML jako `moduleManager`, a testy `ui/tests/UiModuleManagerTest.cpp` weryfikują rejestrację
widoków i serwisów.

Widoki zarejestrowane przez pluginy są prezentowane w zakładce „Moduły”
(`ModuleBrowser.qml`) dostępnej z głównego okna (`BotAppWindow.qml`). Lista pozwala
filtrować widoki po kategorii, podglądać deklarowane metadane i ładować pliki QML
źródłowe w ramach aplikacji. Model `UiModuleViewsModel` udostępnia API do
wyszukiwania widoków i kategorii wykorzystywane przez interfejs.【F:ui/src/app/UiModuleViewsModel.cpp†L64-L147】【F:ui/qml/components/ModuleBrowser.qml†L1-L409】

## Uruchomienie ze stubem gRPC

W pierwszym terminalu uruchom stub z wieloassetowym datasetem i pętlą strumieniową:

```bash
python scripts/run_trading_stub_server.py \
  --dataset data/trading_stub/datasets/multi_asset_performance.yaml \
  --stream-repeat --stream-interval 0.25 \
  --enable-metrics --metrics-jsonl artifacts/metrics.jsonl \
  --metrics-auth-token dev-secret --print-metrics-address
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
  --jank-threshold-ms 12.0 \
  --overlay-disable-secondary-fps 55 \
  --metrics-endpoint 127.0.0.1:50061 \
  --metrics-tag desktop-shell-dev \
  --metrics-auth-token dev-secret
```

Aby włączyć TLS/mTLS gRPC, dodaj dodatkowe opcje:

```bash
  --use-tls \
  --tls-root-cert secrets/mtls/ca/ca.pem \
  --tls-server-name trading.local \
  --tls-client-cert secrets/mtls/client/client.crt \
  --tls-client-key secrets/mtls/client/client.key \
  --tls-pinned-sha256 0123deadbeef...
```

> **Uwaga:** `Application::applyParser()` weryfikuje kompletność konfiguracji.
> Brak pliku datasetu w trybie `in-process` lub brakujące materiały TLS
> (root CA / certyfikat / klucz klienta) skutkują natychmiastowym błędem
> logowanym w kategorii `bot.shell.app.metrics`, co ułatwia diagnostykę
> niepoprawnych release'ów OEM.

Domyślne parametry są zgodne z plikiem `ui/config/example.yaml`. Wartości `--max-samples` oraz `--history-limit` pozwalają kontrolować rozmiar buforów i wpływają na wymagania pamięciowe.

## Tryb in-process (offline)

Powłoka może działać bez lokalnego demona gRPC, korzystając z datasetu OHLCV
wczytywanego z pliku CSV lub potoku named pipe. Aby aktywować ten tryb,
ustaw `--transport-mode in-process` oraz wskaż dataset przez `--transport-dataset`.
UI wymusza także przełączenie telemetrii i modułu Health na transport `in-process`,
ignorując ręcznie ustawione endpointy:

```bash
ui/build/bot_trading_shell \
  --transport-mode in-process \
  --transport-dataset data/sample_ohlcv/trend.csv \
  --disable-metrics
```

W trybie offline TLS jest wyłączony, a `MetricsClient`/`HealthClient` korzystają
z lokalnych stubów wstrzykiwanych przez `Application`. Dzięki temu można
budować w pełni odcięte środowiska demonstracyjne bez zależności od gRPC.

## Integracja z backendem produkcyjnym

Projekt korzysta z tych samych stubów gRPC co demon produkcyjny – generowanych
automatycznie podczas konfiguracji CMake (`trading.grpc.pb.cc/h`). Po
uruchomieniu zrealizowany jest pełny przepływ TLS/mTLS, pinning fingerprintów
oraz retry obsługiwane w `TradingClient` (kanał rynkowy/risk) i
`MetricsClient` (kanał telemetrii). Klient rynku weryfikuje pliki root CA,
certyfikaty klienta oraz odcisk SHA-256 przed zbudowaniem kanału【F:ui/src/grpc/TradingClient.cpp†L88-L175】【F:ui/src/grpc/MetricsClient.cpp†L47-L123】.

* Parametry TLS (`tls.*`, `grpc.tls.*`) pochodzą z `ui/config/*.yaml` lub
  odpowiadających im flag CLI – aplikacja ponownie zestawia połączenia po
  zmianie certyfikatów bez restartu procesu.【F:ui/src/grpc/TradingClient.cpp†L146-L189】【F:ui/src/grpc/MetricsClient.cpp†L63-L99】
* RBAC i tokeny dostępowe są wstrzykiwane przez metadane gRPC – kanał
  telemetrii dodaje `authorization`, `x-bot-scope`, `x-bot-role`, a status UI
  reaguje na błędy `connectionStateChanged`, co pozwala QML-owi wyświetlać
  komunikaty o próbie ponownego połączenia i błędach autoryzacji.【F:ui/src/grpc/MetricsClient.cpp†L125-L138】【F:ui/src/app/Application.cpp†L205-L226】
* Kanał tradingowy obsługuje tokeny i role RBAC przekazywane przez
  `--grpc-auth-token`, `--grpc-auth-token-file`, `--grpc-rbac-role` oraz
  `--grpc-rbac-scopes` (lub zmienne `BOT_CORE_UI_GRPC_*`). Zmiana metadanych
  restartuje strumień z automatycznym backoffem i ponownym snapshotem,
  zachowując synchronizację modeli QML. Pliki przekazane w `--*-auth-token-file`
  są monitorowane i każda aktualizacja jest stosowana bez restartu
  aplikacji.【F:ui/src/app/Application.cpp†L470-L520】【F:ui/src/app/Application.cpp†L2948-L3033】【F:ui/src/grpc/TradingClient.cpp†L225-L392】
  zachowując synchronizację modeli QML.【F:ui/src/app/Application.cpp†L470-L520】【F:ui/src/grpc/TradingClient.cpp†L225-L392】
* Kanał market data/Risk odpytywany jest synchronicznie i strumieniowo;
  snapshot historii OHLCV oraz strumień incrementów wypełniają modele QML, a
  `refreshRiskState()` umożliwia manualne/okresowe odpytywanie Decision Engine
  o profil ryzyka.【F:ui/src/grpc/TradingClient.cpp†L177-L289】【F:ui/src/app/Application.cpp†L238-L253】
* HealthService udostępnia status backendu (wersja, commit, uptime) i jest
  monitorowany przez `HealthClient` oraz `HealthStatusController`. Flagi
  `--health-endpoint`, `--health-auth-token`, `--health-auth-token-file`,
  `--health-rbac-role`, `--health-rbac-scopes`, `--health-refresh-interval` oraz
  `--health-disable-auto-refresh` (lub zmienne `BOT_CORE_UI_HEALTH_*`) pozwalają
  sterować zachowaniem panelu. Dodatkowe przełączniki
  `--health-use-tls`/`--health-disable-tls` i `--health-tls-*`
  (`root-cert`, `client-cert`, `client-key`, `server-name`, `target-name`,
  `pinned-sha256`, `--health-tls-require-client-auth`) umożliwiają niezależną od
  kanału tradingowego konfigurację certyfikatów i pinningu (również przez
  zmienne środowiskowe `BOT_CORE_UI_HEALTH_TLS_*`). Checklisty TLS/RBAC weryfikują
  kompletność materiału kryptograficznego i ostrzegają przed niespójnymi
  ustawieniami jeszcze przed zestawieniem kanału.【F:ui/src/grpc/HealthClient.cpp†L18-L268】【F:ui/src/app/Application.cpp†L540-L742】

W środowisku OEM rekomendowane jest przechowywanie certyfikatów w
`secrets/mtls/<rola>/` i wskazywanie ich przez flagi `--tls-*` oraz
`--metrics-*`. Aplikacja raportuje status kanałów na panelu bocznym (sekcja
„Połączenie”) – błędy TLS/RBAC są mapowane na komunikaty w stopce i alerty,
przez co operator natychmiast widzi problemy z konfiguracją.

## Aktywacja licencji OEM

Przy pierwszym uruchomieniu powłoka wyświetla ekran aktywacyjny licencji OEM. Aby odblokować UI:

1. Przygotuj podpisany pakiet `.lic` (`payload_b64` + `signature_b64`) z `scripts/generate_license.py` lub zeskanuj payload z kodu QR.
2. Zweryfikuj, że fingerprint HWID z licencji odpowiada wartości oczekiwanej (`config/fingerprint.expected.json`).
3. Wczytaj plik z nośnika USB lub wklej tekst pakietu (JSON albo base64) do pola tekstowego i zatwierdź.
4. Po poprawnej weryfikacji licencja zostanie zapisana w `var/licenses/active/license.json`, a stopka pokaże aktywną edycję, status utrzymania i HWID licencji.

Opcje CLI `--license-storage` oraz `--expected-fingerprint-path` pozwalają wskazać niestandardowe lokalizacje docelowej licencji oraz pliku fingerprintu (np. w środowisku produkcyjnym bundla OEM).

## Konfiguracja strategii i Decision Engine

Zakładka „Strategia” udostępnia teraz pełną konfigurację DecisionOrchestratora oraz schedulera
multi-strategy. Sekcja *DecisionOrchestrator* pozwala edytować globalne limity kosztów, progi
prawdopodobieństwa i parametry latencji wraz z nadpisaniami dla poszczególnych profili ryzyka.
Przycisk **Zapisz DecisionOrchestrator** zapisuje dane do `core.yaml`, korzystając z mostka
`scripts/ui_config_bridge.py`, który waliduje zmiany przy pomocy `bot_core.config.loader`.

Druga sekcja prezentuje listę schedulerów wraz z zadaniami (`schedules`). Operator może
aktualizować m.in. `health_check_interval`, przypisany `portfolio_governor` oraz szczegóły
każdego zadania (cadence, profil ryzyka, limit sygnałów). UI utrzymuje synchronizację z backendem
i w przypadku błędów (np. nieistniejącej nazwy zadania) komunikat z mostka wyświetlany jest w
panelu.

Mostek udostępnia również raporty `StrategyRegimeWorkflow`. Wywołanie

```bash
python scripts/ui_config_bridge.py \
  --config config/core.yaml \
  --describe-regime-workflow \
  --regime-workflow-dir var/data/strategy_regime_workflow
```

zwraca w JSON informacje o gotowości presetów (hash, podpis HMAC, brakujące dane,
blokady harmonogramu, wymagane licencje) oraz statystyki aktywacji i historię
fallbacków. Domyślnie mostek oczekuje plików `availability.json` oraz
`activation_history.json` w katalogu wskazanym przez `--regime-workflow-dir` – UI
może je odczytywać bezpośrednio, aby zasilić widok mapowania strategii na reżimy.

Ścieżki i interpreter mostka można dostosować flagami CLI:

```bash
ui/build/bot_trading_shell \
  --core-config /etc/bot_core/core.yaml \
  --strategy-config-python /usr/bin/python3 \
  --strategy-config-bridge /opt/oem/ui_config_bridge.py
```

Analogiczne wartości mogą pochodzić ze zmiennych środowiskowych
`BOT_CORE_UI_CORE_CONFIG_PATH`, `BOT_CORE_UI_STRATEGY_PYTHON` oraz
`BOT_CORE_UI_STRATEGY_BRIDGE`.

## Pakowanie desktopowe

Skrypt `scripts/packaging/qt_bundle.py` automatyzuje konfigurację CMake, build oraz
tworzenie archiwów (`.zip` na Windows, `.tar.gz` na Linux/macOS). Przykład:

```bash
python scripts/packaging/qt_bundle.py \
  --platform auto \
  --build-dir ui/build-release \
  --install-dir ui/install-release \
  --artifact-dir artifacts/ui/linux
```

Workflow GitHub Actions `ui-packaging` uruchamia pakowanie dla `ubuntu-latest`,
`windows-latest` i `macos-latest`, publikując artefakty w katalogu `artifacts/`.

## Pakiet wsparcia i eksport logów

Zakładka „Wsparcie” w panelu administratora umożliwia przygotowanie pakietu
pomocowego dla zespołu L2. UI uruchamia skrypt
`scripts/export_support_bundle.py`, który archiwizuje katalogi `logs/`,
`var/reports`, `var/licenses`, `var/metrics` oraz – opcjonalnie – `var/audit`
do formatu `tar.gz` lub `zip`. Manifest (`bundle_manifest.json`) zawiera
podsumowanie wielkości i liczby plików oraz metadane środowiskowe (hostname,
instrument, status połączenia).

Najważniejsze flagi CLI sterujące pakietem wsparcia:

```bash
ui/build/bot_trading_shell \
  --support-bundle-python /usr/bin/python3 \
  --support-bundle-script /opt/oem/export_support_bundle.py \
  --support-bundle-output-dir /var/support \
  --support-bundle-format zip \
  --support-bundle-basename customer-support \
  --support-bundle-include extra=/var/custom_artifacts \
  --support-bundle-disable audit
```

Analogiczne ustawienia można wstrzyknąć zmiennymi środowiskowymi
`BOT_CORE_UI_SUPPORT_*` (np. `BOT_CORE_UI_SUPPORT_INCLUDE` jako lista
`label=ścieżka` oddzielona średnikami). Skrypt nie wymaga dodatkowych
zależności poza systemowym Pythonem 3.11+, dlatego można go bundlować razem z
dystrybucją OEM.

## Monitorowanie HealthService

Zakładka „Monitorowanie” prezentuje sekcję **Status backendu**, która wyświetla
wynik ostatniego zapytania HealthService: wersję, skrócony commit, czas
uruchomienia (UTC/lokalny), bieżący uptime oraz stempel ostatniego sprawdzenia.
Operator może ręcznie odświeżyć dane, przełączyć auto-odświeżanie i zmienić
interwał bez restartu aplikacji. Kontroler QML (`HealthStatusController`)
deleguje zapytania do `HealthClient`, kolejkując retry w tle i publikując
wyniki do QML. Domyślnie kanał dziedziczy konfigurację TLS/mTLS klienta tradingowego,
ale zestaw `--health-use-tls`/`--health-disable-tls` oraz `--health-tls-*`
pozwala w razie potrzeby wstrzyknąć odrębne certyfikaty i fingerprinty
HealthService. Jeżeli dostarczony fingerprint SHA-256 nie zgadza się z
odczytanym z materiału TLS, `HealthClient` nie tworzy kanału gRPC, a próba
`check()` zwraca błąd do czasu aktualizacji certyfikatów lub konfiguracji
pinningu.【F:ui/qml/components/AdminPanel.qml†L2067-L2166】【F:ui/src/health/HealthStatusController.cpp†L15-L180】【F:ui/src/grpc/HealthClient.cpp†L88-L293】【F:ui/src/app/Application.cpp†L540-L742】

## Architektura komponentów

* `src/grpc/TradingClient.*` – cienki klient gRPC pobierający historię i strumień OHLCV.
* `src/models/OhlcvListModel.*` – ring-buffer świec udostępniony QML jako `ListModel` z metodami `candleAt()` i `latestClose()`.
* `src/app/Application.*` – warstwa klejąca CLI ↔ gRPC ↔ QML. Udostępnia `appController` w kontekście QML oraz synchronizuje parametry instrumentu/guardu pomiędzy oknami.
* `src/utils/FrameRateMonitor.*` – monitoruje `frameSwapped` głównego okna i emituje `reduceMotionActive`, gdy FPS spada poniżej progów guardu (np. 55 FPS @60 Hz), co pozwala UI automatycznie wygasić animacje/overlaya. Emituje także próbki FPS, które trafiają do telemetrii.
* `src/telemetry/UiTelemetryReporter.*` – raportuje zdarzenia UI (`reduce motion`, budżet nakładek) do `MetricsService` demona gRPC i dopisuje metadane okien/nakładek.
* `qml/components/CandlestickChartView.qml` – widok wykresu z krzyżem celowniczym, autoprzeskalowaniem, nakładkami EMA12/EMA26/VWAP sterowanymi `PerformanceGuard` oraz mechanizmem sample-at-x.
* `qml/components/SidePanel.qml` – wizualizacja parametrów performance guard i statusu połączenia, szybkie otwieranie dodatkowych okien.
* `qml/components/ChartWindow.qml` – niezależne okno wykresu (multi-window/multi-monitor) z zapamiętywaniem geometrii przez `Qt.labs.settings`.
* `qml/components/BotAppWindow.qml` – okno główne z menu kontekstowym, skrótem `Ctrl+N` do otwierania nowych okien i automatycznym przywracaniem profilu workspace.

Po uruchomieniu głównego okna można otwierać kolejne wykresy (`Nowe okno` lub `Ctrl+N`). Aplikacja zapamiętuje pozycję i liczbę okien między sesjami (`Qt.labs.settings`), a stopka informuje o aktywnym trybie „reduce motion”, kiedy `FrameRateMonitor` zasygnalizuje spadek FPS.

## Harmonogram ryzyka i konfiguracja Decision Engine

Panel administratora pozwala operatorowi ustawić, czy UI ma okresowo
odświeżać stan risk/AI – parametry (włącz/wyłącz, interwał w sekundach) są
przechowywane w ustawieniach użytkownika i synchronizowane pomiędzy sesjami.
Zmiana interwału restartuje wewnętrzny timer i natychmiast planuje kolejne
wywołanie `refreshRiskState()` – dzięki temu harmonogram Decision Engine jest
zgodny z polityką compliance bez restartu aplikacji.【F:ui/src/app/Application.cpp†L238-L256】【F:ui/src/app/Application.cpp†L516-L686】

Modele ryzyka (`RiskHistoryModel`, `RiskStateModel`) eksportują snapshoty CSV
z zachowaniem limitów i automatycznego eksportu do katalogu roboczego. Operator
może wymusić natychmiastowe wykonanie eksportu, wskazać katalog docelowy z
poziomu UI lub CLI oraz kontrolować limit próbek, co ułatwia synchronizację z
DecisionOrchestrator/Schedulerem backendowym.【F:ui/src/app/Application.cpp†L827-L906】【F:ui/src/app/Application.cpp†L1233-L1314】

## Licencjonowanie OEM end-to-end

Pierwsze uruchomienie prowadzi przez kontroler aktywacji licencji, który
odczytuje oczekiwany fingerprint, nasłuchuje katalogu provisioning (`var/licenses/inbox`) i zapisuje podpisane pakiety `.lic`
do `var/licenses/active`. Kontroler obsługuje pliki, payload base64 i skan
hot-folderu, a błędy zapisu są przekazywane użytkownikowi w formie komunikatu.
Przy każdej zmianie fingerprintu aktualizowany jest dokument HMAC
`fingerprint.expected.json`, co domyka przepływ OEM (UI → backend → storage).【F:ui/src/license/LicenseActivationController.cpp†L66-L216】【F:ui/src/license/LicenseActivationController.cpp†L240-L360】

Kontroler śledzi również opóźnione dostarczenie dokumentu fingerprintu oraz
aktywnej licencji – obserwuje katalogi nadrzędne i automatycznie przeładowuje
konfigurację, gdy `fingerprint.expected.json` lub `var/licenses/active/*.json`
zostaną utworzone, zmodyfikowane bądź usunięte. Dzięki temu UI natychmiast
odświeża widok licencji przy reinstalacji lub wymianie nośnika OEM, a
niezgodności fingerprintu ponownie uruchamiają proces provisioning bez
restartu aplikacji.【F:ui/src/license/LicenseActivationController.cpp†L66-L216】【F:ui/src/license/LicenseActivationController.cpp†L360-L552】【F:ui/src/license/LicenseActivationController.cpp†L900-L1054】

## Telemetria, alerty i logi

`UiTelemetryReporter` buforuje i ponawia wysyłki metryk FPS/janku, za każdym
razem wzbogacając payload o parametry ekranów oraz informację o backlogu retry.
Kanał telemetryczny korzysta z TLS/mTLS i RBAC opisanych powyżej; wszystkie
niepowodzenia są logowane i widoczne w panelu administracyjnym, a bufor retry
jest emitowany jako metryka do QML.【F:ui/src/telemetry/UiTelemetryReporter.cpp†L200-L272】

Administratorzy mogą zarządzać profilami RBAC za pomocą mostka Pythona – każda
operacja zapisuje log do `logs/security_admin.log` oraz emituje zdarzenie,
które UI wyświetla w sekcji audytów. Dzięki temu operator ma pełny wgląd w
zmiany uprawnień i może eksportować logi wsparcia razem z raportami risk.【F:ui/src/security/SecurityAdminController.cpp†L31-L115】【F:ui/src/security/SecurityAdminController.cpp†L116-L175】

## Pakiet OEM i instalacja offline

Artefakty desktopowe są dostarczane w bundlu `core-oem-<wersja>-<platforma>`
budowanym przez `deploy/packaging/build_core_bundle.py`. Manifest podpisany
HMAC opisuje wszystkie pliki (`daemon`, `ui`, `config`, `bootstrap`), a skrypty
instalacyjne weryfikują fingerprint urządzenia zanim licencja zostanie
zapisana. Szczegółowy proces instalacji, walidacji podpisów oraz wymagane
artefakty znajdują się w runbooku `docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md` i
powiązanych checklistach OEM.【F:deploy/packaging/README.md†L1-L61】【F:docs/runbooks/OEM_INSTALLER_ACCEPTANCE.md†L1-L88】

## Testy

Warstwa desktopowa posiada bogaty zestaw testów jednostkowych (`ctest`),
obejmujących aktywację licencji, kontroler raportów, modele alertów oraz
regresję FPS. Dodatkowe testy end-to-end (Qt Quick Test) sprawdzają przepływ
aktywacji licencji i podstawowe scenariusze tradingowe – warto wykonywać je
przed przygotowaniem bundla OEM: `ctest --test-dir ui/build` oraz wybrane testy
regexem (`ctest --test-dir ui/build --tests-regex LicenseActivation`). Backend
mostkujący może być weryfikowany poprzez `pytest` w katalogu głównym projektu
(`pytest tests/ui_bridge`).

## Panel administratora – raporty i archiwizacja

Zakładka **Raporty** w panelu administratora korzysta z mostka
`bot_core.reporting.ui_bridge`. Operacje „Usuń” i „Archiwizuj” są mapowane na
komendy `delete` oraz `archive`, a tryb podglądu w UI uruchamia flagę
`--dry-run`. Konfiguracja wymaga ustawienia katalogu raportów (`var/reports` lub
ścieżka wskazana w `BOT_CORE_UI_REPORTS_DIR`) oraz interpretera Pythona z
zainstalowanym modułem `bot_core`.

Szczegółowe procedury, scenariusze krok-po-kroku i checklisty bezpieczeństwa
znajdują się w runbooku
[`docs/runbooks/report_maintenance.md`](../docs/runbooks/report_maintenance.md).

## Kolejne kroki

* Podpięcie realnego demona C++ (`/core`) przez TLS i RBAC.
* Dodanie wskaźników ATR/RSI oraz konfiguracji nakładek z poziomu UI przy zachowaniu ograniczeń `PerformanceGuard`.
* Integracja z docelowym demona `MetricsService` (mTLS, RBAC) oraz raporty guardu na kanały alertowe.
* Benchmark QML Profiler 60/120 Hz + automatyczne raportowanie do `MetricsService` wraz z detekcją adaptacji animacji.
* Dodanie warstwy animacji (Transitions/States) oraz adaptacji „reduce motion” na podstawie metryk gRPC.
* Benchmark QML Profiler 60/120 Hz + automatyczne raportowanie do `MetricsService`.
