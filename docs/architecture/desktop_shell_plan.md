# Plan architektury aplikacji desktopowej (Qt/QML + core gRPC)

## Cel projektu

Celem jest dostarczenie w pełni natywnej aplikacji tradingowej działającej na Windows, macOS oraz Linux,
która łączy ciężki demon rdzeniowy (core) z lekką powłoką graficzną napisaną w Qt 6 (Qt Quick/QML).
Rozwiązanie ma zapewniać płynność 60/120 Hz, pełną obsługę profili ryzyka i audytu oraz brak bezpośrednich
połączeń UI z giełdami. Komunikacja między powłoką a rdzeniem odbywa się wyłącznie przez gRPC (HTTP/2)
oraz mechanizmy IPC oferowane przez Qt/OS.

## Podział na komponenty repozytorium

| Katalog | Zakres | Język | Uwagi |
| --- | --- | --- | --- |
| `/proto` | Kontrakty Protobuf v1 dla komunikacji shell↔core | `.proto` | Stały zestaw komunikatów: `GetOhlcvHistory`, `StreamOhlcv`, `SubmitOrder`, `CancelOrder`, `RiskState`, `Metrics`. |
| `/core` | Demon C++17/20 implementujący logikę rynku, ryzyka, zamówień i metryk | C++ | gRPC + OpenTelemetry, Prometheus, mTLS. |
| `/ui` | Powłoka Qt Quick + Qt Charts + QML | QML/Qt | Brak logiki domenowej, jedynie prezentacja i interakcje. |
| `/packaging` | Skrypty budujące instalatory (MSI, DMG, AppImage, .deb/.rpm) | Python/CMake/Qt IFW | Kanały `alpha/beta/stable`, podpisy EV, notarization. |
| `/ops` | Dashboardy obserwowalności, alerty i skrypty wdrożeniowe | Terraform/Helm/etc. | Wsparcie dla Prometheus + Grafana, konfiguracja alertów compliance. |

Repozytorium `bot_core/` pozostaje źródłem logiki tradingowej; demon C++ korzysta z tych samych modułów
przez stabilny interfejs C++ (bindingi) lub przez FFI zmodularyzowane w kolejnych iteracjach.

## Architektura rdzenia (daemon)

* **Język i narzędzia:** C++17/20, CMake, gRPC (HTTP/2) z generowanymi stubami z katalogu `/proto`.
* **Moduły:**
  * `MarketDataService` – streaming OHLCV, zarządzanie buforami, downsampling (LTTB) przed wysłaniem do UI.
  * `RiskService` – ekspozycja bieżącego stanu ryzyka (profile: konserwatywny, zbalansowany, agresywny, manualny) oraz
    powiadomienia o przekroczeniach limitów; integracja z `ThresholdRiskEngine` z Pythonowego rdzenia przez gRPC Gateway
    lub wspólne biblioteki.
  * `OrderService` – przyjmowanie zleceń, idempotencja poprzez `client_order_id`, retry i backpressure.
  * `MetricsService` – telemetryka i health-checki dla UI, w tym metryki wykorzystania CPU/RAM, latencje pipeline’u.
* **Bezpieczeństwo:** mTLS (client cert shell, server cert core), pinning certyfikatów, RBAC (`viewer`, `trader`, `admin`, `auditor`).
* **Obserwowalność:** OpenTelemetry traces + Prometheus metrics; logi JSON z `trace_id`, `span_id` i korelacją do decision logu.
* **Buforowanie:** ring-buffery w pamięci na ostatnie N świec per instrument; mechanizmy backpressure przy 5k msg/s.
* **Idempotencja:** dedykowana tabela `orders_requests` (SQLite/LMDB) z TTL na retry; obsługa `SubmitOrder` oraz `CancelOrder` z 
potwierdzeniem statusów.

## Powłoka UI (Qt Quick + QML)

* **Struktura UI:** Router (StackView) dla ekranów: Trading Dashboard, Risk Monitor, Order Blotter, Alert Center, Settings. Multi-window z synchronizacją kontekstu (gRPC session token) i pamięcią układu doków.
* **Komponenty bazowe:**
  * Biblioteka QML z komponentami: `BotAppWindow`, `SidePanel`, `InstrumentPicker`, `OrderTicket`, `RiskBanner`, `AlertToast`, `MetricsPanel`, `LatencyIndicator`.
  * Styleguide animacji (easing OutCubic/OutQuint) i mikrozachowania (Behaviors) zapisany w `ui/styleguide/animations.md` (to-do) wraz z przykładowymi nagraniami „before/after”.
  * Styleguide kolorów/typografii w `ui/styleguide/theme.md` z wariantami dark/light/high-contrast.
  * Biblioteka ikon i ilustracji (SVG) zoptymalizowana pod HiDPI/Retina (1×,2×,3×) oraz tryb „reduce motion”.
* **Wykresy:**
  * Dla ≤10k punktów – `QtCharts::CandlestickSeries` + `LineSeries` dla wskaźników.
  * Powyżej 20k punktów lub wielu overlayów – dedykowany `QQuickItem` oparty o `QSGGeometryNode` z batchingiem (≤2 draw-calls warstwa) i ring-buffery.
  * Aktualizacja intra-bar modyfikuje wyłącznie ostatnią świecę; zamknięcie baru (`closed=true`) inicjuje krótką interpolację ≤100 ms.
  * Downsampling (LTTB) oraz docinanie do rozdzielczości widoku wykonywane w demonie przed wysyłką.
* **Animacje (60/120 Hz):**
  * Nawigacja – przejścia 120–250 ms, fade/slide.
  * Panele i dialogi – fade/slide 120–180 ms, spójny easing OutCubic.
  * Crosshair/tooltipy – fade 60–120 ms, Behavior dla podświetleń świecy/pozycji.
  * Zoom/pan – kinetic scroll + „rubber band”, brak migotania.
  * Adaptacja pod obciążeniem: monitor FPS → poniżej 55/110 wyłącza animacje wtórne (ustawienia „reduce motion”).
  * Brak animowania każdego ticka/metryki – tylko zdarzenia `bar close` lub interakcje.
  * Tryb „performance guard”: monitor FPS, CPU i occupancy GPU; poniżej progów adaptacyjnie skracamy/wyłączamy animacje wtórne oraz przełączamy QML na mniej zasobożerne komponenty (np. uproszczone gradienty, brak rozmyć tła).
* **Wydajność:** QSG w trybie `threaded`, backendy D3D11/Metal/Vulkan per platforma; docelowe KPI: p95 event→frame <150 ms, jank <1%, RAM UI <300 MB.
* **A11y:** tryby dark/light/high-contrast, wsparcie screen readerów (Accessible.name), focus ring, pełna obsługa klawiatury.
* **I18n:** PL/EN, lokalne formaty liczb, jednostki rynkowe (np. PLN/EUR) w UI.
* **Multiwindow:** wsparcie dla wielu monitorów, dokowalne panele (Qt Quick Docking), profile workspace zapisywane w `~/.bot_core/ui_profiles/`.

## Kontrakty Protobuf v1

Pakiet `proto/` zawiera definicje usług z zamrożonymi polami (brak breaking change):

* `MarketDataService` – `GetOhlcvHistory`, `StreamOhlcv`.
* `OrderService` – `SubmitOrder`, `CancelOrder` (z `client_order_id`, `time_in_force`, `slippage_bps`).
* `RiskService` – `RiskState` (limity, użycie, profile), strumień incydentów.
* `MetricsService` – `MetricsSnapshot`, `Heartbeat`.

Pliki `.proto` generują stuby C++ (core) oraz Python (bot_core) – spójne testy kontraktowe golden files.
Artefakty tworzymy skryptem `scripts/generate_trading_stubs.py`, a wzorcowy workflow
`deploy/ci/github_actions_proto_stubs.yml` buduje je w CI i publikuje jako artefakt.

### Stub developerski

* Skrypt `scripts/run_trading_stub_server.py` uruchamia lokalny serwer gRPC bezpośrednio na danych z YAML-a
  lub na domyślnym datasetcie. Parametryzacja obejmuje host/port, wielokrotne `--dataset`, tryb `--shutdown-after`
  (przydatny w CI), `--stream-repeat` do symulacji ciągłego feedu (loop na incrementach) oraz `--stream-interval`
  pozwalający kontrolować kadencję aktualizacji (0 = natychmiast, >0 = odstęp w sekundach). W razie potrzeby można
  pominąć dane startowe poprzez `--no-default-dataset`. Log startowy prezentuje również aktualną konfigurację
  performance guard, co pozwala błyskawicznie zweryfikować oczekiwane progi FPS i ograniczenia overlayów.
* Opcja `--enable-metrics` startuje w tym samym procesie lekki serwer `MetricsService` (domyślnie `127.0.0.1:50061`),
  który udostępnia telemetrię UI powłoce Qt. Można dostroić host/port, rozmiar historii (`--metrics-history-size`),
  zapisy JSONL (`--metrics-jsonl`, `--metrics-jsonl-fsync`), wyłączyć logowanie do stdout (`--metrics-disable-log-sink`)
  oraz wskazać log alertów UI (`--metrics-ui-alerts-jsonl`).
  Przełącznik `--disable-metrics-ui-alerts` pozwala całkowicie wyłączyć sink `UiTelemetryAlertSink`, a dodatkowe
  flagi `--metrics-ui-alerts-*-mode/category/severity/critical-threshold` umożliwiają spójne z core’owym runtime
  sterowanie kategoriami i progami alertów reduce-motion/overlay. Tryb `mode` wspiera teraz warianty `enable`
  (log + dispatch), `jsonl` (wyłącznie JSONL bez dispatchu) oraz `disable` (brak logowania i dispatchu). `--metrics-print-address`
  wypisuje faktyczny adres serwera – przydatne w pipeline CI i przy pracy na wielu instancjach stubu.
  Dostępny jest także przełącznik `--metrics-ui-alerts-risk-profile` (ENV: `RUN_TRADING_STUB_METRICS_UI_ALERTS_RISK_PROFILE`),
  który ładuje te same presety co watcher/verifier – wartości severity, progi overlay/jank i tryby dispatchu są ustawiane
  automatycznie, a w runtime planie `metrics.ui_alerts.risk_profile` zapisuje pełną charakterystykę profilu dla audytu demo→paper→live.
  Jeśli potrzebujemy rozszerzyć lub nadpisać presety telemetryczne, możemy wskazać dodatkowy plik JSON/YAML przez
  `--metrics-risk-profiles-file` (ENV: `RUN_TRADING_STUB_METRICS_RISK_PROFILES_FILE`).  Załadowane profile zostają oznaczone
  w planie runtime (`metrics.ui_alerts.risk_profiles_file`) wraz z listą nazw i ścieżką źródła.  Od tej iteracji
  obsługiwane są również katalogi zawierające wiele plików JSON/YAML – plan raportuje typ (`file`/`directory`) oraz
  listę plików zarejestrowanych presetów, co ułatwia audyt zmian w pipeline demo→paper→live.
  Szybki audyt presetów bez startu serwera umożliwia `--metrics-print-risk-profiles` (ENV:
  `RUN_TRADING_STUB_METRICS_PRINT_RISK_PROFILES`), które wypisuje pełny JSON z dostępnymi profilami oraz metadanymi
  plików/katalogów źródłowych.
  Audyt alertów UI można przełączyć na backend plikowy za pomocą `--metrics-ui-alerts-audit-dir/pattern/retention-days`, `--metrics-ui-alerts-audit-backend`
  (auto/file/memory) i `--metrics-ui-alerts-audit-fsync` (lub zmiennych `RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_*`). Gdy backend plikowy jest niedostępny (brak `FileAlertAuditLog`
  w środowisku), skrypt samoczynnie degraduje się do audytu w pamięci i oznacza plan konfiguracji notatką `file_backend_unavailable`.
* Stub potrafi równocześnie wystartować lekki `MetricsService` (`--enable-metrics`) – te same dane, które powłoka
  wysyła do core, można odebrać lokalnie. Dostępne są przełączniki `--metrics-host/--metrics-port`, zapis JSONL
  (`--metrics-jsonl`, `--metrics-jsonl-fsync`) oraz opcja `--metrics-disable-log-sink` tłumiąca logowanie snapshotów.
  Adres serwera można pobrać z `--print-metrics-address`, co ułatwia podpięcie powłoki i pipeline’ów CI.
* Stub wykorzystuje `bot_core.testing.TradingStubServer` oraz helper `merge_datasets`, dzięki czemu można
  łączyć wiele plików YAML bez konieczności modyfikacji kodu.
* W repozytorium dostarczamy przykładowy zestaw `data/trading_stub/datasets/multi_asset_performance.yaml`
  zawierający BTC/USDT (1m) i ETH/EUR (5m) oraz parametry `performance_guard` (target 120 Hz, jank ≤12 ms,
  automatyczne ograniczenie overlayów). Dataset służy jako punkt startowy dla scenariuszy multi-window i
  benchmarków animacji (60/120 Hz).
* Workflow CI `deploy/ci/github_actions_proto_stubs.yml` po wygenerowaniu artefaktów może uruchomić stub
  z `--shutdown-after`, aby przeprowadzić szybki smoke test UI lub komponentów gRPC.
* Skrypt `scripts/run_metrics_service.py` startuje dedykowany serwer MetricsService (host/port, rozmiar historii,
  opcjonalny LoggingSink oraz zapis do JSONL z `--jsonl`/`--jsonl-fsync`) – wykorzystywany w CI i lokalnie do
  obserwacji zdarzeń reduce-motion, budżetu overlayów **oraz wykrytych klatek jank** wysyłanych z powłoki.
  Flagi `--ui-alerts-audit-dir/pattern/retention-days`, `--ui-alerts-audit-backend` (auto/file/memory) oraz `--ui-alerts-audit-fsync`
  (i odpowiadające im zmienne środowiskowe `RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_*`) pozwalają zapisać audyt alertów UI do rotowanych plików JSONL.
  Profil ryzyka można dobrać jednym przełącznikiem `--ui-alerts-risk-profile` (ENV: `RUN_METRICS_SERVICE_UI_ALERTS_RISK_PROFILE`),
  co automatycznie ustawia tryby dispatch/logowania oraz severity i progi (np. konserwatywny = severity `critical`, próg overlay 1),
  a w planie konfiguracji i metadanych runtime pojawia się sekcja `risk_profile` z pełnym opisem wymuszonych limitów.
  Dodatkowe lub nadpisane presety można załadować przed startem serwera flagą `--risk-profiles-file`
  (ENV: `RUN_METRICS_SERVICE_RISK_PROFILES_FILE`) lub zdefiniować je w konfiguracji YAML poprzez
  pole `runtime.metrics_service.ui_alerts_risk_profiles_file`.  Efektywnie użyte profile są raportowane w sekcji
  `ui_alerts.risk_profiles_file` planu konfiguracji oraz w metadanych runtime (`runtime_state.ui_alerts_sink.config.risk_profiles_file`).
  Do szybkiej inspekcji dostępnych presetów (łącznie z tymi pobranymi z plików/katalogów) służy tryb `--print-risk-profiles`
  lub zmienna `RUN_METRICS_SERVICE_PRINT_RISK_PROFILES`, który wypisuje JSON z profilami, informacją o źródłach i metadanymi
  `core_config` bez uruchamiania serwera.
  Jeżeli backend plikowy nie jest dostępny, narzędzie loguje degradację do audytu w pamięci (również oznaczoną w planie konfiguracji jako
  `file_backend_unavailable` lub `directory_ignored_memory_backend`), aby operatorzy mogli odnotować brak trwałego archiwum.
  Runtime `bootstrap_environment` propaguje te informacje dalej – w `BootstrapContext.metrics_ui_alerts_settings` znajduje się sekcja
  `audit` z rozstrzygnięciem realnego backendu (`memory` lub `file`), z notatkami `inherited_environment_router`, `file_backend_unavailable`
  albo `memory_backend_not_selected` (np. gdy operator wymusił tryb `memory`, ale router środowiskowy wciąż zapisuje do pliku). Dzięki temu
  pipeline demo→paper→live ma jednoznaczny obraz, czy alerty UI trafiają do trwałego audytu, czy też działamy w trybie degradacji.
* Narzędzie `scripts/watch_metrics_stream.py` podgląda `MetricsSnapshot` wprost z gRPC (filtry `--event`, `--severity`,
  `--severity-min`, `--since`, `--until`, `--screen-index`, `--screen-name`, format `table/json`, limit rekordów) i służy do
  debugowania telemetrycznego w CI oraz na stanowiskach operatorów. Wspiera TLS/mTLS przez flagi `--use-tls`, `--root-cert`,
  `--client-cert`, `--client-key`, `--server-name` oraz pinning `--server-sha256`.  Aby zapobiec przypadkowemu braku
  szyfrowania, CLI wymaga jawnego `--use-tls` gdy operator poda którąkolwiek z flag TLS – w przeciwnym razie zakończy się
  błędem i przypomni o konieczności włączenia kanału szyfrowanego.  Te same parametry można zasilić zmiennymi środowiskowymi
  `BOT_CORE_WATCH_METRICS_*` (np. `..._ROOT_CERT`, `..._SERVER_SHA256`, `..._USE_TLS`, `..._SCREEN_INDEX`, `..._SCREEN_NAME`,
  `..._SEVERITY`, `..._SEVERITY_MIN`, `..._SINCE`, `..._UNTIL`) oraz `..._FROM_JSONL` wskazującym artefakt JSONL z pipeline’u.
  W trybie offline można użyć flagi `--from-jsonl`, aby przejrzeć zapisane snapshoty bez gRPC (TLS i tokeny są wówczas
  ignorowane, a filtry po zdarzeniach/monitorach nadal działają); narzędzie rozpoznaje także artefakty `.jsonl.gz`
  (dekompresja w locie) oraz potrafi czytać dane ze standardowego wejścia (`--from-jsonl -`), co ułatwia analizę w potokach
  CI/CD.  Token RBAC można przekazać bezpiecznie z pliku
  (`--auth-token-file`) lub zmiennej `..._AUTH_TOKEN` bez logowania wartości.  Wypisywany strumień zawiera podsumowanie
  monitora (`screen=#1 (Main Display), 1920x1080 px, 60 Hz`), co upraszcza audyt kontekstu multi-monitorowego wraz z
  alertami reduce-motion/overlay/jank zarówno online, jak i podczas analizy artefaktów CI.  Filtry czasowe `--since/--until`
  (i odpowiadające im zmienne środowiskowe) pozwalają analizować konkretne okna czasowe bez potrzeby dodatkowego narzędzia,
  a `--severity-min`/`BOT_CORE_WATCH_METRICS_SEVERITY_MIN` ograniczają audyt do alertów o zadanym poziomie istotności lub
  wyższym (np. tylko `warning`+ i `critical`).  Jeżeli operator równocześnie poda listę `--severity`, CLI wymusza spójność –
  próg `severity_min` nie może być niższy niż wartości na liście; w przeciwnym razie narzędzie zakończy się błędem i przypomni
  o korekcie filtra.  Dodatkowo flaga `--risk-profile` (oraz `BOT_CORE_WATCH_METRICS_RISK_PROFILE`) pozwala jednym przełącznikiem
  załadować preset ryzyka (conservative/balanced/aggressive/manual) – watcher automatycznie włącza podsumowanie, wymusza minimalny
  próg severity, a w metadanych decision logu i podpisanym podsumowaniu zapisuje nazwę profilu wraz z narzuconymi limitami.
  Gdy operator potrzebuje jedynie przejrzeć dostępne presety i ich parametry bez uruchamiania streamingu lub analizy artefaktów,
  może użyć `--print-risk-profiles` (albo zmiennej `BOT_CORE_WATCH_METRICS_PRINT_RISK_PROFILES`), które wypisują pełny JSON
  z opisem limitów KPI, wymaganych podpisów i minimalnych progów severity; taki zrzut można zachować w decision logu lub
  dołączyć do audytu CI.  W razie potrzeby rozszerzenia/nadpisania presetów watcher potrafi wczytać dodatkowy plik
  JSON/YAML przez `--risk-profiles-file` (lub `BOT_CORE_WATCH_METRICS_RISK_PROFILES_FILE`).  Załadowane profile są oznaczane
  w metadanych polem `origin=watcher:…`, dzięki czemu audyt jednoznacznie wskazuje źródło definicji (repozytorium, artefakt CI).
  o korekcie filtra.
  Dodatkowo flaga `--summary` (lub zmienna `..._SUMMARY=true/false`) oblicza zbiorcze statystyki (liczba snapshotów, rozkład zdarzeń,
  agregaty FPS, lista ekranów oraz rozkład severity) zarówno dla strumienia gRPC, jak i odczytu JSONL, co ułatwia operatorom szybkie
  porównanie stanowisk w pipeline demo→paper→live.  Jeśli potrzeba zachować wynik audytu, flaga `--summary-output` lub zmienna
  `..._SUMMARY_OUTPUT` zapisują podsumowanie do wskazanego pliku JSON (kanał gRPC/offline), przy czym kolekcjonowanie
  danych odbywa się nawet wtedy, gdy operator wyłączył wypis na STDOUT, co upraszcza automatyczne raportowanie w CI.  Gdy
  równocześnie dostarczono klucz HMAC (`--decision-log-hmac-key`/`..._DECISION_LOG_HMAC_KEY(_FILE)`), watcher podpisuje także
  wygenerowane podsumowanie (`signature.algorithm = HMAC-SHA256`, opcjonalny `key_id`) i zapisuje parametry podpisu w metadanych
  decision logu (`summary_signature`).  Dzięki temu pipeline demo→paper→live ma jednolity materiał do audytu (decision log +
  summary JSON) z gwarancją integralności kryptograficznej.  Nowa flaga `--decision-log` (oraz `..._DECISION_LOG`) zapisuje każdy przefiltrowany snapshot do pliku JSONL w formacie decision
  danych odbywa się nawet wtedy, gdy operator wyłączył wypis na STDOUT, co upraszcza automatyczne raportowanie w CI.  Nowa
  flaga `--decision-log` (oraz `..._DECISION_LOG`) zapisuje każdy przefiltrowany snapshot do pliku JSONL w formacie decision
  log (źródło gRPC/JSONL, event, severity, FPS, metadane monitora, pełne `notes`). Pozwala to archiwizować decyzje
  operacyjne z audytów reduce-motion/overlay/jank, także podczas pracy offline (`--from-jsonl`). Każdy plik decision log
  rozpoczyna się wpisem `metadata` z kontekstem uruchomienia (tryb online/offline, endpoint lub ścieżka JSONL, aktywne
  filtry – w tym okno czasowe `since/until`, ustawiony próg `severity_min` oraz listy severity – stan TLS/tokenów, wymuszone podsumowanie), dzięki czemu operatorzy mogą odtworzyć parametry audytu bez sięgania
  do historii poleceń.  Decision log można podpisywać kryptograficznie: flagi `--decision-log-hmac-key`/`--decision-log-hmac-key-file`
  (oraz zmienne `..._DECISION_LOG_HMAC_KEY(_FILE)`) ładują klucz HMAC-SHA256, a opcjonalny identyfikator klucza (`--decision-log-key-id`,
  `..._DECISION_LOG_KEY_ID`) trafia zarówno do metadanych, jak i do podpisów pojedynczych wpisów.  Podpisy (pole `signature`
  z algorytmem, wartością Base64 i opcjonalnym `key_id`) są dodawane do wpisu `metadata` i każdego snapshotu, dzięki czemu pipeline demo→paper→live
  może weryfikować integralność decision logów i łączyć je z rotacją kluczy operacyjnych.
* Narzędzie uzupełniające `scripts/verify_decision_log.py` służy do walidacji podpisów HMAC
  w decision logach wygenerowanych przez watcher.  Przyjmuje pliki `.jsonl`, `.jsonl.gz` lub
  dane ze standardowego wejścia, obsługuje te same sekretne klucze przez flagi/zmienne (`--hmac-key`,
  `--hmac-key-file`, `BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY(_FILE)`) oraz wymusza spójność
  identyfikatorów kluczy (`--hmac-key-id`, `--expected-key-id`).  Operatorzy mogą równocześnie
  zweryfikować metadane audytu: `--expect-mode grpc/jsonl`, `--expect-summary-enabled`, wielokrotne
  `--expect-filter klucz=wartość`, wymagania TLS/tokenów (`--require-tls`, `--require-auth-token`),
  a także oczekiwany endpoint (`--expect-endpoint`) lub ścieżkę pliku wejściowego (`--expect-input-file`).
  Flaga `--require-screen-info` (oraz zmienna środowiskowa `BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_SCREEN_INFO`)
  wymusza, by każdy wpis snapshot posiadał sekcję `screen` z co najmniej jednym kluczowym polem (np. `index`, `name`,
  `refresh_hz` lub `resolution`), dzięki czemu audyt potwierdza kompletność kontekstu multi-monitorowego.
  Wbudowana walidacja dopasowuje każdy snapshot do filtrów opisanych w metadanych decision logu
  (lista/severity_min, okno czasowe `since`/`until`, limity liczby wpisów, filtry ekranu i eventu) i
  zatrzymuje weryfikację, jeśli którakolwiek obserwacja wykracza poza deklarowany zakres.  To pozwala
  na szybkie wykrycie niespójnych logów już na etapie audytu bezpieczeństwa.
  Te same warunki można zdefiniować przez zmienne środowiskowe (`BOT_CORE_VERIFY_DECISION_LOG_EXPECT_*`
  oraz `...EXPECT_FILTERS_JSON`).  W przypadku poprawnej weryfikacji wypisywane jest podsumowanie wraz z
  metadanymi audytu, a log podpisany kryptograficznie zostaje uznany za zgodny z parametrami pipeline'u.
  Dodatkowe flagi `--max-event-count ZDARZENIE=LIMIT` oraz `--min-event-count ZDARZENIE=MINIMUM`
  (wraz ze zmiennymi `BOT_CORE_VERIFY_DECISION_LOG_MAX_EVENT_COUNTS_JSON` i
  `BOT_CORE_VERIFY_DECISION_LOG_MIN_EVENT_COUNTS_JSON`) pozwalają natychmiast zablokować logi, w których
  zliczona liczba zdarzeń (np. `reduce_motion`, `overlay_budget`, `jank_spike`) przekracza operacyjne
  limity KPI albo nie osiąga wymaganego minimum (np. oczekiwany alert sanity-check podczas testów
  reduce-motion).  Wymusza to zbieranie lokalnego podsumowania i powiązuje audyt decision logu z limitami
  ustalonymi dla profili ryzyka konserwatywny/zbalansowany/agresywny/manualny.
  Operatorzy mogą skorzystać z predefiniowanych profili ryzyka (`--risk-profile conservative/balanced/aggressive/manual`
  lub `BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILE`) – każda konfiguracja automatycznie wymusza kombinację limitów KPI
  (max/min liczby zdarzeń), minimalny próg severity oraz obligatoryjną obecność metadanych monitora.
  Profil konserwatywny ogranicza np. `overlay_budget` do zera i wymaga severity ≥ `warning`, balanced dopuszcza pojedyncze
  piki janku przy severity ≥ `notice`, a agresywny pracuje z progiem `info` – wszystkie te wartości trafiają także do raportu
  audytowego.  Wariant `manual` pozostawia ustawienia bez zmian dla niestandardowych scenariuszy.  Wprowadzone rozszerzenie
  `--summary-json` (oraz zmienna `BOT_CORE_VERIFY_DECISION_LOG_SUMMARY_JSON`)
  pozwala dodatkowo przekazać plik wygenerowany przez `watch_metrics_stream --summary-output`.  Skrypt
  przelicza własne statystyki (łączna liczba snapshotów, agregaty FPS na event, zestaw ekranów,
  rozkład severity oraz pierwsza/ostatnia obserwacja) i porównuje je z artefaktem – wliczając w to
  podpis HMAC (jeśli obecny) oraz deklarację w metadanych decision logu (`summary_signature`).  Każda
  rozbieżność – brak zadeklarowanego monitora, różne liczniki severity, niespójne wartości FPS, brak
  oczekiwanego podpisu bądź niezgodny identyfikator klucza – kończy audyt kodem błędu, dzięki czemu
  pipeline demo→paper→live może automatycznie wychwycić manipulacje w podsumowaniu, także w scenariuszach
  offline (`--from-jsonl`, artefakty `.jsonl.gz`).  Jeżeli potrzebne są niestandardowe presety telemetryczne,
  walidator pozwala je wczytać przez `--risk-profiles-file` lub zmienną `BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILES_FILE`.
  Nowe profile są oznaczane w raporcie jako `origin=verify:…`, co ułatwia rozpoznanie, z którego artefaktu/repozytorium
  pochodził zestaw progów KPI.
  Dla szybkiego audytu można użyć flagi `--print-risk-profiles` (lub zmiennej
  `BOT_CORE_VERIFY_DECISION_LOG_PRINT_RISK_PROFILES`), która wypisuje bieżącą listę presetów wraz
  z metadanymi pochodzenia – wliczając pliki/katalogi zadeklarowane przez `--risk-profiles-file`
  oraz wartości z `--core-config`.  Operatorzy mogą dzięki temu przed walidacją potwierdzić, jaki
  profil zostanie zastosowany i czy niestandardowe presety zostały poprawnie zarejestrowane.
  offline (`--from-jsonl`, artefakty `.jsonl.gz`).
  Wynik walidacji można zarchiwizować w ustrukturyzowanej postaci: flaga `--report-output` (oraz zmienna
  `BOT_CORE_VERIFY_DECISION_LOG_REPORT_OUTPUT`) zapisuje raport JSON zawierający `report_version`, znacznik czasu
  generacji, metadane decision logu, lokalnie przeliczone podsumowanie oraz – gdy dostępne – wynik weryfikacji
  podsumowania (`summary_validation` z informacją o podpisie).  Sekcje `enforced_event_limits` oraz
  `enforced_event_minimums` dokumentują obowiązujące limity KPI (zarówno maksymalne, jak i minimalne), a nowy blok
  `risk_profile` opisuje zastosowany profil wraz z progami severity i aktywnymi limitami.  Takie dane pozwalają pipeline'owi
  historycznie porównywać profile ryzyka.  Raport może być
  kierowany na STDOUT (`-`) lub do pliku w repozytorium artefaktów CI, dzięki czemu zespoły ryzyka mają jeden spójny
  dokument audytowy obejmujący metadane TLS/tokenów, filtry severity/screen/time oraz wszystkie statystyki FPS.
  To rozszerzenie pozwala automatycznie porównywać audyty decision logów między etapami demo→paper→live, bez potrzeby
  manualnego parsowania JSONL.
  Błędne podpisy, brak oczekiwanych metadanych lub niespełnienie wymagań kończą się kodem 2 i szczegółowym
  logiem diagnostycznym.  To narzędzie stanowi obowiązkowy krok audytowy w pipeline demo→paper→live przed
  eskalacją alertów.
  do historii poleceń.

### Powłoka Qt/QML – MVP

* `ui/` zawiera projekt CMake (Qt 6) budujący binarkę `bot_trading_shell` oraz bibliotekę QML z komponentami bazowymi
  (`BotAppWindow`, `CandlestickChartView`, `SidePanel`, `StatusFooter`).
* Klient gRPC (`TradingClient`) korzysta wyłącznie z kanału HTTP/2, pobiera historię (`GetOhlcvHistory`) oraz strumień
  (`StreamOhlcv`) w tle, a następnie aktualizuje model `OhlcvListModel` (ring-buffer 10k+ świec, `candleAt()` dla QML).
* QML implementuje krzyż celowniczy, tooltipy oraz dynamiczne skalowanie osi; CandlestickChartView renderuje nakładki
  EMA12/EMA26/VWAP (LineSeries) sterowane przez `PerformanceGuard` i automatycznie redukuje liczbę overlayów przy aktywnym
  trybie „reduce motion” lub spadku FPS poniżej `disable_secondary_when_fps_below`.
* `SidePanel` prezentuje parametry guardu, status połączenia oraz zrzut `RiskState` (profil, wartość portfela, drawdown, dźwignia)
  wraz z listą limitów ekspozycji pobieranych z `RiskService`; przekroczenia progów są wyróżniane kolorystycznie i raportowane do
  telemetrii (`overlay_budget`) oraz powiązanego alertingu.  Zdarzenia jank przekraczające budżet (`frame_ms > jank_threshold_ms`)
  są emitowane jako osobny event telemetryjny, który zasila `UiTelemetryAlertSink` i JSONL.
* `FrameRateMonitor` (C++) nasłuchuje `frameSwapped` głównego okna i po spadku FPS poniżej progów guardu (np. 55 FPS @60 Hz,
  110 FPS @120 Hz) emituje `reduceMotionActive`; właściwość jest eksponowana do QML i powoduje natychmiastowe wygaszenie
  animacji wtórnych oraz ograniczenie overlayów w każdym oknie.
* `UiTelemetryReporter` wysyła zdarzenia UI do `MetricsService` (`PushMetrics`): wejście/wyjście z trybu reduce motion, budżety
  overlayów oraz liczbę aktywnych okien multi-window; konfiguracja odbywa się przez flagi CLI (`--metrics-endpoint`,
  `--metrics-tag`, `--metrics-auth-token`, `--metrics-auth-token-file`) lub wpis w YAML. Powłoka respektuje także zmienne
  środowiskowe `BOT_CORE_UI_METRICS_*` (m.in. `ENDPOINT`, `TAG`, `ENABLED/DISABLED`, `USE_TLS`, `ROOT_CERT`, `CLIENT_CERT`,
  `CLIENT_KEY`, `SERVER_NAME`, `SERVER_SHA256`, `AUTH_TOKEN`, `AUTH_TOKEN_FILE`) i automatycznie wymusza TLS, gdy dostarczono
  materiał certyfikacyjny; token autoryzacyjny może zostać wczytany bezpośrednio z pliku.
* Preferowany monitor można wybrać flagami CLI `--screen-name`, `--screen-index` lub `--primary-screen`.  To samo zachowanie
  jest dostępne przez zmienne środowiskowe `BOT_CORE_UI_SCREEN_NAME`/`BOT_CORE_UI_SCREEN_INDEX`/`BOT_CORE_UI_SCREEN_PRIMARY`
  (z rozróżnieniem wartości pustych jako „brak preferencji”).  Powłoka próbuje dopasować nazwę ekranu niezależnie od
  wielkości liter, a przy żądaniu indeksu poza zakresem loguje ostrzeżenie i pozostawia okno na bieżącym monitorze.  Operatorzy
  multi-monitor mogą dzięki temu przypiąć główne okno do wyświetlacza transakcyjnego już w pipeline demo→paper→live.
* `UiTelemetryReporter` dołącza do zdarzeń JSON kontekst aktywnego ekranu (nazwa, producent, model, indeks, geometrię i odświeżanie).
  Dane są aktualizowane przy każdej zmianie monitora i stanowią część audytu telemetryjnego, co ułatwia diagnozowanie problemów z FPS/jank
  na stanowiskach demo→paper→live.
* `UiTelemetryAlertSink` przenosi metadane ekranu oraz aktywny profil ryzyka do kontekstu alertów i wpisów JSONL (`screen_index`, rozdzielczość, odświeżanie, DPR, `risk_profile`, `risk_profile_origin`)
* `UiTelemetryAlertSink` przenosi metadane ekranu do kontekstu alertów i wpisów JSONL (`screen_index`, rozdzielczość, odświeżanie, DPR)
  oraz dopisuje skrócony opis monitora w treści powiadomień, dzięki czemu operatorzy wiedzą, na którym stanowisku pipeline'u demo→paper→live
  wystąpiła degradacja wydajności.
* Połączenie telemetrii może być zabezpieczone TLS/mTLS – powłoka obsługuje `--metrics-use-tls`, ścieżki certów/kluczy oraz
  pinning SHA-256 (`--metrics-server-sha256`), a stuby developerskie (`run_metrics_service.py`, `run_trading_stub_server.py`)
  potrafią wystartować serwer z materiałem TLS i opcjonalnym wymaganiem certyfikatu klienta.
* Sekcja `runtime.metrics_service` w `config/core.yaml` ustawia host/port serwera telemetrii, rozmiar historii (`history_size`), aktywność log sinka (`log_sink`), parametry eksportu JSONL (`jsonl_path`, `jsonl_fsync`), opcjonalny token autoryzacyjny (`auth_token`), ścieżkę logu alertów UI (`ui_alerts_jsonl_path`) oraz nowy klucz `ui_alerts_risk_profile`.  Loader normalizuje nazwę profilu do małych liter i waliduje ją względem sekcji `risk_profiles`, dzięki czemu runtime i narzędzia telemetryczne otrzymują spójne metadane audytowe.
* Sekcja `runtime.metrics_service` w `config/core.yaml` ustawia host/port serwera telemetrii, rozmiar historii (`history_size`), aktywność log sinka (`log_sink`), parametry eksportu JSONL (`jsonl_path`, `jsonl_fsync`), opcjonalny token autoryzacyjny (`auth_token`) oraz ścieżkę logu alertów UI (`ui_alerts_jsonl_path`).
* Pola `reduce_motion_alerts`/`reduce_motion_mode`/`reduce_motion_category`/`reduce_motion_severity_*` sterują zachowaniem sinka reduce-motion (tryby `enable`/`jsonl`/`disable`), który deduplikuje zdarzenia spadku FPS i loguje je do JSONL. Loader konfiguracji normalizuje wartości trybów do małych liter i odrzuca inne warianty, dzięki czemu błędna konfiguracja zostaje wykryta przed startem runtime.
* Pola `overlay_alerts`/`overlay_alert_mode`/`overlay_alert_category`/`overlay_alert_severity_*`/`overlay_alert_severity_critical`/`overlay_alert_critical_threshold` kontrolują eskalację przekroczeń budżetu nakładek (jsonl-only vs pełne alerty) oraz próg krytyczny. Tryb jest walidowany przez loader (`enable`/`jsonl`/`disable` – bez rozróżniania wielkości liter), więc błędna wartość zostanie zatrzymana przed uruchomieniem usług.
* Pola `jank_alerts`/`jank_alert_mode`/`jank_alert_category`/`jank_alert_severity_spike`/`jank_alert_severity_critical`/`jank_alert_critical_over_ms` konfigurują alerty „jank spike” (przekroczenie budżetu klatki) i logowanie JSONL, umożliwiając np. eskalację krytyczną po przekroczeniu limitu ms. Loader normalizuje tryb i zgłasza błąd, jeśli YAML zawiera wartość spoza zbioru `enable`/`jsonl`/`disable`.
* Wsparcie multi-window: `BotAppWindow` potrafi otwierać dodatkowe `ChartWindow` (`Ctrl+N`/przycisk), zapamiętywać liczbę i geometrię okien
  (`Qt.labs.settings`) oraz synchronizować guard/instrument pomiędzy wszystkimi widokami – spełnia wymagania pracy na wielu monitorach.
* `ui/config/example.yaml` oraz flagi CLI (w tym `--overlay-disable-secondary-fps`) pozwalają spiąć powłokę z dowolnym datasetem
  stubu (`--dataset`), kontrolować budżet animacji i overlayów oraz przyspieszać iteracje multi-window/120 Hz bez uruchamiania
  całego core.

## Pipeline danych i synchronizacja z UI

1. Demon core subskrybuje publiczne dane (REST + cache Parquet) i publikuje strumień OHLCV (gRPC streaming).
2. UI utrzymuje ring-buffer (ostatnie 10k świec widocznych) oraz persistent cache (historyczne okna) zasilany przez `GetOhlcvHistory`.
3. Zlecenia wysyłane są przez gRPC (idempotentny `SubmitOrder`) z asynchronicznymi potwierdzeniami. Retry idempotentny (client-side tokens).
4. Risk Engine udostępnia `RiskState` i triggeruje alerty (Telegram/Signal/SMS) poprzez istniejącą warstwę `bot_core/alerts`.
5. Decision log (JSONL) pozostaje w core; UI pobiera metadane i potwierdzenia podpisów.
6. Telemetria UI (latencje renderu, FPS, drop rate) trafia do core przez gRPC `MetricsService` i jest agregowana w Prometheusie; UI nie przechowuje danych giełdowych lokalnie poza ring-bufferem.

## Bezpieczeństwo i obserwowalność

* mTLS shell↔core (mutual TLS), pinned cert chain; rotacja certów co 90 dni.
* RBAC – polityki w core z tokenami krótkiego życia, UI uzyskuje `session_token` po autoryzacji operatora.
* OpenTelemetry – trace’y od UI (gRPC client) do core (server) z eksportem do Jaeger/Grafana Tempo.
* Prometheus – metryki latencji, FPS UI, usage CPU/RAM; alerty w `/ops/alerts/smoke_rules.yml`.
* Decision log – JSONL z podpisami cyfrowymi; UI umożliwia eksport/druk.

## Testy i CI/CD

* **Kontrakty** – golden tests dla Protobuf (stabilne `.pb.bin` w `/proto/tests/golden`) oraz stub serwera do UI (`/tests/ui_stubs`). Weryfikacja braku breaking changes przez `buf breaking --against`.
* **Perf** – soak 24h z 5k msg/s, chaos (latencja, błędy 5xx), benchmark animacji (QML Profiler, FPS >58 @60 Hz), testy downsamplingu LTTB i pan/zoom 60 FPS dla 10k świec + 3 overlayów.
* **E2E** – `submit_order → fill → portfolio update` z idempotentnym retry oraz pipeline alertów (decision log, risk update, UI toast) w jednym przebiegu.
* **Lint** – zakaz WebSocketów/przeglądarkowych zależności (CI step `lint_no_web`), `clang-tidy`/`cppcheck` dla core, `qmllint`/`qmlformat` dla UI, `reuse lint` dla licencji ikon.
* **CI pipeline** – generowanie stubów, build core (CMake + Ninja), build UI (Qt), pakiety (IFW/Sparkle/AppImage + .deb/.rpm), testy automatyczne (kontrakty, unit, E2E), raporty QML Profiler, publikacja logów FPS i latencji.
* **Release gating** – staging `alpha/beta/stable`, smoke test papierowy (CI) musi przejść, `validate_paper_smoke_summary` oraz walidacja podpisów instalatorów; brak release jeśli `publish.required` = true i auto-publikacja się nie powiedzie.

## Wdrożenia i packaging

* **Windows** – MSI (Qt IFW/WiX), podpis EV, auto-update (WinSparkle) z delta patch.
* **macOS** – `.app` + DMG, Hardened Runtime, notarization, Sparkle do aktualizacji.
* **Linux** – AppImage + `.deb/.rpm`, repo aktualizacji, sygnatury.
* **Auto-update** – staged rollout, rollback w 1 klik, detekcja niespójności wersji core↔shell (API handshake na starcie).
* **Telemetria** – anonimowe wersje, crash reporting (minidumps + symbol server); integracja z Sentry/Breakpad.
* **Repo aktualizacji** – `/ops/update_feed` publikuje manifesty delta patch (WinSparkle/Sparkle) oraz AppImage update info. Wersje odczytują kanały (`alpha/beta/stable`) i raportują status instalacji do telemetrii.
* **Zgodność core↔shell** – handshake gRPC `GetCompatibilityInfo`; UI odmawia startu, jeśli wersje nie są kompatybilne. Automatyczny rollback poprzez pobranie poprzedniego wydania i sygnał do update daemon.

## Roadmapa wysokiego poziomu

1. **Specyfikacja kontraktów** – finalizacja `.proto`, golden tests, ADR dot. braku WebSocketów i wyboru gRPC/IPC.
2. **Core daemon MVP** – implementacja MarketData/Risk/Order/Metrics z gRPC i buforami.
3. **UI shell MVP** – ekran Trading Dashboard (Qt Quick + Charts), integracja streamingowa (stub core), podstawowe animacje i A11y.
4. **Integracja core↔shell** – pełny przepływ gRPC, kontrola wersji API, mTLS i RBAC.
5. **Performance & Animations** – optymalizacja custom QQuickItem, testy 60/120 Hz, adaptacja „reduce motion”, automatyczne skracanie animacji przy spadku FPS, raporty QML Profiler.
6. **Packaging & Auto-update** – buildy MSI/DMG/AppImage + delta updates, podpisy EV/notarization, staging rollout, repo aktualizacji, telemetria wersji.
7. **Observability & Compliance** – telemetry, decision log signing, dashboardy ops, powiadomienia compliance (Slack/Telegram/Signal), audyt RBAC.
8. **E2E certification** – smoke testy submit→fill, testy chaos, soak 24h, raporty QML Profiler, release alpha/beta/stable, weryfikacja handshake core↔shell i RBAC.
9. **UX & A11y polish** – finalizacja komponentów (dark/light/high-contrast), testy screen readerów, multiwindow, profile workspace, integracja i18n (pl/en) i lokalizacji rynkowych.

## Deliverables i artefakty

1. **Repozytorium** – katalogi `/core`, `/ui`, `/proto`, `/packaging`, `/ops`, `/docs/styleguide`, `/docs/adr`. Każdy zawiera README z instrukcją build/run/test.
2. **Styleguide animacji** – `ui/styleguide/animations.md` + katalog `ui/styleguide/screencasts/` z przykładami 60/120 Hz (before/after).
3. **Biblioteka komponentów QML** – dokumentacja `ui/components/README.md` z przykładami i testami wizualnymi (screenshot diff lub Squish).
4. **Skrypty build/release** – `packaging/build_all.py`, `packaging/signing/*.py`, `packaging/notarize.sh` + definicje kanałów `alpha/beta/stable`.
5. **Dashboardy ops** – `ops/dashboards/*.json` dla Grafany, `ops/alerts/*.yml` (Prometheus/Alertmanager), integracja z compliance (webhook).
6. **CI workflows** – GitHub Actions/GitLab CI definicje: generowanie stubów, build core/ui, smoke tests, walidacja summary, publikacja artefaktów, render Markdown.
7. **ADR** – kluczowe decyzje (brak WebSocketów, gRPC vs IPC fallback, adaptacja animacji) opisane w `docs/adr/adr-00x-*.md`.

## KPI i Definition of Done

* KPI: event→frame p95 < 150 ms, jank <1%, UI drop rate <0.1% @5k msg/s, RAM UI <300 MB, CPU p95 <25% (i7/RTX laptop),
  chart pan/zoom 60 FPS dla 10k świec + 3 overlayów.
* DoD: wszystkie KPI spełnione, testy E2E i perf zaliczone, instalatory podpisane/notarized, UI 120 Hz bez dropów (raport z QML Profiler),
  brak WebSocketów, Protobuf v1 zamrożone, ADR dla kluczowych decyzji.

