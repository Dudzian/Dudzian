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

