# FUNCTIONAL-PREVIEW-6.9 — BLOK D anti-duplication integration audit

## Existing launch path

- Główny widoczny launcher istniejącej aplikacji preview to `run_ui_preview_visible_doubleclick.bat`; wykrywa repo, bootstrappuje lokalne `.venv`, sprawdza PySide6 i uruchamia `python -m ui.pyside_app --config ui/config/preview_local.yaml`.
- Krótszy wrapper Windows dla tego samego widocznego UI znajduje się w `scripts/windows/run_ui_preview_visible.bat` i uruchamia tę samą komendę modułową z tym samym profilem `ui/config/preview_local.yaml`.
- Pozostałe pliki `.bat` (`build_preview_exe_windows*.bat`, `make_audit_bundle.bat`) są builderami/audytami artefaktów albo pakietów, nie główną ścieżką klikanego PySide/QML preview.
- Po przyszłej integracji `.bat` powinien nadal uruchamiać tę samą aplikację i ten sam Python entrypoint; nie należy tworzyć drugiego launchera ani drugiego frontendu.

## Existing PySide/QML entrypoint

- Faktyczny Python entrypoint to `ui/pyside_app/__main__.py`, który deleguje do `ui.pyside_app.app.main`.
- `ui/pyside_app/app.py` definiuje `AppOptions`, `BotPysideApplication`, `main()` i tryb `--smoke`.
- Domyślny QML entrypoint pochodzi z konfiguracji aplikacji, a launcher preview podaje `ui/config/preview_local.yaml`; bez override `--qml` ładowany jest entrypoint skonfigurowany jako `ui/pyside_app/qml/MainWindow.qml`.

## Existing QML engine/context setup

- `QGuiApplication` jest tworzony albo odzyskiwany w `BotPysideApplication._ensure_qt_app()`.
- `QQmlApplicationEngine` jest tworzony w `BotPysideApplication.load()`.
- Import paths QML są składane w `BotPysideApplication.load()` przed instalacją bridge'a.
- Istniejący kontekst QML jest instalowany przez `QmlContextBridge.install()` wywołane przed `engine.load(...)`.
- Root QML jest ładowany przez `engine.load(QUrl.fromLocalFile(qml_file.as_posix()))` w `BotPysideApplication.load()`.
- Po załadowaniu root obiektu `BotPysideApplication.load()` odczytuje `engine.rootContext()` i przenosi wybrane kontrolery context-property na root properties tylko dla `dashboardSettingsController`, `complianceController`, `reportController`.

## Existing context properties / bridges

Aktualne `setContextProperty` w realnym setupie PySide/QML są w jednym miejscu: `ui/pyside_app/qml_bridge.py::QmlContextBridge.install()`:

- `uiConfig` → wariant konfiguracji UI.
- `cloudRuntimeEnabled` → flaga cloud runtime.
- `grpcBridge` → `UiGrpcBridge`, adapter do `RuntimeService` dla QML.
- `runtimeState` → `UiRuntimeState`, wspólny store statusu runtime/feed/cloud.
- `licensingController` → `LicensingController`.
- `diagnosticsController` → `DiagnosticsController`.
- `layoutController` → `LayoutProfileController`.
- `modeWizardController` → `ModeWizardController`.
- `strategyManagementController` → `StrategyManagementController`.
- `theme` → `ThemeBridge`.
- `typedPreviewBridge` → `LocalPreviewStateBridge`.

Istniejące podobne bridge/provider/snapshot:

| Plik | Klasa/funkcja | Context property | Overlap | Decyzja |
| --- | --- | --- | --- | --- |
| `ui/pyside_app/preview_state_bridge.py` | `LocalPreviewStateBridge` | `typedPreviewBridge` | Read-only lokalne snapshoty paper/scanner/governor/portfolio/alert/runtime boundary. Nie kataloguje intencji akcji i nie obsługuje wyboru source-control. | Compose / do not touch. |
| `ui/pyside_app/controllers/ui_grpc_bridge.py` | `UiGrpcBridge` | `grpcBridge` | Udostępnia `RuntimeService`, nie jest preview-only action dispatch catalog. | Do not touch. |
| `ui/pyside_app/controllers/ui_runtime_state.py` | `UiRuntimeState` | `runtimeState` | Status runtime/feed/cloud, nie dispatch intencji paper controls. | Do not touch. |
| `ui/backend/runtime_service.py` | `RuntimeService` | pośrednio przez `grpcBridge` i kontrolery | Backend service dla UI, nie QML-safe source-only action dispatch. | Do not touch. |

Wniosek: `PaperRuntimeActionDispatchQtBridge` nie dubluje istniejącego context bridge'a. Uzupełnia lukę: brakuje QML-safe, source-only, fail-closed mostka opisującego intencje start/stop/pause/resume/refresh dla paper runtime bez wykonywania komend.

## Existing preview/paper controls and mock/placeholder areas

- `ui/pyside_app/qml/MainWindow.qml` zawiera lokalny preview state paper/session: `paperSessionState`, `paperSessionStatus`, `paperSessionTicks`, helpery snapshotów i funkcje `startLiveLikePaperSimulation`, `pausePaperSession`, `stopPaperSession`, `resetPaperSession`, `generatePaperTick`, `startPaperPreview`.
- `ui/pyside_app/qml/views/OperatorDashboard.qml` ma przyciski `Start Paper Preview` i `Generate Next Tick`, które wywołują lokalne metody `previewState.startPaperPreview()` oraz `previewState.generatePaperTick()`.
- `ui/pyside_app/qml/views/PaperTerminal.qml` ma aktywne lokalne kontrolki BUY/SELL, LIMIT/MARKET, percent chips, timeframe, bottom tabs, order book price click i `paperTerminalSimulateOrderButton`, ale opisy mówią jednoznacznie: local-only, no real order, no network/API call, runtime loop not started.
- `paperTerminalLifecycleReservedPlaceholder` w `PaperTerminal.qml` jest jawnie zarezerwowanym placeholderem dla lifecycle/live states i nie powinien być traktowany jako realne miejsce dispatchu.
- Istniejące elementy do ewentualnej przyszłej konsumpcji nowego bridge'a to istniejący panel `runtimeSessionControlPanel` oraz istniejące paper controls w `OperatorDashboard.qml`/`PaperTerminal.qml`; w tym kroku nie zmieniono QML i nie dodano handlerów.

## New Block D bridge pipeline summary

- `preview_action_dispatch_contract.py`: klasyfikuje dozwolone i odrzucone intencje paper runtime; brak PySide/QML i brak wykonania.
- `preview_action_dispatch_audit.py`: buduje audytowy envelope dla intencji; accepted means not executed.
- `preview_action_dispatch_catalog.py`: buduje katalog UI-safe akcji start/stop/pause/resume/snapshot-refresh dla `runtimeSessionControlPanel`.
- `preview_action_dispatch_selection.py`: mapuje action/source-control do wyniku wyboru fail-closed.
- `preview_action_dispatch_bridge_snapshot.py`: serializuje katalog i wybór do QML-safe plain dict.
- `preview_action_dispatch_bridge_provider.py`: source-only provider przechowujący ostatni lokalny wybór preview.
- `preview_action_dispatch_qt_bridge.py`: cienki `QObject` z `snapshot`, `previewSelectAction`, `previewSelectSourceControl`, `resetPreviewSelection`; nie rejestruje się sam.
- `preview_action_dispatch_qt_bridge_registration.py`: kontrolowany helper preflight dla jednego `context.setContextProperty`, ale nie jest wywołany w startupie.

## Duplication analysis

- Nie ma istniejącego bridge'a, który robi tę samą rzecz co `PaperRuntimeActionDispatchQtBridge`.
- Największy overlap jest z `typedPreviewBridge`, ale to snapshot/read-only mirror, a nie catalog/selection/audit dispatch-intention bridge.
- Nie należy zastępować `typedPreviewBridge`; przyszła integracja powinna komponować nowy bridge obok istniejącego context setupu, dopiero po testach source-only i runtime smoke.
- Nie wolno dodać alternatywnego `QQmlApplicationEngine`, drugiego `QmlContextBridge`, drugiego `.bat`, osobnego QML root ani równoległego frontendowego panelu.

## Recommended single integration point

Rekomendowane jedyne miejsce przyszłej integracji helpera z 6.8: `ui/pyside_app/qml_bridge.py::QmlContextBridge.install()`, po istniejących instancjach bridge/provider i przed `engine.load(...)`, przez jedną kontrolowaną rejestrację context property `paperRuntimeActionDispatchBridge`.

Dlaczego to miejsce:

- Jest to aktualnie jedyny realny centralny punkt `setContextProperty` dla PySide/QML preview.
- Jest wywoływane przed załadowaniem root QML.
- Nie tworzy drugiego frontendu, bo wykorzystuje istniejący engine, istniejący context i istniejącą aplikację startowaną przez `.bat`.
- Pozwala utrzymać jeden kontrakt nazewnictwa context property oraz jeden punkt audytu.

## Explicit non-integration points

Nie integrować w:

- `run_ui_preview_visible_doubleclick.bat` ani `scripts/windows/run_ui_preview_visible.bat` — launchery mają pozostać launcherami, nie miejscem wiring UI bridge.
- `ui/pyside_app/app.py` — nie dodawać tam ad-hoc `setContextProperty`, bo centralnym miejscem jest `QmlContextBridge.install()`.
- `ui/pyside_app/qml/MainWindow.qml` — nie dodawać QML handlerów ani `Connections` w tym kroku.
- `ui/pyside_app/qml/views/OperatorDashboard.qml` i `ui/pyside_app/qml/views/PaperTerminal.qml` — nie zmieniać istniejących lokalnych mock controls w tym kroku.
- `ui/pyside_app/preview_state_bridge.py` — nie mieszać read-only snapshot mirror z action-dispatch-intention bridge.
- `ui/backend/runtime_service.py`, `bot_core/runtime/*`, `TradingController`, `DecisionEnvelope`, adaptery live/testnet, market/account fetch, secrets, order/fill paths — poza zakresem i ryzykowne.

## Risk assessment

- Ryzyko duplikacji: średnie, jeśli ktoś doda rejestrację poza `QmlContextBridge.install()` albo stworzy nowy launcher/frontend.
- Ryzyko UI/runtime: średnie, bo QML ma już lokalne paper controls; realne wiring bez mapy może pomylić mock state z runtime execution.
- Ryzyko bezpieczeństwa: wysokie przy niekontrolowanym podpięciu start/stop/pause jako realnych lifecycle commands; obecne moduły BLOKU D celowo zostawiają execution disabled.
- Ryzyko regresji: niskie w tym audycie, bo nie zmieniono QML ani startupu.

## Required tests before real wiring

Przed realnym wiringiem wymagane minimum:

- Aktualne testy BLOKU D: contract source guard, bridge snapshot, provider, Qt bridge, registration.
- Source-only test potwierdzający, że `paperRuntimeActionDispatchBridge` jest rejestrowany dokładnie raz i tylko w `QmlContextBridge.install()`.
- Test, że istniejące context properties (`typedPreviewBridge`, `runtimeState`, `grpcBridge`) pozostają bez zmiany.
- QML/runtime smoke po obowiązkowym ensure/install PySide6 i UI runtime deps.
- Test, że `.bat` nadal uruchamia `python -m ui.pyside_app --config ui/config/preview_local.yaml` i nie przełącza na drugi frontend.
- Test boundary, że wybór akcji z bridge'a nadal ma `execution_allowed=False`, `execution_performed=False`, brak order generation/submission i brak lifecycle execution.

## Decision: proceed to wiring vs close Block D first

`CLOSE_BLOCK_D_AS_BRIDGE_READY_NOT_WIRED`

Uzasadnienie: nowy bridge nie dubluje istniejącego bridge'a, ale aplikacja ma już rozbudowany lokalny mock/preview paper UI. Najbezpieczniejszy następny krok to zamknąć BLOK D jako „bridge ready, not wired” i dopiero w kolejnym, osobnym kroku wykonać kontrolowane wiring w jednym miejscu (`QmlContextBridge.install()`), z testem dokładnie jednej rejestracji i bez zmian launcherów/startupu/QML mock controls.
