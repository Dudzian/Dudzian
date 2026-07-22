# CryptoHunter M0.1 — current-state inventory

## Executive summary

- Obecny Windows EXE jest budowany przez PyInstaller z `ui/pyside_app/__main__.py`, a aplikacja uruchamia PySide6 `QGuiApplication` i `QQmlApplicationEngine` dla `ui/pyside_app/qml/MainWindow.qml` z importami z `ui/qml`.
- Repo zawiera kilka generacji UI: aktywny `ui/pyside_app`, współdzielone QML `ui/qml`, alternatywny/legacy `ui/desktop` oraz brak zweryfikowanej ścieżki `ui/src` w obecnym EXE.
- Execution ma wieloadapterowy `LiveExecutionRouter`, ale aktualny bootstrap GUI opakowuje `ExchangeManager` jako pojedynczy adapter `primary` i ustawia `default_route=("primary",)`.
- Znaczna część roadmapowych ekranów i bramek ma postać `preview_*`, fixture lub read model; oznaczono je jako `PREVIEW_ONLY`, jeśli nie znaleziono realnego runtime I/O.
- Nie wykonano prawdziwego tradingu, nie użyto sekretów i nie uruchomiono Live.

## Baseline commit

`repository_baseline_commit`: `d8de0acba975af035df296d2d39d36ea58979c65`

## Faktyczny entrypoint obecnego CryptoHunter.exe

`CryptoHunter.spec` przekazuje do PyInstaller `ui/pyside_app/__main__.py`, nadaje nazwę `CryptoHunter` i zbiera aplikację jako katalog `CryptoHunter`. `ui/pyside_app/__main__.py` deleguje do `ui.pyside_app.app.main`. `AppOptions` domyślnie rozwiązuje QML do `ui/pyside_app/qml/MainWindow.qml`, a `BotPysideApplication.load` tworzy `QQmlApplicationEngine` i ścieżki importów QML.

Dowody: `CryptoHunter.spec` (`app_datas`, `Analysis`, `EXE`), `ui/pyside_app/__main__.py`, `ui/pyside_app/app.py` (`AppOptions`, `BotPysideApplication.load`), `.github/workflows/windows-build.yml`, `scripts/build_windows.ps1`, `tests/ui_pyside/test_source_smoke.py`, `tests/windows_build/test_windows_build_pipeline.py`.

## Mapa procesów i komponentów

| Proces/warstwa | Obecny dowód | Status | Uwagi |
| --- | --- | --- | --- |
| Desktop EXE | `CryptoHunter.spec`, `scripts/build_windows.ps1`, `.github/workflows/windows-build.yml` | IMPLEMENTED | PyInstaller onedir, nazwa `CryptoHunter.exe`, smoke `--smoke-test --offscreen`. |
| PySide/QML shell | `ui/pyside_app/app.py`, `ui/pyside_app/qml/MainWindow.qml` | IMPLEMENTED | Aktualny entrypoint EXE. |
| Shared QML | `ui/qml/components`, `ui/qml/dashboard` | PARTIAL | Bundlowane i importowane, ale część ekranów jest preview/read-only. |
| Runtime service dla UI | `ui/backend/runtime_service.py` | PARTIAL | Dostarcza modele do QML; wymaga dalszego rozdziału fixture vs runtime. |
| ExchangeManager | `bot_core/exchanges/manager.py`, `config/core.yaml` | PARTIAL | Rejestr adapterów i paper/testnet helpery istnieją; aktywacja zależy od konfiguracji. |
| Live execution | `bot_core/execution/live_router.py`, `bot_core/execution/execution_service.py` | PARTIAL | Router multi-adapter, ale frontend bootstrap używa pojedynczego `primary`. |
| Paper execution | `bot_core/execution/paper.py` | PARTIAL | Ledger/symulacja istnieją; M0 musi ustalić kontrakt trwałości. |
| Risk/security | `bot_core/risk`, `bot_core/security`, `core/licensing` | PARTIAL | Reusable foundation, wymaga rozróżnienia bramek runtime od preview. |
| Updater/package | `ui/backend/update_controller.py`, `tests/update/test_update_package_cli.py` | PARTIAL | Obecne mechanizmy istnieją, finalny rollback/update contract jeszcze nieustalony. |

## Mapa frontendów

| Frontend | W obecnym EXE | Aktywnie używany | Ocena | Unikalne funkcje do przeglądu |
| --- | --- | --- | --- | --- |
| `ui/pyside_app` | Tak | Tak | IMPLEMENTED | App bootstrap, preview gates, controllers, theme bridge. |
| `ui/qml` | Tak | Częściowo | PARTIAL | FirstRunWizard, dashboard, runtime/risk panels. |
| `ui/desktop` | Nie | Nie w PyInstaller EXE | PARTIAL/legacy candidate | React/Electron renderer; wymaga przeglądu UX przed decyzją. |
| `ui/src` | Nie znaleziono | Nie zweryfikowano | UNKNOWN | Brak aktywnego dowodu w obecnym EXE. |
| `ui/backend` | Pośrednio | Tak dla QML context | PARTIAL | Runtime/licensing/update/onboarding controllers. |

## Tabela komponentów ze statusem

Szczegółowa lista komponentów, ścieżek dowodowych i symboli znajduje się w [current_state_inventory.json](current_state_inventory.json). Statusy użyte w M0.1: `IMPLEMENTED`, `PARTIAL`, `PREVIEW_ONLY`, `PLANNED`, `CONFLICTING`, `UNKNOWN`.

## Verified working

- Windows packaging workflow instaluje zależności, uruchamia `tests/windows_build`, buduje PyInstaller, weryfikuje artefakt i uruchamia izolowany smoke test EXE.
- Lokalny source smoke dla PySide/QML jest objęty `tests/ui_pyside/test_source_smoke.py`.
- `LiveExecutionRouter` ma testy failover i testy podpisywania/withdrawal bez prawdziwego API.

## Static contract / preview only

- Moduły `ui/pyside_app/preview_*` opisują read modele i statyczne fixture dla paper orders, read-only market data, decision dry-run, testnet/sandbox, packaging gates i safety gates.
- Trading universe/AI candidates w UI nie są dowodem na runtime live universe automation.

## Documented but not verified

- Tray/system tray, autostart, HUD/chmurka, reconnect GUI do Core i zachowanie po zamknięciu okna wymagają osobnego M0.
- Pełny updater rollback i backup przed aktualizacją są udokumentowane fragmentarycznie, ale nie zostały uznane za kompletne bez finalnego kontraktu.
- Multi-account na jednej giełdzie nie został potwierdzony jako aktualny model obecnego EXE.

## Conflicts with accepted roadmap

- Dokumenty i preview flows mówią o demo/paper/live/testnet, ale kod rozdziela live przez `runtime.execution.live.enabled`, `DUDZIAN_TEST_MODE` i adaptery; nazewnictwo środowisk jest niespójne.
- Roadmapowe multi-exchange możliwości istnieją w routerze i ExchangeManagerze, lecz aktualna ścieżka frontend bootstrap rejestruje realnie jeden adapter `primary`.

## Reusable foundations

- PySide6/QML shell i Windows PyInstaller packaging.
- `LiveExecutionRouter`, `RoutingPlan`, route overrides, circuit breakers i decision log.
- `ExchangeManager`, dynamic native adapter registry, sandbox helper i paper simulators.
- Risk/security/licensing/signing primitives.
- Journal/ledger/update/reporting test foundations.

## Likely legacy or duplicate paths

- `ui/desktop` jako alternatywny Electron/React frontend poza obecnym EXE.
- Liczne `preview_*` read modele powielające pojęcia runtime; są użyteczne jako kontrakty źródłowe, ale nie jako dowód wykonania.

## Unknowns requiring later M0 decisions

- Docelowa liczba kont na giełdę i model routingu danych oraz zleceń.
- Kanoniczny model persistence dla order/fill/portfolio/recovery.
- Operator shell lifecycle: tray, autostart, reconnect, background survival.
- Które funkcje preview przechodzą do runtime, a które pozostają wyłącznie dokumentacją produktu.

## High-risk reconstruction areas

- Rozdzielenie demo/paper/testnet/live bez przypadkowego otwarcia Live.
- Migracja UI bez utraty unikalnych ekranów z `ui/qml` i potencjalnie `ui/desktop`.
- Ujednolicenie multi-exchange routera z obecnym `primary` bootstrapem.
- Sekrety, licencje, hardware wallet i live gate jako twarde bramki runtime, nie tylko statusy UI.

## Recommended order for remaining M0 elements

1. Zamknąć słownik środowisk i statusów.
2. Ustalić proces desktop/core/background.
3. Zmapować wszystkie UI do kanonicznych ekranów.
4. Zdefiniować execution/account/exchange contract.
5. Dopiero potem podjąć decyzje migracyjne i legacy cleanup.
