# Interfejs użytkownika – kreator startowy i dashboard portfela

## Kreator konfiguracji (Setup Wizard)

* Lokalizacja: `ui/qml/views/SetupWizard.qml`, dostępny z zakładki **Kreator** w głównym oknie (`BotAppWindow.qml`).
* Kroki kreatora prowadzą przez aktywację licencji, wybór giełdy i instrumentu startowego, przegląd strategii oraz preferencje UI.
* Kreator korzysta z nowych metod `Application` (`supportedExchanges()`, `personalizationSnapshot()` oraz setterów `setUiTheme`, `setUiLayoutMode`, `setAlertToastsEnabled`).
* Ukończenie kreatora odblokowuje kolejne zakładki – po aktywnej licencji użytkownik trafia automatycznie na zakładkę **Wykres**.

## Dashboard portfela

* Lokalizacja: `ui/qml/views/PortfolioDashboard.qml`, dostępny w zakładce **Dashboard**.
* Wykorzystuje modele `riskModel`, `riskHistoryModel` i `alertsModel`, aby prezentować:
  - historię wartości portfela (wykres P&L),
  - ekspozycję per giełda i per strategia (na podstawie kodów `exchange:*` i `strategy:*`),
  - listę aktywnych alertów wraz z akcjami potwierdzenia.
* Widok aktualizuje się automatycznie po każdej zmianie snapshotu ryzyka lub historii.
* Nowy panel **Explainability** korzysta z `bot_core.ui.api.build_explainability_feed`, aby prezentować listę ostatnich decyzji AI wraz z wiodącymi cechami modelu. Dane pobierane są z dziennika decyzji (`TradingDecisionJournal`) i zawierają nazwę modelu, metodę wyjaśnienia oraz top cechy pozytywne/negatywne.
* Sekcja **Runtime Overview** otrzymała panel **Risk Journal**, który łączy się z usługą `RuntimeService.riskMetrics`/`riskTimeline` i prezentuje zagregowane blokady, zamrożenia oraz stress overrides. Panel zawiera wykres aktywności decyzji, filtry po strategii i sygnałach ryzyka, żetony z najczęściej flagowanymi strategiami oraz najczęściej pojawiającymi się flagami/stress failure, podsumowanie ostatniej akcji operatora, sygnalizację ostatnich blokad/zamrożeń/override'ów, karty ze statusem strategii wymagających uwagi (liczniki blokad/zamrożeń/override'ów + ostatnie zdarzenie) oraz modal do drill-downu (z akcjami zamrożenia/odblokowania portfela wywoływanymi na `RuntimeService`).

## Offline Runtime API dla UI

* Backend udostępnia funkcję `bot_core.ui.api.build_runtime_snapshot`, która zbiera migawkę portfela, statusy giełd, katalog strategii oraz najnowsze wpisy explainability/alertów. Snapshot jest zgodny z wymaganiami panelu dashboardu i konfiguratora strategii.
* `bot_core.ui.api.describe_strategy_catalog` serializuje wpisy katalogu strategii (`StrategyCatalog` + `StrategyDefinition`) do formatu wykorzystywanego w QML (nazwa, engine, licencja, klasy ryzyka, tagi i metadane).
* `bot_core.ui.api.RuntimeStateSync` realizuje lokalny polling (bez WebSocketów) i przekazuje kolejne snapshoty do zarejestrowanych słuchaczy w UI. Polling domyślnie odbywa się co 2 sekundy, można go spersonalizować przy inicjalizacji mostka.
* W trybie testowym wystarczy utworzyć `RuntimeStateSync`, dodać listener i wywołać `start()` – komponent zadba o regularne odświeżanie stanu i można go zatrzymać poprzez `stop()` (np. w `Component.onDestruction`).

## Alerty – powiadomienia toast

* Komponent `ui/qml/components/AlertToastOverlay.qml` reaguje na sygnał `AlertsModel::alertRaised` i prezentuje powiadomienia toast.
* Preferencja `alertToastsEnabled` jest zapisywana wraz z pozostałymi ustawieniami w pliku `~/.dudzian/ui_settings.json` (lub w lokalizacji wskazanej przez `BOT_CORE_UI_SETTINGS_PATH`); przełączenie jej w aplikacji natychmiast czyści kolejkę toastów.

## Personalizacja interfejsu

* Preferencje motywu (`dark`, `light`, `midnight`), układu (`classic`, `compact`, `advanced`) i powiadomień są budowane przez `Application::buildPersonalizationPayload()` i zapisywane w `~/.dudzian/ui_settings.json` (lub w ścieżce nadpisanej zmienną `BOT_CORE_UI_SETTINGS_PATH`).
* `Application::loadUiSettings()` wymaga, aby ustawienia były dostępne w aktualnym magazynie (domyślnie `~/.dudzian/ui_settings.json` lub ścieżka z `BOT_CORE_UI_SETTINGS_PATH`). Pliki `var/state/ui_settings.json` nie są już ładowane – aplikacja loguje komunikat o koniecznej migracji opisanej w [docs/migrations/2024-stage5-storage-removal.md](../migrations/2024-stage5-storage-removal.md).

## Panel Strategy Management

* Lokalizacja: `ui/qml/views/StrategyManagement.qml`, dostępny z zakładki **Strategy Workbench**.
* Sekcja **Biblioteka presetów** umożliwia podgląd różnic względem championa, symulację wyników (`previewStrategyPreset`) oraz zapis duplikatu presetów wprost z UI. Mostek offline (`OfflineRuntimeService.previewPreset`) dostarcza metryki P&L oraz status walidacji.
* Widok **Raporty i automatyzacja jakości** prezentuje listę raportów championów budowaną przez `python -m bot_core.reporting.ui_bridge overview`. Każdy wpis zawiera skrócone metryki (score, skuteczność kierunkowa, MAE, Sharpe/Sortino) oraz przycisk „Otwórz lokalizację” (wykorzystuje `ReportCenterController::openReportLocation`).
* Przyciski „Podgląd archiwizacji” i „Archiwizuj” wywołują odpowiednio `previewArchiveReports()` i `archiveReports()` (delegowane do `ui_bridge archive`). Dzięki temu operator może najpierw wykonać dry-run, a następnie zarchiwizować raporty championów bez opuszczania panelu.
* Panel monitoruje sygnały `ReportCenterController` (`reportsChanged`, `archivePreviewReady`, `archiveFinished`, `lastNotificationChanged`) i prezentuje status operacji w pasku komunikatów widoku.

## Testy

* `tests/ui/test_setup_wizard.py` – weryfikuje kroki kreatora, aktualizację list giełd/instrumentów i zapis preferencji UI.
* `tests/ui/test_portfolio_dashboard.py` – sprawdza przetwarzanie ekspozycji i historii P&L oraz renderowanie alertów.
* Uruchomienie: `QT_QPA_PLATFORM=offscreen pytest tests/ui/test_setup_wizard.py tests/ui/test_portfolio_dashboard.py`.
* Runbook testów QML z opisem artefaktów: [qml_testing.md](qml_testing.md).

## Wskazówki QA

1. Uruchom `scripts/run_local_bot.py` i zweryfikuj, że zakładki **Kreator** i **Dashboard** pojawiają się w głównym oknie.
2. Podczas testów manualnych zasymuluj alerty (np. z `bot_core/alerts/dispatcher.py`), aby sprawdzić pojawianie się toastów.
3. Po zmianie motywu w kreatorze sprawdź, czy paleta Qt przełącza się globalnie (ciemny/jasny/midnight).
