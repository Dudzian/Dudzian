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

## Offline Runtime API dla UI

* Backend udostępnia funkcję `bot_core.ui.api.build_runtime_snapshot`, która zbiera migawkę portfela, statusy giełd, katalog strategii oraz najnowsze wpisy explainability/alertów. Snapshot jest zgodny z wymaganiami panelu dashboardu i konfiguratora strategii.
* `bot_core.ui.api.describe_strategy_catalog` serializuje wpisy katalogu strategii (`StrategyCatalog` + `StrategyDefinition`) do formatu wykorzystywanego w QML (nazwa, engine, licencja, klasy ryzyka, tagi i metadane).
* `bot_core.ui.api.RuntimeStateSync` realizuje lokalny polling (bez WebSocketów) i przekazuje kolejne snapshoty do zarejestrowanych słuchaczy w UI. Polling domyślnie odbywa się co 2 sekundy, można go spersonalizować przy inicjalizacji mostka.
* W trybie testowym wystarczy utworzyć `RuntimeStateSync`, dodać listener i wywołać `start()` – komponent zadba o regularne odświeżanie stanu i można go zatrzymać poprzez `stop()` (np. w `Component.onDestruction`).

## Alerty – powiadomienia toast

* Komponent `ui/qml/components/AlertToastOverlay.qml` reaguje na sygnał `AlertsModel::alertRaised` i prezentuje powiadomienia toast.
* Preferencja `alertToastsEnabled` jest zapisywana w `config/ui_prefs.json`; przełączenie jej w aplikacji natychmiast czyści kolejkę toastów.

## Personalizacja interfejsu

* Preferencje motywu (`dark`, `light`, `midnight`), układu (`classic`, `compact`, `advanced`) i powiadomień są budowane przez `Application::buildPersonalizationPayload()` i przechowywane w `config/ui_prefs.json`.
* Migracja ze starej ścieżki `var/state/ui_settings.json` wykonywana jest w `Application::loadUiSettings()` – testy `tests/ui/test_setup_wizard.py` pokrywają scenariusz zapisu/odczytu podstawowych ustawień.

## Testy

* `tests/ui/test_setup_wizard.py` – weryfikuje kroki kreatora, aktualizację list giełd/instrumentów i zapis preferencji UI.
* `tests/ui/test_portfolio_dashboard.py` – sprawdza przetwarzanie ekspozycji i historii P&L oraz renderowanie alertów.
* Uruchomienie: `QT_QPA_PLATFORM=offscreen pytest tests/ui/test_setup_wizard.py tests/ui/test_portfolio_dashboard.py`.

## Wskazówki QA

1. Uruchom `scripts/run_local_bot.py` i zweryfikuj, że zakładki **Kreator** i **Dashboard** pojawiają się w głównym oknie.
2. Podczas testów manualnych zasymuluj alerty (np. z `bot_core/alerts/dispatcher.py`), aby sprawdzić pojawianie się toastów.
3. Po zmianie motywu w kreatorze sprawdź, czy paleta Qt przełącza się globalnie (ciemny/jasny/midnight).
