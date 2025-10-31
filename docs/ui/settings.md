# Ustawienia interfejsu i dashboardu

Plik `core/config/ui_settings.py` wprowadza zunifikowany model ustawień interfejsu użytkownika, który jest przechowywany lokalnie w katalogu `~/.kryptolowca/ui_settings.json`. W pliku przechowywane są m.in. kolejność kart telemetry, ukryte widoki, interwał odświeżania danych oraz preferowany motyw kolorystyczny.

## Najważniejsze elementy

- **DashboardSettingsController** (`ui/backend/dashboard_settings.py`) – warstwa pośrednia udostępniana QML, odpowiedzialna za walidację, normalizację i trwały zapis ustawień. Umożliwia zmianę kolejności kart, ukrywanie widoków, konfigurację interwału odświeżania oraz wybór motywu UI.
- **PrivacySettingsController** (`ui/backend/privacy_settings.py`) – zarządza zgodą na telemetrię anonimową, udostępnia liczbę zdarzeń w kolejce oraz funkcje eksportu/wyczyszczenia danych w widoku `PrivacySettings.qml`.
- **RuntimeOverview.qml** – panel telemetry odczytuje preferencje z kontrolera i w czasie rzeczywistym dostosowuje kolejność kart, a także interwał auto-odświeżania.
- **DashboardSettings.qml** – nowy widok ustawień, w którym użytkownik może zarządzać kartami, motywem i parametrami odświeżania. Formularz korzysta z `DashboardSettingsController` i automatycznie zapisuje zmiany.

## Ścieżka pliku ustawień

Domyślnie ustawienia są zapisywane w `~/.kryptolowca/ui_settings.json`. Ścieżka może być nadpisana przy tworzeniu `UISettingsStore`, co jest wykorzystywane w testach UI.

Plik JSON ma następującą strukturę:

```json
{
  "version": 1,
  "dashboard": {
    "card_order": ["io_queue", "guardrails", "retraining"],
    "hidden_cards": [],
    "refresh_interval_ms": 4000,
    "theme": "system"
  }
}
```

## Integracja z UI

1. `DashboardSettingsController` ładuje ustawienia podczas startu aplikacji oraz emituje sygnały po każdej zmianie.
2. `RuntimeOverview.qml` i `DashboardSettings.qml` otrzymują kontroler poprzez właściwość kontekstową i reagują na sygnały (`cardOrderChanged`, `visibleCardOrderChanged`, `refreshIntervalMsChanged`, `themeChanged`).
3. Ustawienia są zapisywane natychmiastowo, a błędy zapisu są bezpiecznie ignorowane (z zachowaniem bieżących wartości w pamięci).

## Testy

- `tests/ui/test_dashboard_settings.py` zawiera testy kontrolera oraz sprawdza, czy komponent QML ładuje się poprawnie. Testy są automatycznie pomijane, gdy środowisko nie udostępnia bibliotek Qt (analogicznie do pozostałych testów UI).
- `tests/ui/test_privacy_settings.py` weryfikuje obsługę zgody na telemetrię i poprawne ładowanie widoku ustawień prywatności.

## Dalsze kroki

- Można rozszerzyć model o dodatkowe preferencje (np. filtr alertów, układ siatki) poprzez rozbudowę `DashboardSettings` i aktualizację kontrolera.
- Włączenie synchronizacji ustawień pomiędzy stacjami roboczymi wymagałoby dodania dodatkowej warstwy (np. eksport/import lub integracja z licencją OEM).
