# Dashboard runtime

Nowy panel `RuntimeOverview` prezentuje w jednym miejscu kluczowe metryki działania bota. Widok składa się z czterech kart:

- **Kolejki I/O** – lista środowisk i kolejek wraz z liczbą timeoutów, średnim czasem oczekiwania oraz poziomem guardrail (normal/info/warning/error). Dane pochodzą bezpośrednio z `core.monitoring.metrics_api.load_runtime_snapshot` i są odświeżane cyklicznie lub manualnie przyciskiem *Odśwież*.
- **Guardrail'e** – podsumowanie ilościowe (liczba kolejek w poszczególnych stanach, suma timeoutów oraz oczekiwań) dla szybkiej oceny kondycji systemu.
- **Retraining** – agregaty cykli retrainingu (liczba uruchomień, średni czas i średni dryf) rozbite według statusów publikowanych przez scheduler.
- **Zgodność (KYC/AML)** – wynik ostatniego audytu zgodności wraz ze statusem reguł KYC, AML i limitów transakcji, listą naruszeń oraz rekomendacjami naprawczymi.

Komponent QML korzysta z klasy `ui.backend.TelemetryProvider`, która serializuje metryki do struktur przyjaznych QML i udostępnia właściwości `ioQueues`, `guardrailSummary`, `retraining`, `lastUpdated` oraz `errorMessage`. Provider potrafi obsłużyć błędy komunikacji – komunikat zostaje wyświetlony w czerwonym banerze.

## Integracja

1. Wstrzyknij instancję `TelemetryProvider` do kontekstu QML pod nazwą `telemetryProvider` (analogicznie do `LicensingController`).
2. Opcjonalnie udostępnij `ComplianceController` pod nazwą `complianceController`, aby karta zgodności mogła automatycznie uruchomić audyt i zaktualizować telemetrię.
3. Osadź `dashboard/RuntimeOverview.qml` w dowolnym widoku (np. w zakładce monitoringu) i ustaw parametry `refreshIntervalMs` według potrzeb.
4. W przypadku potrzeby testów jednostkowych możesz przekazać własny `snapshot_loader` do `TelemetryProvider`, co umożliwia podmianę danych na deterministyczne.

## Panel zgodności (CompliancePanel)

Komponent `dashboard/CompliancePanel.qml` może być również użyty samodzielnie. Prezentuje wynik ostatniego audytu oraz statystyki naruszeń na podstawie metryk `TelemetryProvider`.

1. Osadź komponent w widoku i przekaż właściwości `telemetryProvider` oraz `complianceController`.
2. `ComplianceController` powinien dostarczać dane strategii, źródeł danych, transakcji i profilu KYC – domyślne implementacje zwracają puste struktury, dlatego w aplikacji należy wstrzyknąć odpowiednie providery.
3. Po ukończeniu audytu kontroler wywoła metodę `TelemetryProvider.updateComplianceSummary`, co pozwala na zsynchronizowanie panelu z metrykami guardrail.
4. Testy referencyjne znajdują się w `tests/ui/test_compliance_panel.py`.

## Pliki

- `ui/qml/dashboard/RuntimeOverview.qml` – definicja layoutu panelu.
- `ui/backend/telemetry_provider.py` – warstwa logiki i mapowanie metryk.
- `tests/ui/test_runtime_overview.py` – scenariusze weryfikujące integrację z QML i obsługę błędów.
- `tests/ui/test_compliance_panel.py` – testy walidujące przepływ audytu zgodności w UI.
- `core/monitoring/metrics_api.py` – źródło zagregowanych metryk wykorzystywanych przez UI.
