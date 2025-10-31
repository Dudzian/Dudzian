# Dashboard runtime

Nowy panel `RuntimeOverview` prezentuje w jednym miejscu kluczowe metryki działania bota. Widok składa się z trzech kart:

- **Kolejki I/O** – lista środowisk i kolejek wraz z liczbą timeoutów, średnim czasem oczekiwania oraz poziomem guardrail (normal/info/warning/error). Dane pochodzą bezpośrednio z `core.monitoring.metrics_api.load_runtime_snapshot` i są odświeżane cyklicznie lub manualnie przyciskiem *Odśwież*.
- **Guardrail'e** – podsumowanie ilościowe (liczba kolejek w poszczególnych stanach, suma timeoutów oraz oczekiwań) dla szybkiej oceny kondycji systemu.
- **Retraining** – agregaty cykli retrainingu (liczba uruchomień, średni czas i średni dryf) rozbite według statusów publikowanych przez scheduler.

Komponent QML korzysta z klasy `ui.backend.TelemetryProvider`, która serializuje metryki do struktur przyjaznych QML i udostępnia właściwości `ioQueues`, `guardrailSummary`, `retraining`, `lastUpdated` oraz `errorMessage`. Provider potrafi obsłużyć błędy komunikacji – komunikat zostaje wyświetlony w czerwonym banerze.

## Integracja

1. Wstrzyknij instancję `TelemetryProvider` do kontekstu QML pod nazwą `telemetryProvider` (analogicznie do `LicensingController`).
2. Osadź `dashboard/RuntimeOverview.qml` w dowolnym widoku (np. w zakładce monitoringu) i ustaw parametry `refreshIntervalMs` według potrzeb.
3. W przypadku potrzeby testów jednostkowych możesz przekazać własny `snapshot_loader` do `TelemetryProvider`, co umożliwia podmianę danych na deterministyczne.

## Pliki

- `ui/qml/dashboard/RuntimeOverview.qml` – definicja layoutu panelu.
- `ui/backend/telemetry_provider.py` – warstwa logiki i mapowanie metryk.
- `tests/ui/test_runtime_overview.py` – scenariusze weryfikujące integrację z QML i obsługę błędów.
- `core/monitoring/metrics_api.py` – źródło zagregowanych metryk wykorzystywanych przez UI.
