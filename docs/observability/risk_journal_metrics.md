# Metryki Risk Journal – Prometheus/OpenMetrics

Poniższy zestaw metryk jest emitowany przez `RiskJournalMetricsExporter` z nazwami
spójnymi z modelem QML (`incompleteEntries` / `incompleteSamples`), tak aby
panel Runtime Overview oraz dashboardy Prometheus/Grafana korzystały z tych
samych kluczy i etykiet. Eksporter normalizuje etykiety do zestawu bazowego
`channel=risk_journal` oraz dodatkowych pól przekazanych z `RuntimeService`,
np. `environment`.

## Nazwy metryk
- `bot_ui_risk_journal_state` – status kompletności (0=ok, 1=warning, 2=critical).
- `bot_ui_risk_journal_incomplete_entries_total` – liczba niekompletnych wpisów
  pomijanych w agregacjach Risk Journal.
- `bot_ui_risk_journal_incomplete_samples_total` – liczba próbek
  niekompletnych wpisów raportowanych w diagnostyce/alertach.

## Etykiety
- `channel` – zawsze `risk_journal` dla łatwego filtrowania (dodawane automatycznie).
- `environment` – aktywny profil runtime/UI (`default`, `paper`, `prod` itd.).

## Zgodność z QML
Model `RuntimeService.riskMetrics` używa pól `incompleteEntries` oraz
`incompleteSamples`. Alerty telemetryczne (`UiTelemetryAlertSink`) publikują
payload z kanonicznymi kluczami camelCase oraz, dla zgodności wstecznej,
ze słowami rozdzielonymi podkreśleniem. Panel QML korzysta z
normalizacji kluczy, więc obsługuje oba warianty bez dodatkowych
mapowań w kodzie backendu. Pole `incompleteSamples` może być listą
próbek lub samą liczbą – eksporter Prometheusa/OpenMetrics konwertuje ją
na liczność (`*_incomplete_samples_total`). Jeżeli diagnostyka przekaże
`incompleteSamplesCount` (`incomplete_samples_count`), eksporter użyje
podanego wolumenu nawet wtedy, gdy lista próbek jest przycięta do kilku
pozycji.

## Dashboard Grafana
Przykładową definicję panelu znajdziesz w
`docs/observability/grafana/risk_journal_health.json` – wykorzystuje powyższe
metryki i grupuje je po etykiecie `environment`.
