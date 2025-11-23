# Metryki Risk Journal – Prometheus/OpenMetrics

Poniższy zestaw metryk jest emitowany przez `RiskJournalMetricsExporter` z nazwami
spójnymi z modelem QML (`incompleteEntries` / `incompleteSamples`), tak aby
panel Runtime Overview oraz dashboardy Prometheus/Grafana korzystały z tych
samych kluczy i etykiet.

## Nazwy metryk
- `bot_ui_risk_journal_state` – status kompletności (0=ok, 1=warning, 2=critical).
- `bot_ui_risk_journal_incomplete_entries_total` – liczba niekompletnych wpisów
  pomijanych w agregacjach Risk Journal.
- `bot_ui_risk_journal_incomplete_samples_total` – liczba próbek
  niekompletnych wpisów raportowanych w diagnostyce/alertach.

## Etykiety
- `channel` – zawsze `risk_journal` dla łatwego filtrowania.
- `environment` – aktywny profil runtime/UI (`default`, `paper`, `prod` itd.).

## Zgodność z QML
Model `RuntimeService.riskMetrics` używa pól `incompleteEntries` oraz
`incompleteSamples`. Alerty telemetryczne (`UiTelemetryAlertSink`) publikują
payload z kluczami `incomplete_entries` **i** `incompleteEntries`, co umożliwia
spójne mapowanie w dashboardach Prometheus/Grafana oraz w panelu QML bez
transformacji nazw.

## Dashboard Grafana
Przykładową definicję panelu znajdziesz w
`docs/observability/grafana/risk_journal_health.json` – wykorzystuje powyższe
metryki i grupuje je po etykiecie `environment`.
