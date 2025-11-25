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
- `bot_ui_risk_journal_risk_flag_entries_total` – liczba wpisów zawierających
  dany znacznik ryzyka (etykieta `riskFlag`).

## Etykiety
- `channel` – zawsze `risk_journal` dla łatwego filtrowania (dodawane automatycznie).
- `environment` – aktywny profil runtime/UI (`default`, `paper`, `prod` itd.).
- `riskFlag` – nazwa flagi ryzyka, automatycznie zerowana, gdy zniknie z
  bieżącego raportu diagnostycznego (zapobiega utrzymywaniu się starych
  etykiet w metrykach).

## Zgodność z QML
Model `RuntimeService.riskMetrics` używa pól `incompleteEntries`,
`incompleteSamples` oraz histogramów `riskFlagCounts`. Alerty telemetryczne
(`UiTelemetryAlertSink`) publikują payload z kanonicznymi kluczami camelCase
oraz, dla zgodności wstecznej, ze słowami rozdzielonymi podkreśleniem. Panel
QML korzysta z normalizacji kluczy, więc obsługuje oba warianty bez
dodatkowych mapowań w kodzie backendu. Pole `incompleteSamples` może być
listą próbek lub samą liczbą – eksporter Prometheusa/OpenMetrics konwertuje ją
na liczność (`*_incomplete_samples_total`). Jeżeli diagnostyka przekaże
`incompleteSamplesCount` (`incomplete_samples_count`), eksporter użyje
podanego wolumenu nawet wtedy, gdy lista próbek jest przycięta do kilku
pozycji. Histogram `riskFlagCounts` jest eksportowany jako metryka
`bot_ui_risk_journal_risk_flag_entries_total` z etykietą `riskFlag`
identyczną jak klucz w modelu QML.

### Tabela zgodności kluczy
| Klucz QML               | Nazwa metryki Prometheus                               | Etykiety                         |
|-------------------------|--------------------------------------------------------|----------------------------------|
| `incompleteEntries`     | `bot_ui_risk_journal_incomplete_entries_total`         | `channel`, `environment`         |
| `incompleteSamples`     | `bot_ui_risk_journal_incomplete_samples_total`         | `channel`, `environment`         |
| `riskFlagCounts`        | `bot_ui_risk_journal_risk_flag_entries_total`          | `channel`, `environment`, `riskFlag` |

### Przykładowe zapytanie PromQL
Średnia liczba niekompletnych wpisów oraz wolumen wpisów z flagą ryzyka
`stress_override` w danym środowisku:

```
avg_over_time(bot_ui_risk_journal_incomplete_entries_total{environment="prod"}[5m])
  or
sum by(environment) (
  bot_ui_risk_journal_risk_flag_entries_total{
    environment="prod",
    riskFlag="stress_override"
  }
)
```

## Dashboard Grafana
Przykładową definicję panelu znajdziesz w
`docs/observability/grafana/risk_journal_health.json` – wykorzystuje powyższe
metryki i grupuje je po etykiecie `environment`.
