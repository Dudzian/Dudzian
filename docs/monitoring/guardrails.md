# Guardrails kolejki I/O

Nowa warstwa `AsyncIOGuardrails` monitoruje zdarzenia z dispatcher-a `AsyncIOTaskQueue`.
Zapewnia:

- metryki Prometheus (`exchange_io_timeout_total`, `exchange_io_rate_limit_wait_total`,
  histrogramy czasu oczekiwania i timeoutów),
- logi ostrzegawcze w `logs/guardrails/events.log`,
- opcjonalne zdarzenia UI zapisywane do pliku JSONL (gdy `metrics_ui_alerts_path` jest dostępne).

## Integracja runtime

Pipeline tworzy guardrails automatycznie, gdy w `config/runtime.yaml`
znajduje się sekcja `io_queue`. Najważniejsze parametry:

```yaml
io_queue:
  max_concurrency: 6
  burst: 16
  rate_limit_warning_seconds: 0.75
  timeout_warning_seconds: 12.0
  log_directory: logs/guardrails
```

- `rate_limit_warning_seconds` – czas oczekiwania w kolejce, po którego przekroczeniu
  generowany jest alert UI i wpis w logu,
- `timeout_warning_seconds` – minimalny czas operacji zakończonej timeoutem, by
  eskalować zdarzenie do poziomu `error`,
- `log_directory` – katalog, w którym zapisywane są zdarzenia guardrails.

Instancja guardrails otrzymuje również ścieżkę `metrics_ui_alerts_path` z kontekstu
bootstrap (jeśli została skonfigurowana), co pozwala na bezpośrednie zasilanie UI.

## Zdarzenia i format UI

Każde ostrzeżenie guardrails jest zapisywane jako linia JSONL zawierająca:

```json
{
  "timestamp": "2024-05-24T09:41:10.418292+00:00",
  "event": "io_rate_limit_wait",
  "severity": "warning",
  "source": "core.monitoring.guardrails",
  "environment": "paper",
  "payload": {
    "queue": "binance",
    "waited_seconds": 0.913512,
    "burst_limit": 8,
    "pending_after": 8,
    "streak": 3
  }
}
```

Dostępne typy zdarzeń:

- `io_rate_limit_wait` – kolejka osiągnęła limit `burst`,
- `io_timeout` – operacja wykonała się z timeoutem (z rozróżnieniem na `warning`/`error`).

## Testy regresyjne

`tests/monitoring/test_guardrails_async.py` zawiera scenariusze `pytest-asyncio`
walidujące:

1. raportowanie oczekiwania na limiter,
2. eskalację timeoutu wraz z wpisami w logach i zdarzeniami UI.

Uruchomienie:

```bash
pytest tests/monitoring/test_guardrails_async.py
```

## Dodatkowe metryki

Metryki dostępne w rejestrze:

- `exchange_io_rate_limit_wait_total` + histogram `exchange_io_rate_limit_wait_seconds`,
- `exchange_io_timeout_total` + histogram `exchange_io_timeout_duration_seconds`.

Można je eksponować w Prometheusie poprzez istniejącą usługę obserwowalności.
