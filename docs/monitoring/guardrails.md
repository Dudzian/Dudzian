# Guardrails kolejki I/O i retrainingu

Warstwa `AsyncIOGuardrails` monitoruje zarówno zdarzenia z dispatcher-a
`AsyncIOTaskQueue`, sygnały retrainingu publikowane przez
`RetrainingScheduler`, jak i – poprzez kontroler kreatora – kontekst
onboardingu. Zapewnia:

- metryki Prometheus (`exchange_io_timeout_total`, `exchange_io_rate_limit_wait_total`,
  histogramy czasu oczekiwania i timeoutów),
- metryki retrainingu (`retraining_duration_seconds`, `retraining_drift_score`),
- metryki onboardingowe (`onboarding_duration_seconds`),
- logi ostrzegawcze w `logs/guardrails/events.log` oraz
  `logs/guardrails/retraining/events.log`,
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
- `io_timeout` – operacja wykonała się z timeoutem (z rozróżnieniem na `warning`/`error`),
- `retraining_cycle_completed` – raport z zakończenia cyklu retrainingu (zawiera
  czas trwania, opcjonalny dryf oraz metadane np. wstrzyknięte opóźnienia),
- `retraining_drift_detected` – wykryty dryf danych w fazie przygotowawczej,
- `retraining_missing_data` – brakujące porcje danych blokujące retraining,
- `retraining_delay_injected` – celowe opóźnienie startu retrainingu,
- `onboarding_completed` / `onboarding_failed` – wynik kreatora onboardingowego
  wraz z czasem jego trwania i identyfikatorem licencji.

Alerty retrainingu zapisywane są dodatkowo w katalogu
`logs/guardrails/retraining/` i mogą korzystać z dedykowanych progów
konfigurowanych przez argumenty:

- `retraining_duration_warning_threshold` – czas trwania cyklu, po którym
  eskalowany jest poziom `warning`,
- `drift_warning_threshold` – globalny próg dryfu (opcjonalny); przy jego braku
  wykorzystywany jest próg przekazany w zdarzeniu.

## Testy regresyjne

`tests/monitoring/test_guardrails_async.py` zawiera scenariusze `pytest-asyncio`
walidujące:

1. raportowanie oczekiwania na limiter,
2. eskalację timeoutu wraz z wpisami w logach i zdarzeniami UI.

`tests/monitoring/test_retraining_guardrails.py` uzupełnia powyższe o testy
obsługi zdarzeń retrainingu: zapisu metryk dryfu, emisji alertów UI oraz logów.

Uruchomienie:

```bash
pytest tests/monitoring/test_guardrails_async.py
pytest tests/monitoring/test_retraining_guardrails.py
```

## Dodatkowe metryki

Metryki dostępne w rejestrze:

- `exchange_io_rate_limit_wait_total` + histogram `exchange_io_rate_limit_wait_seconds`,
- `exchange_io_timeout_total` + histogram `exchange_io_timeout_duration_seconds`,
- `retraining_duration_seconds` – histogram czasu trwania cyklu retrainingu,
- `retraining_drift_score` – histogram wartości dryfu danych,
- `onboarding_duration_seconds` – histogram czasu przejścia kreatora
  onboardingowego.

## Logowanie onboardingowe

Onboarding desktopowy posiada dedykowany logger (`ui/backend/logging.py`), który
tworzy katalog `logs/ui/onboarding/` oraz plik `onboarding.log`. Każde
zakończenie kreatora (sukces lub błąd) zapisuje ustrukturyzowaną linię logu,
emitując równocześnie zdarzenia `OnboardingCompleted` / `OnboardingFailed`
widoczne dla guardrail’i i UI. Dzięki temu panel runbooków może prezentować
ostatnie problemy z aktywacją licencji bez konieczności analizy logów systemu.

Można je eksponować w Prometheusie poprzez istniejącą usługę obserwowalności.
