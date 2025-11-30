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

## Scenariusze DR

- **Multi-region Alertmanager/Prometheus** – utrzymuj kopię reguł
  (`deploy/prometheus/rules/cloud_worker_alerts.yml`) i wartości
  `alertmanager.yml` w repozytorium DR. Raz na tydzień: (1) synchronizuj
  paczkę reguł i tajnych danych receiverów między regionami, (2) przełącz
  scrape’y `bot_cloud_health_status` oraz `bot_cloud_worker_last_error` na
  węzeł zapasowy, (3) wykonaj testowy trigger alertów
  `CloudWorkerHealthDegraded`/`CloudWorkerErrorCritical` i potwierdź, że
  webhooki HyperCare/CloudAlertService zwracają HTTP 2xx.
- **Kopie konfiguracji i rehydratacja alertów** – zestaw konfiguracyjny
  Prometheus/Alertmanagera (reguły, `alertmanager.yml`, sekrety receiverów)
  powinien być zrzucany do `reports/ci/dr_alerts/` po każdej zmianie i
  odtwarzany na węźle DR w ramach ćwiczeń. Po przełączeniu upewnij się, że
  `_lastError` z poprzedniego regionu jest widoczny w nowym (`rehydratedFromPrevious=true`
  w wynikach sond) – pozwala to HyperCare na kontynuację trwających alertów.
- **Procedura `_lastError`/`_health` CloudOrchestratora** – synthetic probes
  (patrz `scripts/dr_synthetic_probes.py` oraz `CloudOrchestrator.run_synthetic_probes`)
  wywołują `HealthService.Check` i oczekują nagłówków
  `x-bot-cloud-health`/`x-bot-cloud-last-error`. Wynik zapisuje się w
  `reports/ci/dr_probes/` i powinien wskazywać status `running` oraz brak
  `_lastError`; jeżeli orchestrator został przełączony, flaga
  `rehydratedFromPrevious=true` powinna kopiować ostatni błąd z regionu
  pierwotnego. Probes raportują opóźnienie RPC < 5 s (limit CLI
  `--latency-threshold-ms`) i flagę `healthOk=true`/`prometheusOk=true`;
  `failoverReady` pozostaje `false`, jeżeli `prometheusOk` nie jest jawnie
  `true` lub wykryto rehydratowany błąd `_lastError`.
  W razie wymuszonego skrócenia okna detekcji można użyć `--health-timeout`
  (domyślnie 5 s) do ograniczenia timeoutu RPC i szybciej wyłapywać degradację.

### Analiza snapshotów failoveru

- Cotygodniowy job `dr-weekly-failover-snapshots` (GA cron `15 4 * * 0`) uruchamia
  `scripts/dr_failover_validation.py` z prefiksem `reports/ci/dr_failover/weekly/failover_snapshot`.
  Tworzy raport porównawczy (`dr_failover_report.json`) oraz parę snapshotów
  `failover_snapshot_before.json` / `failover_snapshot_after.json` zawierających
  `_lastError`, `rulesDigest` oraz licznik aktywnych alertów. Te same pola są
  rejestrowane w raporcie w sekcji `summary.states`/`states`, dzięki czemu
  można porównać wartości bezpośrednio z artefaktu JSON.
- Różnice analizujemy przez porównanie snapshotów: brak zmian `rulesDigest`
  oznacza zgodność reguł, identyczny `_lastError` + `firingAlerts>0` potwierdza
  rehydratację alertów. Każdy dryf (`rulesDigest` różny lub `_lastError` się zmienił)
  wymaga akcji korygujących.
- Akcje korygujące:
  - **Dryf reguł** (`rulesDigest` inny, `rulesInSync=false`) → zsynchronizuj paczki
    Alertmanager/Prometheus między regionami (rsync/artefakt Helm), powtórz job.
  - **Brak rehydratacji** (`rehydrated=false` lub `firingAlerts=0` przy identycznym
    `_lastError`) → wymuś federację `federate_rules_digest`, odtwórz archiwalne alerty
    (import `silences.json`/`alerts.json`) i wykonaj ponowny failover.
  - **Regresja `_lastError`** (nowa wartość w `after`) → eskaluj do L2, zbierz logi
    z CloudOrchestratora oraz wynik health checków, przygotuj rollback DR.

### Walidacja regresji `_lastError`

- Cotygodniowy job CI `dr-failover-validation` wymusza przełączenie Alertmanagera
  i Prometheusa w region DR, wykonując dwa przebiegi sond: `before` (primary)
  oraz `after` (DR). Raport (`reports/ci/dr_failover/dr_failover_report.json`)
  musi zawierać porównanie `_lastError` oraz digestu reguł (`rulesDigest`).
- **Scenariusze oczekiwane:**
  - `rehydrated=true`, `rulesInSync=true` – DR przejmuje poprzedni `_lastError`
    i widzi identyczny zestaw reguł alertowych (brak utraty alertów).
  - `cleared=true`, `rulesInSync=true` – `_lastError` został naprawiony po
    przełączeniu; alerty wygaszone, ale reguły zgodne między regionami.
  - `unchanged=true`, `rulesInSync=true` i `firing_alerts` w obu przebiegach –
    brak zmian stanu (kontrola bazowa, brak regresji).
- **Scenariusze regresji:**
  - `regression=true` – pojawił się nowy lub zmieniony `_lastError` po
    przełączeniu; wymaga eskalacji do L2 wraz z snapshotem `after`.
  - `rulesInSync=false` – dryf konfiguracji reguł; zsynchronizuj pakiety DR i
    powtórz sondę.
  - `rehydrated=true` **i** `firing_alerts=0` po stronie DR – alerty nie zostały
    odtworzone w Alertmanagerze; uruchom ręczną replikację historycznych alertów
    lub sprawdź federację `federate_rules_digest`.
  - `beforeOk=false` / `afterOk=false` – sonda przed lub po przełączeniu
    zakończyła się błędem (`HealthService.Check`, Prometheus, Alertmanager);
    raportuje `failureReasons` odpowiednio `before_probe_failed` i
    `after_probe_failed`.
  - każda z powyższych flag jest raportowana jako `failureReasons` w
    `summary.status=fail` (`regression_last_error`/`rules_drift`/
    `alerts_missing_after_rehydration`/`before_probe_failed`/
    `after_probe_failed`).
- **Matryca akcji:**
  - `regression=true` → eskalacja L2, rollback planu przełączeniowego (job CI
    kończy się kodem ≠ 0, `failureReasons` zawiera `regression_last_error`).
  - `rulesInSync=false` → synchronizacja reguł + ponowny dry run failoveru (job
    CI kończy się kodem ≠ 0, `failureReasons` zawiera `rules_drift`).
  - `alertsMissing=true` (rehydratacja `_lastError`, ale brak firing alertów)
    → weryfikacja federacji i rehydrate backlogu alertów, dopiero potem
    `healthOk=true` (job CI kończy się kodem ≠ 0, `failureReasons` zawiera
    `alerts_missing_after_rehydration`).
  - `beforeOk=false` / `afterOk=false` → diagnoza niedostępności endpointów
    lub błędów HTTP; pipeline kończy się kodem ≠ 0 z powodu
    `before_probe_failed`/`after_probe_failed`.
  - wszystkie flagi zdrowe (`rehydrated`/`regression`/`rulesInSync` OK) →
    zatwierdź wynik w raporcie CI.
