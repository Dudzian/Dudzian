# OKX – Runbook go-live

## Runbook go-live
1. W `config/runtime.yaml` upewnij się, że `okx_desktop_paper` wskazuje środowisko `okx_paper` oraz że przypisane są tagi `[desktop, paper, okx]`.
2. Sprawdź plik `config/core.yaml` (sekcja `okx_paper`) w celu potwierdzenia, że zmapowane poświadczenia `okx_paper_trading` są dostępne w magazynie tajemnic.
3. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint okx_desktop_paper` i potwierdź, że `paper_exchange_metrics.summary.status` raportuje `"ok"` (lub `"unknown"`, jeśli metryki jeszcze nie napłynęły), `paper_exchange_metrics.okx_desktop_paper.status` pozostaje w stanie `"ok"`, sekcja `breaches` jest pusta, a `paper_exchange_metrics.summary.breached_entrypoint_names` == `[]`, `paper_exchange_metrics.summary.breached_thresholds_entrypoints` == `{}` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` == `{}`. Dodatkowo upewnij się, że `paper_exchange_metrics.summary.breach_counts_by_metric` oraz `paper_exchange_metrics.summary.threshold_breach_counts` zwracają `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}`, a `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_errors_total` oraz `.bot_exchange_health_status` zawierają `"okx_desktop_paper"`; `paper_exchange_metrics.summary.monitored_entrypoint_names` obejmuje wszystkie papierowe entrypointy, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_health_status` == `1.0`, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_errors_total` == `1.0` po ustabilizowaniu metryk, a `paper_exchange_metrics.summary.metric_coverage_score` oraz `paper_exchange_metrics.summary.threshold_coverage_score` raportują `1.0`. Zwróć uwagę, aby `paper_exchange_metrics.summary.network_error_severity_totals` raportowało zera dla `warning`, `error` i `critical`, `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.warning` oraz `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.error` obejmowały `"okx_desktop_paper"`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `1.0` dla każdej severity, a `paper_exchange_metrics.summary.missing_error_severity_counts` oraz `paper_exchange_metrics.summary.missing_error_severity_entrypoints` pozostały puste w scenariuszu referencyjnym.

4. Jeśli raport zawiera ostrzeżenia z fragmentem `network_errors_max` lub `health_min`, przerwij publikację i przeanalizuj logi adaptera.
5. Monitoruj `logs/guardrails` oraz metrykę `bot_exchange_errors_total{exchange="okx"}` w celu wykrycia potencjalnych błędów sieciowych.
6. Zweryfikuj `execution.paper_profiles.okx_paper` oraz `execution.trading_profiles.okx_desktop` – muszą wskazywać właściwy entrypoint, adapter (`okx_spot`) i listę guardrailowych tagów.

## Checklist licencyjna
- [ ] Licencja OEM obejmuje profil `desktop` oraz uprawnienie na wykorzystanie OKX (sekcja `licensing.license.allowed_profiles`).
- [ ] Fingerprint oraz lista odwołań zostały zsynchronizowane przed publikacją (`var/licenses/active/*`).
- [ ] Dokumenty zgodności (KYC/KYB) dla OKX są zarchiwizowane w repozytorium audytowym.

## Checklist specyficzna dla OKX
- [ ] Limit kolejki I/O (`io_queue.exchanges.okx_spot`) posiada `max_concurrency=3` i `burst=6`.
- [ ] `execution.paper_profiles.okx_paper.metrics.thresholds` mają ustawione `rate_limit_max=0`, `network_errors_max=1` oraz `health_min=1.0`.
- [ ] Raport z `run_local_bot` potwierdza `paper_exchange_metrics.summary.status` = `"ok"` (ew. `"unknown"` przy braku danych) oraz `paper_exchange_metrics.okx_desktop_paper.status` = `"ok"`; sekcja `breaches` pozostaje pusta, a `paper_exchange_metrics.summary.breached_entrypoint_names` == `[]`, `paper_exchange_metrics.summary.breached_thresholds_entrypoints` == `{}` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` == `{}`; ponadto `paper_exchange_metrics.summary.breach_counts_by_metric` == `{}`, `paper_exchange_metrics.summary.threshold_breach_counts` == `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}`, a `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status` zawiera `"okx_desktop_paper"`; `paper_exchange_metrics.summary.monitored_entrypoint_names` obejmuje komplet papierowych entrypointów, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_health_status` == `1.0`, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_errors_total` == `1.0`, a `paper_exchange_metrics.summary.metric_coverage_score` oraz `paper_exchange_metrics.summary.threshold_coverage_score` raportują `1.0`. Dodatkowo `paper_exchange_metrics.summary.network_error_severity_totals` == `{warning: 0, error: 0, critical: 0}`, `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.warning` oraz `.error` obejmują `"okx_desktop_paper"`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `{warning: 1.0, error: 1.0, critical: 1.0}`, a `paper_exchange_metrics.summary.missing_error_severity_counts` oraz `paper_exchange_metrics.summary.missing_error_severity_entrypoints` są puste.

- [ ] Test `tests/integration/exchanges/test_okx.py::test_okx_spot_rate_limit` przechodzi na środowisku QA, potwierdzając działanie retry.
- [ ] Test `tests/exchanges/test_okx_signing.py::test_okx_adapter_populates_credentials_and_signs_request` potwierdza, że podpisy CCXT wykorzystują przekazane klucze i hasło.
- [ ] Podpisy CCXT są aktualne – `sandbox_mode` jest automatycznie ustawiany dla środowiska paper/testnet w adapterze.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_okx.py
```

Spodziewany wynik: brak błędów oraz pozytywny scenariusz rate-limit retry.
