# Bybit – Runbook go-live

## Runbook go-live
1. Skontroluj `config/runtime.yaml`, aby `bybit_desktop_paper` wskazywał na środowisko `bybit_paper` z aktywnym profilem `multi_strategy_default` oraz tagami `[desktop, paper, bybit]`.
2. Zweryfikuj sekcje `execution.paper_profiles.bybit_paper` oraz `execution.trading_profiles.bybit_desktop` – muszą wskazywać adapter `bybit_spot`, poprawne limity guardrailowe i powiązany entrypoint.
3. Upewnij się, że `config/core.yaml` zawiera poprawne mapowanie `bybit_paper` oraz że klucz `bybit_paper_trading` jest aktywny w magazynie tajemnic.
4. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint bybit_desktop_paper` i sprawdź, że `paper_exchange_metrics.summary.status` wskazuje `"ok"` (lub `"unknown"` dla świeżych sesji), `paper_exchange_metrics.bybit_desktop_paper.status` pozostaje `"ok"`, `breaches` są puste, `warnings` nie zawiera wpisów z `rate_limit_max`/`network_errors_max`, a `paper_exchange_metrics.summary.breached_entrypoint_names` == `[]`, `paper_exchange_metrics.summary.breached_thresholds_entrypoints` == `{}` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` == `{}`. Dodatkowo upewnij się, że `paper_exchange_metrics.summary.breach_counts_by_metric` i `paper_exchange_metrics.summary.threshold_breach_counts` zwracają `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}`, a `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_rate_limited_total` obejmuje `"bybit_desktop_paper"` po zakończeniu smoke testu; `paper_exchange_metrics.summary.monitored_entrypoint_names` pokrywa wszystkie papierowe entrypointy, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_rate_limited_total` == `1.0`, `paper_exchange_metrics.summary.metric_coverage_score` == `1.0`, a `paper_exchange_metrics.summary.threshold_coverage_score` == `1.0`. Sprawdź również, że `paper_exchange_metrics.summary.network_error_severity_totals` raportuje zera dla `warning`, `error` i `critical`, `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.warning` oraz `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.error` obejmują `"bybit_desktop_paper"`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `1.0` dla każdej severity, a `paper_exchange_metrics.summary.missing_error_severity_counts` oraz `paper_exchange_metrics.summary.missing_error_severity_entrypoints` wskazują brak braków próbkowania.

5. Po smoke teście wykonaj inspekcję `logs/guardrails` i eksportu raportu Markdown, upewniając się, że nie odnotowano throttlingu.

## Checklist licencyjna
- [ ] Licencja OEM pozwala na handel na Bybit (profil `desktop`, wystawca `dudzian`).
- [ ] Fingerprint (`var/licenses/active/fingerprint.json`) został odświeżony po aktualizacji sprzętu.
- [ ] Klucze API są zarejestrowane i przypisane do konta produkcyjnego, a dostęp offline jest udokumentowany.

## Checklist specyficzna dla Bybit
- [ ] `io_queue.exchanges.bybit_spot` ma ustawione `max_concurrency=3` oraz `burst=6`.
- [ ] W `execution.paper_profiles.bybit_paper.metrics.thresholds` limity `rate_limit_max` i `network_errors_max` pozostają równe `0`, a `health_min` wynosi `1.0`.
- [ ] Raport papierowy wskazuje `paper_exchange_metrics.summary.status` = `"ok"` (ew. `"unknown"` przy braku prób) oraz `paper_exchange_metrics.bybit_desktop_paper.status` = `"ok"`; lista `breaches` pozostaje pusta, a `paper_exchange_metrics.summary.breached_entrypoint_names` == `[]`, `paper_exchange_metrics.summary.breached_thresholds_entrypoints` == `{}` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` == `{}`; uzupełniająco `paper_exchange_metrics.summary.breach_counts_by_metric` == `{}`, `paper_exchange_metrics.summary.threshold_breach_counts` == `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}`, a `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status` zawiera `"bybit_desktop_paper"` po wypełnieniu metryk; `paper_exchange_metrics.summary.monitored_entrypoint_names` obejmuje wszystkie papierowe entrypointy, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_rate_limited_total` == `1.0`, a `paper_exchange_metrics.summary.metric_coverage_score` oraz `paper_exchange_metrics.summary.threshold_coverage_score` raportują `1.0`. Ponadto `paper_exchange_metrics.summary.network_error_severity_totals` == `{warning: 0, error: 0, critical: 0}`, `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.warning` oraz `.error` obejmują `"bybit_desktop_paper"`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `{warning: 1.0, error: 1.0, critical: 1.0}`, a `paper_exchange_metrics.summary.missing_error_severity_counts` i `paper_exchange_metrics.summary.missing_error_severity_entrypoints` pozostają puste.

- [ ] Test `tests/integration/exchanges/test_bybit.py::test_bybit_spot_rate_limit` przechodzi lokalnie i w CI, potwierdzając poprawne retry.
- [ ] Test `tests/exchanges/test_bybit_signing.py::test_bybit_adapter_populates_credentials_and_signs_request` potwierdza poprawność podpisów REST.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_bybit.py
```

Test powinien zakończyć się sukcesem, a metryki `bot_exchange_rate_limited_total{exchange="bybit"}` pozostać na 0.
