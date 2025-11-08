# Kraken – Runbook go-live

## Runbook go-live
1. Zweryfikuj konfigurację `config/runtime.yaml`, upewniając się, że punkt wejścia `kraken_desktop_paper` wskazuje na środowisko `kraken_paper` oraz że profil ryzyka odpowiada wymaganiom klienta.
2. W `config/core.yaml` sprawdź mapowanie `kraken_paper` → `kraken_spot` oraz klucz `kraken_paper_trading`; potwierdź, że poświadczenia są dostępne w menedżerze tajemnic.
3. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint kraken_desktop_paper`, obserwując sekcję `paper_exchange_metrics` w raporcie JSON – `paper_exchange_metrics.summary.status` powinien mieć wartość `"ok"` (dopuszczalne `"unknown"` podczas inicjalizacji), `paper_exchange_metrics.kraken_desktop_paper.status` powinien raportować `"ok"`, a lista `breaches` pozostać pusta; dodatkowo `paper_exchange_metrics.summary.breached_entrypoint_names` powinno być puste, a obiekty `paper_exchange_metrics.summary.breached_thresholds_entrypoints` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` powinny być puste (`{}`). Zweryfikuj również, że `paper_exchange_metrics.summary.breach_counts_by_metric` i `paper_exchange_metrics.summary.threshold_breach_counts` zwracają `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}`, natomiast `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status` zawiera `"kraken_desktop_paper"`, a `paper_exchange_metrics.summary.monitored_metric_names` obejmuje wszystkie metryki limitów i zdrowia; `paper_exchange_metrics.summary.monitored_entrypoint_names` obejmuje wszystkie papierowe entrypointy, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_health_status` == `1.0`, `paper_exchange_metrics.summary.metric_coverage_score` == `1.0`, a `paper_exchange_metrics.summary.threshold_coverage_score` == `1.0` (po ustabilizowaniu metryk). Dodatkowo upewnij się, że `paper_exchange_metrics.summary.network_error_severity_totals` raportuje wartości `0` dla `warning`, `error` i `critical`, `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.warning` oraz `paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints.error` obejmują `"kraken_desktop_paper"`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `1.0` dla każdej severity, a `paper_exchange_metrics.summary.missing_error_severity_counts` oraz `paper_exchange_metrics.summary.missing_error_severity_entrypoints` pozostają puste w stabilnym scenariuszu.

4. Jeżeli w raporcie pojawi się wpis w `warnings` zawierający `rate_limit_max` lub `network_errors_max`, zatrzymaj wdrożenie i powtórz testy po analizie przyczyny.
5. Weryfikuj logi `logs/guardrails` pod kątem błędów sieciowych lub ostrzeżeń związanych z limitem zapytań.
6. Zweryfikuj sekcję `execution.paper_profiles.kraken_paper` i `execution.trading_profiles.kraken_desktop` w `config/runtime.yaml`, aby upewnić się, że wskazują właściwy entrypoint i limity kolejek I/O.

## Checklist licencyjna
- [ ] Licencja OEM (`var/licenses/active/license.json`) zawiera wpis na listę dozwolonych profili obejmujący `desktop` oraz `kraken`.
- [ ] Plik `config/runtime.yaml` ma ustawione `licensing.enforcement: true`, a fingerprint jest zaktualizowany (`var/licenses/active/fingerprint.json`).
- [ ] Revocation list (`var/licenses/active/revocations.json`) została zsynchronizowana z centralnym repozytorium.

## Checklist specyficzna dla Kraken
- [ ] Limity I/O z sekcji `io_queue.exchanges.kraken_spot` utrzymują `max_concurrency=3` oraz `burst=6`.
- [ ] W `execution.paper_profiles.kraken_paper.metrics.thresholds` wartości `rate_limit_max` oraz `network_errors_max` wynoszą `0`, a `health_min` równe `1.0`.
- [ ] Raport z `scripts/run_local_bot.py` potwierdza `paper_exchange_metrics.summary.status` = `"ok"` (ew. `"unknown"` przy braku metryk) oraz `paper_exchange_metrics.kraken_desktop_paper.status` = `"ok"`; listy `breaches` pozostają puste, a `paper_exchange_metrics.summary.breached_entrypoint_names` == `[]`, `paper_exchange_metrics.summary.breached_thresholds_entrypoints` == `{}` oraz `paper_exchange_metrics.summary.missing_metrics_entrypoints` == `{}`; dodatkowo `paper_exchange_metrics.summary.breach_counts_by_metric` == `{}`, `paper_exchange_metrics.summary.threshold_breach_counts` == `{}`, `paper_exchange_metrics.summary.missing_metric_counts` == `{}` oraz `paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status` zawiera `"kraken_desktop_paper"`; `paper_exchange_metrics.summary.monitored_entrypoint_names` pokrywa wszystkie papierowe entrypointy, `paper_exchange_metrics.summary.metric_coverage_ratio.bot_exchange_health_status` == `1.0`, a `paper_exchange_metrics.summary.metric_coverage_score` i `paper_exchange_metrics.summary.threshold_coverage_score` raportują `1.0`. Dodatkowo `paper_exchange_metrics.summary.network_error_severity_totals` == `{warning: 0, error: 0, critical: 0}`, `paper_exchange_metrics.summary.network_error_severity_coverage_ratio` == `{warning: 1.0, error: 1.0, critical: 1.0}`, a `paper_exchange_metrics.summary.missing_error_severity_counts` oraz `paper_exchange_metrics.summary.missing_error_severity_entrypoints` pozostają puste.

- [ ] Poświadczenia mają uprawnienia `trade` i `read`; test `tests/exchanges/test_kraken_signing.py::test_spot_private_request_signature` potwierdza poprawne podpisywanie żądań.
- [ ] Health-checki (`bot_exchange_health_status`) raportują wartość 1 dla `kraken` po zakończeniu smoke testów.
- [ ] `execution.trading_profiles.kraken_desktop` zawiera tagi `[desktop, paper, kraken]`, co pozwala raportom Guardrail poprawnie filtrować scenariusze papierowe.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_kraken.py
```

Wynik powinien potwierdzić przejście scenariuszy retry/rate-limit oraz walidację podpisów.
