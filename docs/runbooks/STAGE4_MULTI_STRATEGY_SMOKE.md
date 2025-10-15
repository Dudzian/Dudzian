# Runbook: Stage4 – smoke test wielostrate-giczny

## Cel
Zapewnienie cyklicznej weryfikacji strategii mean reversion, volatility targeting oraz cross-exchange
arbitrage wraz z harmonogramem multi-strategy i raportami Paper Labs. Runbook opisuje pełny
proces end-to-end – od ręcznego uruchomienia workflow GitHub Actions po archiwizację artefaktów
w `var/audit/acceptance` oraz aktualizację decision logu.

## Procedura
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Zweryfikuj aktualność konfiguracji `config/core.yaml` (`python scripts/validate_config.py --profile demo`) | Inżynier Runtime | Raport walidacji, hash SHA-384 konfiguracji | Raport PASS, hash dopisany do decision logu Stage4 |
| 2. Uruchom workflow **Stage4 multi-strategy smoke** (`deploy/ci/github_actions_stage4_multi_strategy.yml`) – ręcznie lub poczekaj na harmonogram tygodniowy | Operator CI | Log wykonania GitHub Actions | Job zakończony sukcesem, brak ostrzeżeń krytycznych |
| 3. Zweryfikuj w logu kroki `Run targeted pytest suite with coverage` oraz `Enforce coverage thresholds` – brak czerwonych ostrzeżeń, raport `coverage.xml` w artefakcie | Operator CI | `coverage.xml`, log kroku | Minimalne pokrycie 85% globalnie oraz 87% dla `bot_core.strategies` spełnione |
| 4. Sprawdź wynik kroku `Validate Prometheus alert rules` – wszystkie reguły Stage4 muszą przejść walidację bez ostrzeżeń | Operator CI | log kroku, `deploy/prometheus/rules/multi_strategy_rules.yml` | Log zawiera komunikaty „Walidacja reguł zakończona sukcesem” |
| 5. Po zakończeniu workflow pobierz artefakt `stage4-multi-strategy-<run>` | Operator CI | `smoke_report.json`, `risk_simulation/*`, `metadata.json`, `load_test.json` | Artefakty zdeponowane lokalnie w katalogu tymczasowym |
| 6. Zweryfikuj paczkę `stage4_strategies/strategy_bundle/stage4-strategies-<version>.zip` i dopasowaną parę `manifest.json`/`.sig`; potwierdź, że wersja w `$STAGE4_STRATEGY_BUNDLE_VERSION` pokrywa się z logiem workflow | Operator OEM | `stage4-strategies-<version>.zip`, `stage4-strategies-<version>.manifest.{json,sig}` | Manifest i sygnatura podpisane HMAC, wyniki `sha256` zgodne z raportem |
| 7. Uruchom `python scripts/export_observability_bundle.py --version <run_id> --output-dir var/audit/acceptance/<TS>/stage4_smoke --signing-key secrets/stage4_observability_signing.key --key-id stage4-observability` i potwierdź, że powstała paczka zawiera dashboardy oraz reguły alertów | Observability | `observability-bundle-<run_id>.tar.gz`, `manifest.json`, `manifest.sig` | Pliki obecne, podpis HMAC-SHA384 zweryfikowany `python scripts/verify_signature.py` |
| 8. Przenieś artefakty do `var/audit/acceptance/<timestamp>/stage4_smoke` oraz dodaj wpis do `audit/decision_logs/demo.jsonl` z kategorią `stage4_smoke` | Operator OEM | Struktura katalogu `var/audit/acceptance/<TS>/stage4_smoke`, wpis decision logu | Pliki skopiowane, wpis podpisany HMAC-SHA384 |
| 9. Uruchom `python scripts/watch_metrics_stream.py --headers-report --output var/audit/acceptance/<TS>/headers_report.json` i sprawdź telemetrię scheduler-a | Observability | `headers_report.json` | Raport zawiera metryki `avg_abs_zscore`, `allocation_error_pct`, `spread_capture_bps` w granicach profilu |
| 10. Uruchom `python scripts/audit_stage4_compliance.py --mtls-bundle-name core-oem --output-json var/audit/acceptance/<TS>/stage4_smoke/compliance.json` | Bezpieczeństwo | Raport audytu Stage4 (`status`, `issues`, `warnings`) | Raport zakończony statusem `ok`/`warn`, brak `fail` |
| 11. Otwórz dashboard Grafany **Stage4 – Multi-Strategy Operations** i zweryfikuj panele `avg_abs_zscore`, `allocation_error_pct`, `spread_capture_bps`, `secondary_delay_ms`, `pnl_drawdown_pct` względem progów profilu | Observability | `deploy/grafana/provisioning/dashboards/stage4_multi_strategy.json`, zrzuty ekranu dashboardu | Wszystkie panele w statusie zielonym/żółtym zgodnie z profilem ryzyka |
| 12. Zweryfikuj raport Paper Labs (`risk_simulation_report.json`) oraz wynik `load_test.json` – brak naruszeń i status zasobów `ok` | Zespół Ryzyka | `risk_simulation_report.json`, `risk_simulation_report.pdf`, `load_test.json` | Raport bez `breach` oraz `resource_status: ok` |
| 13. Aktualizuj checklistę `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` (sekcja Etap 1, krok 3) odnotowując numer joba GitHub oraz timestamp archiwizacji | Operator Runtime | Checklisty, decision log | Pole Akceptacja oznaczone `[x]` |

## Artefakty/Akceptacja
- Raport z workflow GitHub Actions (`stage4-multi-strategy-<run>`), skopiowany do
  `var/audit/acceptance/<timestamp>/stage4_smoke`.
- `smoke_report.json` z `scripts/smoke_demo_strategies.py`, `risk_simulation_report.{json,pdf}`
  z Paper Labs oraz `load_test.json` z `scripts/load_test_scheduler.py`.
- `stage4-strategies-<version>.zip` wraz z `stage4-strategies-<version>.manifest.{json,sig}`
  potwierdzonymi podpisem HMAC-SHA384.
- `observability-bundle-<run_id>.tar.gz` wraz z `manifest.json` i `manifest.sig`
  wygenerowanymi przez `scripts/export_observability_bundle.py`.
- Log kroku GitHub Actions `Validate Prometheus alert rules` potwierdzający poprawność
  reguł w `deploy/prometheus/rules/multi_strategy_rules.yml`.
- Raport nagłówków `headers_report.json` z `watch_metrics_stream.py`.
- Wpis w decision logu (`audit/decision_logs/demo.jsonl`) z kategorią `stage4_smoke`, podpisany
  HMAC-SHA384 i zweryfikowany `python scripts/verify_decision_log.py --strict`.
- Uaktualniona checklista `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` z numerem joba oraz
  podpisem operatora i compliance.
