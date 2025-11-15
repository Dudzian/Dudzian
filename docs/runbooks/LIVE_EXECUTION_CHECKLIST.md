# Live Execution – Checklista uruchomienia

## Cel
Zapewnienie bezpiecznego startu trybu live (daemon + UI) po spełnieniu wymogów Paper Labs, licencjonowania OEM i zabezpieczeń mTLS/RBAC. Wszystkie kroki muszą zostać potwierdzone podpisanym decision logiem.

## Lista kontrolna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| [ ] Zweryfikuj ważność pakietu mTLS (`python scripts/generate_mtls_bundle.py`) oraz aktualność rejestru rotacji | SecOps | `secrets/mtls/*`, `var/security/tls_rotation.json` | [ ] |
| [ ] Sprawdź licencję OEM oraz fingerprint urządzenia (`oem_provision_license.py --verify`) | Operator OEM | `var/licenses/registry.jsonl`, `config/fingerprint.expected.json` | [ ] |
| [ ] Uruchom `python scripts/run_risk_simulation_lab.py` i dołącz raport Paper Labs do decision logu | Zespół Ryzyka | `reports/paper_labs/*.json`, `audit/decision_logs/live_execution.jsonl` | [ ] |
| [ ] Skonfiguruj live router (`config/core.yaml`) i potwierdź alerty fallback/latency w Prometheusie | Inżynier Runtime | `config/core.yaml`, `metrics/live_router.prom` | [ ] |
| [ ] Uruchom demona z parametrem `--live` i włącz mTLS (`config.execution.mtls`) | Inżynier Runtime | `logs/runtime/live_bootstrap.jsonl` | [ ] |
| [ ] (Opcjonalnie) wystartuj usługę cloudową (`python scripts/run_cloud_service.py --config config/cloud/server.yaml --emit-stdout`), wykonaj testowy handshake `CloudAuthService.AuthorizeClient` i potwierdź wpisy `cloud_auth_granted`/`cloud_session_*` w `logs/security_admin.log` | Inżynier Runtime + SecOps | `config/cloud/`, `var/runtime/cloud_ready.json`, `logs/security_admin.log` | [ ] |
| [ ] Zainicjuj UI z wymuszonym mTLS (`--grpc-use-mtls`) i przejdź checklistę pre-live (FPS/jank/overlay) | Operator UI | `logs/ui/startup.jsonl`, screenshot monitorów | [ ] |
| [ ] Zweryfikuj wpis decision logu (podpis HMAC, identyfikator operatora, latencja < progu) | Compliance | `audit/decision_logs/live_execution.jsonl` | [ ] |
| [ ] Potwierdź gotowość komunikacją do zespołu incident-response (kanał `#oem-live`) | L2 On-Call | `communications/go_live_announcement.md` | [ ] |

### Szybkie testy regresyjne

Przed finalnym uruchomieniem wykonaj lokalnie następujące polecenia (w katalogu repozytorium):

```bash
python scripts/lint_paths.py
mypy
pytest -m e2e_demo_paper --maxfail=1 --disable-warnings
pytest tests/test_paper_execution.py
pytest tests/integration/test_execution_router_failover.py
```

Zwróć uwagę na metryki `live_orders_fallback_total` oraz limitowanie per giełda – test integracyjny routera potwierdza, że fallback CCXT działa, a rate limit wymusza sekwencyjność.

> **Uwaga:** lint layoutu kończy się błędem przy każdej pozostałości `KryptoLowca/**` lub pliku wykonywalnym w `archive/**`. Usuń takie pliki przed kontynuowaniem procedury.

## Artefakty / Akceptacja
- Podpisany decision log (`audit/decision_logs/live_execution.jsonl`) z wpisami: przygotowanie mTLS, wyniki Paper Labs, start demona, start UI.
- Raport Paper Labs oraz stres testów zarchiwizowany w `reports/paper_labs/` i dołączony do ticketu compliance.
- Zrzut konfiguracji `config/core.yaml` (sekcja `execution`) z hashami SHA-384.
- Log bootstrapu potwierdzający, że `bootstrap_environment` oraz `bootstrap_frontend_services` dostarczyły realny `ExecutionService` (np. `LiveExecutionRouter` dla live) i wspólny `MarketIntelAggregator`; brak ostrzeżeń „paper-only mode” z AutoTradera.
- Snapshot dashboardu Prometheus/Grafana z metrykami `live_orders_total`, `live_orders_fallback_total`, `live_execution_latency_seconds`.
- Potwierdzenie kanałowe (chatops) z identyfikatorami operatorów oraz timestampem startu.

