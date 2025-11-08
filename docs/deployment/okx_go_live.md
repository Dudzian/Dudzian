# OKX – Runbook go-live

## Runbook go-live
1. W `config/runtime.yaml` upewnij się, że `okx_desktop_paper` wskazuje środowisko `okx_paper` oraz że przypisane są tagi `[desktop, paper, okx]`.
2. Sprawdź plik `config/core.yaml` (sekcja `okx_paper`) w celu potwierdzenia, że zmapowane poświadczenia `okx_paper_trading` są dostępne w magazynie tajemnic.
3. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint okx_desktop_paper` i potwierdź, że w raporcie `paper_exchange_metrics.okx_desktop_paper.rate_limited_events` pozostaje poniżej progu operacyjnego (0 w testach QA).
4. Monitoruj `logs/guardrails` oraz metrykę `bot_exchange_errors_total{exchange="okx"}` w celu wykrycia potencjalnych błędów sieciowych.
5. Zweryfikuj `execution.paper_profiles.okx_paper` oraz `execution.trading_profiles.okx_desktop` – muszą wskazywać właściwy entrypoint, adapter (`okx_spot`) i listę guardrailowych tagów.

## Checklist licencyjna
- [ ] Licencja OEM obejmuje profil `desktop` oraz uprawnienie na wykorzystanie OKX (sekcja `licensing.license.allowed_profiles`).
- [ ] Fingerprint oraz lista odwołań zostały zsynchronizowane przed publikacją (`var/licenses/active/*`).
- [ ] Dokumenty zgodności (KYC/KYB) dla OKX są zarchiwizowane w repozytorium audytowym.

## Checklist specyficzna dla OKX
- [ ] Limit kolejki I/O (`io_queue.exchanges.okx_spot`) posiada `max_concurrency=3` i `burst=6`.
- [ ] Test `tests/integration/exchanges/test_okx.py::test_okx_spot_rate_limit` przechodzi na środowisku QA, potwierdzając działanie retry.
- [ ] Test `tests/exchanges/test_okx_signing.py::test_okx_adapter_populates_credentials_and_signs_request` potwierdza, że podpisy CCXT wykorzystują przekazane klucze i hasło.
- [ ] Podpisy CCXT są aktualne – `sandbox_mode` jest automatycznie ustawiany dla środowiska paper/testnet w adapterze.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_okx.py
```

Spodziewany wynik: brak błędów oraz pozytywny scenariusz rate-limit retry.
