# Bybit – Runbook go-live

## Runbook go-live
1. Skontroluj `config/runtime.yaml`, aby `bybit_desktop_paper` wskazywał na środowisko `bybit_paper` z aktywnym profilem `multi_strategy_default` oraz tagami `[desktop, paper, bybit]`.
2. Zweryfikuj sekcje `execution.paper_profiles.bybit_paper` oraz `execution.trading_profiles.bybit_desktop` – muszą wskazywać adapter `bybit_spot`, poprawne limity guardrailowe i powiązany entrypoint.
3. Upewnij się, że `config/core.yaml` zawiera poprawne mapowanie `bybit_paper` oraz że klucz `bybit_paper_trading` jest aktywny w magazynie tajemnic.
4. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint bybit_desktop_paper` i zweryfikuj, że raport `paper_exchange_metrics.bybit_desktop_paper` nie raportuje błędów sieciowych, a `rate_limited_events` pozostaje równe 0.
5. Po smoke teście wykonaj inspekcję `logs/guardrails` i eksportu raportu Markdown, upewniając się, że nie odnotowano throttlingu.

## Checklist licencyjna
- [ ] Licencja OEM pozwala na handel na Bybit (profil `desktop`, wystawca `dudzian`).
- [ ] Fingerprint (`var/licenses/active/fingerprint.json`) został odświeżony po aktualizacji sprzętu.
- [ ] Klucze API są zarejestrowane i przypisane do konta produkcyjnego, a dostęp offline jest udokumentowany.

## Checklist specyficzna dla Bybit
- [ ] `io_queue.exchanges.bybit_spot` ma ustawione `max_concurrency=3` oraz `burst=6`.
- [ ] Test `tests/integration/exchanges/test_bybit.py::test_bybit_spot_rate_limit` przechodzi lokalnie i w CI, potwierdzając poprawne retry.
- [ ] Test `tests/exchanges/test_bybit_signing.py::test_bybit_adapter_populates_credentials_and_signs_request` potwierdza poprawność podpisów REST.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_bybit.py
```

Test powinien zakończyć się sukcesem, a metryki `bot_exchange_rate_limited_total{exchange="bybit"}` pozostać na 0.
