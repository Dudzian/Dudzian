# Kraken – Runbook go-live

## Runbook go-live
1. Zweryfikuj konfigurację `config/runtime.yaml`, upewniając się, że punkt wejścia `kraken_desktop_paper` wskazuje na środowisko `kraken_paper` oraz że profil ryzyka odpowiada wymaganiom klienta.
2. W `config/core.yaml` sprawdź mapowanie `kraken_paper` → `kraken_spot` oraz klucz `kraken_paper_trading`; potwierdź, że poświadczenia są dostępne w menedżerze tajemnic.
3. Uruchom `scripts/run_local_bot.py --mode paper --entrypoint kraken_desktop_paper`, obserwując sekcję `paper_exchange_metrics` w raporcie JSON – wskaźnik `rate_limited_events` powinien pozostać na poziomie 0 podczas testów akceptacyjnych.
4. Weryfikuj logi `logs/guardrails` pod kątem błędów sieciowych lub ostrzeżeń związanych z limitem zapytań.
5. Zweryfikuj sekcję `execution.paper_profiles.kraken_paper` i `execution.trading_profiles.kraken_desktop` w `config/runtime.yaml`, aby upewnić się, że wskazują właściwy entrypoint i limity kolejek I/O.

## Checklist licencyjna
- [ ] Licencja OEM (`var/licenses/active/license.json`) zawiera wpis na listę dozwolonych profili obejmujący `desktop` oraz `kraken`.
- [ ] Plik `config/runtime.yaml` ma ustawione `licensing.enforcement: true`, a fingerprint jest zaktualizowany (`var/licenses/active/fingerprint.json`).
- [ ] Revocation list (`var/licenses/active/revocations.json`) została zsynchronizowana z centralnym repozytorium.

## Checklist specyficzna dla Kraken
- [ ] Limity I/O z sekcji `io_queue.exchanges.kraken_spot` utrzymują `max_concurrency=3` oraz `burst=6`.
- [ ] Poświadczenia mają uprawnienia `trade` i `read`; test `tests/exchanges/test_kraken_signing.py::test_spot_private_request_signature` potwierdza poprawne podpisywanie żądań.
- [ ] Health-checki (`bot_exchange_health_status`) raportują wartość 1 dla `kraken` po zakończeniu smoke testów.
- [ ] `execution.trading_profiles.kraken_desktop` zawiera tagi `[desktop, paper, kraken]`, co pozwala raportom Guardrail poprawnie filtrować scenariusze papierowe.

## Smoke test
Uruchom polecenie:

```bash
pytest tests/integration/exchanges/test_kraken.py
```

Wynik powinien potwierdzić przejście scenariuszy retry/rate-limit oraz walidację podpisów.
