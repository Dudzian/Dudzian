# BitMEX Futures – runbook go-live

## Zakres
- Adapter `bot_core.exchanges.bitmex.futures:BitmexFuturesAdapter` (CCXT + watchdog + long-poll fallback).
- Środowiska: `bitmex_futures_testnet`, `bitmex_futures_live` w `config/core.yaml`.
- Streaming: `https://stream.sandbox.dudzian.ai/exchanges` (testnet/paper) i `https://stream.hyperion.dudzian.ai/exchanges` (live) – ścieżki `/bitmex/futures/public`, `/bitmex/futures/private`.
- HyperCare: podpisana checklist `stage6-bitmex-futures-2024q4`, dokumenty `kyc_2024q3`, `license_attestation`, `alerting_playbook`, `hypercare_runbook`.

## Pre-check
1. `python scripts/list_exchange_adapters.py --output reports/exchanges/bitmex_go_live.csv` i potwierdź obecność wpisów z aktualnymi URL strumieni (`stream_base_url`) oraz `live_readiness_signed=True`.
2. Zweryfikuj secrets `bitmex_futures_testnet_trading`, `bitmex_futures_live_trading`.
3. Sprawdź podpis checklisty HyperCare:
   ```bash
   python scripts/verify_decision_log.py --profile bitmex_futures_live \
       --checklist compliance/live/bitmex_futures/checklist.sig
   ```
4. Uruchom testy failoveru: `pytest tests/integration/test_exchange_manager_failover.py -k bitmex -n auto`.

## Deploy
1. `python scripts/run_stage6_hypercare_cycle.py --exchange bitmex_futures_live --export var/audit/hypercare/bitmex/$(date +%F).json`
2. Zweryfikuj streaming: `python scripts/run_stream_gateway.py --exchange bitmex_futures_live` oraz panel „Fallback long-poll”.
3. Udokumentuj wynik `python scripts/list_exchange_adapters.py --output reports/exchanges/post_deploy_bitmex.csv` (załącz do HyperCare).

## Rollback
- Wymuś CCXT failover: `manager.configure_failover(enabled=True, failure_threshold=1, cooldown_seconds=180)` w runtime.
- Przywróć `stream.base_url` do sandboxa i usuń klucze z produkcyjnego keychain.
- Dodaj wpis rollback w `var/audit/hypercare/bitmex/rollback.json` i zaktualizuj checklistę `stage6-bitmex-futures-2024q4` (status cofnięty).
