# Deribit Futures – runbook go-live

## Zakres
- Adapter `bot_core.exchanges.deribit.futures:DeribitFuturesAdapter` (REST + fallback long-poll `LocalLongPollStream`).
- Środowiska: `deribit_futures_testnet`, `deribit_futures_live` (konfiguracja w `config/core.yaml`).
- Streaming: `https://stream.sandbox.dudzian.ai/exchanges` (testnet/paper) oraz `https://stream.hyperion.dudzian.ai/exchanges` (live) z kanałami `/deribit/futures/public` i `/deribit/futures/private`.
- HyperCare: podpisana checklist `stage6-deribit-futures-2024q4` + komplet dokumentów (`kyc_2024q3`, `license_attestation`, `alerting_playbook`, `hypercare_runbook`).

## Pre-check
1. `python scripts/list_exchange_adapters.py --output reports/exchanges/deribit_go_live.csv`
   - zweryfikuj, że w raporcie widnieją wpisy `deribit/testnet` i `deribit/live` z poprawnymi URL strumieni oraz `live_readiness_signed=True`.
2. Potwierdź obecność kluczy w `secrets/keychain.yaml`: `deribit_futures_testnet_trading`, `deribit_futures_live_trading`.
3. Zweryfikuj podpisy HyperCare:
   ```bash
   python scripts/verify_decision_log.py --profile deribit_futures_live \
       --checklist compliance/live/deribit_futures/checklist.sig
   ```
4. Testy integracyjne failoveru: `pytest tests/integration/test_exchange_manager_failover.py -k deribit -n auto`.

## Deploy
1. `python scripts/run_stage6_hypercare_cycle.py --exchange deribit_futures_live --export var/audit/hypercare/deribit/$(date +%F).json`
2. `python scripts/run_stream_gateway.py --exchange deribit_futures_live` – obserwuj metryki `LocalLongPollStream` (panel „Fallback long-poll”).
3. Uruchom `python scripts/list_exchange_adapters.py --output reports/exchanges/post_deploy_deribit.csv` i załącz raport do HyperCare.

## Rollback
- Wyłącz native adapter `deribit_futures` w `config/runtime.yaml` (`enabled: false`) i przełącz failover menedżera na CCXT: `manager.configure_failover(enabled=True, failure_threshold=1, cooldown_seconds=120)`.
- Przywróć streaming sandboxowy (`base_url` → `https://stream.sandbox.dudzian.ai/exchanges`) i usuń klucze produkcyjne z keychain.
- Oznacz checklistę `stage6-deribit-futures-2024q4` jako cofniętą (dodaj wpis w `var/audit/hypercare/deribit/rollback.json`).
