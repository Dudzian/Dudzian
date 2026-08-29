# CryptoHunter Architecture Baseline Freeze

## M0.2-M0.11 — FROZEN

Cel: `INTEGRITY_GUARD_ONLY_NOT_RUNTIME_AUTHORITY`.

| Milestone | Canonical artifact | Canonical schema version | SHA-256 |
| --- | --- | --- | --- |
| M0.2 | `docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json` | `cryptohunter.canonical_domain_vocabulary.v1` | `4b25ce887432590bcb0647fcc10b774b68c49f2fb207637399201a7bc88087f8` |
| M0.3 | `docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json` | `cryptohunter.process_topology_and_lifecycle.v1` | `5eed0ce5d5c7a2c94ef607412ea39ec3df9b595f2c6a59b12b6a744a94014b7c` |
| M0.4 | `docs/architecture/cryptohunter_product_architecture/environment_and_product_capabilities.json` | `cryptohunter.environment_and_product_capabilities.v1` | `0fb844d86817746fee3efb7ba7e19794ab2c4f321b70082557bdc6e2ea80b9c4` |
| M0.5 | `docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json` | `cryptohunter.exchange_accounts_and_instruments.v1` | `83049f1f492c55d31e6ea8fb20c2389016f39e3db5aa1cd365c03636bb1c5081` |
| M0.6 | `docs/architecture/cryptohunter_product_architecture/strategy_market_data_and_execution_routing.json` | `1.0.0` | `39a9b4f3b5e0c1311f788b061170bf5dc6b5e9c0fe3cec691b7094455fe4ceb7` |
| M0.7 | `docs/architecture/cryptohunter_product_architecture/commands_events_order_lifecycle_and_idempotency.json` | `cryptohunter.commands_events_order_lifecycle_and_idempotency.v1` | `e9cb8bfc0300174e03a0bbd557f7c5055634b0fd2f00bcf2f989625b6b896cb0` |
| M0.8 | `docs/architecture/cryptohunter_product_architecture/ledger_portfolio_capital_and_pnl.json` | `1.0.0` | `bdf70ab825298ee689718de794ee6f9d39211405db0f735db19816119c2d46fd` |
| M0.9 | `docs/architecture/cryptohunter_product_architecture/risk_hierarchy_kill_switch_and_execution_lease.json` | `1.0.0` | `a595d7baf3ca8581df7686dd86551f70394b75f072e3dafc97ae5ac1d1e4bc37` |
| M0.10 | `docs/architecture/cryptohunter_product_architecture/identity_device_authentication_and_secrets.json` | `1.0.0` | `0c82b5b468e4d362cddaa0abaab32be2e11eff45ebc1c25a2c395a2220af2d0a` |
| M0.11 | `docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json` | `1.0.0` | `d8fcdc59e53f17c4aea8f68f388bd6398e6260532fa2a1eccadc9f9f9344ca0e` |

Baseline M0.2–M0.11 jest zamrożony. Zmiana któregokolwiek canonical JSON wymaga jawnej aktualizacji freeze manifestu.

Hash dowodzi wyłącznie integralności. Freeze manifest nie jest runtime authority.

Późniejsza implementacja produkcyjna ma implementować istniejące kontrakty, a nie reinterpretować je.
