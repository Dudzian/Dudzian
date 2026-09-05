# CryptoHunter Architecture Baseline Freeze

## M0.2-M0.11 — FROZEN

Cel: `INTEGRITY_GUARD_ONLY_NOT_RUNTIME_AUTHORITY`.

| Milestone | Canonical artifact | Canonical schema version | SHA-256 |
| --- | --- | --- | --- |
| M0.2 | `docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json` | `cryptohunter.canonical_domain_vocabulary.v1` | `4b25ce887432590bcb0647fcc10b774b68c49f2fb207637399201a7bc88087f8` |
| M0.3 | `docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json` | `cryptohunter.process_topology_and_lifecycle.v1` | `f17cca82cdeab53c9e45eb4a67aa4605197bf8a9b576098315eb294423a6b0db` |
| M0.4 | `docs/architecture/cryptohunter_product_architecture/environment_and_product_capabilities.json` | `cryptohunter.environment_and_product_capabilities.v1` | `0fb844d86817746fee3efb7ba7e19794ab2c4f321b70082557bdc6e2ea80b9c4` |
| M0.5 | `docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json` | `cryptohunter.exchange_accounts_and_instruments.v1` | `82fe0548c889aa0f203059db158e324ff1680b5b5c1b24395542244363f2174b` |
| M0.6 | `docs/architecture/cryptohunter_product_architecture/strategy_market_data_and_execution_routing.json` | `1.0.0` | `39a9b4f3b5e0c1311f788b061170bf5dc6b5e9c0fe3cec691b7094455fe4ceb7` |
| M0.7 | `docs/architecture/cryptohunter_product_architecture/commands_events_order_lifecycle_and_idempotency.json` | `cryptohunter.commands_events_order_lifecycle_and_idempotency.v1` | `e9cb8bfc0300174e03a0bbd557f7c5055634b0fd2f00bcf2f989625b6b896cb0` |
| M0.8 | `docs/architecture/cryptohunter_product_architecture/ledger_portfolio_capital_and_pnl.json` | `1.0.0` | `bdf70ab825298ee689718de794ee6f9d39211405db0f735db19816119c2d46fd` |
| M0.9 | `docs/architecture/cryptohunter_product_architecture/risk_hierarchy_kill_switch_and_execution_lease.json` | `1.0.0` | `f7b3fb943d3aa503b4ea6810ea084f17f5bc830f82eab7ce98b3ea5d63eb0537` |
| M0.10 | `docs/architecture/cryptohunter_product_architecture/identity_device_authentication_and_secrets.json` | `1.0.0` | `dd7a23a4f40001a929ab16135876b5d1cb850d3a4f136c7e7d21e01a1c38e61e` |
| M0.11 | `docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json` | `1.0.0` | `cb623d2a6af31b9ea07e2738d0749b782bf2250768b8e26c1f43294618c22077` |

Baseline M0.2–M0.11 jest zamrożony. Zmiana któregokolwiek canonical JSON wymaga jawnej aktualizacji freeze manifestu.

Hash dowodzi wyłącznie integralności. Freeze manifest nie jest runtime authority.

Późniejsza implementacja produkcyjna ma implementować istniejące kontrakty, a nie reinterpretować je.
