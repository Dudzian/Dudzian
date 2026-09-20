# CryptoHunter Architecture Baseline Freeze

## M0.2-M0.11 — FROZEN

Cel: `INTEGRITY_GUARD_ONLY_NOT_RUNTIME_AUTHORITY`.

| Milestone | Canonical artifact | Canonical schema version | SHA-256 |
| --- | --- | --- | --- |
| M0.2 | `docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json` | `cryptohunter.canonical_domain_vocabulary.v1` | `f50b4deae2ba220fae20804901a83cfafb7b9dbc569f6cb00ac737aaa7879ff3` |
| M0.3 | `docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json` | `cryptohunter.process_topology_and_lifecycle.v1` | `aa68cc7b844374a99121b3f6c792f6eda19f83cbba5e2a0116054557a0311be3` |
| M0.4 | `docs/architecture/cryptohunter_product_architecture/environment_and_product_capabilities.json` | `cryptohunter.environment_and_product_capabilities.v1` | `0fb844d86817746fee3efb7ba7e19794ab2c4f321b70082557bdc6e2ea80b9c4` |
| M0.5 | `docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json` | `cryptohunter.exchange_accounts_and_instruments.v1` | `9be7e3b5f5f6bf48738ab758e216822f0c3ae02d3edc6024bc81d7502ca59510` |
| M0.6 | `docs/architecture/cryptohunter_product_architecture/strategy_market_data_and_execution_routing.json` | `1.0.0` | `e5651d55fefd5b3ce9ebf33431aa1561d7bf3339075e1b0ce1b008a50475cf7f` |
| M0.7 | `docs/architecture/cryptohunter_product_architecture/commands_events_order_lifecycle_and_idempotency.json` | `cryptohunter.commands_events_order_lifecycle_and_idempotency.v1` | `790b9110f52fca2afab573f5130b003f18f74e2f441b3a58eb8078dfa66ae25d` |
| M0.8 | `docs/architecture/cryptohunter_product_architecture/ledger_portfolio_capital_and_pnl.json` | `1.0.0` | `e4e078013789ace2e283e544e2dccd93b3bd5e2930e0b7fe74f6bedb371710d8` |
| M0.9 | `docs/architecture/cryptohunter_product_architecture/risk_hierarchy_kill_switch_and_execution_lease.json` | `1.0.0` | `cbf7f6c07e7c0238753e8eea68690668731c175d5e4d2ffe615d3f045210754f` |
| M0.10 | `docs/architecture/cryptohunter_product_architecture/identity_device_authentication_and_secrets.json` | `1.5.0` | `a532fb292ea9e2f18e56c50a45202e8369400ab1179b57a186851dbf08219d71` |
| M0.11 | `docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json` | `1.0.0` | `31776894972835b37f1a9640f0ddea3f675cab116c8c96892c12c22cc2c246b1` |

Baseline M0.2–M0.11 jest zamrożony. Zmiana któregokolwiek canonical JSON wymaga jawnej aktualizacji freeze manifestu.

Hash dowodzi wyłącznie integralności. Freeze manifest nie jest runtime authority.

Późniejsza implementacja produkcyjna ma implementować istniejące kontrakty, a nie reinterpretować je.
