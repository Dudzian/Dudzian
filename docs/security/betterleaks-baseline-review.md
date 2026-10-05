# Betterleaks baseline review

The working-tree baseline contains only reviewed public digests, explicit
examples, and synthetic development/test fixtures. It must never contain an
entry for `.env`, `secrets/*.json`, or `var/secrets/*.json` runtime credential
stores. Findings are identified below without reproducing their matched values.

| ID | Baseline location and rule | Safe rationale |
| --- | --- | --- |
| BL-001 | `deployment/stage9_expected_public_authorities.json:14`, generic API key | Public authority digest used for comparison, not secret material. |
| BL-002 | `deployment/stage9_expected_public_authorities.json:22`, generic API key | Public authority digest used for comparison, not secret material. |
| BL-003 | `deployment/windows_stage9_production_trust.py:43`, generic API key | Public expected digest mirrored by the verifier, not secret material. |
| BL-004 | `deployment/windows_stage9_production_trust.py:44`, generic API key | Public expected digest mirrored by the verifier, not secret material. |
| BL-005 | `docs/offline_updates.md:24`, generic API key | Documentation-only example digest. |
| BL-006 | `docs/offline_updates.md:37`, generic API key | Documentation-only example digest. |
| BL-007 | `docs/offline_updates.md:53`, generic API key | Documentation-only example digest. |
| BL-008 | `deploy/docker-compose.yml:47`, generic password | Explicit local-development database placeholder. |
| BL-009 | `tests/test_alerts.py:385`, generic password | Synthetic authentication fixture confined to a test. |
| BL-010 | `tests/persistence/test_state_store_records.py:510`, generic API key | Synthetic persistence fixture confined to a test. |
| BL-011 | `tests/persistence/test_persistence_records.py:940`, generic API key | Synthetic persistence fixture confined to a test. |
| BL-012 | `deploy/packaging/README.md:69`, generic password | Documentation-only command placeholder. |
| BL-013 | `tests/architecture/test_cryptohunter_persistence_versioning_migrations_backup_and_recovery.py:1986`, generic API key | Synthetic recovery fixture confined to a test. |
| BL-014 | `config/marketplace/keys/dev-presets-ed25519.key:1`, private key | Intentional development preset key; production policy explicitly rejects reserved DEV identities. |
| BL-015 | `tests/update/test_update_package_cli.py:22`, generic API key | Synthetic update-package fixture confined to a test. |
| BL-016 | `tests/test_security_tls_audit.py:35`, private key | Synthetic TLS fixture confined to a test. |
| BL-017 | `tests/test_security_tls_audit.py:65`, private key | Synthetic TLS fixture confined to a test. |
| BL-018 | `tests/test_runtime_bootstrap.py:500`, private key | Synthetic bootstrap fixture confined to a test. |
| BL-019 | `tests/test_runtime_bootstrap.py:691`, generic password | Synthetic bootstrap passphrase confined to a test. |
| BL-020 | `tests/test_runtime_bootstrap.py:2963`, generic password | Synthetic bootstrap passphrase confined to a test. |

Any line movement requires regenerating and re-reviewing both this inventory and
`.betterleaks-baseline.json`; adding an exception merely to make CI pass is
forbidden.
