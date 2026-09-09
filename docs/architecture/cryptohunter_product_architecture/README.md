# CryptoHunter Product Architecture Contract — M0

Ten katalog jest wersjonowanym źródłem prawdy dla bloku M0 „Product Architecture Contract”. Starsze dokumenty w `docs/`, `archive/`, workflowach i testach pozostają materiałem źródłowym, ale nie są nadrzędne wobec finalnego kontraktu produktu powstającego w M0.

## Status M0.1 — closed

M0.1 jest zamkniętym audytem aktualnego stanu repozytorium. Nie implementuje docelowej architektury, nie zmienia runtime'u, nie uruchamia Live i nie wybiera jeszcze modułów do usunięcia. Inwentaryzacja bazuje na kodzie, testach, konfiguracji, workflowach i packagingu istniejących w commicie bazowym zapisanym w JSON.

## Artefakty M0.1

- [current_state_inventory.md](current_state_inventory.md) — opisowa inwentaryzacja obecnego stanu.
- [current_state_inventory.json](current_state_inventory.json) — maszynowo walidowany kontrakt inwentaryzacji.


## Status M0.2 — closed

M0.2 jest kolejną warstwą źródła prawdy po zamkniętym M0.1: definiuje kanoniczny słownik domeny, publiczne środowiska tradingowe, niezależne osie stanu, politykę trwałych identyfikatorów, granice bezpieczeństwa oraz konflikty legacy. M0.2 nie zmienia runtime'u, produkcyjnych enumów ani konfiguracji środowisk.

## Artefakty M0.2

- [canonical_domain_vocabulary.md](canonical_domain_vocabulary.md) — opisowy kanoniczny słownik domeny M0.2.
- [canonical_domain_vocabulary.json](canonical_domain_vocabulary.json) — maszynowo walidowany kontrakt słownika domeny M0.2.


## Status M0.3 — closed

Closed M0.3 includes the protected external restore-freshness authority contract that M0.11 will consume later.

M0.3 definiuje kontrakt topologii procesów i lifecycle’u aplikacji: role CoreHost/TrayAgent/DesktopShell/Bootstrapper, zasady IPC/discovery, background po zamknięciu GUI, shutdown intents, first-run readiness, autostart oraz failure/restart policy. M0.3 jest zamkniętym kontraktem i nie implementuje osobnych procesów, tray, QML, proto ani Windows Service.

## Artefakty M0.3

- [process_topology_and_lifecycle.md](process_topology_and_lifecycle.md) — opisowy kontrakt topologii procesów i lifecycle’u M0.3.
- [process_topology_and_lifecycle.json](process_topology_and_lifecycle.json) — maszynowo walidowany kontrakt topologii procesów i lifecycle’u M0.3.



## Status M0.4 — closed

M0.4 definiuje kontrakt środowisk wykonawczych, ProductCapabilities, fail-closed fallback, capability trust, credential/endpoint policy, readiness, recovery oraz wielowarstwową blokadę Live. M0.4 jest zamkniętym kontraktem i nie implementuje runtime, adapterów, sekretów, endpointów, IPC, QML ani wykonywania zleceń.

## Artefakty M0.4

- [environment_and_product_capabilities.md](environment_and_product_capabilities.md) — opisowy kontrakt środowisk i ProductCapabilities M0.4.
- [environment_and_product_capabilities.json](environment_and_product_capabilities.json) — maszynowo walidowany kontrakt środowisk i ProductCapabilities M0.4.



## Status M0.5 — closed

M0.5 definiuje kontrakt multi-exchange: build-time Exchange Registry, ExchangeAccount, CredentialProfile, Instrument, katalog instrumentów, aliasy symboli, TradingUniverse, denial codes, audit events oraz zgodność z M0.2–M0.4. M0.5 jest zamkniętym kontraktem i nie implementuje runtime, adapterów giełdowych, endpointów, sekretów, routingu, strategii ani wykonywania zleceń.

## Artefakty M0.5

- [exchange_accounts_and_instruments.md](exchange_accounts_and_instruments.md) — opisowy kontrakt kont giełdowych i instrumentów M0.5.
- [exchange_accounts_and_instruments.json](exchange_accounts_and_instruments.json) — maszynowo walidowany kontrakt kont giełdowych i instrumentów M0.5.

## Status M0.6 — closed

M0.6 jest zamkniętym kontraktem architektonicznym definiującym StrategyDefinition, StrategyInstance, MarketDataRoute i ExecutionRoute, trusted validation context, readiness, kompatybilność oraz deterministyczny fail-closed routing. Jest kontraktem architektury bez runtime, adapterów, endpointów, sekretów ani wykonywania zleceń.

## Artefakty M0.6

- [strategy_market_data_and_execution_routing.md](strategy_market_data_and_execution_routing.md) — opisowy kontrakt strategii i routingu M0.6.
- [strategy_market_data_and_execution_routing.json](strategy_market_data_and_execution_routing.json) — maszynowo walidowany kontrakt strategii i routingu M0.6, będący źródłem prawdy.

## Status M0.7 — closed

M0.7 definiuje zamknięty kontrakt commands, immutable events, order lifecycle, idempotency, retry/duplicate handling, fills oraz bezpieczną granicę wykonania wspólną dla PAPER/TESTNET/LIVE. LIVE jest first-class target, lecz w current edition pozostaje zablokowane przez upstream M0.4/M0.6 policy. M0.7 nie implementuje runtime ani wykonywania zleceń.

## Artefakty M0.7

- [commands_events_order_lifecycle_and_idempotency.md](commands_events_order_lifecycle_and_idempotency.md) — opisowy kontrakt M0.7.
- [commands_events_order_lifecycle_and_idempotency.json](commands_events_order_lifecycle_and_idempotency.json) — maszynowo walidowany kontrakt M0.7, będący źródłem prawdy.

## Planowane elementy M0.1–M0.14

1. M0.1 — current-state audit/inventory
2. M0.2 — canonical domain model and durable IDs
3. M0.3 — Core/Tray/UI boundary and process lifecycle
4. M0.4 — Paper/Testnet/Live and ProductCapabilities
5. M0.5 — multi-exchange, accounts and instruments
6. M0.6 — strategy, market-data and execution routing
7. M0.7 — commands, events, order lifecycle and idempotency
8. M0.8 — ledger, portfolio, capital and P&L
9. M0.9 — risk hierarchy, kill switch and execution lease
10. M0.10 — identity, device, PIN, biometrics and secrets
11. M0.11 — persistence, versioning, migrations, backup and recovery
12. M0.12 — audit, observability, alerts and updater
13. M0.13 — final architecture contract and validators
14. M0.14 — closing audit

## Status M0.8 — closed

M0.8 closes the target architecture for an immutable, atomic, per-asset-balanced `LedgerEntry`
journal and rebuildable Portfolio balance, capital, SPOT FIFO position/P&L, valuation/NAV, and
reconciliation projections. It preserves venue-scoped asset identity, separates capital and fees
from trading P&L, and uses one isolated accounting core for PAPER, TESTNET, and LIVE without enabling
LIVE execution or implementing persistence, runtime execution, or M0.9 risk.

Artifacts:

- [`ledger_portfolio_capital_and_pnl.json`](ledger_portfolio_capital_and_pnl.json)
- [`ledger_portfolio_capital_and_pnl.md`](ledger_portfolio_capital_and_pnl.md)

## Status M0.9 — closed

M0.9 definiuje deterministyczną hierarchię risk policy, exact current-SPOT pre-trade limits,
hierarchiczny kill switch z monotonic fencing oraz krótko żyjący, exact-bound i one-shot
`ExecutionLease`. Zachowuje M0.7 idempotency, wymaga M0.8 order-bound reservation, izoluje środowiska i
pozostawia LIVE target-capable, ale zablokowane przez current upstream M0.4/M0.6. Nie implementuje
runtime'u, exchange side effects, M0.10 security ani M0.11 persistence.

## Artefakty M0.9

- [risk_hierarchy_kill_switch_and_execution_lease.md](risk_hierarchy_kill_switch_and_execution_lease.md)
- [risk_hierarchy_kill_switch_and_execution_lease.json](risk_hierarchy_kill_switch_and_execution_lease.json)

## Status M0.10 — closed

M0.10 zamyka Core-owned identity, device trust, PIN, platform-biometric assertion, Core-issued short-lived authentication proof, non-resurrectable proof fencing epochs, Core-policy-derived operation entitlement, actual-mutation-bound session-generation fencing, non-resurrectable current-security-epoch-fenced, fresh-distinct-ID, installation/policy-scoped LiveAccessGrant oraz external secret-reference oraz LiveAccessGrant security semantics bez implementacji runtime'u i persistence.

## Artefakty M0.10

- [identity_device_authentication_and_secrets.md](identity_device_authentication_and_secrets.md)
- [identity_device_authentication_and_secrets.json](identity_device_authentication_and_secrets.json)

## Status M0.11 — CLOSED

M0.11 zamyka neutralny implementacyjnie, projekcyjny kontrakt reprezentacji persistence, zamknięty i ściśle wykonywalnie walidowany kontrakt architektoniczny persistence, versioning, migrations, backup i recovery. Nie implementuje produkcyjnego silnika persistence, adaptera bazy danych ani UI.

## Artefakty M0.11

- [persistence_versioning_migrations_backup_and_recovery.json](persistence_versioning_migrations_backup_and_recovery.json) — kanoniczne maszynowe źródło prawdy.
- [persistence_versioning_migrations_backup_and_recovery.md](persistence_versioning_migrations_backup_and_recovery.md) — dokładna projekcja Markdown.
- [test_cryptohunter_persistence_versioning_migrations_backup_and_recovery.py](../../../tests/architecture/test_cryptohunter_persistence_versioning_migrations_backup_and_recovery.py) — wykonywalny model semantyczny i testy adwersarialne.

W M0.11 nośnik `PersistenceRecord` ma wykonywalne klucze wyprowadzane z faktów kanonicznych, walidację payloadu zależną od kategorii oraz dwustopniową rewalidację kandydata restore; chroniony `PREPARE G+1` pozostaje nieodwracalnym progiem świeżości podczas recovery.
Bezpośrednie rekordy upstream są walidowane według rzeczywistych zamkniętych pól i reguł wskazanego kontraktu upstream; ogólny `semantic_object`/`canonical_fields` nie jest nośnikiem sukcesu.
Immutable history przechowuje zamknięty upstream payload/projection i recomputed payload fingerprint; fact oraz direct carriers wykonują literalne kontrakty typów, prefixów, kluczy i nested schemas.

### M0.11 — immutable upstream provenance closure

Immutable history carriers są wyprowadzane metodą **source first → carrier second**. `RuntimeSession` zachowuje wyłącznie kanoniczne fakty M0.2: `runtime_session_id` (`run_<uuidv7>`) oraz rodzica `device_installation_id`; bootstrap M0.3 nie jest właścicielem tej tożsamości, a aktywny proces nie jest odtwarzaną authority. `RiskPolicy` zachowuje dokładny zamknięty rekord M0.9: `risk_policy_id`, `revision`, `environment`, `scope_type`, `scope_id`, `action`, `limits` i `semantic_fingerprint_sha256`. Integralność wrappera jest sprawdzana osobno od rewalidacji canonical provenance i semantic fingerprint payloadu.

Wszystkie 14 nośników immutable history korzysta z pełnego zamkniętego DTO upstream albo z jawnej, bezstratnej projekcji z wykonywalnymi selectorami per-field. Stany są ograniczone do rejestrów właściwych ownerów, `StrategyDefinition` zachowuje literalny rekord `sdef` i `definition_version`, a historia rezerwacji jest dyskryminowaną sumą dokładnych M0.8 `capital_reservation` i `capital_release`. Wrapper hash pozostaje wyłącznie dowodem integralności i nie zastępuje ponownej walidacji semantic fingerprint ani accepted/current membership.

Semantic fingerprints mają jawne wykonywalne tryby zgodne z owner contracts: canonical JSON object, ordered canonical JSON array, NFC event envelope, M0.5 domain separator + newline + canonical object oraz M0.6 surowe bajty domain separatora poprzedzające canonical configuration UTF-8. Test provenance rozwiązuje również constraint pointers i niezależnie porównuje prefiksy ID, enumy, constants, nested schemas, typy oraz nullable semantics z actual upstream.

Historia rezerwacji wykonuje oba warianty M0.8 bez generycznych skrótów: `capital_reservation.quantity` jest dodatnim canonical decimal, `asset_reference` jest dokładnym zamkniętym M0.5/M0.8 `AssetReference.trusted`, `provenance` jest niepustym stringiem, a środowisko, discriminator i terminal states są wiązane z literalnymi rejestrami. `source_fingerprint_sha256` jest ponownie liczony z kompletnego DTO bez pola fingerprint po canonical NFC JSON, lecz authority nadal wymaga wcześniejszego `CoreAcceptedAccountingFactProjection` membership.

M0.11 rewaliduje autorytet historii rezerwacji poza integralnością rekordu: restore wymaga istniejącego wcześniej, nieprzenoszonego w backupie członkostwa `CoreAcceptedAccountingFactProjection` oraz zapieczętowanego kontekstu zaakceptowanej komendy M0.7 `SUBMIT_ORDER` albo zaakceptowanego terminalnego zdarzenia M0.7. Kandydat, jego source hash ani wrapper hash nie mogą samodzielnie utworzyć tego członkostwa (no self-enrollment).

## Status M0.12 — IN PROGRESS

M0.12 S9A ustanawia foundation contract dla durable audit evidence, observability, alertów oraz uprzywilejowanego updatera/rollbacku. Nie implementuje runtime'u M1, nie aktywuje LIVE i nie zmienia zamrożonego baseline'u M0.2–M0.11.

## Artefakty M0.12 S9A

- [audit_observability_alerts_and_updater.json](audit_observability_alerts_and_updater.json) — kanoniczne maszynowe źródło prawdy foundation contract.
- [audit_observability_alerts_and_updater.md](audit_observability_alerts_and_updater.md) — deterministyczna projekcja Markdown.
- [test_cryptohunter_audit_observability_alerts_and_updater.py](../../../tests/architecture/test_cryptohunter_audit_observability_alerts_and_updater.py) — wykonywalny test architektury i cross-contract bindings.

## FROZEN ARCHITECTURE BASELINE — M0.2–M0.11

M0.2–M0.11 stanowią zamrożony baseline architektury chroniony przez [canonical freeze manifest](architecture_baseline_freeze.json) oraz jego [projekcję Markdown](architecture_baseline_freeze.md). Późniejsze prace produkcyjne nie mogą potajemnie zmieniać tych kontraktów. Zmiana baseline'u wymaga jawnego architecture change, aktualizacji właściwego canonical JSON, a następnie świadomej aktualizacji freeze manifestu.
