# M0.8 — Ledger, Portfolio, Capital and P&L

Status: **closed**. This is a target architecture contract, not runtime, storage, exchange synchronization, risk authority, or LIVE enablement. The machine-readable companion is `ledger_portfolio_capital_and_pnl.json`; independently authored immutable expectations attest it.

## Upstream and scope

M0.2–M0.7 are closed. M0.7 supplies the composite trusted accepted Full Economic Fill with identity, Portfolio/account/environment scope, exact historical Instrument binding, side, quantity, price, economic time, and multi-asset fee. Raw event payloads, adapter responses, current Instrument fallback, parsed symbols, and current fee schedules are not accounting authority.

M0.6 permits readiness and activation only for `SPOT/SPOT_PAIR` in the current edition. M0.8 closes SPOT_PAIR accounting and fails closed for MARGIN_PAIR, PERPETUAL_CONTRACT, DELIVERY_FUTURE, and OPTION. Those are structurally representable upstream but are not current-edition execution-capable and lack complete borrow/collateral/funding or expiry/exercise/settlement accounting authority.

## Immutable LedgerEntry journal

Canonical M0.2 `LedgerEntry` is the only durable accounting identity. No Posting, Batch, Balance, Position, Lot, Reservation, PnL, Asset, or Valuation durable entity is introduced. Its exact record is:

1. `ledger_entry_id`; `workspace_id`; `portfolio_id`; `environment`;
2. nullable `exchange_account_id`; nullable `strategy_instance_id`; `asset_reference`;
3. `account_role`; `direction`; `quantity`; `source_type`;
4. `accounting_source_identity`; `accounting_source_fingerprint_sha256`;
5. `accounting_rule_version`; `posting_index`; `posting_role`; `batch_fingerprint_sha256`;
6. `effective_at_utc`; `append_sequence`;
7. nullable `order_id`; nullable `fill_id`; nullable `audit_event_id`; nullable `correction_reason`.

Identity remains `led_<uuidv7>`. Quantity is a strictly positive canonical M0.5 decimal string. Direction is `DEBIT` or `CREDIT`; signed and zero quantities are forbidden. Economic time is immutable metadata distinct from strict append sequence.

The journal is append-only. Records cannot be updated, deleted, backdated in place, or rewritten after a late Fill. A correction is a new trusted, reasoned, balanced `reconciliation_correction` batch. Source identity plus deterministic posting keys preserves unambiguous audit without a new reversal identity or unsupported reference.

## Atomic batches and idempotency

The non-durable posting key is `(source_type, accounting_source_identity, ACCOUNTING_SPOT_FIFO_V1, posting_index, posting_role)`. The batch fingerprint hashes ordered exact projections excluding allocated LedgerEntry IDs and append sequences.

Before allocating the first ID or appending, the core validates trusted context, schema, source identity/fingerprint, transitions, positive quantities, and balance. Append is all-or-nothing. For every exact AssetReference independently:

`sum(DEBIT quantities) == sum(CREDIT quantities)`.

BTC, ETH, and USDT are never combined in one equation. A Fill creates base received/base clearing, quote clearing/quote paid, and, for `CHARGE`, fee expense/exact owned fee asset. Same identity, fingerprint, and rule returns `REPLAY_SUCCESS` with zero entries. Changed economics under the identity returns `ACCOUNTING_IDENTITY_CONFLICT`, requires reconciliation, and mutates nothing.

## Roles, balances, capital, and fees

`OWNED_AVAILABLE` and `OWNED_RESERVED` are the only owned roles. `TRADE_CLEARING`, `FEE_EXPENSE`, `REALIZED_PNL_CLASSIFICATION`, `EXTERNAL_CAPITAL`, `TRANSFER_CLEARING`, and `RECONCILIATION_CLEARING` balance/classify economics and never create another owned asset.

Balance keys preserve workspace, Portfolio, environment, ExchangeAccount, and exact AssetReference. For SPOT, `owned_total = available + reserved`. Reservation moves the same quantity from available to reserved; release reverses it. Neither changes total, NAV, or P&L. Replay cannot double-reserve/release. Insufficiency is an accounting transition failure, not risk authority; M0.9 owns authorization.

External contribution increases owned assets and net external capital, never trading P&L. Withdrawal decreases both, never becoming a trading loss. Deposit is external only with Portfolio-boundary provenance. Internal transfer creates no P&L and binds both legs, environments, accounts, references, quantities, and provenance. Cross-environment migration is never implicit.

Fee `NONE` has no posting. Fee `CHARGE` debits classification and credits the exact owned base, quote, or third asset. Base fees reduce received inventory before FIFO lot creation; third-asset quantity is preserved. Classification is excluded from NAV, and fee is subtracted at most once in net reporting. Missing fee valuation is not zero. Rebates remain unsupported upstream.

## Asset identity and aggregation

Asset identity is the full M0.5 venue-scoped object: `venue_asset_code`, `canonical_display_code`, `asset_namespace`, and `mapping_status`. Only `EXACT` and `EXPLICIT_ALIAS` are authoritative. There is no normalization, global alias, stablecoin parity, or durable Asset; display code is never identity.

Physical quantities remain per-account/per-namespace. BTC at two venues cannot be summed because upstream has no global equivalence authority. Portfolio value may aggregate only when every namespace-aware asset has an explicit trusted conversion path. Cross-namespace transfer likewise requires explicit equivalence or fails closed.

## FIFO positions and P&L

`ACCOUNTING_SPOT_FIFO_V1` is deterministic product accounting, not tax policy. Lots are projections keyed by Fill identity and posting index. Order is append sequence then posting index. Late/backfilled facts append later and never rewrite history; equal effective times use the same sequence rule. `Fraction` preserves exact ratios without rounding or global Decimal context.

A buy increases gross spot inventory; sell consumes oldest lots. Partial reduction preserves exact remainder and full close leaves no phantom lot. Negative inventory, borrowing, short, leverage, and flip-through-zero fail closed. Spot position is an analytical inventory/cost view and never another NAV asset.

Gross realized P&L is proceeds minus consumed FIFO cost. Fee effect is separate and subtracted once after trusted conversion. Net realized is gross minus once-valued fees plus supported funding/interest; those are unsupported in SPOT scope. `realized_pnl` is same-Fill-batch classification, never independent economics. Capital flows and marks do not alter realized P&L. Unrealized is projection-only marked value minus remaining FIFO cost; a mark changes unrealized/NAV but never journal or realized P&L.

## Valuation, NAV, and reconciliation

A valuation edge binds exact subject, valuation unit, positive exact rate, source identity, observed/effective/as-of/stale-after times, and fingerprint. Paths are explicit, ordered, acyclic, source-scoped, and exact. Missing, stale, and unsupported paths have distinct failures and are never zero. Any material unvalued asset yields `PARTIALLY_UNVALUED`.

The sole NAV equation is the sum of every `OWNED_AVAILABLE + OWNED_RESERVED` quantity converted by complete trusted paths, minus supported liabilities (none in current SPOT). Clearing/classification roles, analytical position, and realized/unrealized classifications are not added. Inventory, fee, position, and gain cannot be double-counted. Capital and P&L metrics remain disjoint; historical capital valuation requires an effective-time valuation, never today's substitute.

A venue snapshot is observed external fact only. Reconciliation returns `MATCH`, `DRIFT`, `MISSING_INTERNAL_FACT`, `MISSING_EXTERNAL_FACT`, `UNMAPPED_ASSET`, or `UNSUPPORTED`. Drift is ordinary `RECONCILIATION_DRIFT`, not `CONTRACT_INCONSISTENT`, and mutates nothing. A trusted correction AuditEvent with exact reason/provenance may append a balanced correction.

## Sources, environments, and M0.9

Fill and same-batch realized classification use trusted M0.7 `fill_id`. Non-Fill accounting uses `audit_event_id` only when immutable AuditEvent contains exact economics/provenance; otherwise it fails closed. Funding and interest are explicitly unsupported in current execution-capable scope.

PAPER, TESTNET, and LIVE share one isolated core. PAPER must emit canonical Full Fill facts; legacy float/mutable `paper.py` is evidence only. TESTNET uses trusted venue facts. LIVE is a first-class accounting target while current-edition execution stays disabled. Cross-environment totals are non-authoritative and cannot feed M0.9.

M0.8 may expose owned/available/reserved balances, spot exposure, NAV, realized/unrealized P&L, valuation completeness, and reconciliation status. It never exposes risk approval, limits, kill switch, ExecutionLease, or execution authority. `CONTRACT_INCONSISTENT` is only machine schema/dependency attestation corruption; ordinary fact, balance, valuation, unsupported-semantics, identity, and venue-drift failures remain distinct.

## Executable closure amendment

The reference model has two explicit levels. `PostingProjection` is a non-durable, pre-append value
with complete workspace/Portfolio/environment/account/AssetReference scope. `LedgerEntry` is the
canonical durable record and validates the exact 23-field schema, UUIDv7 prefixes, nullable and
source-reference rules, authoritative AssetReference mappings, closed role/direction/source/posting
role registries, positive decimal, both SHA-256 values, effective UTC time, posting index, and strictly
monotonic append sequence. Invalid enum values are rejected directly; boolean indexing is not enum
validation. IDs and sequences are allocated only after source, posting, fingerprint, balance, state,
and idempotency validation.

Trade entry accepts only the nominal `M07PrevalidatedAcceptedFillContext`, representing successful
M0.7 structural validation, Fill fingerprinting, historical Instrument resolution, and canonical
accepted history. A raw mapping is rejected. The context supplies the complete Fill economics and
resolved historical SPOT base/quote references. BUY derives base-owned/base-clearing and
quote-clearing/quote-owned legs. SELL derives base-owned/base-clearing and
quote-owned/quote-clearing legs, consumes reconstructed FIFO lots, and adds balanced gross-realized
classification legs. Every entry in that result retains `source_type = fill`; the M0.2
`realized_pnl` registry value is reserved and independent M0.8 use fails with
`UNSUPPORTED_ACCOUNTING_SEMANTICS`. Thus there is one Fill economic authority and no second P&L
source.

The closed non-durable AccountingEconomicFact registry specifies exact payload fields for deposit,
withdrawal, internal transfer, reservation, release, and reconciliation correction. The independent
`fee`, `funding`, `interest`, and `realized_pnl` shapes are closed forbidden inputs, not authorities.
Each derives identity from its durable `audit_event_id`, validates a canonical payload fingerprint,
and binds complete scope, exact assets/quantities, effective time and provenance/reason. Funding and
interest remain unsupported for current SPOT. Internal transfer contains source and destination
environments/accounts/assets/quantities and produces four atomic legs. Different environments,
namespaces, assets, or quantities fail closed.

The engine computes posting indices and its batch fingerprint itself. Per-source posting keys are
unique. A replay must match identity, source fingerprint, rule, and the previously derived batch;
otherwise it is an identity or contract-integrity failure with no append. Prospective owned available
and reserved balances are checked before commit. Withdrawals, trades, Fill fees, and reservations
cannot make either projection negative. Reservation effects are keyed by accepted SUBMIT_ORDER
command identity, exact order, scope, account, asset, and quantity; accepted terminal evidence cannot
release another order/account/environment or more than the exact open remainder.

Valuation uses an immutable nominal `PrevalidatedValuationContext` of exact edges containing subject
AssetReference, valuation-unit AssetReference, positive rate, source ID, observed/effective/as-of/
stale-after times and source fingerprint. Freshness is derived from time, not a caller boolean. The
resolver follows explicit deterministic fingerprint-ordered, source-scoped, acyclic paths. Raw,
missing, stale, cyclic, namespace-mismatched, negative-rate, or fingerprint-tampered input fails with
an explicit non-zero outcome. NAV consumes only this trusted context.

Reconciliation similarly consumes a nominal prevalidated observed balance context with exact
workspace, Portfolio, environment, account, AssetReference, observed quantity, as-of time, source ID,
and fingerprint. Drift never mutates the journal. Correction requires the closed exact-economic
AuditEvent schema and appends a reasoned balanced batch.

## Direct dependency fingerprint attestation

Every dependency entry contains a real RFC 6901 JSON Pointer and SHA-256 of canonical JSON
(`sort_keys`, UTF-8, compact separators). The validator resolves the actual upstream document,
rejects nonexistent pointers, hashes content, and exact-compares independently authored expected
entries. Collection roots use stable semantic extraction for Portfolio, LedgerEntry, allowed source
types, and the Portfolio-to-LedgerEntry relationship; no array index is an authority.

| Contract | JSON Pointer | SHA-256 |
|---|---|---|
| `canonical_domain_vocabulary.json` | `/public_trading_environments` | `b0114e386bf72439199ec8155d65a57dc7e00cabe79dba805f53221ba7713103` |
| `canonical_domain_vocabulary.json` | `/entity_kinds` | `bb92436a67abe42975d763b06717962009d3f4bc8a6066b227f34c8e7d178e9b` |
| `canonical_domain_vocabulary.json` | `/relationships` | `03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9` |
| `canonical_domain_vocabulary.json` | `/identifier_policy` | `44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20` |
| `exchange_accounts_and_instruments.json` | `/asset_reference_contract` | `6aaa3985e58acd4f7ee82cbd941e67507dd76a3186e7969bf8c62db1d4d41ee0` |
| `exchange_accounts_and_instruments.json` | `/decimal_policy` | `e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48` |
| `exchange_accounts_and_instruments.json` | `/instrument_contract/record_fields` | `39b5d32a1d887863cbc7f7a652e163b6b78d02a6ae21b1e4447e97437e18bf16` |
| `exchange_accounts_and_instruments.json` | `/instrument_contract/trusted_history_contract` | `d3bdb205695163ec2834a065bb872d9ca39f1ad1b0d60c592d7273f42c380c65` |
| `exchange_accounts_and_instruments.json` | `/instrument_type_registry` | `e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8` |
| `strategy_market_data_and_execution_routing.json` | `/current_edition_execution_pair_policy` | `7dd12317c8dab7bc2d19e751f39800db7895f737a42335df1e22927d465ec0b9` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/fill_contract` | `1846393d14f684fc2462eb12b5d92f9c18cfa8a0163b2a2f41fe88c913324d1b` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/command_registry/SUBMIT_ORDER` | `864bc27bf55228d08b6592f2042a3f9a1f447eae661382bde0b380602369d748` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/event_contract` | `c63d514a4546161798de2f7f90441821cd71653fba433c3577c6d7b630e4b6b2` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/order_lifecycle` | `48a514419aaa0863078e69ffc50c3acd4b37a06d873d257275fb68874cc840dd` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/idempotency_contract` | `43ba37d976eb5948970d289cc71cd06da82cb90803f9e5f61cc84109f00e9a62` |
| `commands_events_order_lifecycle_and_idempotency.json` | `/closed_request_policy` | `f3a523b01cbce2bafabeaad261777e3a97fc4f60db751deee0e77cb912f93e7c` |

The independently deep-frozen expected contract covers the complete LedgerEntry schema, roles, batch,
source and economic-fact registries, balance, reservation, asset identity, instrument coverage, FIFO,
P&L, valuation, NAV, reconciliation, ordering, environment, M0.9 outputs, failures, forbidden scope,
and dependency manifest. `validate_contract` exact-binds every root. Semantic weakening and upstream
content, manifest removal, pointer, fingerprint, or nonexistent-pointer mutations all return
`CONTRACT_INCONSISTENT`.

## Final trust-composition and projection closure

M0.8 does not construct M0.7 trust from a local Fill subset. Its nominal
`M07PrevalidatedAcceptedFillContext` represents the output of final M0.7 composite resolution and
contains the complete canonical Full Fill, the resolved historical Instrument, and the nominal
Core-owned accepted-Fill history projection. Reference fixture attestation requires the exact M0.7
`fact_fields`: `fill_id`, `order_id`, `environment`, `workspace_id`, `portfolio_id`,
`exchange_account_id`, `exchange_id`, `instrument_id`, `instrument_metadata_version`,
`execution_route_id`, `venue_trade_id`, `side`, `executed_quantity`, `execution_price`,
`executed_at_utc`, `fee_kind`, `fee_quantity`, `fee_asset_reference`, and
`fill_fingerprint_sha256`. It recomputes the canonical M0.7 fingerprint over the exact M0.7 input
fields and requires `M0.8 accounting_source_fingerprint == Fill.fill_fingerprint_sha256`. Missing
route/exchange/trade identity, changed economics under an old fingerprint, scope mismatch, invalid
accepted history, or a legacy subset is `TRUSTED_CONTEXT_FAILURE`.

Non-Fill source validation executes source-specific constraints before accounting. Deposit accepts
only `EXTERNAL_CONTRIBUTION`; withdrawal accepts only `EXTERNAL_WITHDRAWAL`; internal movement is
only the atomic four-leg `internal_transfer`. Reservation requires the nominal, already accepted M0.7
`M07PrevalidatedAcceptedCommandContext`; release requires
`M07PrevalidatedAcceptedTerminalOrderEventContext`.
Neither identifiers nor caller terminal strings confer authority. Reconciliation correction requires
an exact accepted target tuple, non-empty reason and provenance, and derives only the direction-inverse
target postings; generic `INCREASE`/`DECREASE` is forbidden. The M0.2
standalone `fee` source type is registry-reserved and currently returns
`UNSUPPORTED_ACCOUNTING_SEMANTICS`: M0.7 Full Fill is the sole trade-fee authority, preventing a
caller-created duplicate fee. Funding, interest, and independent realized-P&L remain likewise
unsupported for current SPOT semantics.

FIFO reconstruction uses an enumerated cursor and preserves every lot after a partial or exact-boundary
consumption. A BUY base fee cannot exceed executed quantity; such a fact is rejected before append.
Lots retain quantity, unit cost, exact `basis_valuation_unit`, source identity, append sequence, and
posting index. They remain ordered by accepted append sequence and posting index; duplicate Fill replay
is effect-free, and full close has no residual lot. A disposal whose lot basis unit differs from its
quote/P&L unit requires an explicit accepted historical valuation path. Its exact rate converts basis;
otherwise the distinct missing, stale, or unsupported valuation result fails closed before append.

Canonical NAV now requires an exact non-durable `PortfolioAccountingScope(workspace_id, portfolio_id,
environment)`. It aggregates ExchangeAccounts only inside that scope. PAPER, TESTNET, LIVE, another
Portfolio, or another Workspace can never enter the same execution-authoritative NAV. No canonical
cross-environment-total API is exposed.

Executable P&L projections rebuild, per exact Portfolio accounting scope, gross realized FIFO
classification, once-valued Fill fee effect, net realized P&L, and remaining-lot unrealized P&L.
Base, quote, and third-asset fees use explicit trusted conversion. Missing fee conversion returns
`MISSING_VALUATION` with no numeric fee/net result; it is never zero. Mark changes unrealized P&L and
NAV only, leaving journal and gross realized P&L unchanged. Capital contributions and withdrawals
remain excluded.

Every durable entry also binds source identity to its durable source reference: Fill entries require
`accounting_source_identity == fill_id`; supported non-Fill and reconciliation entries require
`accounting_source_identity == audit_event_id`. Independent `realized_pnl` remains forbidden.

A structurally valid valuation edge is not authority. `PrevalidatedValuationContext` additionally
requires the nominal, Core-owned, closed non-durable `CoreAcceptedValuationSourceRegistry`; arbitrary
self-hashed `valsrc_*` input cannot enroll itself. Resolution enumerates authorized acyclic paths,
discards stale paths, and selects the fresh winner by its ordered edge-fingerprint tuple. Only when no
fresh path exists does failure precedence produce `STALE_VALUATION`, then
`UNSUPPORTED_VALUATION_PATH` for cycles, then `MISSING_VALUATION`. A stale or cyclic branch cannot
block a fresh authorized alternative.

Observed balances use the equivalent nominal `CoreAcceptedObservedBalanceSourceRegistry`; a raw
self-hashed snapshot is not trusted venue evidence. Reconciliation implements all declared outcomes:
`MATCH`, `DRIFT`, `MISSING_INTERNAL_FACT`, `MISSING_EXTERNAL_FACT`, `UNMAPPED_ASSET`, and
`UNSUPPORTED`. Absence of journal history is distinct from an authoritative projected zero. Drift
still performs zero mutation.

Finally, immutable expectations and `validate_contract` now also exact-bind `schema_version`,
`m0_element`, `status`, `authority`, `source_of_truth`, `contract_inconsistent_scope`, and
`closure_conditions`. UI authority, mutable venue balance truth, M0.9 relabeling, broad misuse of
`CONTRACT_INCONSISTENT`, or best-effort closure all fail as `CONTRACT_INCONSISTENT`.

## Order-bound reservations, rebuild, and inventory closure

A reservation is admitted only with nominal Core-owned `M07PrevalidatedAcceptedCommandContext`
evidence and
is bound to the existing M0.7 `order_id` and canonical SUBMIT_ORDER `command_id`;
M0.7 defines `command_id` as the immutable idempotency identity and keeps it distinct from `order_id`.
Exact M0.7 attestation enforces `idempotency_key == command_id`; `correlation_id` alone is excluded
from the command fingerprint, while command, idempotency and nullable causation identities remain
semantic. Those rules are directly attested at `/idempotency_contract`; canonical decimal,
timestamp, null, UTF-8, sorted-key, compact-separator and NFC behavior is directly attested at
`/closed_request_policy` rather than inferred from the SUBMIT_ORDER registry alone.
The accepted reservation fact binds command, order, workspace, Portfolio, environment, account, exact
instrument and route. M0.7 Order `quantity` is order economics, not reservation quantity, and M0.7 has
no reservation-asset field. The exact reservation `asset_reference` and `quantity` belong exclusively
to the separately accepted M0.8 accounting instruction. M0.8 records that instruction but does not
claim M0.7 approved it and does not create M0.9 risk authority. The command identity deduplicates
identical reservation economics even if a new
AuditEvent ID is presented; changed economics conflict. A matching Fill consumes its exact spend from
`OWNED_RESERVED` atomically (including a fee in the reserved spend asset), partial Fills leave the
remainder, and another order/scope/asset cannot consume it. A nominal Core-owned accepted terminal
event context preserves the exact canonical envelope and safe-payload value schema, fingerprint,
order, scope, account, instrument, and route. Its sealed opaque M0.7 lifecycle-ingestion proof binds
event identity/type, resulting terminal target, legal predecessor state, aggregate version and exact
previous contiguous version; stale and gap events cannot receive release authority. `ORDER_REJECTED`, `ORDER_FILLED`,
`ORDER_CANCEL_CONFIRMED`, `ORDER_REPLACE_CONFIRMED`, and `ORDER_EXPIRED` map respectively to
`REJECTED`, `FILLED`, `CANCELLED`, `REPLACED`, and `EXPIRED`. Only that accepted mapping releases the exact
remainder once. This is accounting state, not M0.9 approval.
Accordingly, neither durable source is authorized by an AuditEvent or self-hash alone: reservation
requires the accepted M0.8 fact plus sealed accepted SUBMIT_ORDER context, and release requires the
accepted M0.8 fact plus sealed accepted terminal lifecycle/event context.

`Engine.fills` is no longer stored mutable authority: it is a derived view over immutable
`CoreAcceptedAccountingSourceHistory`. Every accepted source record is exact-bound to journal entries
by source identity, source fingerprint, rule version, and batch fingerprint. A fresh rebuild from the
journal plus that immutable accepted-source history reconstructs balances, reservations, FIFO lots,
gross/fee/net P&L, and unrealized P&L identically; valuation remains an explicit input.
For every record rebuild replays its canonical immutable context against preceding reconstructed
state, rederives the complete posting set, contiguous indices, rule, source and batch fingerprints,
and exact first sequence, then compares every resulting `LedgerEntry`. A coordinated legal-looking
journal mutation plus attacker-recomputed batch and source-record hashes therefore still differs from
the canonical rederivation and yields `CONTRACT_INCONSISTENT`.

FIFO inventory now processes every accepted asset-changing source in append order. External deposit
requires an exact effective-time unit basis and creates a product-accounting lot. Withdrawal consumes
FIFO basis without trading P&L. Internal transfer moves the exact consumed lot slices and basis between
accounts, preserving Portfolio basis. Fill spends and base/quote/third-asset fees consume the
corresponding asset inventory; insufficient or missing basis fails closed, and complete consumption
leaves no zero lot. Reconciliation reversal removes the exact target source effect from rebuild.

Unknown reconciliation drift still cannot mutate accounting. Correction is no longer a generic
increase/decrease: its AuditEvent must bind target source type, accounting source identity, source
fingerprint, and batch fingerprint. The engine validates the existing accepted batch and derives exact
direction-inverted postings. Missing canonical deposit is recorded as deposit, not correction.
The correction workspace, Portfolio, and environment must equal every target entry. Mandatory
target-aware batch validation runs on candidate entries before journal or accepted-map mutation, then
compares one-to-one cardinality/order, account, AssetReference,
role, posting role and quantity, and requires opposite direction. An isolated correction LedgerEntry
is only structurally well-formed; it is not executable-valid without this accepted target context.
The supported target set is exactly `fill`, `deposit`, `withdrawal`, `internal_transfer`,
`capital_reservation`, and `capital_release`. An exact target tuple can be effectively reversed once:
the same correction identity replays, while a second identity conflicts with zero mutation. Rebuild
removes the reversed economic effect, so Fill reversal restores consumed reservation, FIFO and
realized classification; reservation and release reversals also restore journal-consistent state.

The independently frozen `source_posting_matrix` forbids fee, funding, interest, and independent
realized-P&L durable entries and closes role, posting-role, and direction combinations for each
supported source. Deposit/withdrawal no longer advertise transfer clearing. Fill replay validates the
accepted source record, stored accepted batch, and every journal entry/batch fingerprint before
returning replay; corruption is `CONTRACT_INCONSISTENT` with zero new append.

## Accepted-fact authority and one reporting time

Non-Fill ingress requires a nominal Core-owned accepted projection mapping `audit_event_id` to the
exact source type, economic fingerprint, and scope. A raw self-hashed fact is only structurally intact,
not accepted. Valuation and observed-balance authorities likewise map accepted identity to the exact
accepted fingerprint and semantics; changing content and recomputing its hash without a new Core
acceptance fails trusted-context validation. This is an architectural trust boundary, not M0.10
signatures or authentication.

Every valuation context carries one `reporting_as_of_utc`. Every edge used by one NAV/P&L projection
must have that exact as-of, and freshness is evaluated against it. Unrealized P&L preserves
`MISSING_VALUATION`, `STALE_VALUATION`, and `UNSUPPORTED_VALUATION_PATH` with deterministic
precedence. Exact zero holdings are non-material for valuation completeness and require no conversion
path. PAPER, TESTNET, and LIVE retain the same isolated accounting core; LIVE execution remains
policy-disabled only by the current upstream edition.
