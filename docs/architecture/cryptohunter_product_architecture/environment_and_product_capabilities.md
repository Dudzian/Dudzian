# CryptoHunter M0.4 — Environment and ProductCapabilities Contract

Status: `under audit`

This document explains the M0.4 machine contract in `environment_and_product_capabilities.json`. The JSON is the source of truth; this Markdown describes intent, invariants and safety boundaries only. M0.4 does not implement runtime adapters, exchange I/O, secret storage, order execution, IPC, QML, persistence, recovery or cryptography.

## Canonical execution environments and endpoint classes

CryptoHunter has exactly three execution environments: **PAPER**, **TESTNET** and **LIVE**. `DEMO` is not a public UX mode and not a machine-contract execution environment. `PREVIEW` can remain only a historical product name or build label. `SIMULATION` and `SANDBOX` are not product environments.

Each environment uses the closed `endpoint_classes` enum. PAPER has exactly `LOCAL_SIMULATION`; TESTNET has exactly `PUBLIC_TESTNET` and `PRIVATE_TESTNET`; LIVE has exactly `PUBLIC_LIVE` and `PRIVATE_LIVE`. Every environment endpoint class must exist in `endpoint_policy.endpoint_classes`, and the current edition still forbids Live endpoint resolution and Testnet-to-Live fallback.

## Live dialog graph, not a linear script

`live_activation_dialog_flow` is a deterministic directed graph with entry step `operator_selects_visible_locked_live` and exactly three terminal outcomes: `CANCELLED`, `AUTHENTICATION_FAILED` and `LIVE_BLOCKED_BY_EDITION`.

Cancel, auth failure and auth success are not consecutive linear steps. The operator first reaches `operator_decision`: `CANCEL` terminates at `cancelled_without_activation_request`, while `CONTINUE` goes to authentication. Authentication then branches: `FAILED` terminates at `authentication_failed`; `SUCCEEDED` records `LIVE_ACTIVATION_AUTH_SUCCEEDED`, does not extend ProductCapabilities, creates an authenticated activation attempt, and only then reaches the policy capability check. The current edition still terminates at `LIVE_BLOCKED_BY_EDITION`, emits `LIVE_ACTIVATION_BLOCKED_BY_EDITION`, leaves `active_environment` unchanged and forbids credential read, endpoint resolution, adapter creation, private connection and execution.

The graph has no cycles, terminal steps have no outgoing transitions, all destinations must exist, every non-entry step is reachable, cancel never reaches authentication, failure never reaches success, success never reaches failure, and only the success path can reach the policy capability check.

## Core activation order and defensive gates

The runtime activation flow is Core-owned: ProductCapabilities schema/signature/trust; requested environment enum validation; operator authorization context if required; edition environment capability; future LiveAccessGrant only for a future Live edition; kill switch/safety; readiness; credential scope; endpoint class; adapter capability; connection policy; Core commit; route binding; and per-order risk gates.

`requested_environment` must be validated before capability lookup. For the current edition LIVE denial occurs at the edition capability gate before credential read, endpoint resolution, adapter creation or connection creation. DesktopShell never changes `active_environment`.

The machine contract also carries `gate_execution_invariants`: no OR-bypass, all applicable gates required, earlier denial forbids later side effects, direct layer invocation revalidates its own gate, later layers do not assume earlier validation, denied environments never commit `active_environment`, and current-edition LIVE is denied at every defensive gate. Direct calls to adapter factory, connection manager, route binding, execution pipeline, reconnect, recovery or reconciliation must still reject Live themselves.

## Current product edition

The current edition is `CRYPTOHUNTER_TESTNET_EDITION`: PAPER is visible/selectable/executable, TESTNET is visible/selectable/executable after readiness gates, and LIVE is visible but locked, non-selectable, non-executable and denied with `LIVE_BLOCKED_BY_EDITION`.

## Signed ProductCapabilities document envelope

M0.4 separates payload, header and envelope:

- `capability_payload_schema` requires exactly `schema_version`, `capabilities_id`, `edition_id`, `issued_at_utc`, `expires_at_utc`, `non_expiring`, `capability_set`, `environment_capabilities`, `feature_flags`, `capability_set_hash`, `source`, `fail_closed_policy` and `revocation_reference`.
- `signature_header_schema` requires exactly `signature_schema_version`, `signature_algorithm_id` and `key_id`.
- `signed_document_envelope_schema` requires exactly `capability_payload`, `signature_header`, `signed_payload_hash` and `signature`.

`capability_payload` is canonicalized deterministically. `signed_payload_hash` is the hash of that canonical payload. The signature covers the canonical signature header plus `signed_payload_hash`; the signature field is not part of the signed payload. `trust_state` is not a trusted input field in the signed document; it is the result of envelope validation. A payload without the full signature envelope is not a verified ProductCapabilities document and cannot be `VALID`. Unknown authority-bearing fields are rejected unless a versioned schema explicitly allows and signs them.

## Executable expiry policy

`expiry_policy` is executable: timestamps must be `RFC3339_UTC` and UTC is required. `NON_EXPIRING` requires boolean `non_expiring=true`, present `expires_at_utc=null`, and a valid UTC `issued_at_utc`. `EXPIRING` requires boolean `non_expiring=false`, non-null valid UTC `expires_at_utc`, valid UTC `issued_at_utc`, and `expires_at_utc > issued_at_utc`.

At validation time, `current_utc >= expires_at_utc` maps to `EXPIRED`; the equality boundary is expired. Malformed timestamps, missing fields, invalid field types and invalid expiry combinations fail closed and map to `UNSUPPORTED_SCHEMA` rather than `VALID`.

## Typed trust-state outcomes and SAFE_LOCAL_ONLY

`trust_state_outcomes` uses one uniform schema for every trust state: `private_testnet_execution_decision`, `live_execution_decision`, `secret_read_decision`, `private_connection_decision`, `fallback_policy`, `denial_code` and `audit_event`.

`VALID` uses enum decisions `PROCEED_TO_READINESS_GATES` for private Testnet and `PROCEED_TO_LATER_TESTNET_GATES` for secret/private connection checks; LIVE remains `DENY`. Every non-VALID state has all decisions `DENY`, fallback `SAFE_LOCAL_ONLY`, a stable ProductCapabilities denial code and `PRODUCT_CAPABILITIES_REJECTED`.

`SAFE_LOCAL_ONLY` means PAPER may remain local-only, TESTNET private execution is blocked, LIVE is blocked, exchange secrets are not read/decrypted, UI shows a capability integrity/permission error, Core starts no external-side-effect strategies, reconciliation stays local-only and the condition is audited.

## Complete 18-gate audit event map

Every Live gate has an exact specialist audit event: `UI_MODE_SELECTOR → LIVE_ACTIVATION_BLOCKED_BY_EDITION`, `CONFIG_DESERIALIZATION → CONFIG_ENVIRONMENT_REJECTED`, `PERSISTED_STATE_RESTORE → PERSISTED_ENVIRONMENT_REJECTED`, `IPC_COMMAND_VALIDATION → IPC_ENVIRONMENT_COMMAND_REJECTED`, `PRODUCT_CAPABILITY_POLICY → LIVE_ACTIVATION_BLOCKED_BY_EDITION`, `LIVE_ACCESS_GRANT_POLICY → LIVE_ACTIVATION_BLOCKED_BY_EDITION`, `CREDENTIAL_PROFILE_RESOLUTION → CREDENTIAL_SCOPE_REJECTED`, `SECRET_READ_OR_DECRYPT → CREDENTIAL_READ_REJECTED`, `ENDPOINT_RESOLUTION → ENDPOINT_POLICY_REJECTED`, `ADAPTER_FACTORY → ADAPTER_FACTORY_REJECTED`, `PRIVATE_CONNECTION_CREATION → PRIVATE_CONNECTION_REJECTED`, `RUNTIME_ENVIRONMENT_ACTIVATION → RUNTIME_ENVIRONMENT_ACTIVATION_REJECTED`, `STRATEGY_ROUTE_BINDING → EXECUTION_ROUTE_REJECTED`, `ORDER_INTENT_ACCEPTANCE → ORDER_INTENT_REJECTED`, `EXECUTION_COMMAND_SUBMISSION → EXECUTION_COMMAND_REJECTED`, `RECONNECT → RECONNECT_ENVIRONMENT_REJECTED`, `RECOVERY → RECOVERY_ENVIRONMENT_REJECTED`, and `RECONCILIATION_WITH_EXTERNAL_VENUE → RECONCILIATION_ENVIRONMENT_REJECTED`.

All gates share `requested_environment`, `active_environment_before`, `product_capabilities_snapshot`, `edition_id` and `correlation_id`, and each gate has its own non-empty layer-specific inputs.

## Strict timestamp profile and canonical signature validation

The machine timestamp profile is `RFC3339_UTC_ZULU_SECONDS`: dates use `YYYY-MM-DD`, the separator is uppercase `T`, time includes hour/minute/second, the timezone is uppercase `Z`, offsets such as `+00:00` are forbidden, basic dates and ISO week dates are forbidden, lowercase `t`/`z` are forbidden, fractional seconds use only `.` with 1–9 digits, comma fractions are forbidden, leap seconds are rejected, and the timestamp must be a real calendar date.

Signature canonicalization is executable and structural. M0.4 uses `RFC8785_JCS`, `UTF-8`, `SHA-256`, lowercase 64-character hex hashes, `BASE64URL_NO_PADDING` signatures, and `UNIQUE_LEXICOGRAPHICALLY_SORTED_STRING_ARRAY` for `capability_set`. `capability_set` must be duplicate-free and sorted before `capability_set_hash`; changing `capability_set` changes both `capability_set_hash` and `signed_payload_hash`.

`signature_input_schema` is defined once in `ProductCapabilities.document_schemas`. Signature input is the RFC 8785/JCS canonicalization of `{ "signature_header": <full signature_header>, "signed_payload_hash": <hash payloadu> }`; no unspecified string concatenation is allowed. The same `document_schemas` registry is the single authoritative source for payload, header, envelope and signature-input schema definitions; other sections reference those schema IDs.

The structural hash tests do not implement real asymmetric cryptography. They canonicalize payloads, compute `capability_set_hash`, compute `signed_payload_hash`, build canonical signature input, verify hash encodings, and reject tampering or unknown authority-bearing fields before any document can be structurally classified as valid.

The success path emits `LIVE_ACTIVATION_AUTH_SUCCEEDED` exactly once and `LIVE_ACTIVATION_BLOCKED_BY_EDITION` exactly once. The terminal `forbid_live_external_side_effects` step requires that prior audit event but does not emit it again.

Graph determinism is strict: ACTION has exactly one outgoing transition; DECISION has at least two uniquely labelled non-empty events; TERMINAL has no transitions and exactly one registered outcome. The interpreter rejects unknown, missing and unused decision events instead of choosing a default transition.

## M0.4 FIX: timestamp precision, JCS and registry validation

The JSON contract remains the machine source of truth. This section clarifies the additional constraints that are now executable in the architecture tests.

### Strict timestamp profile

M0.4 timestamps use `RFC3339_UTC_ZULU_SECONDS`: `YYYY-MM-DD`, uppercase `T`, required hour/minute/second, uppercase `Z`, optional fractional seconds with a dot and 1–9 digits, and no UTC offsets such as `+00:00`. Leap seconds and invalid calendar dates are rejected. Test interpreters preserve nanosecond precision separately from the whole-second UTC timestamp, so `.000000001Z` is one nanosecond and boundary comparisons such as `current_utc == expires_at_utc` deterministically return `EXPIRED`.

### RFC 8785/JCS canonicalization

The contract continues to name `RFC8785_JCS`, but the tests no longer treat Python `json.dumps(sort_keys=True)` as a JCS implementation. They use a test-only current-schema canonicalizer for the ProductCapabilities JSON types plus independent vectors for UTF-16 property ordering, escaping, and determinism. Numbers are forbidden in the current signed capability payload schema, so the current-schema canonicalizer rejects integers and floats and does not implement or simulate RFC 8785 number serialization.

### Signed payload type policy and hashes

The signed payload currently allows objects, arrays, strings, booleans, and null. Numbers are `FORBIDDEN_IN_CURRENT_SCHEMA` and any integer or float inside the payload fails structural validation. `capability_set` is a unique lexicographically sorted string array; its JCS bytes are hashed with SHA-256 to produce a 64-character lowercase hex `capability_set_hash`. The full `capability_payload` JCS bytes are hashed with SHA-256 to produce `signed_payload_hash`, so changing `capability_set`, `environment_capabilities`, `feature_flags`, expiry, revocation, source, or fail-closed policy changes the structural hash.

### Signature input and registries

The deterministic signature input is the JCS canonical object containing exactly `signature_header` and `signed_payload_hash`; the `signature` field never signs itself. Structural/hash validation is separate from trust validation: the test digest is named a structural fixture and never proves trust state `VALID`. A separate signature verification registry contract requires recognized algorithm IDs and recognized key IDs from trusted build/runtime registries. GUI, config, CLI, environment variables, IPC, operator authentication, and the signed document itself cannot add algorithms or trusted keys; unknown or missing registries fail closed.

### Single schema source

`ProductCapabilities.document_schemas` is the only authoritative source for payload, signature header, envelope, and signature-input field sets. Other sections refer to those schemas by stable `*_schema_ref` values and do not carry fallback copies of `required_fields` or the canonical signature-input object shape.

### Base64url and deterministic dialog auditing

`BASE64URL_NO_PADDING` validation is structural and decoding-based: values must be non-empty, padding-free, URL-safe, decodable with controlled padding, and canonical after re-encoding. The Live activation dialog remains a deterministic graph rather than a linear list. Its success path emits `LIVE_ACTIVATION_AUTH_SUCCEEDED` exactly once and `LIVE_ACTIVATION_BLOCKED_BY_EDITION` exactly once; the terminal side-effect-forbidding step depends on that prior audit event instead of emitting it again.

## M0.4 FIX: typed document validation and registry outcomes

M0.4 now separates signature validation into explicit stages: document structure, canonical hashes, registry policy, future cryptographic signature verification, expiry/revocation, and validated snapshot creation. Stage results are not trust states. In particular, `REGISTRY_ACCEPTED` only means that the algorithm and key references were recognized; it does not create `VALID`, because real asymmetric verification is intentionally outside M0.4.

Registry failures produce diagnostic reason codes (`UNKNOWN_SIGNATURE_ALGORITHM`, `UNKNOWN_KEY_ID`, `MISSING_SIGNATURE_ALGORITHM_REGISTRY`, `MISSING_VERIFICATION_KEY_REGISTRY`) that map consistently to trust state `INVALID_SIGNATURE` and denial code `PRODUCT_CAPABILITIES_INVALID_SIGNATURE`. The JSON contract is the source for that mapping, and tests read it from the contract rather than hardcoding an inconsistent validator.

`capability_set` uses the `UTF16_CODE_UNIT_LEXICOGRAPHIC` order used by JCS property-name ordering. The order is not locale-aware, does not use Python code-point sorting without an explicit key, and does not normalize Unicode. Duplicate entries are rejected and the hash is computed over the already-canonical UTF-16 ordered array.

The signed document schemas now contain field-level contracts for the capability payload, signature header, envelope, and signature input. Those contracts type-check strings, booleans, nullable fields, arrays, objects, hash formats, base64url signatures, environment keys and current-schema feature flag values before hash validation.

Raw JSON must be parsed as UTF-8 with duplicate object member names rejected before canonicalization. No parser may silently apply last-key-wins. The rule applies to the envelope, payload, header and nested objects, and malformed UTF-8, non-finite values and trailing data fail closed.

The current-schema canonicalizer is complete only for the JSON types allowed by the current signed payload schema: object, array, string, boolean and null. It rejects integers and floats, and it does not contain a partial `.15g` or special-case number serializer. Future use of JSON numbers requires a new schema version plus full RFC 8785 number serialization and dedicated conformance vectors.

The M0.4 architecture tests again cover the full safety surface: execution environments, current edition matrix, endpoint integrity, trust states, `SAFE_LOCAL_ONLY`, activation ordering, persistence/recovery, audit and denial registries, typed signature contracts, the deterministic Live dialog graph, nanosecond expiry, strict base64url and all 18 defensive Live gates.

## M0.4 FIX: separated pipeline, closed schemas and revocation

The validation pipeline is now separated into six machine stages. `DOCUMENT_STRUCTURE_VALIDATION` parses raw JSON, rejects duplicate members, checks required field sets, field contracts, closed nested schemas, hash string formats and base64url syntax only. It does not recompute hashes, does not compare the `signature` field with a fixture, does not perform registry lookup and never creates a trust state. `CANONICAL_HASH_VALIDATION` is the first stage that recomputes `capability_set_hash` and `signed_payload_hash`. `REGISTRY_POLICY_VALIDATION` only checks trusted algorithm and key registries after hashes have been accepted. `CRYPTOGRAPHIC_SIGNATURE_VERIFICATION` is explicitly not implemented in M0.4, so `VALIDATED_SNAPSHOT_CREATION` and trust state `VALID` are unreachable in this milestone.

Environment capability values are closed nested schemas. `PAPER` contains exactly `private_execution_allowed=false` and `local_execution_allowed=true`; `TESTNET` contains exactly `private_execution_allowed_when_trust_state_valid=true`; `LIVE` contains exactly `private_execution_allowed=false` and `public_live_market_data_implies_execution=false`. Additional fields, missing fields or type changes are rejected by structure validation.

The current feature flag registry is also closed for schema version `cryptohunter.product_capabilities.payload.v1`: `live_activation_dialog=true`, `live_execution=false` and `testnet_execution=true`. Unknown flags, missing flags or non-boolean values are rejected before hash validation, and no feature flag can widen Live execution rights.

The version policy supports only `cryptohunter.product_capabilities.payload.v1` and `cryptohunter.product_capabilities.signature.v1`. Unknown or empty schema versions map to `UNSUPPORTED_SCHEMA`, `PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA` and `PRODUCT_CAPABILITIES_REJECTED`; payloads cannot override the schema registry from inside the signed document.

Revocation lookup is a separate fail-closed policy. `REVOKED` maps to trust state `REVOKED` and denial `PRODUCT_CAPABILITIES_REVOKED`; unknown status or unavailable registries map to existing non-VALID fail-closed outcomes. A document cannot provide its own trusted revocation status, and `NOT_REVOKED` still cannot create a VALID snapshot without future real cryptographic signature verification.

## M0.4 FIX: raw entrypoint and strict stage transitions

The test interpreter now has a single public raw-document entrypoint: `validate_raw_document_structure(raw_document)`. It accepts only bytes or a string containing the complete raw JSON document. It decodes UTF-8, parses JSON with duplicate-member rejection, rejects non-finite values and trailing data, and only then calls the internal parsed-document structure helper. Passing a dict directly to the raw entrypoint is rejected so a last-key-wins `json.loads` result cannot be treated as a safely parsed ProductCapabilities document.

Stage results are a closed machine registry. `REJECTED` means the stage ran and denied its own responsibility; `NOT_REACHED` means a prerequisite stage did not accept and the stage did not perform its checks. `CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4` is not an accepted crypto result, so the normal M0.4 path stops before expiry/revocation and snapshot creation.

Expiry and revocation are split into a pipeline gate and a pure future policy evaluator. The gate runs only after `CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED`; the pure evaluator accepts explicit `current_utc` and explicit revocation registry data for future-policy tests but does not create `VALID` or a snapshot.

`SNAPSHOT_CREATED` requires every prerequisite result: `STRUCTURE_ACCEPTED`, `HASHES_ACCEPTED`, `REGISTRY_ACCEPTED`, `CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED`, `EXPIRY_REVOCATION_ACCEPTED`, no mapped non-VALID trust state and no denial code. Any hash, registry, crypto, expiry or revocation rejection leaves `SNAPSHOT_NOT_CREATED`.

`revocation_reference` is strictly either null or `caprev_[A-Za-z0-9_-]{8,64}`. Unknown or invalid revocation registry statuses fail closed to `INVALID_REVOCATION_REGISTRY_STATUS`. A successful `NOT_REVOKED` lookup uses `PRODUCT_CAPABILITIES_REVOCATION_CHECK_PASSED`; `PRODUCT_CAPABILITIES_VALIDATED` is reserved for after future real cryptographic verification, successful expiry/revocation and immutable snapshot creation.

The version registry is an explicit prerequisite. Missing, empty or unknown payload/signature schema registries fail closed to `STRUCTURE_REJECTED` with `MISSING_SCHEMA_VERSION_REGISTRY` or unknown-version diagnostics; the mapped trust state is a fail-closed decision mapping, not a trusted state created by structure validation.

## Raw entrypoint and stage-output ownership

The only public pipeline entrypoint for ProductCapabilities structure validation is `validate_raw_document_structure`. It accepts only `str` or `bytes` raw JSON input. Every other input type, including already parsed `dict` objects, lists, tuples, numeric values, booleans, `None`, mutable byte buffers and custom objects, fails closed with `STRUCTURE_REJECTED` and `RAW_DOCUMENT_TYPE_NOT_SUPPORTED` before parsing or canonicalization starts.

The parsed-document validator is an internal helper used only after the raw parser has enforced UTF-8 decoding, duplicate-member rejection, non-finite-value rejection and trailing-data rejection. It is not a pipeline entrypoint and must not be used as a production bypass around raw-document validation.

Each validation stage owns a closed set of outputs declared by the machine contract. The registry-policy stage can return only `REGISTRY_ACCEPTED`, `REGISTRY_REJECTED` or `REGISTRY_VALIDATION_NOT_REACHED`; it never returns structure-stage or hash-stage results. If it receives a malformed document despite an asserted `HASHES_ACCEPTED` prerequisite, it fails closed with `REGISTRY_REJECTED`, `REGISTRY_INPUT_DOCUMENT_INVALID`, `UNSUPPORTED_SCHEMA` and `PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA`.

The normal M0.4 path remains deterministic: structure and hashes may be accepted, registry lookup may be accepted, real cryptographic verification is still `CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4`, expiry/revocation is therefore `EXPIRY_REVOCATION_NOT_REACHED`, and snapshot creation remains `SNAPSHOT_NOT_CREATED`.


## Edition-bound evidence and snapshot safety

The signed payload is bound to `CRYPTOHUNTER_TESTNET_EDITION`. Its `capability_set` must exactly match the closed current-edition capability registry, its `environment_capabilities` and `feature_flags` must match the current ProductCapabilities policy, and `fail_closed_policy` must remain `SAFE_LOCAL_ONLY`. The document cannot add capability IDs, reorder the canonical capability set, supply future Live-enabled capabilities, or change the trusted edition-policy registry.

Each accepted structure result creates a document-bound validation context containing `validation_context_id`, `capabilities_id`, `edition_id`, `signed_payload_hash`, `capability_set_hash`, `payload_schema_version`, and `signature_schema_version`. Every subsequent stage evidence carries its `stage_id`, `stage_result`, validation context identity, document identity, diagnostics, mapped trust state, and mapped denial code. Later stages reject evidence from another document, another context, a different capability ID, or a different signed payload hash.

The normal M0.4 pipeline has no `force_future_crypto` switch. Future cryptographic acceptance exists only inside the full `_run_future_contract_pipeline` closure after raw structure, hash and registry validation. There is no standalone future issuer and no fixture authority object is passed as a runtime argument.

`SNAPSHOT_CREATED` checks all stage evidence globally: correct stage ownership, accepted prerequisite results, one shared validation context, one signed payload hash, no diagnostic reason, no mapped non-VALID trust state, no denial code, accepted expiry/revocation, future real cryptographic acceptance, and current-edition policy consistency. Any mismatch, trust mapping, denial, diagnostic, or M0.4 `NOT_IMPLEMENTED` crypto result fails closed to `SNAPSHOT_NOT_CREATED`.

Malformed registry inputs fail closed without uncontrolled `TypeError`, `KeyError`, or `AttributeError`: malformed documents use `REGISTRY_INPUT_DOCUMENT_INVALID`, malformed algorithm/key references use `REGISTRY_REFERENCE_TYPE_INVALID`, and malformed registry containers use `REGISTRY_CONTAINER_INVALID`. All map to `UNSUPPORTED_SCHEMA` and `PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA`.

Surrounding JSON whitespace follows the machine raw parse policy: leading whitespace is allowed, trailing whitespace is allowed, and trailing non-whitespace data such as a second JSON document is rejected before canonicalization.

## Document fingerprint evidence binding

M0.4 validation evidence is now bound to a canonical `document_fingerprint`, defined as SHA-256 over the RFC8785/JCS canonical bytes of the complete signed-document envelope: the full `capability_payload`, `signature_header`, `signed_payload_hash`, and `signature`. The fingerprint is not computed from raw JSON bytes, so legal JSON whitespace and member ordering differences do not change the semantic document identity.

`validation_context_id` is derived from a domain-separated hash of that `document_fingerprint`. Stage evidence carries the full context fields directly: `validation_context_id`, `document_fingerprint`, `capabilities_id`, `edition_id`, `signed_payload_hash`, `capability_set_hash`, `payload_schema_version`, and `signature_schema_version`. Each transition rejects prior evidence with diagnostics, trust mappings, denial codes, missing fields, bad field types, wrong stage ownership, or any context mismatch.

Snapshot creation revalidates the final document, not only the edition id. It requires the complete direct stage chain, identical context fields across all evidence, revalidated canonical hashes, current-edition capability-set/environment/feature-flag/source/fail-closed policy consistency, unchanged signature header/signature as represented by the document fingerprint, and no diagnostic, mapped non-VALID trust state, or denial code.

The current source policy is exact: signed payloads use `signed_build_resource_after_validation`, matching `ProductCapabilities.current_edition_capability_policy.source` and `current_edition_signed_payload_policy.source`. Other source values are rejected by structure validation and final snapshot validation.

There are no public or standalone future-crypto acceptance producers. `_future_crypto_accepted_evidence` does not exist after interpreter initialization; future acceptance occurs only inside the private `_run_future_contract_pipeline` closure after the earlier validators succeed. The public M0.4 `run_pipeline` always reaches `CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4`, then `EXPIRY_REVOCATION_NOT_REACHED`, then `SNAPSHOT_NOT_CREATED`.

Strict surrounding JSON whitespace is limited to SPACE, TAB, LF, and CR. Vertical tab, form feed, non-breaking space, em space, a leading BOM, or a second document after legal whitespace are rejected before canonicalization.

## M0.4 FIX: deterministic stage evidence chain

M0.4 treats each validation stage output as document-bound evidence rather
than as a free-form status string. The raw document remains the only public
pipeline entrypoint. After structure acceptance, every stage evidence carries
an evidence schema version, deterministic `stage_evidence_id`, direct
`predecessor_stage_evidence_id`, stage identifier/result, and the complete
validation context fields directly on the evidence object. There is no
unverified nested context copy.

`stage_evidence_id` is SHA-256 over the RFC8785/JCS canonical object made from
its declared evidence-ID fields. The chain is linear: structure has a `null`
predecessor; hashes point to structure; registry points to hashes; crypto
points to registry; expiry/revocation points to crypto; snapshot creation
requires the expiry/revocation evidence. Any mutation of an evidence field,
predecessor ID, context field, diagnostic reason, mapped trust state, or denial
code invalidates the chain and fails closed.

Future cryptographic acceptance is isolated inside the full
`_run_future_contract_pipeline` closure. No public function can produce
arbitrary accepted stage results, and ordinary calls cannot manufacture
`CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED`. The future path first runs raw
structure, hash, and registry validation, then lexically issues future crypto
evidence through `issue_future_crypto(context, registry_evidence)`. No sentinel
token, runtime authority object, or authority argument is involved.
`fixture_availability=TEST_FIXTURE_FUTURE_ONLY` remains only a diagnostic marker
covered by the evidence ID. The normal public M0.4 pipeline still ends with
`CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4`, then
`EXPIRY_REVOCATION_NOT_REACHED`, then `SNAPSHOT_NOT_CREATED`.

The expiry/revocation policy evaluator is private and pure: it returns a policy
decision only, not stage evidence. Only the expiry/revocation stage can turn
that decision into evidence, and only when it receives clean crypto predecessor
evidence. Snapshot creation verifies the full evidence hash chain, the fixture
marker for future crypto acceptance, the final document fingerprint,
canonical-hash revalidation, and current-edition policy consistency before any
future `SNAPSHOT_CREATED` result.

## M0.4 FIX: stage evidence provenance and producer authenticity

The stage evidence hash chain is an integrity and deterministic-identity
mechanism, not a producer-authenticity mechanism. A rehashed dictionary can
prove only that its public fields are internally consistent; it cannot prove
that the structure, hash, registry, crypto fixture, expiry/revocation, or
snapshot stage validator issued that evidence.

Accepted stage evidence is therefore represented in tests by
`_IssuedStageEvidence`, a read-only `Mapping` composition object rather than a
`dict` subclass. It stores only a private copy of the public representation in
`_public_fields`; it does not store an issuer token, does not store a producer
stage ID, and does not store a predecessor reference as private attributes.

Attestation and provenance live in the closure-local registry. Each registry
record stores the exact strong `evidence_ref`, diagnostic `id(evidence)`,
`stage_id`, `stage_result`, full validation context, exact `predecessor_ref`,
and canonical `public_digest`. Validation requires
`meta["evidence_ref"] is evidence`, matching canonical public digest, matching
stage/result/context, the exact predecessor reference, matching public
predecessor ID, and full recursive predecessor-chain validation. The generic
`_stage_result` helper is retained only for rejected, not-reached,
not-implemented, and not-created outputs; it cannot issue `STRUCTURE_ACCEPTED`,
`HASHES_ACCEPTED`, `REGISTRY_ACCEPTED`,
`CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED`, `EXPIRY_REVOCATION_ACCEPTED`,
or `SNAPSHOT_CREATED`.

`fixture_availability=TEST_FIXTURE_FUTURE_ONLY` remains a diagnostic marker and
is included in the stage evidence schema and evidence-ID fields so any marker
change invalidates integrity. It is not authority. Future crypto accepted
evidence is recognized only when the closure-local future pipeline records it
against the exact accepted registry evidence object.

A plain `dict`, `dict(evidence)`, ordinary copy, JSON round-trip, manually
copied public fields, or fully rehashed fabricated chain does not preserve
producer attestation and must fail closed to `SNAPSHOT_NOT_CREATED`. Public APIs
also do not accept issuer tokens, producer attestations, private predecessor
references, arbitrary accepted stage-result strings, or a future-crypto force
switch.

## M0.4 FIX: immutable issued evidence and recursive chain semantics

Issued stage evidence is sealed before it leaves a stage-specific issuer. Public
mapping mutators and private provenance attributes are blocked after issue, so a
previous stage result, diagnostic, trust mapping, denial code, context field, or
predecessor link cannot be edited in place and then rehashed into an accepted
successor.

Accepted evidence is no longer produced by a generic accepted-result factory.
Accepted results are issued only inside the stage validation closures after the
real prerequisite checks succeed. Caller-supplied `stage_result`, issuer
authority, fixture token, construction authority, `producer_stage_id`, or raw
predecessor reference cannot create attestation. Contractual issuer IDs in JSON
are names for audit and tests only; runtime authority objects are not published
in the machine contract and are not passed through a public or generic helper
API.

Recursive chain validation checks the full semantics of every predecessor, not
only wrapper shape and evidence hash. It verifies the exact accepted result for
each stage, clean diagnostic/trust/denial metadata, all validation-context
fields, fixture-marker semantics, public predecessor IDs, and private predecessor
object references back to the structure stage. Direct wrapper construction,
plain dicts, serialized evidence, rehashed forged evidence, and name-prefix
conventions do not create producer attestation.

## M0.4 FIX — sealed evidence factory and realistic provenance boundary

The test interpreter now treats public stage evidence mappings as integrity data,
not as producer authority.  `stage_evidence_id` remains a deterministic SHA-256
identifier over the canonical public fields, but it is not an attestation that a
stage validator produced the evidence.  Accepted evidence is recognized only when
it is issued through the closed validator flow and recorded in a closure-local
attestation registry.

Issued evidence is represented by a read-only `Mapping` composition object rather
than a `dict` subclass.  The public fields are copied when the object is issued,
so `dict(evidence)`, shallow/deep copies, JSON round-trips and manually rehashed
public dictionaries carry no producer attestation.  Python `dict` base-class
mutators cannot target the issued object because it is not a `dict`, and ordinary
mapping mutators are unavailable or fail closed.

The generic sealer, raw stage-specific issuer functions, construction authority
objects, standalone future issuer and runtime authority globals are not part of
the module API.  Issuer authority is not accepted as a function argument and a
direct wrapper constructor cannot create a trusted attestation.  The
future-contract harness is the only isolated path that can exercise future
crypto acceptance; it first runs the real structure, hash and registry validators
and then binds future crypto evidence to the issued registry predecessor.

The closure-local registry binds object identity, canonical public-field digest,
stage identifier, exact accepted result, document context and predecessor object
identity.  This makes ordinary tampering fail closed: if public fields are copied,
serialized, manually rehashed or altered through `object.__setattr__`, recursive
chain validation no longer matches the registry entry and snapshot creation is
rejected.

Threat model: M0.4 covers untrusted ProductCapabilities documents, JSON input,
configuration/CLI/environment/IPC/public API inputs, evidence copying and
serialization, manual evidence-ID recomputation and ordinary construction
attempts through available APIs.  It does not claim to sandbox arbitrary trusted
code execution inside the same Python process, debugger modification of closure
cells, whole-module monkeypatching or raw process-memory modification.  Production
authenticity still requires real cryptography and appropriate process/module
boundaries; the architecture interpreter only models fail-closed provenance
semantics for M0.4 tests.

## M0.4 FINAL FIX — closure-local evidence authority boundary

A module-global evidence runtime object is an issuer escape and is forbidden. The
interpreter no longer exposes `_EVIDENCE_RUNTIME`, `_make_evidence_runtime`, a
future-fixture token, an authority bundle, or any global object with callable
stage-issuer methods such as `structure_validated`, `hashes_validated`,
`registry_validated`, `future_fixture_validated`, `expiry_validated`,
`snapshot_validated`, `seal`, `issue`, or `issue_accepted`.

The final interpreter is initialized once into a shared closure. That closure owns
the attestation registry, seal/issue functions, future-fixture authority, chain
validator and snapshot issuer, but it returns no runtime object, token, issuer
bundle or accepted-evidence factory. Only the validation functions and read-only
helpers are assigned as module globals.

Attestation is bound to the exact current evidence object. Each registry entry
stores a strong `evidence_ref`, the exact `predecessor_ref`, the stage/result,
full validation context and canonical public digest. Validation requires
`meta["evidence_ref"] is evidence`; numeric `id()` values are diagnostic only and
cannot substitute for object identity. Directly constructed wrappers, public
copies with identical digests, deepcopy, JSON round-trips and `object.__setattr__`
tampering remain fail-closed, while the isolated future harness still creates a
future snapshot only through the full issued chain.

## M0.4 FINAL CLOSURE FIX — no standalone future issuer

Future crypto acceptance is no longer exposed as `_future_crypto_accepted_evidence`
or any other standalone callable. It can occur only inside the closure-local
implementation of `_run_future_contract_pipeline`, after that pipeline has run raw
structure validation, canonical hash validation and registry policy validation.
The same closure owns the future-fixture authority and never returns it.

The one-time bootstrap removes itself from module globals after installing the
final functions. After import, globals must not contain `_initialize_closure_local_interpreter`,
`_future_crypto_accepted_evidence`, `_EVIDENCE_RUNTIME`, `_make_evidence_runtime`
or `_FUTURE_CRYPTO_FIXTURE_TOKEN`, and no global object may expose callable issuer
methods. This keeps normal M0.4 on the not-implemented crypto path while allowing
only the isolated future-contract pipeline to reach a future `SNAPSHOT_CREATED`.

The closure-local future fixture is an execution path, not a materialized authority token. `issue_future_crypto` has the data-only signature `issue_future_crypto(context, registry_evidence)` inside the closure; it does not accept `authority`, `fixture_token`, `issuer_token`, `construction_authority` or similar authority parameters.
