"""Pure executable M0.10 authority model; it implements no runtime or durability."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE_PATH = DOCS / "identity_device_authentication_and_secrets.json"
UUID7 = "01890f3a-2b4c-7abc-8def-0123456789ab"
ACCOUNT = "acct_" + UUID7
OPERATOR = "op_" + UUID7
DEVICE = "dev_" + UUID7
DEVICE_2 = "dev_01890f3a-2b4c-7abc-9def-0123456789ab"
SESSION = "run_" + UUID7
CREDENTIAL = "cred_" + UUID7
EXCHANGE_ACCOUNT = "xacc_" + UUID7
LIVE_GRANT = "lgrant_" + UUID7
FP_A, FP_B, FP_C = "a" * 64, "b" * 64, "c" * 64
T0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class OperatorIdentitySecurityProjection:
    account_id: Any
    operator_id: Any
    state: Any
    identity_revision: Any
    security_generation: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class DeviceTrustProjection:
    account_id: Any
    device_installation_id: Any
    state: Any
    trust_revision: Any
    security_generation: Any
    platform_enrollment_revision: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class PinVerifierRecord:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    algorithm_id: Any
    parameter_policy_version: Any
    salt_reference: Any
    verifier: Any
    pin_revision: Any
    failed_attempts: Any
    lockout_until_utc: Any
    security_generation: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class PlatformBiometricAssertion:
    account_id: Any
    device_installation_id: Any
    platform_authenticator_source: Any
    platform_enrollment_revision: Any
    challenge_fingerprint_sha256: Any
    outcome: Any
    verified_at_utc: Any
    expires_at_utc: Any
    assertion_fingerprint_sha256: Any


@dataclass(frozen=True)
class AuthenticationProof:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    factor_set: Any
    issued_at_utc: Any
    expires_at_utc: Any
    identity_revision: Any
    device_trust_revision: Any
    pin_revision: Any
    platform_enrollment_revision: Any
    security_generation: Any
    session_generation: Any
    environment: Any
    operation: Any
    scope_fingerprint_sha256: Any
    mutation_fingerprint_sha256: Any
    causation_id: Any
    correlation_id: Any
    proof_fingerprint_sha256: Any


@dataclass(frozen=True)
class CoreIssuedAuthenticationProofBinding:
    proof_fingerprint_sha256: Any
    complete_proof_content_fingerprint_sha256: Any
    authority_source: Any
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    identity_revision: Any
    device_trust_revision: Any
    pin_revision: Any
    platform_enrollment_revision: Any
    security_generation: Any
    session_generation: Any


@dataclass(frozen=True)
class SessionSecurityState:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    runtime_session_id: Any
    state: Any
    session_generation: Any
    security_generation: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class OperationEntitlementProjection:
    account_id: Any
    operator_id: Any
    operation: Any
    environment: Any
    authorization_scope: Any
    entitlement_revision: Any
    security_generation: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class SecretMetadataProjection:
    secret_reference: Any
    secret_kind: Any
    exchange_account_id: Any
    credential_profile_id: Any
    exchange_id: Any
    environment: Any
    permitted_operations: Any
    secret_revision: Any
    state: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class LiveAccessGrantSecurityProjection:
    live_access_grant_id: Any
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    policy_scope_fingerprint_sha256: Any
    state: Any
    grant_revision: Any
    security_generation: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class InitialSecurityState:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    security_generation: Any
    session_generation: Any
    state: Any
    bootstrap_claim_fingerprint_sha256: Any
    content_fingerprint_sha256: Any


@dataclass(frozen=True)
class InitialSecurityEstablishmentResult:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    security_generation: Any
    session_generation: Any
    bootstrap_claim_fingerprint_sha256: Any
    result: Any
    result_fingerprint_sha256: Any


@dataclass(frozen=True)
class CoreAcceptedPlatformBiometricAssertionBinding:
    assertion_fingerprint_sha256: Any
    complete_assertion_content_fingerprint_sha256: Any
    authority_source: Any
    account_id: Any
    device_installation_id: Any
    platform_enrollment_revision: Any
    challenge_fingerprint_sha256: Any


@dataclass(frozen=True)
class M03BootstrapAuthorityView:
    claim_fingerprint_sha256: Any
    account_id: Any
    device_installation_id: Any
    operator_id: Any
    bootstrap_generation: Any
    bootstrap_revision: Any
    purpose: Any
    pre_state_fingerprint_sha256: Any
    post_state_fingerprint_sha256: Any
    consumed_authority_fingerprint_sha256: Any
    consumed_claim_fingerprint_sha256: Any
    consumed_challenge_fingerprint_sha256: Any


@dataclass(frozen=True)
class M03AcceptedBootstrapAuthorityBinding:
    view_fingerprint_sha256: Any
    complete_view_content_fingerprint_sha256: Any
    claim_fingerprint_sha256: Any
    account_id: Any
    device_installation_id: Any
    operator_id: Any
    bootstrap_generation: Any
    bootstrap_revision: Any
    purpose: Any
    accepted_pre_fingerprint_sha256: Any
    current_post_fingerprint_sha256: Any
    consumed_authority_fingerprint_sha256: Any
    consumed_claim_fingerprint_sha256: Any
    consumed_challenge_fingerprint_sha256: Any
    authority_source: Any


M010_AUTHORITY_DATACLASSES = (
    OperatorIdentitySecurityProjection,
    DeviceTrustProjection,
    PinVerifierRecord,
    PlatformBiometricAssertion,
    AuthenticationProof,
    CoreIssuedAuthenticationProofBinding,
    SessionSecurityState,
    OperationEntitlementProjection,
    SecretMetadataProjection,
    LiveAccessGrantSecurityProjection,
    InitialSecurityState,
    InitialSecurityEstablishmentResult,
    CoreAcceptedPlatformBiometricAssertionBinding,
    M03BootstrapAuthorityView,
    M03AcceptedBootstrapAuthorityBinding,
)

EXPECTED_PROTOCOL_LITERAL = {
    "schema_version": "1.0.0",
    "m0_element": "M0.10",
    "status": "closed",
    "authority": {
        "owner": "CoreHost",
        "integrity_is_authority": False,
        "authority_rule": "accepted/current Core membership; fingerprints prove content "
        "integrity only",
        "non_authorities": [
            "UI",
            "TrayAgent",
            "DesktopShell",
            "raw caller",
            "caller boolean",
            "caller role",
            "nominal dataclass/class",
            "self-hash",
            "PIN input",
            "caller factor_set",
            "caller biometric bool/string",
            "RuntimeSession existence",
            "DeviceInstallation identity alone",
            "LiveAccessGrant presence alone",
        ],
        "current_designation": "Core-owned scope-to-accepted-fingerprint maps only; "
        "is_current content never establishes authority",
        "public_authorization_inputs": [
            "untrusted AuthenticationProof",
            "untrusted exact authorization request",
            "now_utc",
        ],
        "public_authorization_forbidden_inputs": [
            "Core-issued binding",
            "current authority objects or registries",
            "entitlement boolean",
            "caller roles or approval booleans",
        ],
        "acceptance_plane": {
            "public_self_enrollment_allowed": False,
            "bootstrap_seed_from_shape_allowed": False,
            "platform_seed_from_shape_allowed": False,
            "entitlement_caller_acceptance_allowed": False,
            "secret_caller_acceptance_allowed": False,
            "live_grant_caller_acceptance_allowed": False,
            "trusted_seed_boundary": "module-private architecture-test "
            "harness represents authority already "
            "accepted by M0.3, external platform "
            "or M0.5; never product API",
        },
        "security_generation_coherence": "identity == device == PIN == session == "
        "entitlement == proof for exact authorization "
        "scope; inconsistent Core bundle is "
        "CONTRACT_INCONSISTENT",
        "canonical_time_boundary": "every public time path requires timezone-aware datetime "
        "with UTC offset exactly zero; naive and non-UTC inputs "
        "fail closed",
        "security_generation_monotonicity": "current OperatorIdentity and DeviceTrust "
        "security_generation are monotonic "
        "non-decreasing per scope; rollback is denied "
        "and cannot resurrect historical grant/proof "
        "authority",
    },
    "dependency_manifest": [
        {
            "contract": "canonical_domain_vocabulary.json",
            "json_pointer": "/entity_kinds",
            "expected_content_fingerprint_sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1",
        },
        {
            "contract": "canonical_domain_vocabulary.json",
            "json_pointer": "/relationships",
            "expected_content_fingerprint_sha256": "03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9",
        },
        {
            "contract": "canonical_domain_vocabulary.json",
            "json_pointer": "/identifier_policy",
            "expected_content_fingerprint_sha256": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
        },
        {
            "contract": "canonical_domain_vocabulary.json",
            "json_pointer": "/entity_kinds/3",
            "expected_content_fingerprint_sha256": "32abdbb5aa443ef2ac55391f3be61a766ffe086940503209f9e49733943fa8a1",
        },
        {
            "contract": "process_topology_and_lifecycle.json",
            "json_pointer": "/first_run_bootstrap_authority_contract",
            "expected_content_fingerprint_sha256": "df90a7d0d6972db16fee67bea285c60758e75615c2f583b57e5a62b0f7df752b",
        },
        {
            "contract": "environment_and_product_capabilities.json",
            "json_pointer": "/ProductCapabilities",
            "expected_content_fingerprint_sha256": "fd8cd8b62827b96d52bd6870985f7e2438a0306cd267048b7745472157877605",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/exchange_account_contract",
            "expected_content_fingerprint_sha256": "f1f7b18513a53d107b6effe169e793e74f5c70c8df6261a25ee96a3012c51ed4",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/credential_profile_contract",
            "expected_content_fingerprint_sha256": "d0f8b6213cd4388f097a2c16f27edc2bd304b36f0118ab0c834088e210432a2a",
        },
        {
            "contract": "strategy_market_data_and_execution_routing.json",
            "json_pointer": "/live_execution_authority_policy",
            "expected_content_fingerprint_sha256": "eb2034a4691a18f3037b2a5f93d1fb41fcb38ef67c6c43c5dfff891dd35fe1de",
        },
        {
            "contract": "commands_events_order_lifecycle_and_idempotency.json",
            "json_pointer": "/event_contract",
            "expected_content_fingerprint_sha256": "c63d514a4546161798de2f7f90441821cd71653fba433c3577c6d7b630e4b6b2",
        },
        {
            "contract": "ledger_portfolio_capital_and_pnl.json",
            "json_pointer": "/reservation_protocol",
            "expected_content_fingerprint_sha256": "b3579e387a02b9f291468f5cc89a9ed650e4024a538cbb7f6f0c0818ed0c4641",
        },
        {
            "contract": "risk_hierarchy_kill_switch_and_execution_lease.json",
            "json_pointer": "/m010_boundary",
            "expected_content_fingerprint_sha256": "7a3df73176cb2fc9c18fd77a4dbf6182f62a5406d55ccad8b609bd3708ac12d9",
        },
    ],
    "bootstrap": {
        "purpose": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        "direct_dependency_pointer": "/first_run_bootstrap_authority_contract",
        "requires_genuine_accepted_current_m03_transition": True,
        "manual_or_self_hashed_result_authority": False,
        "one_shot": True,
        "terminally_fenced_after_establishment": True,
        "may_establish": [
            "first current OperatorIdentity",
            "initial current TRUSTED device",
            "first PinVerifierRecord",
            "initial security generation",
            "initial session security state",
        ],
        "never_authorizes": [
            "second device",
            "ordinary privileged operation",
            "RiskPolicy",
            "kill switch",
            "ProductCapabilities",
            "secret rotation",
            "ExecutionLease",
            "LIVE",
        ],
        "executable_consumer": "M03BootstrapAuthorityView resolves accepted historical PRE, "
        "accepted/current POST and exact consumed authority; "
        "transition creates all initial accepted/current M0.10 "
        "projections and atomically fences one-shot authority",
        "semantic_atomicity": "prepare and validate complete "
        "identity/device/PIN/session/initial POST in shadow semantic "
        "state; publish all plus bootstrap consumption together, or "
        "publish zero",
        "bridge_membership": "pre-existing M03AcceptedBootstrapAuthorityBinding from "
        "trusted M0.3 bridge; binds complete view, PRE, current POST, "
        "consumed authority, consumed claim/challenge, exact "
        "generation/revision/scope and upstream source",
        "bridge_semantic_validation": "view and binding validate canonical identifiers, "
        "positive non-bool generation/revision, initial-only "
        "purpose, lowercase fingerprints, exact PRE/current "
        "POST/consumed authority/claim/challenge and external "
        "provisioning source; recomputed malformed pairs "
        "denied",
        "consumed_challenge_source": "explicit module-private "
        "_M03TrustedConsumedBootstrapEvidence supplied before "
        "projection; view is derived from evidence; no local "
        "claim/challenge/authority default or derivation "
        "exists",
    },
    "registries": {
        "identity_states": ["ACTIVE", "REVOKED"],
        "device_trust_states": ["ENROLLED_UNTRUSTED", "TRUSTED", "REVOKED", "REPLACED"],
        "session_states": ["LOCKED", "UNLOCKED", "LOGGED_OUT"],
        "biometric_outcomes": ["SUCCESS", "FAILED", "CANCELLED", "UNAVAILABLE"],
        "factors": ["PIN", "BIOMETRIC"],
        "factor_policies": ["PIN", "BIOMETRIC", "PIN_AND_BIOMETRIC"],
        "environments": ["PAPER", "TESTNET", "LIVE"],
        "secret_kinds": ["API_KEY", "API_SECRET", "PASSPHRASE", "PRIVATE_KEY"],
        "secret_states": ["AVAILABLE", "ROTATED", "REVOKED", "REPLACED"],
        "grant_states": ["ACTIVE", "SUSPENDED", "REVOKED"],
        "failure_codes": [
            "MALFORMED_UNTRUSTED_CONTEXT",
            "IDENTITY_INVALID",
            "IDENTITY_REVOKED",
            "DEVICE_NOT_TRUSTED",
            "DEVICE_REVOKED",
            "AUTHENTICATION_REQUIRED",
            "AUTHENTICATION_FAILED",
            "FACTOR_UNAVAILABLE",
            "PIN_LOCKED",
            "PROOF_EXPIRED",
            "PROOF_STALE",
            "AUTHORIZATION_DENIED",
            "SECRET_INVALID",
            "SECRET_UNAVAILABLE",
            "SECRET_REVOKED",
            "SECRET_STALE",
            "OPERATION_UNSUPPORTED",
            "CONTRACT_INCONSISTENT",
        ],
        "device_trust_transition_graph": {
            "TRUST_DEVICE": ["ABSENT->TRUSTED", "ENROLLED_UNTRUSTED->TRUSTED"],
            "REVOKE_DEVICE": ["TRUSTED->REVOKED"],
            "terminal_states": ["REVOKED", "REPLACED"],
            "denied": ["TRUSTED->TRUSTED", "REVOKED->TRUSTED", "REPLACED->TRUSTED"],
        },
        "identity_transition_rule": "REVOKED is terminal for the same OperatorIdentity; "
        "ACTIVE cannot be restored for that identity",
        "secret_use_operation_registry": ["PRIVATE_DATA", "ORDER_ENTRY"],
    },
    "operation_policy_registry": {
        "TRUST_DEVICE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "trust_device",
            "environments": ["PAPER", "TESTNET"],
        },
        "REVOKE_DEVICE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "revoke_device",
            "environments": ["PAPER", "TESTNET"],
        },
        "SETUP_PIN": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "setup_pin",
            "environments": ["PAPER", "TESTNET"],
        },
        "CHANGE_PIN": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "change_pin",
            "environments": ["PAPER", "TESTNET"],
        },
        "RESET_PIN": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "reset_pin",
            "environments": ["PAPER", "TESTNET"],
        },
        "LOCK_SESSION": {
            "factor_policy": "PIN",
            "freshness_seconds": 60,
            "authorization_scope": "lock_session",
            "environments": ["PAPER", "TESTNET"],
        },
        "UNLOCK_SESSION": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "unlock_session",
            "environments": ["PAPER", "TESTNET"],
        },
        "LOGOUT_SESSION": {
            "factor_policy": "PIN",
            "freshness_seconds": 60,
            "authorization_scope": "logout_session",
            "environments": ["PAPER", "TESTNET"],
        },
        "ROTATE_SECRET_REFERENCE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "rotate_secret_reference",
            "environments": ["PAPER", "TESTNET"],
        },
        "REBIND_SECRET_REFERENCE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "rebind_secret_reference",
            "environments": ["PAPER", "TESTNET"],
        },
        "ACTIVATE_CREDENTIAL_PROFILE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "activate_credential_profile",
            "environments": ["PAPER", "TESTNET"],
        },
        "DEACTIVATE_CREDENTIAL_PROFILE": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "deactivate_credential_profile",
            "environments": ["PAPER", "TESTNET"],
        },
        "CHANGE_RISK_POLICY": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "change_risk_policy",
            "environments": ["PAPER", "TESTNET"],
        },
        "CHANGE_KILL_SWITCH": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "change_kill_switch",
            "environments": ["PAPER", "TESTNET"],
        },
        "CHANGE_PRODUCT_CAPABILITIES": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "change_product_capabilities",
            "environments": ["PAPER", "TESTNET"],
        },
        "GRANT_LIVE_ACCESS": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "grant_live_access",
            "environments": ["LIVE"],
        },
        "SUSPEND_LIVE_ACCESS": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "suspend_live_access",
            "environments": ["LIVE"],
        },
        "REVOKE_LIVE_ACCESS": {
            "factor_policy": "PIN_AND_BIOMETRIC",
            "freshness_seconds": 60,
            "authorization_scope": "revoke_live_access",
            "environments": ["LIVE"],
        },
    },
    "pin_policy": {
        "reference_algorithm": "M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF",
        "max_failed_attempts": 3,
        "lockout_seconds": 300,
        "raw_pin_serialized": False,
        "bool_as_int_rejected": True,
        "change_or_reset_increments_pin_revision": True,
        "success_after_lockout_expiry": "successful comparison resets failed_attempts to "
        "zero and lockout_until_utc to null by a new "
        "accepted/current record without changing "
        "pin_revision",
    },
    "biometric_policy": {
        "stores_biometric_material": False,
        "acceptance": [
            "SUCCESS",
            "exact account/device",
            "current enrollment revision",
            "expected challenge",
            "verified_at_utc <= now_utc <= expires_at_utc",
        ],
        "acceptance_authority": "pre-existing external-platform accepted membership "
        "represented by "
        "CoreAcceptedPlatformBiometricAssertionBinding; "
        "nominal SUCCESS and self-hash denied",
        "trusted_seed_boundary": "module-private fixture represents pre-existing "
        "external platform acceptance; public Core API only "
        "consumes membership",
        "expected_challenge": "Core-derived canonical fingerprint of "
        "account/operator/device/environment/operation/scope/mutation/causation/correlation/enrollment/security/session; "
        "caller cannot supply expected challenge",
    },
    "proof_policy": {
        "factor_set": "derived Core output only",
        "membership": "pre-existing CoreIssuedAuthenticationProofBinding",
        "self_hash_authority": False,
        "time_rule": "issued_at_utc <= now_utc <= expires_at_utc; expires_at_utc > "
        "issued_at_utc; freshness age <= operation freshness_seconds "
        "(inclusive)",
        "always_revalidate_current": [
            "identity",
            "device trust",
            "PIN revision",
            "biometric enrollment revision",
            "security generation",
            "session generation",
            "environment",
            "operation",
            "account",
            "scope",
            "mutation",
            "causation",
            "correlation",
        ],
        "issuance": "Core resolves all accepted/current registries, validates operation "
        "policy and factor evidence, derives factor_set, emits proof and "
        "records exact CoreIssuedAuthenticationProofBinding",
        "binding_validation": "registry key and every ordered binding field exact; "
        "authority_source exactly CoreHost",
    },
    "secret_reference_policy": {
        "upstream_pointer": "/credential_profile_contract/secure_store_reference_grammar",
        "prefix": "secure-store://",
        "locator": "non-empty opaque string",
        "forbidden_characters": ["whitespace", "?", "#", "="],
        "forbidden_payload_markers": [
            "api_key",
            "apikey",
            "secret",
            "password",
            "token",
            "private_key",
            "credential_value",
            "plaintext",
        ],
        "keyring_scheme_result": "SECRET_INVALID",
        "raw_secret_in_domain_record": False,
        "current_authority": "accepted SecretMetadataProjection plus "
        "canonical Core-owned "
        "scope-to-accepted-fingerprint map for exact "
        "revision",
        "upstream_seed_boundary": "module-private fixture represents already "
        "accepted/current M0.5 projection; public "
        "Core API only validates use; ROTATE/REBIND "
        "only return authorized security request to "
        "M0.5 owner",
    },
    "live_policy": {
        "target_architecture_supports_live": True,
        "current_live": "DENIED_BY_UPSTREAM_POLICY",
        "testnet_to_live_fallback": False,
        "security_alone_enables_live": False,
        "grant_alone_enables_live": False,
        "future_required_gates": [
            "M0.4 ProductCapabilities",
            "LiveAccessGrant",
            "M0.5 account/credential",
            "M0.6 readiness/routing",
            "M0.7 execution/event authority",
            "M0.8 accounting/reservation",
            "M0.9 RiskDecision/ExecutionLease",
            "M0.10 security",
        ],
        "grant_authority": "accepted LiveAccessGrantSecurityProjection plus canonical "
        "Core-owned scope-to-accepted-fingerprint map; nominal payload "
        "denied",
        "grant_transitions": "GRANT_LIVE_ACCESS, SUSPEND_LIVE_ACCESS and "
        "REVOKE_LIVE_ACCESS are M0.10-owned authorized transitions; "
        "no public acceptance method",
        "current_designation_key": [
            "account_id",
            "device_installation_id",
            "policy_scope_fingerprint_sha256",
        ],
        "grant_identity_history": "live_access_grant_id identity cannot be reparented; "
        "account/device/policy scope and operator parent remain "
        "immutable; security_generation is the authorization "
        "epoch of each projection, not immutable grant identity "
        "lineage",
        "transition_predecessors": {
            "GRANT_LIVE_ACCESS": ["ABSENT"],
            "SUSPEND_LIVE_ACCESS": ["ACTIVE"],
            "REVOKE_LIVE_ACCESS": ["ACTIVE", "SUSPENDED"],
        },
        "one_current_rule": "at most one current ACTIVE grant per account/device/policy "
        "scope; second ACTIVE identity for occupied scope denied; "
        "accepted historical projections remain fenced",
        "security_generation_epoch": "each new projection binds current coherent "
        "identity/device/proof generation; historical "
        "older-generation projection is stale; generation "
        "change never reactivates or self-mints grant "
        "authority",
        "non_resurrection": "once parent is terminally revoked/replaced or authorization "
        "epoch advances, historical grant authority never becomes "
        "active again through later value equality; fresh authority "
        "requires a new legal grant transition on legal parents",
        "grant_id_lifecycle": {
            "NEW_UNSEEN_ID": "ACTIVE revision 1",
            "ACTIVE": "SUSPENDED or REVOKED",
            "SUSPENDED": "REVOKED",
            "REVOKED": "terminal; ID reuse denied",
        },
        "scope_eligibility": {
            "ABSENT": "fresh unseen grant ID allowed",
            "ACTIVE": "new grant denied",
            "SUSPENDED": "new grant denied",
            "REVOKED": "fresh distinct unseen grant ID allowed",
        },
        "history_preservation": "all accepted projections remain immutable history; "
        "current scope designation may move from revoked G1 to "
        "fresh active G2 without rewriting G1",
    },
    "audit_safe_payload": {
        "allowed_categories": [
            "canonical IDs",
            "opaque secure-store references",
            "fingerprints",
            "reason codes",
            "causation_id",
            "correlation_id",
        ],
        "forbidden_categories": [
            "raw PIN",
            "verifier",
            "biometric material",
            "API secret",
            "private key",
            "passphrase",
            "bearer token",
            "plaintext secure-store payload",
        ],
    },
    "identifier_policy": {
        "prefixes": ["acct_", "op_", "dev_", "cred_", "xacc_", "lgrant_"],
        "uuid_version": 7,
    },
    "canonical_integrity_fingerprint_policy": {
        "algorithm": "SHA-256",
        "input_shape": "EXACT_CLOSED_JSON_OBJECT_OF_EXPLICIT_SEMANTIC_INPUT_FIELDS",
        "json_canonicalization": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "encoding": "UTF-8",
        "array_policy": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER; "
        "NEVER_SORT_UNLESS_OWNING_CONTRACT_REQUIRES",
        "object_key_policy": "KEY_ORDER_NON_SEMANTIC; SORT_KEYS_CANONICALIZES",
        "number_policy": "NO_FLOAT_COERCION; VALIDATE_OWNING_FIELD_CONTRACT_BEFORE_HASHING",
        "decimal_string_policy": "VALIDATE_CANONICAL_OWNING_FIELD_CONTRACT; "
        "NO_FINGERPRINT_LAYER_NORMALIZATION",
        "timestamp_policy": "VALIDATE_CANONICAL_OWNING_TIMESTAMP_CONTRACT; "
        "NO_FINGERPRINT_LAYER_NORMALIZATION",
        "unicode_policy": "HASH_EXACT_VALIDATED_STRINGS; NO_HIDDEN_NFC_OR_NFD_TRANSFORMATION",
        "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
        "validation": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
        "authority_boundary": "INTEGRITY_ONLY; "
        "NEVER_CREATES_ACCEPTED_MEMBERSHIP, "
        "CURRENT_AUTHORITY, "
        "LIVE_READINESS, OR "
        "SELF_ENROLLMENT",
    },
    "executable_boundary_terminal_fingerprints": {
        "SessionSecurityState": {
            "field": "content_fingerprint_sha256",
            "algorithm": "SHA-256",
            "input_fields": [
                "account_id",
                "operator_id",
                "device_installation_id",
                "runtime_session_id",
                "state",
                "session_generation",
                "security_generation",
            ],
            "excluded_fields": ["content_fingerprint_sha256"],
            "input_shape": "JSON_OBJECT",
            "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
            "canonicalization": {
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": False,
                "allow_nan": False,
            },
            "encoding": "UTF-8",
            "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
            "unicode_normalization": "NONE",
            "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
            "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
            "authority_boundary": "INTEGRITY_ONLY; "
            "DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
            "exact_fields_source_pointer": "/executable_boundary_schemas/SessionSecurityState",
        },
        "SecretMetadataProjection": {
            "field": "content_fingerprint_sha256",
            "algorithm": "SHA-256",
            "input_fields": [
                "secret_reference",
                "secret_kind",
                "exchange_account_id",
                "credential_profile_id",
                "exchange_id",
                "environment",
                "permitted_operations",
                "secret_revision",
                "state",
            ],
            "excluded_fields": ["content_fingerprint_sha256"],
            "input_shape": "JSON_OBJECT",
            "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
            "canonicalization": {
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": False,
                "allow_nan": False,
            },
            "encoding": "UTF-8",
            "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
            "unicode_normalization": "NONE",
            "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
            "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
            "authority_boundary": "INTEGRITY_ONLY; "
            "DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
            "exact_fields_source_pointer": "/executable_boundary_schemas/SecretMetadataProjection",
        },
    },
    "executable_boundary_schemas": {
        "OperatorIdentitySecurityProjection": [
            "account_id",
            "operator_id",
            "state",
            "identity_revision",
            "security_generation",
            "content_fingerprint_sha256",
        ],
        "DeviceTrustProjection": [
            "account_id",
            "device_installation_id",
            "state",
            "trust_revision",
            "security_generation",
            "platform_enrollment_revision",
            "content_fingerprint_sha256",
        ],
        "PinVerifierRecord": [
            "account_id",
            "operator_id",
            "device_installation_id",
            "algorithm_id",
            "parameter_policy_version",
            "salt_reference",
            "verifier",
            "pin_revision",
            "failed_attempts",
            "lockout_until_utc",
            "security_generation",
            "content_fingerprint_sha256",
        ],
        "PlatformBiometricAssertion": [
            "account_id",
            "device_installation_id",
            "platform_authenticator_source",
            "platform_enrollment_revision",
            "challenge_fingerprint_sha256",
            "outcome",
            "verified_at_utc",
            "expires_at_utc",
            "assertion_fingerprint_sha256",
        ],
        "AuthenticationProof": [
            "account_id",
            "operator_id",
            "device_installation_id",
            "factor_set",
            "issued_at_utc",
            "expires_at_utc",
            "identity_revision",
            "device_trust_revision",
            "pin_revision",
            "platform_enrollment_revision",
            "security_generation",
            "session_generation",
            "environment",
            "operation",
            "scope_fingerprint_sha256",
            "mutation_fingerprint_sha256",
            "causation_id",
            "correlation_id",
            "proof_fingerprint_sha256",
        ],
        "CoreIssuedAuthenticationProofBinding": [
            "proof_fingerprint_sha256",
            "complete_proof_content_fingerprint_sha256",
            "authority_source",
            "account_id",
            "operator_id",
            "device_installation_id",
            "identity_revision",
            "device_trust_revision",
            "pin_revision",
            "platform_enrollment_revision",
            "security_generation",
            "session_generation",
        ],
        "SessionSecurityState": {
            "exact_fields": [
                "account_id",
                "operator_id",
                "device_installation_id",
                "runtime_session_id",
                "state",
                "session_generation",
                "security_generation",
                "content_fingerprint_sha256",
            ],
            "terminal_fingerprint": {
                "field": "content_fingerprint_sha256",
                "algorithm": "SHA-256",
                "input_fields": [
                    "account_id",
                    "operator_id",
                    "device_installation_id",
                    "runtime_session_id",
                    "state",
                    "session_generation",
                    "security_generation",
                ],
                "excluded_fields": ["content_fingerprint_sha256"],
                "input_shape": "JSON_OBJECT",
                "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
                "canonicalization": {
                    "sort_keys": True,
                    "separators": [",", ":"],
                    "ensure_ascii": False,
                    "allow_nan": False,
                },
                "encoding": "UTF-8",
                "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
                "unicode_normalization": "NONE",
                "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
                "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
                "authority_boundary": "INTEGRITY_ONLY; "
                "DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
                "exact_fields_source_pointer": "/executable_boundary_schemas/SessionSecurityState",
            },
        },
        "OperationEntitlementProjection": [
            "account_id",
            "operator_id",
            "operation",
            "environment",
            "authorization_scope",
            "entitlement_revision",
            "security_generation",
            "content_fingerprint_sha256",
        ],
        "SecretMetadataProjection": {
            "exact_fields": [
                "secret_reference",
                "secret_kind",
                "exchange_account_id",
                "credential_profile_id",
                "exchange_id",
                "environment",
                "permitted_operations",
                "secret_revision",
                "state",
                "content_fingerprint_sha256",
            ],
            "terminal_fingerprint": {
                "field": "content_fingerprint_sha256",
                "algorithm": "SHA-256",
                "input_fields": [
                    "secret_reference",
                    "secret_kind",
                    "exchange_account_id",
                    "credential_profile_id",
                    "exchange_id",
                    "environment",
                    "permitted_operations",
                    "secret_revision",
                    "state",
                ],
                "excluded_fields": ["content_fingerprint_sha256"],
                "input_shape": "JSON_OBJECT",
                "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
                "canonicalization": {
                    "sort_keys": True,
                    "separators": [",", ":"],
                    "ensure_ascii": False,
                    "allow_nan": False,
                },
                "encoding": "UTF-8",
                "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
                "unicode_normalization": "NONE",
                "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
                "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
                "authority_boundary": "INTEGRITY_ONLY; "
                "DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY",
                "exact_fields_source_pointer": "/executable_boundary_schemas/SecretMetadataProjection",
            },
            "field_schemas": {
                "permitted_operations": {
                    "type": "canonical_unique_array_of_enum",
                    "items_source_pointer": "/registries/secret_use_operation_registry",
                    "min_items": 1,
                    "unique": True,
                    "canonical_order": "REGISTRY_ORDER",
                    "validator_behavior": "REJECT_NON_CANONICAL_ORDER_OR_DUPLICATES_NEVER_SORT_OR_DEDUPLICATE",
                    "validate_before_terminal_fingerprint": True,
                    "domain_separation": "SECRET_USE_OPERATIONS_NOT_ADMIN_OPERATION_POLICY_OR_M05_CREDENTIAL_PERMISSIONS",
                    "authority_boundary": "MEMBERSHIP_IN_INTRINSIC_METADATA_DOES_NOT_ESTABLISH_CURRENT_ACCEPTED_SECRET_AUTHORITY",
                }
            },
        },
        "LiveAccessGrantSecurityProjection": [
            "live_access_grant_id",
            "account_id",
            "operator_id",
            "device_installation_id",
            "policy_scope_fingerprint_sha256",
            "state",
            "grant_revision",
            "security_generation",
            "content_fingerprint_sha256",
        ],
        "InitialSecurityState": [
            "account_id",
            "operator_id",
            "device_installation_id",
            "security_generation",
            "session_generation",
            "state",
            "bootstrap_claim_fingerprint_sha256",
            "content_fingerprint_sha256",
        ],
        "InitialSecurityEstablishmentResult": [
            "account_id",
            "operator_id",
            "device_installation_id",
            "security_generation",
            "session_generation",
            "bootstrap_claim_fingerprint_sha256",
            "result",
            "result_fingerprint_sha256",
        ],
        "CoreAcceptedPlatformBiometricAssertionBinding": [
            "assertion_fingerprint_sha256",
            "complete_assertion_content_fingerprint_sha256",
            "authority_source",
            "account_id",
            "device_installation_id",
            "platform_enrollment_revision",
            "challenge_fingerprint_sha256",
        ],
        "M03BootstrapAuthorityView": [
            "claim_fingerprint_sha256",
            "account_id",
            "device_installation_id",
            "operator_id",
            "bootstrap_generation",
            "bootstrap_revision",
            "purpose",
            "pre_state_fingerprint_sha256",
            "post_state_fingerprint_sha256",
            "consumed_authority_fingerprint_sha256",
            "consumed_claim_fingerprint_sha256",
            "consumed_challenge_fingerprint_sha256",
        ],
        "M03AcceptedBootstrapAuthorityBinding": [
            "view_fingerprint_sha256",
            "complete_view_content_fingerprint_sha256",
            "claim_fingerprint_sha256",
            "account_id",
            "device_installation_id",
            "operator_id",
            "bootstrap_generation",
            "bootstrap_revision",
            "purpose",
            "accepted_pre_fingerprint_sha256",
            "current_post_fingerprint_sha256",
            "consumed_authority_fingerprint_sha256",
            "consumed_claim_fingerprint_sha256",
            "consumed_challenge_fingerprint_sha256",
            "authority_source",
        ],
    },
    "deferred_to_m011": ["persistence", "durable atomicity", "migrations", "backup", "recovery"],
    "core_owned_registry_semantics": {
        "durability": "deferred to M0.11; pure semantic registries only",
        "accepted_registries": [
            "operator identities",
            "device trust projections",
            "PIN verifier records",
            "session security states",
            "operation entitlements",
            "secret metadata",
            "LIVE grant projections",
            "initial security states",
            "platform biometric assertion bindings",
            "Core-issued proof bindings",
        ],
        "current_designations": "canonical designation is only "
        "Core-owned scope -> accepted "
        "projection fingerprint; no "
        "Current*Binding authority schemas",
        "public_registry_injection": False,
    },
    "operation_ownership": {
        "M0.10_OWNED_TRANSITION": [
            "TRUST_DEVICE",
            "REVOKE_DEVICE",
            "CHANGE_PIN",
            "RESET_PIN",
            "LOCK_SESSION",
            "UNLOCK_SESSION",
            "LOGOUT_SESSION",
            "GRANT_LIVE_ACCESS",
            "SUSPEND_LIVE_ACCESS",
            "REVOKE_LIVE_ACCESS",
        ],
        "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER": [
            "SETUP_PIN",
            "ROTATE_SECRET_REFERENCE",
            "REBIND_SECRET_REFERENCE",
            "ACTIVATE_CREDENTIAL_PROFILE",
            "DEACTIVATE_CREDENTIAL_PROFILE",
            "CHANGE_RISK_POLICY",
            "CHANGE_KILL_SWITCH",
            "CHANGE_PRODUCT_CAPABILITIES",
        ],
        "rule": "every registered operation has exactly one executable owner; "
        "upstream-owned operations stop after M0.10 authorization",
    },
    "transition_binding_invariants": {
        "actual_mutation_must_match_fingerprint": True,
        "session_operation_exact_transition": True,
        "m03_bridge_semantic_validation": True,
        "consumed_challenge_from_upstream": True,
        "device_authorization_requires_trusted": True,
        "unlock_generation_coherence": True,
        "core_owned_biometric_challenge": True,
        "malformed_public_inputs_fail_closed": True,
        "m010_owned_scope": "canonical fingerprint of "
        "account/operator/actor-device/environment/operation/authorization-scope",
        "m010_owned_mutation": "canonical safe descriptor fingerprint; "
        "raw PIN/verifier excluded; actual "
        "transition payload must match request "
        "and proof",
        "upstream_owned_mutation": "opaque fingerprint handoff bound by "
        "M0.10; upstream owner MUST "
        "independently derive actual "
        "upstream mutation fingerprint and "
        "require equality before mutation",
        "session_mapping": {
            "LOCK_SESSION": "LOCKED",
            "LOGOUT_SESSION": "LOGGED_OUT",
            "UNLOCK_SESSION": "dedicated factor path only",
        },
        "unlock_requires_canonical_m010_scope": True,
        "m03_consumed_challenge_is_upstream_evidence": True,
        "m03_no_local_consumed_challenge_default": True,
        "live_current_designation_is_installation_policy_scoped": True,
        "live_grant_parent_is_immutable": True,
        "live_suspend_revoke_require_current_predecessor": True,
        "one_current_active_live_grant_per_installation_policy_scope": True,
        "live_grant_requires_current_active_identity": True,
        "live_grant_requires_current_trusted_device": True,
        "live_grant_requires_current_security_generation": True,
        "live_grant_security_fence_is_non_resurrectable": True,
        "device_revoked_or_replaced_cannot_restore_old_grant_authority": True,
        "security_generation_epoch_cannot_roll_back": True,
        "device_revoked_and_replaced_are_terminal_for_installation_identity": True,
        "operator_identity_revoked_is_terminal": True,
        "revoked_grant_id_is_terminal": True,
        "historical_grant_id_reuse_denied": True,
        "revoked_scope_accepts_fresh_distinct_grant": True,
        "active_or_suspended_scope_blocks_second_grant": True,
        "fresh_grant_revision_starts_at_one": True,
    },
    "proof_fencing_epoch_policy": {
        "proof_fencing_epochs_are_monotonic": True,
        "stale_authentication_proof_is_non_resurrectable": True,
        "identity_revision_cannot_roll_back": True,
        "device_trust_revision_cannot_roll_back": True,
        "platform_enrollment_revision_cannot_roll_back": True,
        "pin_revision_cannot_roll_back": True,
        "session_generation_cannot_roll_back": True,
        "same_epoch_rule": "exact same projection is idempotent; different "
        "authority content at same "
        "identity/device/session epoch denied; PIN same "
        "revision only permits failed_attempts/lockout "
        "updates with immutable credential content",
        "scope": "Core-owned current registries; no caller override",
    },
}


def deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(deep_freeze(item) for item in value)
    return value


EXPECTED_PROTOCOL = deep_freeze(EXPECTED_PROTOCOL_LITERAL)
EXPECTED_OPERATION_POLICIES = cast(
    MappingProxyType[str, Any], EXPECTED_PROTOCOL["operation_policy_registry"]
)


def thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def content(record: Any, fingerprint_field: str) -> dict[str, Any]:
    return {key: value for key, value in record.__dict__.items() if key != fingerprint_field}


def pointer(document: Any, raw: str) -> Any:
    value = document
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError("CONTRACT_INCONSISTENT")
    for part in raw[1:].split("/"):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("CONTRACT_INCONSISTENT") from error
    return value


def attest(machine: Any) -> str:
    if not isinstance(machine, dict) or machine != thaw(EXPECTED_PROTOCOL):
        return "CONTRACT_INCONSISTENT"
    if set(machine) != set(thaw(EXPECTED_PROTOCOL)):
        return "CONTRACT_INCONSISTENT"
    try:
        for item in machine["dependency_manifest"]:
            upstream = json.loads((DOCS / item["contract"]).read_text())
            if (
                fingerprint(pointer(upstream, item["json_pointer"]))
                != item["expected_content_fingerprint_sha256"]
            ):
                return "CONTRACT_INCONSISTENT"
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return "CONTRACT_INCONSISTENT"
    return "OK"


def utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo == timezone.utc else None


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def valid_utc_datetime(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def exact_int(value: Any, *, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def valid_id(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and bool(UUID7_RE.fullmatch(value[len(prefix) :]))
    )


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def valid_secret_reference(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("secure-store://"):
        return False
    locator = value[len("secure-store://") :]
    forbidden = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "private_key",
        "credential_value",
        "plaintext",
    )
    return (
        bool(locator)
        and not any(char.isspace() or char in "?#=" for char in locator)
        and not any(marker in locator.lower() for marker in forbidden)
    )


def record_fp(record: Any, field: str) -> bool:
    value = getattr(record, field, None)
    return valid_sha(value) and value == fingerprint(content(record, field))


def reference_verifier(pin: str, salt_reference: str) -> str:
    return fingerprint(["M010-REFERENCE-NOT-PRODUCTION-KDF", salt_reference, pin])


@dataclass(frozen=True)
class AuthorizationRequest:
    account_id: Any
    operator_id: Any
    device_installation_id: Any
    environment: Any
    operation: Any
    scope_fingerprint_sha256: Any
    mutation_fingerprint_sha256: Any
    causation_id: Any
    correlation_id: Any


class CoreSecurityModel:
    """Private registries model Core/external accepted/current membership, not persistence."""

    def __init__(self) -> None:
        self._accepted_identities: dict[str, OperatorIdentitySecurityProjection] = {}
        self._current_identity: dict[tuple[str, str], str] = {}
        self._accepted_devices: dict[str, DeviceTrustProjection] = {}
        self._current_device: dict[tuple[str, str], str] = {}
        self._accepted_pins: dict[str, PinVerifierRecord] = {}
        self._current_pin: dict[tuple[str, str, str], str] = {}
        self._accepted_sessions: dict[str, SessionSecurityState] = {}
        self._current_session: dict[tuple[str, str, str], str] = {}
        self._accepted_entitlements: dict[str, OperationEntitlementProjection] = {}
        self._current_entitlement: dict[tuple[str, str, str, str, str], str] = {}
        self._accepted_platform: dict[str, CoreAcceptedPlatformBiometricAssertionBinding] = {}
        self._issued_proofs: dict[str, CoreIssuedAuthenticationProofBinding] = {}
        self._accepted_secrets: dict[str, SecretMetadataProjection] = {}
        self._current_secret: dict[tuple[str, str], str] = {}
        self._accepted_grants: dict[str, LiveAccessGrantSecurityProjection] = {}
        self._current_grant: dict[tuple[str, str, str], str] = {}
        self._accepted_bootstrap: dict[str, M03AcceptedBootstrapAuthorityBinding] = {}
        self._bootstrap_views: dict[str, M03BootstrapAuthorityView] = {}
        self._current_bootstrap: dict[tuple[str, str], str] = {}
        self._consumed_bootstrap: set[str] = set()
        self._accepted_initial: dict[str, InitialSecurityState] = {}
        self._current_initial: dict[tuple[str, str], str] = {}

    @staticmethod
    def _identity_valid(value: Any) -> bool:
        return (
            isinstance(value, OperatorIdentitySecurityProjection)
            and valid_id(value.account_id, "acct_")
            and valid_id(value.operator_id, "op_")
            and value.state in {"ACTIVE", "REVOKED"}
            and exact_int(value.identity_revision)
            and exact_int(value.security_generation)
            and record_fp(value, "content_fingerprint_sha256")
        )

    @staticmethod
    def _device_valid(value: Any) -> bool:
        return (
            isinstance(value, DeviceTrustProjection)
            and valid_id(value.account_id, "acct_")
            and valid_id(value.device_installation_id, "dev_")
            and value.state in {"ENROLLED_UNTRUSTED", "TRUSTED", "REVOKED", "REPLACED"}
            and exact_int(value.trust_revision)
            and exact_int(value.security_generation)
            and exact_int(value.platform_enrollment_revision)
            and record_fp(value, "content_fingerprint_sha256")
        )

    @staticmethod
    def _pin_valid(value: Any) -> bool:
        lockout_valid = value.lockout_until_utc is None or utc(value.lockout_until_utc) is not None
        return (
            isinstance(value, PinVerifierRecord)
            and valid_id(value.account_id, "acct_")
            and valid_id(value.operator_id, "op_")
            and valid_id(value.device_installation_id, "dev_")
            and value.algorithm_id == "M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF"
            and exact_int(value.parameter_policy_version)
            and valid_secret_reference(value.salt_reference)
            and valid_sha(value.verifier)
            and exact_int(value.pin_revision)
            and exact_int(value.failed_attempts, minimum=0)
            and lockout_valid
            and exact_int(value.security_generation)
            and record_fp(value, "content_fingerprint_sha256")
        )

    @staticmethod
    def _session_valid(value: Any) -> bool:
        return (
            isinstance(value, SessionSecurityState)
            and valid_id(value.account_id, "acct_")
            and valid_id(value.operator_id, "op_")
            and valid_id(value.device_installation_id, "dev_")
            and valid_id(value.runtime_session_id, "run_")
            and value.state in {"LOCKED", "UNLOCKED", "LOGGED_OUT"}
            and exact_int(value.session_generation)
            and exact_int(value.security_generation)
            and record_fp(value, "content_fingerprint_sha256")
        )

    def _resolve(
        self, accepted: dict[str, Any], current: dict[Any, str], scope: Any, validator: Any
    ) -> Any | None:
        key = current.get(scope)
        value = accepted.get(key) if isinstance(key, str) else None
        return (
            value
            if value is not None
            and validator(value)
            and key == getattr(value, "content_fingerprint_sha256", None)
            else None
        )

    def _identity(self, account: str, operator: str) -> OperatorIdentitySecurityProjection | None:
        return self._resolve(
            self._accepted_identities,
            self._current_identity,
            (account, operator),
            self._identity_valid,
        )

    def _device(self, account: str, device: str) -> DeviceTrustProjection | None:
        return self._resolve(
            self._accepted_devices, self._current_device, (account, device), self._device_valid
        )

    def _pin(self, account: str, operator: str, device: str) -> PinVerifierRecord | None:
        return self._resolve(
            self._accepted_pins, self._current_pin, (account, operator, device), self._pin_valid
        )

    def _session(self, account: str, operator: str, device: str) -> SessionSecurityState | None:
        return self._resolve(
            self._accepted_sessions,
            self._current_session,
            (account, operator, device),
            self._session_valid,
        )

    def _accept_identity(self, value: OperatorIdentitySecurityProjection) -> bool:
        if not self._identity_valid(value):
            return False
        current = self._identity(value.account_id, value.operator_id)
        if (
            current is not None
            and value != current
            and (
                value.security_generation < current.security_generation
                or value.identity_revision <= current.identity_revision
                or current.state == "REVOKED"
            )
        ):
            return False
        self._accepted_identities[value.content_fingerprint_sha256] = value
        self._current_identity[(value.account_id, value.operator_id)] = (
            value.content_fingerprint_sha256
        )
        return True

    def _accept_device(self, value: DeviceTrustProjection) -> bool:
        if not self._device_valid(value):
            return False
        current = self._device(value.account_id, value.device_installation_id)
        if (
            current is not None
            and value != current
            and (
                value.security_generation < current.security_generation
                or value.trust_revision <= current.trust_revision
                or value.platform_enrollment_revision < current.platform_enrollment_revision
                or current.state in {"REVOKED", "REPLACED"}
            )
        ):
            return False
        self._accepted_devices[value.content_fingerprint_sha256] = value
        self._current_device[(value.account_id, value.device_installation_id)] = (
            value.content_fingerprint_sha256
        )
        return True

    def _accept_pin(self, value: PinVerifierRecord) -> bool:
        if not self._pin_valid(value):
            return False
        current = self._pin(value.account_id, value.operator_id, value.device_installation_id)
        if current is not None and value != current:
            if value.pin_revision < current.pin_revision:
                return False
            if value.pin_revision == current.pin_revision:
                immutable = (
                    "account_id",
                    "operator_id",
                    "device_installation_id",
                    "algorithm_id",
                    "parameter_policy_version",
                    "salt_reference",
                    "verifier",
                    "pin_revision",
                    "security_generation",
                )
                if any(getattr(value, field) != getattr(current, field) for field in immutable):
                    return False
        self._accepted_pins[value.content_fingerprint_sha256] = value
        self._current_pin[(value.account_id, value.operator_id, value.device_installation_id)] = (
            value.content_fingerprint_sha256
        )
        return True

    def _accept_session(self, value: SessionSecurityState) -> bool:
        if not self._session_valid(value):
            return False
        current = self._session(value.account_id, value.operator_id, value.device_installation_id)
        if (
            current is not None
            and value != current
            and value.session_generation <= current.session_generation
        ):
            return False
        self._accepted_sessions[value.content_fingerprint_sha256] = value
        self._current_session[
            (value.account_id, value.operator_id, value.device_installation_id)
        ] = value.content_fingerprint_sha256
        return True

    def _seed_m03_bridge(
        self, view: M03BootstrapAuthorityView, binding: M03AcceptedBootstrapAuthorityBinding
    ) -> bool:
        expected = bootstrap_binding(view)
        if not valid_m03_bridge(view, binding) or binding != expected:
            return False
        self._accepted_bootstrap[view.claim_fingerprint_sha256] = binding
        self._bootstrap_views[view.claim_fingerprint_sha256] = view
        self._current_bootstrap[(view.account_id, view.device_installation_id)] = (
            view.claim_fingerprint_sha256
        )
        return True

    def establish_initial_security(
        self, untrusted_view: Any, raw_pin: Any
    ) -> tuple[str, InitialSecurityEstablishmentResult | None]:
        if not isinstance(untrusted_view, M03BootstrapAuthorityView) or not isinstance(
            raw_pin, str
        ):
            return "AUTHORIZATION_DENIED", None
        member = self._accepted_bootstrap.get(untrusted_view.claim_fingerprint_sha256)
        scope = (untrusted_view.account_id, untrusted_view.device_installation_id)
        if (
            member != bootstrap_binding(untrusted_view)
            or self._bootstrap_views.get(untrusted_view.claim_fingerprint_sha256) != untrusted_view
            or self._current_bootstrap.get(scope) != untrusted_view.claim_fingerprint_sha256
        ):
            return "AUTHORIZATION_DENIED", None
        if untrusted_view.claim_fingerprint_sha256 in self._consumed_bootstrap:
            return "AUTHORIZATION_DENIED", None
        if self._identity(untrusted_view.account_id, untrusted_view.operator_id) is not None:
            return "AUTHORIZATION_DENIED", None
        identity = make_identity(
            untrusted_view.account_id, untrusted_view.operator_id, "ACTIVE", 1, 1
        )
        device = make_device(
            untrusted_view.account_id, untrusted_view.device_installation_id, "TRUSTED", 1, 1, 1
        )
        pin = make_pin(
            untrusted_view.account_id,
            untrusted_view.operator_id,
            untrusted_view.device_installation_id,
            raw_pin,
            1,
            1,
        )
        session = make_session(
            untrusted_view.account_id,
            untrusted_view.operator_id,
            untrusted_view.device_installation_id,
            "UNLOCKED",
            1,
            1,
        )
        initial_data = dict(
            account_id=untrusted_view.account_id,
            operator_id=untrusted_view.operator_id,
            device_installation_id=untrusted_view.device_installation_id,
            security_generation=1,
            session_generation=1,
            state="ESTABLISHED",
            bootstrap_claim_fingerprint_sha256=untrusted_view.claim_fingerprint_sha256,
        )
        initial = InitialSecurityState(
            **initial_data, content_fingerprint_sha256=fingerprint(initial_data)
        )
        if not (
            self._identity_valid(identity)
            and self._device_valid(device)
            and self._pin_valid(pin)
            and self._session_valid(session)
        ):
            return "CONTRACT_INCONSISTENT", None
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    pin.security_generation,
                    session.security_generation,
                    initial.security_generation,
                }
            )
            != 1
        ):
            return "CONTRACT_INCONSISTENT", None
        # Semantic atomic commit: complete POST was prepared and validated before any mutation.
        self._accepted_identities[identity.content_fingerprint_sha256] = identity
        self._current_identity[(identity.account_id, identity.operator_id)] = (
            identity.content_fingerprint_sha256
        )
        self._accepted_devices[device.content_fingerprint_sha256] = device
        self._current_device[(device.account_id, device.device_installation_id)] = (
            device.content_fingerprint_sha256
        )
        self._accepted_pins[pin.content_fingerprint_sha256] = pin
        self._current_pin[(pin.account_id, pin.operator_id, pin.device_installation_id)] = (
            pin.content_fingerprint_sha256
        )
        self._accepted_sessions[session.content_fingerprint_sha256] = session
        self._current_session[
            (session.account_id, session.operator_id, session.device_installation_id)
        ] = session.content_fingerprint_sha256
        self._accepted_initial[initial.content_fingerprint_sha256] = initial
        self._current_initial[scope] = initial.content_fingerprint_sha256
        self._consumed_bootstrap.add(untrusted_view.claim_fingerprint_sha256)
        del self._current_bootstrap[scope]
        result_data = dict(
            account_id=initial.account_id,
            operator_id=initial.operator_id,
            device_installation_id=initial.device_installation_id,
            security_generation=1,
            session_generation=1,
            bootstrap_claim_fingerprint_sha256=initial.bootstrap_claim_fingerprint_sha256,
            result="INITIAL_SECURITY_ESTABLISHED",
        )
        return "INITIAL_SECURITY_ESTABLISHED", InitialSecurityEstablishmentResult(
            **result_data, result_fingerprint_sha256=fingerprint(result_data)
        )

    def _derive_product_policy_entitlement(
        self,
        operation: str,
        environment: str = "TESTNET",
        *,
        generation: int = 1,
        revision: int = 1,
    ) -> bool:
        if operation not in EXPECTED_OPERATION_POLICIES:
            return False
        value = make_entitlement(operation, environment, revision, generation)
        valid = (
            isinstance(value, OperationEntitlementProjection)
            and valid_id(value.account_id, "acct_")
            and valid_id(value.operator_id, "op_")
            and value.operation in EXPECTED_OPERATION_POLICIES
            and value.environment in {"PAPER", "TESTNET", "LIVE"}
            and value.authorization_scope
            == EXPECTED_OPERATION_POLICIES[value.operation]["authorization_scope"]
            and value.environment in EXPECTED_OPERATION_POLICIES[value.operation]["environments"]
            and exact_int(value.entitlement_revision)
            and exact_int(value.security_generation)
            and record_fp(value, "content_fingerprint_sha256")
        )
        if not valid:
            return False
        self._accepted_entitlements[value.content_fingerprint_sha256] = value
        scope = (
            value.account_id,
            value.operator_id,
            value.operation,
            value.environment,
            value.authorization_scope,
        )
        self._current_entitlement[scope] = value.content_fingerprint_sha256
        return True

    def _seed_external_platform_assertion(self, assertion: PlatformBiometricAssertion) -> bool:
        if not valid_assertion_shape(assertion):
            return False
        binding = CoreAcceptedPlatformBiometricAssertionBinding(
            assertion.assertion_fingerprint_sha256,
            fingerprint(assertion.__dict__),
            "external_platform_authenticator",
            assertion.account_id,
            assertion.device_installation_id,
            assertion.platform_enrollment_revision,
            assertion.challenge_fingerprint_sha256,
        )
        self._accepted_platform[assertion.assertion_fingerprint_sha256] = binding
        return True

    def verify_current_pin(
        self, account: Any, operator: Any, device: Any, raw_pin: Any, now: Any
    ) -> str:
        if (
            not all(
                (valid_id(account, "acct_"), valid_id(operator, "op_"), valid_id(device, "dev_"))
            )
            or not isinstance(raw_pin, str)
            or not valid_utc_datetime(now)
        ):
            return "MALFORMED_UNTRUSTED_CONTEXT"
        record = self._pin(account, operator, device)
        if record is None:
            return "AUTHENTICATION_FAILED"
        locked = utc(record.lockout_until_utc) if record.lockout_until_utc else None
        if locked is not None and now < locked:
            return "PIN_LOCKED"
        if reference_verifier(raw_pin, record.salt_reference) == record.verifier:
            if record.failed_attempts or record.lockout_until_utc is not None:
                updated = replace(
                    record, failed_attempts=0, lockout_until_utc=None, content_fingerprint_sha256=""
                )
                updated = replace(
                    updated,
                    content_fingerprint_sha256=fingerprint(
                        content(updated, "content_fingerprint_sha256")
                    ),
                )
                if not self._accept_pin(updated):
                    return "CONTRACT_INCONSISTENT"
            return "PIN_ACCEPTED"
        failures = record.failed_attempts + 1
        until = utc_text(now + timedelta(seconds=300)) if failures >= 3 else None
        updated = replace(
            record, failed_attempts=failures, lockout_until_utc=until, content_fingerprint_sha256=""
        )
        updated = replace(
            updated,
            content_fingerprint_sha256=fingerprint(content(updated, "content_fingerprint_sha256")),
        )
        self._accept_pin(updated)
        return "PIN_LOCKED" if until else "AUTHENTICATION_FAILED"

    def verify_platform_assertion(self, assertion: Any, request: Any, now: Any) -> str:
        malformed = validate_request(request)
        if malformed or not valid_assertion_shape(assertion) or not valid_utc_datetime(now):
            return "MALFORMED_UNTRUSTED_CONTEXT"
        identity = self._identity(request.account_id, request.operator_id)
        current = self._device(request.account_id, request.device_installation_id)
        session = self._session(
            request.account_id, request.operator_id, request.device_installation_id
        )
        if identity is None or current is None or session is None:
            return "AUTHENTICATION_FAILED"
        expected_challenge = core_expected_challenge(
            request,
            current.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
        )
        binding = self._accepted_platform.get(assertion.assertion_fingerprint_sha256)
        exact = CoreAcceptedPlatformBiometricAssertionBinding(
            assertion.assertion_fingerprint_sha256,
            fingerprint(assertion.__dict__),
            "external_platform_authenticator",
            assertion.account_id,
            assertion.device_installation_id,
            assertion.platform_enrollment_revision,
            assertion.challenge_fingerprint_sha256,
        )
        if binding != exact:
            return "AUTHENTICATION_FAILED"
        if assertion.outcome == "UNAVAILABLE":
            return "FACTOR_UNAVAILABLE"
        if assertion.outcome != "SUCCESS":
            return "AUTHENTICATION_FAILED"
        start, end = utc(assertion.verified_at_utc), utc(assertion.expires_at_utc)
        exact_values = (
            assertion.account_id,
            assertion.device_installation_id,
            assertion.platform_enrollment_revision,
            assertion.challenge_fingerprint_sha256,
        )
        expected_values = (
            request.account_id,
            request.device_installation_id,
            current.platform_enrollment_revision,
            expected_challenge,
        )
        if exact_values != expected_values:
            return "AUTHENTICATION_FAILED"
        return (
            "BIOMETRIC_ACCEPTED"
            if start is not None and end is not None and start <= now <= end
            else "AUTHENTICATION_FAILED"
        )

    def issue_authentication_proof(
        self,
        request: Any,
        now: Any,
        expires_at: Any,
        *,
        raw_pin: Any = None,
        assertion: Any = None,
    ) -> tuple[str, AuthenticationProof | None]:
        if not valid_utc_datetime(now):
            return "MALFORMED_UNTRUSTED_CONTEXT", None
        malformed = validate_request(request)
        if malformed:
            return malformed, None
        policy = EXPECTED_OPERATION_POLICIES.get(request.operation)
        if policy is None:
            return "OPERATION_UNSUPPORTED", None
        if request.environment not in policy["environments"]:
            return "AUTHORIZATION_DENIED", None
        identity = self._identity(request.account_id, request.operator_id)
        device = self._device(request.account_id, request.device_installation_id)
        session = self._session(
            request.account_id, request.operator_id, request.device_installation_id
        )
        pin = self._pin(request.account_id, request.operator_id, request.device_installation_id)
        if identity is None or identity.state != "ACTIVE":
            return "IDENTITY_INVALID", None
        generations = {
            identity.security_generation,
            device.security_generation if device else None,
            pin.security_generation if pin else None,
            session.security_generation if session else None,
        }
        if len(generations) != 1:
            return "CONTRACT_INCONSISTENT", None
        if device is None or device.state != "TRUSTED":
            return "DEVICE_NOT_TRUSTED", None
        if session is None or session.state != "UNLOCKED" or pin is None:
            return "AUTHENTICATION_REQUIRED", None
        factors = []
        if (
            raw_pin is not None
            and self.verify_current_pin(
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                raw_pin,
                now,
            )
            == "PIN_ACCEPTED"
        ):
            factors.append("PIN")
        if (
            assertion is not None
            and self.verify_platform_assertion(assertion, request, now) == "BIOMETRIC_ACCEPTED"
        ):
            factors.append("BIOMETRIC")
        required = policy["factor_policy"]
        if (
            (required == "PIN" and "PIN" not in factors)
            or (required == "BIOMETRIC" and "BIOMETRIC" not in factors)
            or (required == "PIN_AND_BIOMETRIC" and set(factors) != {"PIN", "BIOMETRIC"})
        ):
            return "AUTHENTICATION_FAILED", None
        factors = {
            "PIN": ["PIN"],
            "BIOMETRIC": ["BIOMETRIC"],
            "PIN_AND_BIOMETRIC": ["PIN", "BIOMETRIC"],
        }[required]
        end = utc(expires_at)
        issued = now if valid_utc_datetime(now) else None
        if issued is None or end is None or end <= issued:
            return "MALFORMED_UNTRUSTED_CONTEXT", None
        data = dict(
            account_id=request.account_id,
            operator_id=request.operator_id,
            device_installation_id=request.device_installation_id,
            factor_set=tuple(factors),
            issued_at_utc=utc_text(issued),
            expires_at_utc=expires_at,
            identity_revision=identity.identity_revision,
            device_trust_revision=device.trust_revision,
            pin_revision=pin.pin_revision,
            platform_enrollment_revision=device.platform_enrollment_revision,
            security_generation=identity.security_generation,
            session_generation=session.session_generation,
            environment=request.environment,
            operation=request.operation,
            scope_fingerprint_sha256=request.scope_fingerprint_sha256,
            mutation_fingerprint_sha256=request.mutation_fingerprint_sha256,
            causation_id=request.causation_id,
            correlation_id=request.correlation_id,
        )
        proof = AuthenticationProof(**data, proof_fingerprint_sha256=fingerprint(data))
        binding = CoreIssuedAuthenticationProofBinding(
            proof.proof_fingerprint_sha256,
            fingerprint(proof.__dict__),
            "CoreHost",
            proof.account_id,
            proof.operator_id,
            proof.device_installation_id,
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        )
        self._issued_proofs[proof.proof_fingerprint_sha256] = binding
        return "AUTHENTICATED", proof

    def authorize(self, proof: Any, request: Any, now: Any) -> str:
        if not valid_utc_datetime(now):
            return "MALFORMED_UNTRUSTED_CONTEXT"
        if not isinstance(proof, AuthenticationProof):
            return "AUTHENTICATION_REQUIRED"
        malformed = validate_request(request)
        if malformed:
            return malformed
        policy = EXPECTED_OPERATION_POLICIES.get(request.operation)
        if policy is None:
            return "OPERATION_UNSUPPORTED"
        if request.environment not in policy["environments"]:
            return "AUTHORIZATION_DENIED"
        owned = set(thaw(EXPECTED_PROTOCOL["operation_ownership"])["M0.10_OWNED_TRANSITION"])
        if (
            request.operation in owned
            and request.scope_fingerprint_sha256
            != canonical_scope_fingerprint(
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                request.operation,
            )
        ):
            return "AUTHORIZATION_DENIED"
        binding = self._issued_proofs.get(proof.proof_fingerprint_sha256)
        exact_binding = CoreIssuedAuthenticationProofBinding(
            proof.proof_fingerprint_sha256,
            fingerprint(proof.__dict__),
            "CoreHost",
            proof.account_id,
            proof.operator_id,
            proof.device_installation_id,
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        )
        if binding != exact_binding:
            return "AUTHENTICATION_REQUIRED"
        if proof.proof_fingerprint_sha256 != fingerprint(
            content(proof, "proof_fingerprint_sha256")
        ):
            return "PROOF_STALE"
        if any(getattr(proof, k) != getattr(request, k) for k in request.__dict__):
            return "AUTHORIZATION_DENIED"
        issued, end = utc(proof.issued_at_utc), utc(proof.expires_at_utc)
        if issued is None or end is None or not isinstance(now, datetime) or end <= issued:
            return "MALFORMED_UNTRUSTED_CONTEXT"
        if now < issued:
            return "AUTHENTICATION_REQUIRED"
        if now > end or (now - issued).total_seconds() > policy["freshness_seconds"]:
            return "PROOF_EXPIRED"
        expected_factors = {
            "PIN": ("PIN",),
            "BIOMETRIC": ("BIOMETRIC",),
            "PIN_AND_BIOMETRIC": ("PIN", "BIOMETRIC"),
        }[policy["factor_policy"]]
        if proof.factor_set != expected_factors:
            return "PROOF_STALE"
        identity = self._identity(proof.account_id, proof.operator_id)
        device = self._device(proof.account_id, proof.device_installation_id)
        pin = self._pin(proof.account_id, proof.operator_id, proof.device_installation_id)
        session = self._session(proof.account_id, proof.operator_id, proof.device_installation_id)
        if identity is None or identity.state == "REVOKED":
            return "IDENTITY_REVOKED"
        if device is None or device.state != "TRUSTED":
            return "DEVICE_NOT_TRUSTED"
        if session is None or session.state != "UNLOCKED":
            return "PROOF_STALE"
        if pin is None or (
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
        ) != (
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        ):
            return "PROOF_STALE"
        scope = (
            proof.account_id,
            proof.operator_id,
            proof.operation,
            proof.environment,
            policy["authorization_scope"],
        )
        key = self._current_entitlement.get(scope)
        entitlement = self._accepted_entitlements.get(key or "")
        if (
            entitlement is None
            or key != entitlement.content_fingerprint_sha256
            or not record_fp(entitlement, "content_fingerprint_sha256")
        ):
            return "AUTHORIZATION_DENIED"
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    pin.security_generation,
                    session.security_generation,
                    entitlement.security_generation,
                    proof.security_generation,
                }
            )
            != 1
        ):
            return "CONTRACT_INCONSISTENT"
        return "AUTHORIZED"

    def transition_device(self, proof: Any, request: Any, now: Any, target_device_id: Any) -> str:
        malformed = validate_request(request)
        if (
            malformed
            or request.operation not in {"TRUST_DEVICE", "REVOKE_DEVICE"}
            or not valid_id(target_device_id, "dev_")
        ):
            return "AUTHORIZATION_DENIED"
        old = self._device(request.account_id, target_device_id)
        current_revision = 0 if old is None else old.trust_revision
        expected = device_mutation_fingerprint(
            request, target_device_id, current_revision, current_revision + 1
        )
        if (
            request.mutation_fingerprint_sha256 != expected
            or self.authorize(proof, request, now) != "AUTHORIZED"
        ):
            return "AUTHORIZATION_DENIED"
        if request.operation == "TRUST_DEVICE":
            if old is not None and old.state != "ENROLLED_UNTRUSTED":
                return "AUTHORIZATION_DENIED"
            value = make_device(
                request.account_id,
                target_device_id,
                "TRUSTED",
                current_revision + 1,
                proof.security_generation,
                1 if old is None else old.platform_enrollment_revision,
            )
        else:
            if old is None or old.state != "TRUSTED":
                return "DEVICE_NOT_TRUSTED"
            value = make_device(
                old.account_id,
                old.device_installation_id,
                "REVOKED",
                current_revision + 1,
                old.security_generation,
                old.platform_enrollment_revision,
            )
        return value.state if self._accept_device(value) else "CONTRACT_INCONSISTENT"

    def transition_live_grant(
        self,
        proof: Any,
        request: Any,
        now: Any,
        grant_id: Any,
        policy_scope_fingerprint_sha256: Any,
    ) -> tuple[str, LiveAccessGrantSecurityProjection | None]:
        operations = {
            "GRANT_LIVE_ACCESS": "ACTIVE",
            "SUSPEND_LIVE_ACCESS": "SUSPENDED",
            "REVOKE_LIVE_ACCESS": "REVOKED",
        }
        malformed = validate_request(request)
        if (
            malformed
            or request.operation not in operations
            or not valid_id(grant_id, "lgrant_")
            or not valid_sha(policy_scope_fingerprint_sha256)
        ):
            return "AUTHORIZATION_DENIED", None
        history = [
            grant
            for grant in self._accepted_grants.values()
            if grant.live_access_grant_id == grant_id
        ]
        if any(
            (
                grant.account_id,
                grant.device_installation_id,
                grant.policy_scope_fingerprint_sha256,
                grant.operator_id,
            )
            != (
                request.account_id,
                request.device_installation_id,
                policy_scope_fingerprint_sha256,
                request.operator_id,
            )
            for grant in history
        ):
            return "AUTHORIZATION_DENIED", None
        scope_key = (
            request.account_id,
            request.device_installation_id,
            policy_scope_fingerprint_sha256,
        )
        current_key = self._current_grant.get(scope_key)
        old = self._accepted_grants.get(current_key or "")
        target_state = operations[request.operation]
        if request.operation == "GRANT_LIVE_ACCESS":
            if history or (old is not None and old.state != "REVOKED"):
                return "AUTHORIZATION_DENIED", None
            current_revision = 0
        else:
            if old is None or old.live_access_grant_id != grant_id:
                return "AUTHORIZATION_DENIED", None
            allowed = {
                "SUSPEND_LIVE_ACCESS": {"ACTIVE"},
                "REVOKE_LIVE_ACCESS": {"ACTIVE", "SUSPENDED"},
            }[request.operation]
            if old.state not in allowed:
                return "AUTHORIZATION_DENIED", None
            current_revision = old.grant_revision
        expected = live_mutation_fingerprint(
            request,
            grant_id,
            target_state,
            policy_scope_fingerprint_sha256,
            current_revision,
            current_revision + 1,
            proof.security_generation if isinstance(proof, AuthenticationProof) else 0,
        )
        if (
            request.mutation_fingerprint_sha256 != expected
            or self.authorize(proof, request, now) != "AUTHORIZED"
        ):
            return "AUTHORIZATION_DENIED", None
        data = dict(
            live_access_grant_id=grant_id,
            account_id=request.account_id,
            operator_id=request.operator_id,
            device_installation_id=request.device_installation_id,
            policy_scope_fingerprint_sha256=policy_scope_fingerprint_sha256,
            state=target_state,
            grant_revision=current_revision + 1,
            security_generation=proof.security_generation,
        )
        value = LiveAccessGrantSecurityProjection(
            **data, content_fingerprint_sha256=fingerprint(data)
        )
        return (
            (value.state, value)
            if self._commit_live_grant(value)
            else ("CONTRACT_INCONSISTENT", None)
        )

    def authorize_upstream_security_request(self, proof: Any, request: Any, now: Any) -> str:
        malformed = validate_request(request)
        if malformed:
            return malformed
        upstream = set(
            thaw(EXPECTED_PROTOCOL["operation_ownership"])[
                "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER"
            ]
        )
        if request.operation not in upstream:
            return "OPERATION_UNSUPPORTED"
        return (
            "AUTHORIZED_SECURITY_REQUEST"
            if self.authorize(proof, request, now) == "AUTHORIZED"
            else "AUTHORIZATION_DENIED"
        )

    def transition_pin(self, proof: Any, request: Any, now: Any, new_pin: Any) -> str:
        malformed = validate_request(request)
        if (
            malformed
            or request.operation not in {"CHANGE_PIN", "RESET_PIN"}
            or not isinstance(new_pin, str)
        ):
            return "AUTHORIZATION_DENIED"
        old = self._pin(request.account_id, request.operator_id, request.device_installation_id)
        if old is None:
            return "AUTHORIZATION_DENIED"
        expected = pin_mutation_fingerprint(request, old.pin_revision, old.pin_revision + 1)
        if (
            request.mutation_fingerprint_sha256 != expected
            or self.authorize(proof, request, now) != "AUTHORIZED"
        ):
            return "AUTHORIZATION_DENIED"
        return (
            "PIN_CHANGED"
            if self._accept_pin(
                make_pin(
                    old.account_id,
                    old.operator_id,
                    old.device_installation_id,
                    new_pin,
                    old.pin_revision + 1,
                    old.security_generation,
                )
            )
            else "CONTRACT_INCONSISTENT"
        )

    def transition_session(self, proof: Any, request: Any, now: Any) -> str:
        malformed = validate_request(request)
        mapping = {"LOCK_SESSION": "LOCKED", "LOGOUT_SESSION": "LOGGED_OUT"}
        if malformed or request.operation not in mapping:
            return "AUTHORIZATION_DENIED"
        old = self._session(request.account_id, request.operator_id, request.device_installation_id)
        if old is None:
            return "AUTHORIZATION_DENIED"
        target = mapping[request.operation]
        expected = session_mutation_fingerprint(
            request, target, old.session_generation, old.session_generation + 1
        )
        if (
            request.mutation_fingerprint_sha256 != expected
            or self.authorize(proof, request, now) != "AUTHORIZED"
        ):
            return "AUTHORIZATION_DENIED"
        updated = make_session(
            old.account_id,
            old.operator_id,
            old.device_installation_id,
            target,
            old.session_generation + 1,
            old.security_generation,
        )
        return target if self._accept_session(updated) else "CONTRACT_INCONSISTENT"

    def unlock_session(self, request: Any, now: Any, raw_pin: Any, assertion: Any) -> str:
        malformed = validate_request(request)
        if malformed or not valid_utc_datetime(now) or request.operation != "UNLOCK_SESSION":
            return "AUTHORIZATION_DENIED"
        canonical_scope = canonical_scope_fingerprint(
            request.account_id,
            request.operator_id,
            request.device_installation_id,
            request.environment,
            request.operation,
        )
        if request.scope_fingerprint_sha256 != canonical_scope:
            return "AUTHORIZATION_DENIED"
        policy = EXPECTED_OPERATION_POLICIES["UNLOCK_SESSION"]
        identity = self._identity(request.account_id, request.operator_id)
        device = self._device(request.account_id, request.device_installation_id)
        session = self._session(
            request.account_id, request.operator_id, request.device_installation_id
        )
        scope = (
            request.account_id,
            request.operator_id,
            request.operation,
            request.environment,
            policy["authorization_scope"],
        )
        entitlement_key = self._current_entitlement.get(scope)
        entitlement = self._accepted_entitlements.get(entitlement_key or "")
        pin = self._pin(request.account_id, request.operator_id, request.device_installation_id)
        if (
            identity is None
            or identity.state != "ACTIVE"
            or device is None
            or device.state != "TRUSTED"
            or session is None
            or session.state != "LOCKED"
            or pin is None
            or entitlement is None
            or entitlement_key != entitlement.content_fingerprint_sha256
            or not record_fp(entitlement, "content_fingerprint_sha256")
            or entitlement.operation != request.operation
            or entitlement.environment != request.environment
            or entitlement.authorization_scope != policy["authorization_scope"]
        ):
            return "AUTHORIZATION_DENIED"
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    pin.security_generation,
                    session.security_generation,
                    entitlement.security_generation,
                }
            )
            != 1
        ):
            return "CONTRACT_INCONSISTENT"
        expected_mutation = session_mutation_fingerprint(
            request, "UNLOCKED", session.session_generation, session.session_generation + 1
        )
        if request.mutation_fingerprint_sha256 != expected_mutation:
            return "AUTHORIZATION_DENIED"
        if (
            self.verify_current_pin(
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                raw_pin,
                now,
            )
            != "PIN_ACCEPTED"
        ):
            return "AUTHENTICATION_FAILED"
        if self.verify_platform_assertion(assertion, request, now) != "BIOMETRIC_ACCEPTED":
            return "AUTHENTICATION_FAILED"
        updated = make_session(
            session.account_id,
            session.operator_id,
            session.device_installation_id,
            "UNLOCKED",
            session.session_generation + 1,
            session.security_generation,
        )
        return "UNLOCKED" if self._accept_session(updated) else "CONTRACT_INCONSISTENT"

    def _seed_m05_secret(self, metadata: SecretMetadataProjection) -> bool:
        valid = valid_secret(metadata)
        if not valid:
            return False
        self._accepted_secrets[metadata.content_fingerprint_sha256] = metadata
        self._current_secret[(metadata.exchange_account_id, metadata.credential_profile_id)] = (
            metadata.content_fingerprint_sha256
        )
        return True

    def validate_secret_use(self, untrusted: Any, operation: Any, environment: Any) -> str:
        if not isinstance(untrusted, SecretMetadataProjection) or not valid_secret(untrusted):
            return "SECRET_INVALID"
        key = self._current_secret.get(
            (untrusted.exchange_account_id, untrusted.credential_profile_id)
        )
        current = self._accepted_secrets.get(key or "")
        if current != untrusted or key != untrusted.content_fingerprint_sha256:
            return "SECRET_STALE"
        if untrusted.state == "REVOKED":
            return "SECRET_REVOKED"
        if untrusted.state in {"ROTATED", "REPLACED"}:
            return "SECRET_STALE"
        return (
            "SECRET_AVAILABLE"
            if operation in untrusted.permitted_operations and environment == untrusted.environment
            else "SECRET_UNAVAILABLE"
        )

    def _commit_live_grant(self, grant: LiveAccessGrantSecurityProjection) -> bool:
        valid = valid_grant(grant)
        if not valid:
            return False
        self._accepted_grants[grant.content_fingerprint_sha256] = grant
        self._current_grant[
            (grant.account_id, grant.device_installation_id, grant.policy_scope_fingerprint_sha256)
        ] = grant.content_fingerprint_sha256
        return True

    def validate_live_grant(self, untrusted: Any) -> str:
        if not isinstance(untrusted, LiveAccessGrantSecurityProjection) or not valid_grant(
            untrusted
        ):
            return "AUTHORIZATION_DENIED"
        key = self._current_grant.get(
            (
                untrusted.account_id,
                untrusted.device_installation_id,
                untrusted.policy_scope_fingerprint_sha256,
            )
        )
        current = self._accepted_grants.get(key or "")
        if current != untrusted or key != untrusted.content_fingerprint_sha256:
            return "PROOF_STALE"
        identity = self._identity(untrusted.account_id, untrusted.operator_id)
        device = self._device(untrusted.account_id, untrusted.device_installation_id)
        if identity is None or identity.state != "ACTIVE":
            return "AUTHORIZATION_DENIED"
        if device is None or device.state != "TRUSTED":
            return "AUTHORIZATION_DENIED"
        if (
            identity.security_generation != untrusted.security_generation
            or device.security_generation != untrusted.security_generation
        ):
            return "PROOF_STALE"
        return "GRANT_ACTIVE" if untrusted.state == "ACTIVE" else "AUTHORIZATION_DENIED"


def make_identity(
    account: str = ACCOUNT,
    operator: str = OPERATOR,
    state: str = "ACTIVE",
    revision: int = 1,
    generation: int = 1,
) -> OperatorIdentitySecurityProjection:
    data = dict(
        account_id=account,
        operator_id=operator,
        state=state,
        identity_revision=revision,
        security_generation=generation,
    )
    return OperatorIdentitySecurityProjection(**data, content_fingerprint_sha256=fingerprint(data))


def make_device(
    account: str = ACCOUNT,
    device: str = DEVICE,
    state: str = "TRUSTED",
    revision: int = 1,
    generation: int = 1,
    enrollment: int = 1,
) -> DeviceTrustProjection:
    data = dict(
        account_id=account,
        device_installation_id=device,
        state=state,
        trust_revision=revision,
        security_generation=generation,
        platform_enrollment_revision=enrollment,
    )
    return DeviceTrustProjection(**data, content_fingerprint_sha256=fingerprint(data))


def make_pin(
    account: str = ACCOUNT,
    operator: str = OPERATOR,
    device: str = DEVICE,
    pin: str = "2468",
    revision: int = 1,
    generation: int = 1,
) -> PinVerifierRecord:
    data = dict(
        account_id=account,
        operator_id=operator,
        device_installation_id=device,
        algorithm_id="M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF",
        parameter_policy_version=1,
        salt_reference="secure-store://opaque/pin-salt",
        verifier=reference_verifier(pin, "secure-store://opaque/pin-salt"),
        pin_revision=revision,
        failed_attempts=0,
        lockout_until_utc=None,
        security_generation=generation,
    )
    return PinVerifierRecord(**data, content_fingerprint_sha256=fingerprint(data))


def make_session(
    account: str = ACCOUNT,
    operator: str = OPERATOR,
    device: str = DEVICE,
    state: str = "UNLOCKED",
    generation: int = 1,
    security: int = 1,
) -> SessionSecurityState:
    data = dict(
        account_id=account,
        operator_id=operator,
        device_installation_id=device,
        runtime_session_id=SESSION,
        state=state,
        session_generation=generation,
        security_generation=security,
    )
    return SessionSecurityState(**data, content_fingerprint_sha256=fingerprint(data))


def make_entitlement(
    operation: str, environment: str = "TESTNET", revision: int = 1, generation: int = 1
) -> OperationEntitlementProjection:
    data = dict(
        account_id=ACCOUNT,
        operator_id=OPERATOR,
        operation=operation,
        environment=environment,
        authorization_scope=EXPECTED_OPERATION_POLICIES[operation]["authorization_scope"],
        entitlement_revision=revision,
        security_generation=generation,
    )
    return OperationEntitlementProjection(**data, content_fingerprint_sha256=fingerprint(data))


def make_assertion(
    request_value: AuthorizationRequest | None = None,
    outcome: str = "SUCCESS",
    device: str = DEVICE,
    enrollment: int = 1,
    *,
    challenge_override: str | None = None,
    security_generation: int = 1,
    session_generation: int = 1,
) -> PlatformBiometricAssertion:
    request_value = request_value or request()
    challenge = challenge_override or core_expected_challenge(
        request_value, enrollment, security_generation, session_generation
    )
    data = dict(
        account_id=request_value.account_id,
        device_installation_id=device,
        platform_authenticator_source="platform-fixture",
        platform_enrollment_revision=enrollment,
        challenge_fingerprint_sha256=challenge,
        outcome=outcome,
        verified_at_utc=utc_text(T0),
        expires_at_utc=utc_text(T0 + timedelta(seconds=60)),
    )
    return PlatformBiometricAssertion(**data, assertion_fingerprint_sha256=fingerprint(data))


def valid_assertion_shape(a: Any) -> bool:
    return (
        isinstance(a, PlatformBiometricAssertion)
        and valid_id(a.account_id, "acct_")
        and valid_id(a.device_installation_id, "dev_")
        and a.outcome in {"SUCCESS", "FAILED", "CANCELLED", "UNAVAILABLE"}
        and exact_int(a.platform_enrollment_revision)
        and valid_sha(a.challenge_fingerprint_sha256)
        and utc(a.verified_at_utc) is not None
        and utc(a.expires_at_utc) is not None
        and record_fp(a, "assertion_fingerprint_sha256")
    )


def canonical_scope_fingerprint(
    account: str, operator: str, actor_device: str, environment: str, operation: str
) -> str:
    policy = EXPECTED_OPERATION_POLICIES.get(operation)
    authorization_scope = policy["authorization_scope"] if policy else "unsupported"
    return fingerprint(
        ["M010-SCOPE", account, operator, actor_device, environment, operation, authorization_scope]
    )


def device_mutation_fingerprint(
    r: AuthorizationRequest, target_device: str, current_revision: int, next_revision: int
) -> str:
    return fingerprint(
        [
            "M010-DEVICE",
            r.account_id,
            r.operator_id,
            r.device_installation_id,
            r.operation,
            target_device,
            current_revision,
            next_revision,
            EXPECTED_OPERATION_POLICIES[r.operation]["authorization_scope"],
        ]
    )


def pin_mutation_fingerprint(
    r: AuthorizationRequest, current_revision: int, next_revision: int
) -> str:
    return fingerprint(
        [
            "M010-PIN-INTENT",
            r.account_id,
            r.operator_id,
            r.device_installation_id,
            r.operation,
            current_revision,
            next_revision,
            EXPECTED_OPERATION_POLICIES[r.operation]["authorization_scope"],
        ]
    )


def session_mutation_fingerprint(
    r: AuthorizationRequest, target_state: str, current_generation: int, next_generation: int
) -> str:
    return fingerprint(
        [
            "M010-SESSION",
            r.account_id,
            r.operator_id,
            r.device_installation_id,
            r.operation,
            target_state,
            current_generation,
            next_generation,
            EXPECTED_OPERATION_POLICIES[r.operation]["authorization_scope"],
        ]
    )


def live_mutation_fingerprint(
    r: AuthorizationRequest,
    grant_id: str,
    target_state: str,
    policy_scope: str,
    current_revision: int,
    next_revision: int,
    security_generation: int,
) -> str:
    return fingerprint(
        [
            "M010-LIVE-GRANT",
            r.account_id,
            r.operator_id,
            r.device_installation_id,
            grant_id,
            r.operation,
            target_state,
            policy_scope,
            current_revision,
            next_revision,
            security_generation,
            EXPECTED_OPERATION_POLICIES[r.operation]["authorization_scope"],
        ]
    )


def core_expected_challenge(
    r: AuthorizationRequest, enrollment: int, security_generation: int, session_generation: int
) -> str:
    return fingerprint(
        [
            "M010-BIOMETRIC-CHALLENGE",
            r.account_id,
            r.operator_id,
            r.device_installation_id,
            r.environment,
            r.operation,
            r.scope_fingerprint_sha256,
            r.mutation_fingerprint_sha256,
            r.causation_id,
            r.correlation_id,
            enrollment,
            security_generation,
            session_generation,
        ]
    )


def request(
    operation: str = "CHANGE_PIN", environment: str = "TESTNET", **changes: Any
) -> AuthorizationRequest:
    target_device = cast(
        str, changes.pop("target_device", DEVICE_2 if operation == "TRUST_DEVICE" else DEVICE)
    )
    policy_scope = cast(str, changes.pop("policy_scope", FP_C))
    grant_id = cast(str, changes.pop("grant_id", LIVE_GRANT))
    current_revision = cast(
        int,
        changes.pop(
            "current_revision", 0 if operation in {"GRANT_LIVE_ACCESS", "TRUST_DEVICE"} else 1
        ),
    )
    security_generation = cast(int, changes.pop("security_generation", 1))
    base = dict(
        account_id=ACCOUNT,
        operator_id=OPERATOR,
        device_installation_id=DEVICE,
        environment=environment,
        operation=operation,
        scope_fingerprint_sha256="",
        mutation_fingerprint_sha256="",
        causation_id="cause-canonical",
        correlation_id="correlation-canonical",
    )
    base.update(changes)
    base["scope_fingerprint_sha256"] = changes.get(
        "scope_fingerprint_sha256",
        canonical_scope_fingerprint(
            base["account_id"],
            base["operator_id"],
            base["device_installation_id"],
            environment,
            operation,
        ),
    )
    provisional = AuthorizationRequest(**base)
    if operation in {"TRUST_DEVICE", "REVOKE_DEVICE"}:
        mutation = device_mutation_fingerprint(
            provisional, target_device, current_revision, current_revision + 1
        )
    elif operation in {"CHANGE_PIN", "RESET_PIN"}:
        mutation = pin_mutation_fingerprint(provisional, current_revision, current_revision + 1)
    elif operation in {"LOCK_SESSION", "LOGOUT_SESSION", "UNLOCK_SESSION"}:
        target = {
            "LOCK_SESSION": "LOCKED",
            "LOGOUT_SESSION": "LOGGED_OUT",
            "UNLOCK_SESSION": "UNLOCKED",
        }[operation]
        mutation = session_mutation_fingerprint(
            provisional, target, current_revision, current_revision + 1
        )
    elif operation in {"GRANT_LIVE_ACCESS", "SUSPEND_LIVE_ACCESS", "REVOKE_LIVE_ACCESS"}:
        target = {
            "GRANT_LIVE_ACCESS": "ACTIVE",
            "SUSPEND_LIVE_ACCESS": "SUSPENDED",
            "REVOKE_LIVE_ACCESS": "REVOKED",
        }[operation]
        mutation = live_mutation_fingerprint(
            provisional,
            grant_id,
            target,
            policy_scope,
            current_revision,
            current_revision + 1,
            security_generation,
        )
    else:
        mutation = cast(str, changes.get("mutation_fingerprint_sha256", FP_B))
    base["mutation_fingerprint_sha256"] = changes.get("mutation_fingerprint_sha256", mutation)
    return AuthorizationRequest(**base)


def validate_request(r: Any) -> str | None:
    if not isinstance(r, AuthorizationRequest):
        return "MALFORMED_UNTRUSTED_CONTEXT"
    if (
        not valid_id(r.account_id, "acct_")
        or not valid_id(r.operator_id, "op_")
        or not valid_id(r.device_installation_id, "dev_")
        or not valid_sha(r.scope_fingerprint_sha256)
        or not valid_sha(r.mutation_fingerprint_sha256)
        or not isinstance(r.causation_id, str)
        or not r.causation_id
        or not isinstance(r.correlation_id, str)
        or not r.correlation_id
    ):
        return "MALFORMED_UNTRUSTED_CONTEXT"
    if r.operation not in EXPECTED_OPERATION_POLICIES:
        return "OPERATION_UNSUPPORTED"
    if r.environment not in {"PAPER", "TESTNET", "LIVE"}:
        return "MALFORMED_UNTRUSTED_CONTEXT"
    return None


def make_secret(state: str = "AVAILABLE", revision: int = 1) -> SecretMetadataProjection:
    data = dict(
        secret_reference="secure-store://opaque/exchange-slot",
        secret_kind="API_SECRET",
        exchange_account_id=EXCHANGE_ACCOUNT,
        credential_profile_id=CREDENTIAL,
        exchange_id="BINANCE",
        environment="TESTNET",
        permitted_operations=("PRIVATE_DATA", "ORDER_ENTRY"),
        secret_revision=revision,
        state=state,
    )
    return SecretMetadataProjection(**data, content_fingerprint_sha256=fingerprint(data))


def valid_secret(s: Any) -> bool:
    return (
        isinstance(s, SecretMetadataProjection)
        and valid_secret_reference(s.secret_reference)
        and s.secret_kind in {"API_KEY", "API_SECRET", "PASSPHRASE", "PRIVATE_KEY"}
        and valid_id(s.exchange_account_id, "xacc_")
        and valid_id(s.credential_profile_id, "cred_")
        and isinstance(s.exchange_id, str)
        and bool(s.exchange_id)
        and s.environment in {"PAPER", "TESTNET", "LIVE"}
        and isinstance(s.permitted_operations, tuple)
        and _valid_permitted_operations(s.permitted_operations)
        and exact_int(s.secret_revision)
        and s.state in {"AVAILABLE", "ROTATED", "REVOKED", "REPLACED"}
        and record_fp(s, "content_fingerprint_sha256")
    )


def make_grant(
    state: str = "ACTIVE", revision: int = 1, generation: int = 1
) -> LiveAccessGrantSecurityProjection:
    data = dict(
        live_access_grant_id=LIVE_GRANT,
        account_id=ACCOUNT,
        operator_id=OPERATOR,
        device_installation_id=DEVICE,
        policy_scope_fingerprint_sha256=FP_C,
        state=state,
        grant_revision=revision,
        security_generation=generation,
    )
    return LiveAccessGrantSecurityProjection(**data, content_fingerprint_sha256=fingerprint(data))


def valid_grant(g: Any) -> bool:
    return (
        isinstance(g, LiveAccessGrantSecurityProjection)
        and valid_id(g.live_access_grant_id, "lgrant_")
        and valid_id(g.account_id, "acct_")
        and valid_id(g.operator_id, "op_")
        and valid_id(g.device_installation_id, "dev_")
        and valid_sha(g.policy_scope_fingerprint_sha256)
        and g.state in {"ACTIVE", "SUSPENDED", "REVOKED"}
        and exact_int(g.grant_revision)
        and exact_int(g.security_generation)
        and record_fp(g, "content_fingerprint_sha256")
    )


@dataclass(frozen=True)
class _M03TrustedConsumedBootstrapEvidence:
    account_id: str
    device_installation_id: str
    operator_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    purpose: str
    accepted_pre_fingerprint_sha256: str
    current_post_fingerprint_sha256: str
    consumed_authority_fingerprint_sha256: str
    consumed_claim_fingerprint_sha256: str
    consumed_challenge_fingerprint_sha256: str
    authority_source: str


def project_m03_view(evidence: _M03TrustedConsumedBootstrapEvidence) -> M03BootstrapAuthorityView:
    return M03BootstrapAuthorityView(
        evidence.consumed_claim_fingerprint_sha256,
        evidence.account_id,
        evidence.device_installation_id,
        evidence.operator_id,
        evidence.bootstrap_generation,
        evidence.bootstrap_revision,
        evidence.purpose,
        evidence.accepted_pre_fingerprint_sha256,
        evidence.current_post_fingerprint_sha256,
        evidence.consumed_authority_fingerprint_sha256,
        evidence.consumed_claim_fingerprint_sha256,
        evidence.consumed_challenge_fingerprint_sha256,
    )


def bootstrap_binding(view: M03BootstrapAuthorityView) -> M03AcceptedBootstrapAuthorityBinding:
    return M03AcceptedBootstrapAuthorityBinding(
        fingerprint(view.__dict__),
        fingerprint(view.__dict__),
        view.claim_fingerprint_sha256,
        view.account_id,
        view.device_installation_id,
        view.operator_id,
        view.bootstrap_generation,
        view.bootstrap_revision,
        view.purpose,
        view.pre_state_fingerprint_sha256,
        view.post_state_fingerprint_sha256,
        view.consumed_authority_fingerprint_sha256,
        view.consumed_claim_fingerprint_sha256,
        view.consumed_challenge_fingerprint_sha256,
        "external_product_provisioning_boundary",
    )


def valid_m03_bridge(view: Any, binding: Any) -> bool:
    if not isinstance(view, M03BootstrapAuthorityView) or not isinstance(
        binding, M03AcceptedBootstrapAuthorityBinding
    ):
        return False
    if not (
        valid_id(view.account_id, "acct_")
        and valid_id(view.device_installation_id, "dev_")
        and valid_id(view.operator_id, "op_")
    ):
        return False
    if (
        not exact_int(view.bootstrap_generation)
        or not exact_int(view.bootstrap_revision)
        or view.purpose != "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    ):
        return False
    fingerprint_fields = (
        "claim_fingerprint_sha256",
        "pre_state_fingerprint_sha256",
        "post_state_fingerprint_sha256",
        "consumed_authority_fingerprint_sha256",
        "consumed_claim_fingerprint_sha256",
        "consumed_challenge_fingerprint_sha256",
    )
    if not all(valid_sha(getattr(view, field)) for field in fingerprint_fields):
        return False
    return (
        binding == bootstrap_binding(view)
        and binding.authority_source == "external_product_provisioning_boundary"
    )


class _TrustedAuthorityHarness:
    """Module-private architecture fixture for authority already accepted outside caller API."""

    _M03_EVIDENCE = _M03TrustedConsumedBootstrapEvidence(
        ACCOUNT,
        DEVICE,
        OPERATOR,
        1,
        1,
        "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "external_product_provisioning_boundary",
    )

    @classmethod
    def m03_view(cls) -> M03BootstrapAuthorityView:
        return project_m03_view(cls._M03_EVIDENCE)

    @classmethod
    def seed_m03(cls, core: CoreSecurityModel, view: M03BootstrapAuthorityView) -> bool:
        projected = project_m03_view(cls._M03_EVIDENCE)
        if (
            view != projected
            or cls._M03_EVIDENCE.authority_source != "external_product_provisioning_boundary"
        ):
            return False
        return core._seed_m03_bridge(view, bootstrap_binding(projected))

    @staticmethod
    def seed_platform(core: CoreSecurityModel, assertion: PlatformBiometricAssertion) -> bool:
        return core._seed_external_platform_assertion(assertion)

    @staticmethod
    def seed_m05_secret(core: CoreSecurityModel, metadata: SecretMetadataProjection) -> bool:
        return core._seed_m05_secret(metadata)

    @staticmethod
    def derive_entitlement(
        core: CoreSecurityModel,
        operation: str,
        environment: str = "TESTNET",
        *,
        generation: int = 1,
        revision: int = 1,
    ) -> bool:
        return core._derive_product_policy_entitlement(
            operation, environment, generation=generation, revision=revision
        )


def initialized(
    operation: str = "CHANGE_PIN", *, entitlement: bool = True, platform: bool = True
) -> tuple[CoreSecurityModel, AuthorizationRequest, PlatformBiometricAssertion]:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    assert _TrustedAuthorityHarness.seed_m03(core, view)
    assert core.establish_initial_security(view, "2468")[0] == "INITIAL_SECURITY_ESTABLISHED"
    req = request(operation)
    if entitlement:
        assert _TrustedAuthorityHarness.derive_entitlement(core, operation)
    assertion = make_assertion(req)
    if platform:
        assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    return core, req, assertion


def issued(
    operation: str = "CHANGE_PIN",
) -> tuple[CoreSecurityModel, AuthorizationRequest, AuthenticationProof]:
    core, req, assertion = initialized(operation)
    result, proof = core.issue_authentication_proof(
        req,
        T0,
        utc_text(T0 + timedelta(seconds=120)),
        raw_pin="2468",
        assertion=assertion,
    )
    assert result == "AUTHENTICATED" and proof is not None
    return core, req, proof


def manual_proof(factors: tuple[str, ...] = ("PIN", "BIOMETRIC")) -> AuthenticationProof:
    data = dict(
        account_id=ACCOUNT,
        operator_id=OPERATOR,
        device_installation_id=DEVICE,
        factor_set=factors,
        issued_at_utc=utc_text(T0),
        expires_at_utc=utc_text(T0 + timedelta(seconds=120)),
        identity_revision=1,
        device_trust_revision=1,
        pin_revision=1,
        platform_enrollment_revision=1,
        security_generation=1,
        session_generation=1,
        environment="TESTNET",
        operation="CHANGE_PIN",
        scope_fingerprint_sha256=FP_A,
        mutation_fingerprint_sha256=FP_B,
        causation_id="cause-canonical",
        correlation_id="correlation-canonical",
    )
    return AuthenticationProof(**data, proof_fingerprint_sha256=fingerprint(data))


def machine() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(MACHINE_PATH.read_text()))


def test_full_expected_protocol_dependencies_and_schema_exactness() -> None:
    value = machine()
    assert attest(value) == "OK"
    assert set(value["executable_boundary_schemas"]) == {
        c.__name__ for c in M010_AUTHORITY_DATACLASSES
    }
    for cls in M010_AUTHORITY_DATACLASSES:
        source_schema = value["executable_boundary_schemas"][cls.__name__]
        source_fields = (
            source_schema["exact_fields"] if isinstance(source_schema, dict) else source_schema
        )
        assert source_fields == [f.name for f in fields(cls)]


MUTATIONS = [
    lambda x: x["bootstrap"].__setitem__("one_shot", False),
    lambda x: x["bootstrap"].__setitem__("manual_or_self_hashed_result_authority", True),
    lambda x: x["bootstrap"]["never_authorizes"].remove("LIVE"),
    lambda x: x["operation_policy_registry"]["CHANGE_PIN"].__setitem__("factor_policy", "PIN"),
    lambda x: x["operation_policy_registry"]["CHANGE_PIN"].__setitem__("freshness_seconds", 3600),
    lambda x: x["operation_policy_registry"]["CHANGE_PIN"]["environments"].append("LIVE"),
    lambda x: x["pin_policy"].__setitem__("raw_pin_serialized", True),
    lambda x: x["pin_policy"].__setitem__("bool_as_int_rejected", False),
    lambda x: x["biometric_policy"].__setitem__("stores_biometric_material", True),
    lambda x: x["proof_policy"].__setitem__("self_hash_authority", True),
    lambda x: x["proof_policy"]["always_revalidate_current"].remove("identity"),
    lambda x: x["secret_reference_policy"]["forbidden_payload_markers"].remove("secret"),
    lambda x: x["secret_reference_policy"].__setitem__("keyring_scheme_result", "ACCEPTED"),
    lambda x: x["live_policy"].__setitem__("current_live", "ENABLED"),
    lambda x: x["live_policy"].__setitem__("testnet_to_live_fallback", True),
    lambda x: x["live_policy"].__setitem__("security_alone_enables_live", True),
    lambda x: x["audit_safe_payload"]["forbidden_categories"].remove("raw PIN"),
    lambda x: x["identifier_policy"].__setitem__("uuid_version", 4),
    lambda x: x["executable_boundary_schemas"].pop("AuthenticationProof"),
    lambda x: x["deferred_to_m011"].remove("persistence"),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "public_self_enrollment_allowed", True
    ),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "bootstrap_seed_from_shape_allowed", True
    ),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "platform_seed_from_shape_allowed", True
    ),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "entitlement_caller_acceptance_allowed", True
    ),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "secret_caller_acceptance_allowed", True
    ),
    lambda x: x["authority"]["acceptance_plane"].__setitem__(
        "live_grant_caller_acceptance_allowed", True
    ),
    lambda x: x["authority"].pop("security_generation_coherence"),
    lambda x: x["authority"].pop("canonical_time_boundary"),
    lambda x: x["bootstrap"].pop("semantic_atomicity"),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "actual_mutation_must_match_fingerprint", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "session_operation_exact_transition", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "m03_bridge_semantic_validation", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "consumed_challenge_from_upstream", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "device_authorization_requires_trusted", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__("unlock_generation_coherence", False),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "core_owned_biometric_challenge", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "malformed_public_inputs_fail_closed", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "unlock_requires_canonical_m010_scope", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "m03_consumed_challenge_is_upstream_evidence", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "m03_no_local_consumed_challenge_default", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_current_designation_is_installation_policy_scoped", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_grant_parent_is_immutable", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_suspend_revoke_require_current_predecessor", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "one_current_active_live_grant_per_installation_policy_scope", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_grant_requires_current_active_identity", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_grant_requires_current_trusted_device", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_grant_requires_current_security_generation", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "live_grant_security_fence_is_non_resurrectable", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "device_revoked_or_replaced_cannot_restore_old_grant_authority", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "security_generation_epoch_cannot_roll_back", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "device_revoked_and_replaced_are_terminal_for_installation_identity", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "operator_identity_revoked_is_terminal", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "proof_fencing_epochs_are_monotonic", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "stale_authentication_proof_is_non_resurrectable", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "identity_revision_cannot_roll_back", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "device_trust_revision_cannot_roll_back", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "platform_enrollment_revision_cannot_roll_back", False
    ),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__("pin_revision_cannot_roll_back", False),
    lambda x: x["proof_fencing_epoch_policy"].__setitem__(
        "session_generation_cannot_roll_back", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__("revoked_grant_id_is_terminal", False),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "historical_grant_id_reuse_denied", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "revoked_scope_accepts_fresh_distinct_grant", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "active_or_suspended_scope_blocks_second_grant", False
    ),
    lambda x: x["transition_binding_invariants"].__setitem__(
        "fresh_grant_revision_starts_at_one", False
    ),
]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_full_machine_semantic_mutations_fail_closed(mutation: Any) -> None:
    value = machine()
    mutation(value)
    assert attest(value) == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("factors", [("PIN",), ("BIOMETRIC",), ("PIN", "BIOMETRIC")])
def test_caller_factor_sets_and_self_hash_do_not_issue(factors: tuple[str, ...]) -> None:
    core, req, _ = initialized()
    assert core.authorize(manual_proof(factors), req, T0) == "AUTHENTICATION_REQUIRED"


def test_manual_proof_and_manual_binding_denied() -> None:
    core, req, _ = initialized()
    p = manual_proof()
    manual = CoreIssuedAuthenticationProofBinding(
        p.proof_fingerprint_sha256,
        fingerprint(p.__dict__),
        "CoreHost",
        p.account_id,
        p.operator_id,
        p.device_installation_id,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert manual and core.authorize(p, req, T0) == "AUTHENTICATION_REQUIRED"


def test_genuine_issuance_and_authorization_succeed() -> None:
    core, req, p = issued()
    assert p.factor_set == ("PIN", "BIOMETRIC")
    assert core.authorize(p, req, T0 + timedelta(seconds=60)) == "AUTHORIZED"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("authority_source", "caller"),
        ("account_id", ACCOUNT + "x"),
        ("operator_id", OPERATOR + "x"),
        ("device_installation_id", DEVICE_2),
        ("identity_revision", 2),
        ("device_trust_revision", 2),
        ("pin_revision", 2),
        ("platform_enrollment_revision", 2),
        ("security_generation", 2),
        ("session_generation", 2),
    ],
)
def test_every_issued_binding_field_is_exact(field: str, bad: Any) -> None:
    core, req, p = issued()
    core._issued_proofs[p.proof_fingerprint_sha256] = replace(
        core._issued_proofs[p.proof_fingerprint_sha256], **{field: bad}
    )
    assert core.authorize(p, req, T0) == "AUTHENTICATION_REQUIRED"


def test_binding_under_wrong_registry_key_denied() -> None:
    core, req, p = issued()
    binding = core._issued_proofs.pop(p.proof_fingerprint_sha256)
    core._issued_proofs[FP_C] = binding
    assert core.authorize(p, req, T0) == "AUTHENTICATION_REQUIRED"


@pytest.mark.parametrize(
    "flag", ["authenticated", "pin_ok", "biometric_ok", "admin", "role", "approved", "entitlement"]
)
def test_each_caller_flag_is_absent_and_manual_authority_is_denied(flag: str) -> None:
    import inspect

    public = [
        member
        for name, member in inspect.getmembers(CoreSecurityModel, inspect.isfunction)
        if not name.startswith("_")
    ]
    assert all(flag not in inspect.signature(member).parameters for member in public)
    core, req, _ = initialized()
    assert core.authorize(manual_proof(), req, T0) == "AUTHENTICATION_REQUIRED"


def test_wrong_pin_and_lockout() -> None:
    core, _, _ = initialized()
    assert core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "bad", T0) == "AUTHENTICATION_FAILED"
    assert core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "bad", T0) == "AUTHENTICATION_FAILED"
    assert core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "bad", T0) == "PIN_LOCKED"
    assert core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "2468", T0) == "PIN_LOCKED"


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("FAILED", "AUTHENTICATION_FAILED"),
        ("CANCELLED", "AUTHENTICATION_FAILED"),
        ("UNAVAILABLE", "FACTOR_UNAVAILABLE"),
    ],
)
def test_platform_closed_outcomes(outcome: str, expected: str) -> None:
    core, _, _ = initialized(platform=False)
    a = make_assertion(outcome=outcome)
    assert _TrustedAuthorityHarness.seed_platform(core, a)
    assert core.verify_platform_assertion(a, request(), T0) == expected


def test_nominal_success_denied_and_genuine_platform_accepted() -> None:
    core, _, a = initialized(platform=False)
    assert core.verify_platform_assertion(a, request(), T0) == "AUTHENTICATION_FAILED"
    assert _TrustedAuthorityHarness.seed_platform(core, a)
    assert core.verify_platform_assertion(a, request(), T0) == "BIOMETRIC_ACCEPTED"


@pytest.mark.parametrize(
    "assertion,device,challenge",
    [
        (make_assertion(device=DEVICE_2), DEVICE, FP_A),
        (make_assertion(enrollment=2), DEVICE, FP_A),
        (make_assertion(challenge_override=FP_B), DEVICE, FP_A),
    ],
)
def test_platform_exact_bindings(
    assertion: PlatformBiometricAssertion, device: str, challenge: str
) -> None:
    core, _, _ = initialized(platform=False)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    assert (
        core.verify_platform_assertion(assertion, request(device_installation_id=device), T0)
        == "AUTHENTICATION_FAILED"
    )


def test_policy_unsupported_wrong_environment_and_factor_enforced() -> None:
    core, _, a = initialized()
    assert (
        core.issue_authentication_proof(
            request("UNKNOWN"), T0, utc_text(T0 + timedelta(seconds=30)), raw_pin="2468"
        )[0]
        == "OPERATION_UNSUPPORTED"
    )
    assert (
        core.issue_authentication_proof(
            request("CHANGE_PIN", "LIVE"),
            T0,
            utc_text(T0 + timedelta(seconds=30)),
            raw_pin="2468",
            assertion=a,
        )[0]
        == "AUTHORIZATION_DENIED"
    )
    assert (
        core.issue_authentication_proof(
            request(), T0, utc_text(T0 + timedelta(seconds=30)), raw_pin="2468"
        )[0]
        == "AUTHENTICATION_FAILED"
    )


def test_freshness_boundaries_before_and_expired() -> None:
    core, req, p = issued()
    assert core.authorize(p, req, T0 + timedelta(seconds=60)) == "AUTHORIZED"
    assert core.authorize(p, req, T0 + timedelta(seconds=61)) == "PROOF_EXPIRED"
    assert core.authorize(p, req, T0 - timedelta(seconds=1)) == "AUTHENTICATION_REQUIRED"
    assert core.authorize(p, req, T0 + timedelta(seconds=121)) == "PROOF_EXPIRED"


def test_authenticated_without_or_stale_entitlement_denied() -> None:
    core, req, a = initialized(entitlement=False)
    result, p = core.issue_authentication_proof(
        req, T0, utc_text(T0 + timedelta(seconds=120)), raw_pin="2468", assertion=a
    )
    assert result == "AUTHENTICATED" and p is not None
    assert core.authorize(p, req, T0) == "AUTHORIZATION_DENIED"
    assert _TrustedAuthorityHarness.derive_entitlement(core, "CHANGE_PIN", revision=1)
    core._current_entitlement[(ACCOUNT, OPERATOR, "CHANGE_PIN", "TESTNET", "change_pin")] = FP_C
    assert core.authorize(p, req, T0) == "AUTHORIZATION_DENIED"


def test_identity_device_and_session_transitions_fence_proofs() -> None:
    core, req, p = issued()
    core._accept_identity(make_identity(state="REVOKED", revision=2, generation=1))
    assert core.authorize(p, req, T0) == "IDENTITY_REVOKED"
    core, req, p = issued()
    core._accept_device(make_device(state="REVOKED", revision=2))
    assert core.authorize(p, req, T0) == "DEVICE_NOT_TRUSTED"
    core, req, p = issued()
    core._accept_device(make_device(state="REPLACED", revision=2))
    assert core.authorize(p, req, T0) == "DEVICE_NOT_TRUSTED"


@pytest.mark.parametrize(
    "target,operation", [("LOCKED", "LOCK_SESSION"), ("LOGGED_OUT", "LOGOUT_SESSION")]
)
def test_session_transitions_increment_and_fence(target: str, operation: str) -> None:
    core, req, p = issued(operation)
    assert core.transition_session(p, req, T0) == target
    current = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert current is not None and current.session_generation == 2 and current.state == target
    assert core.authorize(p, req, T0) == "PROOF_STALE"


def test_unlock_session_uses_current_factors_entitlement_and_increments() -> None:
    core, lock_request, proof = issued("LOCK_SESSION")
    assert core.transition_session(proof, lock_request, T0) == "LOCKED"
    unlock_request = request("UNLOCK_SESSION", current_revision=2)
    assert _TrustedAuthorityHarness.derive_entitlement(core, "UNLOCK_SESSION")
    assertion = make_assertion(unlock_request, session_generation=2)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    assert core.unlock_session(unlock_request, T0, "2468", assertion) == "UNLOCKED"
    current = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert current is not None and current.state == "UNLOCKED" and current.session_generation == 3


@pytest.mark.parametrize("operation", ["CHANGE_PIN", "RESET_PIN"])
def test_pin_transitions_increment_and_fence(operation: str) -> None:
    core, req, p = issued(operation)
    assert core.transition_pin(p, req, T0, "8642") == "PIN_CHANGED"
    current = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert current is not None and current.pin_revision == 2
    assert core.authorize(p, req, T0) == "PROOF_STALE"


def test_genuine_bootstrap_establishes_all_and_replay_second_device_denied() -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    assert _TrustedAuthorityHarness.seed_m03(core, view)
    status, result = core.establish_initial_security(view, "2468")
    assert status == "INITIAL_SECURITY_ESTABLISHED" and result is not None
    assert (
        core._identity(ACCOUNT, OPERATOR) is not None
        and core._device(ACCOUNT, DEVICE) is not None
        and core._pin(ACCOUNT, OPERATOR, DEVICE) is not None
        and core._session(ACCOUNT, OPERATOR, DEVICE) is not None
    )
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"
    assert (
        core.establish_initial_security(
            replace(_TrustedAuthorityHarness.m03_view(), device_installation_id=DEVICE_2), "2468"
        )[0]
        == "AUTHORIZATION_DENIED"
    )


def test_manual_self_hashed_and_noncurrent_bootstrap_denied() -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"
    assert _TrustedAuthorityHarness.seed_m03(core, view)
    core._current_bootstrap.clear()
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"


def test_bootstrap_cannot_authorize_normal_or_live() -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    assert _TrustedAuthorityHarness.seed_m03(core, view)
    assert core.authorize(manual_proof(), request(), T0) == "AUTHENTICATION_REQUIRED"
    assert (
        core.authorize(manual_proof(), request("GRANT_LIVE_ACCESS", "LIVE"), T0)
        == "AUTHENTICATION_REQUIRED"
    )


def test_secret_current_registry_and_states() -> None:
    core = CoreSecurityModel()
    available = make_secret()
    assert core.validate_secret_use(available, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
    assert _TrustedAuthorityHarness.seed_m05_secret(core, available)
    assert core.validate_secret_use(available, "PRIVATE_DATA", "TESTNET") == "SECRET_AVAILABLE"
    newer = make_secret(revision=2)
    assert _TrustedAuthorityHarness.seed_m05_secret(core, newer)
    assert core.validate_secret_use(available, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
    for state, expected in (
        ("ROTATED", "SECRET_STALE"),
        ("REPLACED", "SECRET_STALE"),
        ("REVOKED", "SECRET_REVOKED"),
    ):
        item = make_secret(state, 3)
        assert _TrustedAuthorityHarness.seed_m05_secret(core, item)
        assert core.validate_secret_use(item, "PRIVATE_DATA", "TESTNET") == expected


@pytest.mark.parametrize(
    "value",
    [
        "keyring://opaque",
        "secure-store://contains-secret",
        "secure-store://x?y",
        "secure-store://x#y",
        "secure-store://x=y",
    ],
)
def test_secret_invalid_grammar(value: str) -> None:
    item = replace(make_secret(), secret_reference=value, content_fingerprint_sha256=FP_A)
    assert (
        CoreSecurityModel().validate_secret_use(item, "PRIVATE_DATA", "TESTNET") == "SECRET_INVALID"
    )


def test_live_grant_registry_states_and_current_live_denial() -> None:
    core, active = _active_live_grant_fixture()
    assert core.validate_live_grant(active) == "GRANT_ACTIVE"
    assert (
        machine()["live_policy"]["current_live"] == "DENIED_BY_UPSTREAM_POLICY"
        and machine()["live_policy"]["target_architecture_supports_live"]
        and not machine()["live_policy"]["testnet_to_live_fallback"]
    )


def test_testnet_proof_cannot_authorize_live() -> None:
    core, _, p = issued()
    assert core.authorize(p, request("GRANT_LIVE_ACCESS", "LIVE"), T0) == "AUTHORIZATION_DENIED"


def safe_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    allowed = {
        "account_id",
        "operator_id",
        "device_installation_id",
        "credential_profile_id",
        "exchange_account_id",
        "live_access_grant_id",
        "secret_reference",
        "reason_code",
        "causation_id",
        "correlation_id",
    }
    result = {}
    for key, value in payload.items():
        if key in allowed and (key != "secret_reference" or valid_secret_reference(value)):
            result[key] = value
    return result


def test_real_sensitive_audit_payload_projects_safely() -> None:
    sensitive = {
        "raw_pin": "2468",
        "verifier": "real-verifier",
        "biometric_material": "face-template",
        "api_secret": "api-real-secret",
        "private_key": "private-real",
        "passphrase": "phrase-real",
        "bearer_token": "bearer-real",
        "plaintext_secure_store_payload": "payload-real",
    }
    projected = safe_payload(
        {
            **sensitive,
            "account_id": ACCOUNT,
            "secret_reference": "secure-store://opaque/audit",
            "reason_code": "AUTHENTICATION_FAILED",
        }
    )
    encoded = canonical(projected)
    assert set(projected) == {"account_id", "secret_reference", "reason_code"}
    assert all(value not in encoded for value in sensitive.values())


@pytest.mark.parametrize(
    "attack",
    [
        replace(make_identity(), account_id="acct_bad"),
        replace(make_device(), trust_revision=True),
        replace(make_pin(), pin_revision=0),
        replace(make_session(), session_generation=-1),
    ],
)
def test_malformed_authority_records_fail_closed_without_exception(attack: Any) -> None:
    core = CoreSecurityModel()
    outcomes = [
        core._accept_identity(attack)
        if isinstance(attack, OperatorIdentitySecurityProjection)
        else False,
        core._accept_device(attack) if isinstance(attack, DeviceTrustProjection) else False,
        core._accept_pin(attack) if isinstance(attack, PinVerifierRecord) else False,
        core._accept_session(attack) if isinstance(attack, SessionSecurityState) else False,
    ]
    assert not any(outcomes)


def test_current_binding_hybrid_schemas_are_removed() -> None:
    names = set(machine()["executable_boundary_schemas"])
    assert (
        not {
            "CurrentOperatorIdentityBinding",
            "CurrentDeviceTrustBinding",
            "CurrentSecretBinding",
            "CurrentLiveAccessGrantBinding",
        }
        & names
    )


def test_public_api_surface_has_no_accept_register_seed_or_current_bypass() -> None:
    import inspect

    public = {
        name: member
        for name, member in inspect.getmembers(CoreSecurityModel, inspect.isfunction)
        if not name.startswith("_")
    }
    assert not any(
        name.startswith(("accept", "register", "seed", "set_current", "external_accept"))
        for name in public
    )
    for member in public.values():
        signature = inspect.signature(member)
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert not {"binding", "current", "entitlement", "registry"} & set(signature.parameters)


def test_real_public_self_enrollment_attacks_are_denied() -> None:
    core = CoreSecurityModel()
    entitlement = make_entitlement("CHANGE_PIN")
    secret = make_secret()
    grant = make_grant()
    view = _TrustedAuthorityHarness.m03_view()
    assertion = make_assertion()
    proof = manual_proof()
    manual_binding = CoreIssuedAuthenticationProofBinding(
        proof.proof_fingerprint_sha256,
        fingerprint(proof.__dict__),
        "CoreHost",
        ACCOUNT,
        OPERATOR,
        DEVICE,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert entitlement and manual_binding
    assert core.validate_secret_use(secret, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
    assert core.validate_live_grant(grant) == "PROOF_STALE"
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"
    assert core.verify_platform_assertion(assertion, request(), T0) == "AUTHENTICATION_FAILED"
    assert core.authorize(proof, request(), T0) == "AUTHENTICATION_REQUIRED"


def test_manual_bootstrap_bridge_binding_cannot_enter_public_api() -> None:
    import inspect

    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    manual = bootstrap_binding(view)
    assert manual
    assert "binding" not in inspect.signature(core.establish_initial_security).parameters
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("accepted_pre_fingerprint_sha256", FP_C),
        ("current_post_fingerprint_sha256", FP_C),
        ("consumed_claim_fingerprint_sha256", FP_B),
        ("consumed_challenge_fingerprint_sha256", FP_B),
        ("bootstrap_generation", 2),
        ("bootstrap_revision", 2),
        ("account_id", "acct_01890f3a-2b4c-7abc-9def-0123456789ab"),
        ("device_installation_id", DEVICE_2),
        ("operator_id", "op_01890f3a-2b4c-7abc-9def-0123456789ab"),
    ],
)
def test_m03_bridge_exact_binding_attacks_denied(field: str, bad: Any) -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    wrong = replace(bootstrap_binding(view), **{field: bad})
    assert not core._seed_m03_bridge(view, wrong)
    assert core.establish_initial_security(view, "2468")[0] == "AUTHORIZATION_DENIED"


def test_initial_security_failure_is_semantically_zero_change() -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    assert _TrustedAuthorityHarness.seed_m03(core, view)
    before = {
        "identities": copy.deepcopy(core._accepted_identities),
        "current_identity": copy.deepcopy(core._current_identity),
        "devices": copy.deepcopy(core._accepted_devices),
        "current_device": copy.deepcopy(core._current_device),
        "pins": copy.deepcopy(core._accepted_pins),
        "current_pin": copy.deepcopy(core._current_pin),
        "sessions": copy.deepcopy(core._accepted_sessions),
        "current_session": copy.deepcopy(core._current_session),
        "initial": copy.deepcopy(core._accepted_initial),
        "consumed": copy.deepcopy(core._consumed_bootstrap),
    }
    assert core.establish_initial_security(view, object())[0] == "AUTHORIZATION_DENIED"
    after = {
        "identities": core._accepted_identities,
        "current_identity": core._current_identity,
        "devices": core._accepted_devices,
        "current_device": core._current_device,
        "pins": core._accepted_pins,
        "current_pin": core._current_pin,
        "sessions": core._accepted_sessions,
        "current_session": core._current_session,
        "initial": core._accepted_initial,
        "consumed": core._consumed_bootstrap,
    }
    assert after == before
    assert core._current_bootstrap[(ACCOUNT, DEVICE)] == view.claim_fingerprint_sha256


@pytest.mark.parametrize("component", ["device", "pin", "session", "entitlement"])
def test_incoherent_current_security_generation_fails_closed(component: str) -> None:
    core, req, proof = issued()
    if component == "device":
        core._accept_device(make_device(revision=2, generation=2))
    elif component == "pin":
        core._accept_pin(make_pin(revision=2, generation=2))
    elif component == "session":
        core._accept_session(make_session(generation=2, security=2))
    else:
        core._derive_product_policy_entitlement("CHANGE_PIN", generation=2, revision=2)
    assert core.authorize(proof, req, T0) != "AUTHORIZED"


def test_issuance_rejects_incoherent_current_security_bundle() -> None:
    core, req, assertion = initialized()
    core._accept_device(make_device(revision=2, generation=2))
    assert (
        core.issue_authentication_proof(
            req,
            T0,
            utc_text(T0 + timedelta(seconds=60)),
            raw_pin="2468",
            assertion=assertion,
        )[0]
        == "CONTRACT_INCONSISTENT"
    )


@pytest.mark.parametrize(
    "bad_now",
    [datetime(2030, 1, 1), datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=1)))],
)
def test_every_public_time_path_rejects_noncanonical_datetime(bad_now: datetime) -> None:
    core, req, assertion = initialized()
    assert (
        core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "2468", bad_now)
        == "MALFORMED_UNTRUSTED_CONTEXT"
    )
    assert core.verify_platform_assertion(assertion, req, bad_now) == "MALFORMED_UNTRUSTED_CONTEXT"
    assert (
        core.issue_authentication_proof(
            req,
            bad_now,
            utc_text(T0 + timedelta(seconds=60)),
            raw_pin="2468",
            assertion=assertion,
        )[0]
        == "MALFORMED_UNTRUSTED_CONTEXT"
    )
    assert core.authorize(manual_proof(), req, bad_now) == "MALFORMED_UNTRUSTED_CONTEXT"


def test_pin_success_after_expired_lockout_resets_failure_state() -> None:
    core, _, _ = initialized()
    for _ in range(3):
        core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "bad", T0)
    assert (
        core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "2468", T0 + timedelta(seconds=301))
        == "PIN_ACCEPTED"
    )
    current = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert (
        current is not None and current.failed_attempts == 0 and current.lockout_until_utc is None
    )


def _issue_live_admin(
    operation: str, core: CoreSecurityModel | None = None
) -> tuple[CoreSecurityModel, AuthorizationRequest, AuthenticationProof]:
    if core is None:
        core, _, _ = initialized(platform=False)
    assert _TrustedAuthorityHarness.derive_entitlement(core, operation, "LIVE")
    current_revision = 0
    current_key = core._current_grant.get((ACCOUNT, DEVICE, FP_C))
    current_grant = core._accepted_grants.get(current_key or "")
    if current_grant is not None:
        current_revision = current_grant.grant_revision
    req = request(operation, "LIVE", current_revision=current_revision)
    current_session = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert current_session is not None
    assertion = make_assertion(req, session_generation=current_session.session_generation)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    status, proof = core.issue_authentication_proof(
        req,
        T0,
        utc_text(T0 + timedelta(seconds=60)),
        raw_pin="2468",
        assertion=assertion,
    )
    assert status == "AUTHENTICATED" and proof is not None
    return core, req, proof


def test_authorized_live_grant_suspend_revoke_transitions_fence_history() -> None:
    core, req, proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    status, active = core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)
    assert status == "ACTIVE" and active is not None
    core, req, proof = _issue_live_admin("SUSPEND_LIVE_ACCESS", core)
    status, suspended = core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)
    assert status == "SUSPENDED" and suspended is not None
    assert core.validate_live_grant(active) == "PROOF_STALE"
    core, req, proof = _issue_live_admin("REVOKE_LIVE_ACCESS", core)
    status, revoked = core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)
    assert status == "REVOKED" and revoked is not None
    assert core.validate_live_grant(suspended) == "PROOF_STALE"
    assert core.validate_live_grant(revoked) == "AUTHORIZATION_DENIED"
    assert machine()["live_policy"]["current_live"] == "DENIED_BY_UPSTREAM_POLICY"


def test_device_trust_and_revoke_are_authorized_transitions() -> None:
    core, req, proof = issued("TRUST_DEVICE")
    assert core.transition_device(proof, req, T0, DEVICE_2) == "TRUSTED"
    assert core._device(ACCOUNT, DEVICE_2) is not None
    core, req, proof = issued("REVOKE_DEVICE")
    assert core.transition_device(proof, req, T0, DEVICE) == "REVOKED"
    assert core.authorize(proof, req, T0) == "DEVICE_NOT_TRUSTED"


def test_every_operation_has_exact_executable_ownership() -> None:
    ownership = thaw(EXPECTED_PROTOCOL["operation_ownership"])
    owned = set(ownership["M0.10_OWNED_TRANSITION"])
    upstream = set(ownership["AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER"])
    assert not owned & upstream
    assert owned | upstream == set(EXPECTED_OPERATION_POLICIES)


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "LIVE"},
        {"purpose": "CHANGE_PIN"},
        {"bootstrap_generation": 0},
        {"bootstrap_generation": True},
        {"bootstrap_revision": 0},
        {"account_id": "acct_bad"},
        {"operator_id": "op_bad"},
        {"device_installation_id": "dev_bad"},
        {"claim_fingerprint_sha256": "bad"},
        {"pre_state_fingerprint_sha256": "d" * 64},
        {"post_state_fingerprint_sha256": "d" * 64},
        {"consumed_authority_fingerprint_sha256": "d" * 64},
        {"consumed_claim_fingerprint_sha256": "d" * 64},
        {"consumed_challenge_fingerprint_sha256": "d" * 64},
    ],
)
def test_self_consistent_recomputed_m03_view_attacks_are_denied(change: dict[str, Any]) -> None:
    core = CoreSecurityModel()
    tampered = replace(_TrustedAuthorityHarness.m03_view(), **change)
    recomputed = bootstrap_binding(tampered)
    assert valid_m03_bridge(tampered, recomputed) is (
        all(
            valid_sha(getattr(tampered, field))
            for field in (
                "claim_fingerprint_sha256",
                "pre_state_fingerprint_sha256",
                "post_state_fingerprint_sha256",
                "consumed_authority_fingerprint_sha256",
                "consumed_claim_fingerprint_sha256",
                "consumed_challenge_fingerprint_sha256",
            )
        )
        and valid_id(tampered.account_id, "acct_")
        and valid_id(tampered.operator_id, "op_")
        and valid_id(tampered.device_installation_id, "dev_")
        and exact_int(tampered.bootstrap_generation)
        and exact_int(tampered.bootstrap_revision)
        and tampered.purpose == "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert not _TrustedAuthorityHarness.seed_m03(core, tampered)
    assert core.establish_initial_security(tampered, "2468")[0] == "AUTHORIZATION_DENIED"


def test_wrong_m03_authority_source_is_denied() -> None:
    core = CoreSecurityModel()
    view = _TrustedAuthorityHarness.m03_view()
    binding = replace(bootstrap_binding(view), authority_source="caller")
    assert not core._seed_m03_bridge(view, binding)


def test_device_actual_target_must_match_authorized_mutation_without_state_change() -> None:
    core, req, proof = issued("TRUST_DEVICE")
    before = (copy.deepcopy(core._accepted_devices), copy.deepcopy(core._current_device))
    assert (
        core.transition_device(proof, req, T0, "dev_01890f3a-2b4c-7abc-adef-0123456789ab")
        == "AUTHORIZATION_DENIED"
    )
    assert (core._accepted_devices, core._current_device) == before


def test_pin_actual_revision_intent_must_match_without_state_change() -> None:
    core, _, assertion = initialized("CHANGE_PIN")
    req = request("CHANGE_PIN", current_revision=2)
    assertion = make_assertion(req)
    _TrustedAuthorityHarness.seed_platform(core, assertion)
    status, proof = core.issue_authentication_proof(
        req, T0, utc_text(T0 + timedelta(seconds=60)), raw_pin="2468", assertion=assertion
    )
    assert status == "AUTHENTICATED" and proof is not None
    before = (copy.deepcopy(core._accepted_pins), copy.deepcopy(core._current_pin))
    assert core.transition_pin(proof, req, T0, "8642") == "AUTHORIZATION_DENIED"
    assert (core._accepted_pins, core._current_pin) == before


def test_session_operation_has_only_exact_derived_target_and_unlock_is_separate() -> None:
    import inspect

    assert "target" not in inspect.signature(CoreSecurityModel.transition_session).parameters
    for operation, expected in (("LOCK_SESSION", "LOCKED"), ("LOGOUT_SESSION", "LOGGED_OUT")):
        core, req, proof = issued(operation)
        assert core.transition_session(proof, req, T0) == expected
        current = core._session(ACCOUNT, OPERATOR, DEVICE)
        assert current is not None and current.state == expected
    core, req, assertion = initialized("UNLOCK_SESSION")
    status, unlock_proof = core.issue_authentication_proof(
        req, T0, utc_text(T0 + timedelta(seconds=60)), raw_pin="2468", assertion=assertion
    )
    assert status == "AUTHENTICATED" and unlock_proof is not None
    before = (copy.deepcopy(core._accepted_sessions), copy.deepcopy(core._current_session))
    assert core.transition_session(unlock_proof, req, T0) == "AUTHORIZATION_DENIED"
    assert (core._accepted_sessions, core._current_session) == before


def test_live_actual_scope_and_grant_id_must_match_without_state_change() -> None:
    core, req, proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert (
        core.transition_live_grant(proof, req, T0, LIVE_GRANT, "d" * 64)[0]
        == "AUTHORIZATION_DENIED"
    )
    other_grant = "lgrant_01890f3a-2b4c-7abc-adef-0123456789ab"
    assert (
        core.transition_live_grant(proof, req, T0, other_grant, FP_C)[0] == "AUTHORIZATION_DENIED"
    )
    assert (core._accepted_grants, core._current_grant) == before


def test_enrolled_untrusted_current_device_never_authorizes() -> None:
    core, req, proof = issued()
    core._accept_device(make_device(state="ENROLLED_UNTRUSTED", revision=2))
    assert core.authorize(proof, req, T0) == "DEVICE_NOT_TRUSTED"


def _locked_unlock_context() -> tuple[
    CoreSecurityModel, AuthorizationRequest, PlatformBiometricAssertion
]:
    core, lock_request, proof = issued("LOCK_SESSION")
    assert core.transition_session(proof, lock_request, T0) == "LOCKED"
    unlock_request = request("UNLOCK_SESSION", current_revision=2)
    assert _TrustedAuthorityHarness.derive_entitlement(core, "UNLOCK_SESSION")
    assertion = make_assertion(unlock_request, session_generation=2)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    return core, unlock_request, assertion


@pytest.mark.parametrize("component", ["device", "pin", "session", "entitlement"])
def test_unlock_revalidates_full_generation_coherence(component: str) -> None:
    core, req, assertion = _locked_unlock_context()
    if component == "device":
        core._accept_device(make_device(revision=2, generation=2))
    elif component == "pin":
        core._accept_pin(make_pin(revision=2, generation=2))
    elif component == "session":
        core._accept_session(make_session(state="LOCKED", generation=3, security=2))
    else:
        core._derive_product_policy_entitlement("UNLOCK_SESSION", generation=2, revision=2)
    assert core.unlock_session(req, T0, "2468", assertion) == "CONTRACT_INCONSISTENT"


def test_unlock_stale_entitlement_key_is_denied() -> None:
    core, req, assertion = _locked_unlock_context()
    scope = (ACCOUNT, OPERATOR, "UNLOCK_SESSION", "TESTNET", "unlock_session")
    core._current_entitlement[scope] = "d" * 64
    assert core.unlock_session(req, T0, "2468", assertion) == "AUTHORIZATION_DENIED"


@pytest.mark.parametrize(
    "field",
    [
        "operation",
        "environment",
        "scope_fingerprint_sha256",
        "mutation_fingerprint_sha256",
        "causation_id",
        "correlation_id",
    ],
)
def test_biometric_assertion_is_bound_to_exact_request_context(field: str) -> None:
    core, req, assertion = initialized()
    changes: dict[str, Any] = {
        field: "d" * 64
        if "fingerprint" in field
        else ("PAPER" if field == "environment" else "different")
    }
    other = replace(req, **changes)
    assert core.verify_platform_assertion(assertion, other, T0) != "BIOMETRIC_ACCEPTED"


@pytest.mark.parametrize(
    "bad_request",
    [
        object(),
        {},
        AuthorizationRequest(
            "bad", OPERATOR, DEVICE, "TESTNET", "CHANGE_PIN", FP_A, FP_B, "c", "r"
        ),
    ],
)
def test_every_public_request_path_fails_closed_without_exception_or_mutation(
    bad_request: Any,
) -> None:
    core, req, proof = issued()
    assertion = make_assertion(req)
    before = copy.deepcopy(core.__dict__)
    results = [
        core.transition_device(proof, bad_request, T0, DEVICE_2),
        core.transition_live_grant(proof, bad_request, T0, LIVE_GRANT, FP_C)[0],
        core.authorize_upstream_security_request(proof, bad_request, T0),
        core.transition_pin(proof, bad_request, T0, "8642"),
        core.transition_session(proof, bad_request, T0),
        core.unlock_session(bad_request, T0, "2468", assertion),
        core.issue_authentication_proof(bad_request, T0, utc_text(T0 + timedelta(seconds=60)))[0],
        core.authorize(proof, bad_request, T0),
    ]
    assert all(isinstance(result, str) for result in results)
    assert core.__dict__ == before


def test_unlock_rejects_self_consistent_noncanonical_scope_before_factors() -> None:
    core, canonical_request, _ = _locked_unlock_context()
    wrong_scope = "d" * 64
    wrong_request = replace(canonical_request, scope_fingerprint_sha256=wrong_scope)
    wrong_request = replace(
        wrong_request,
        mutation_fingerprint_sha256=session_mutation_fingerprint(wrong_request, "UNLOCKED", 2, 3),
    )
    assertion = make_assertion(wrong_request, session_generation=2)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    before = (copy.deepcopy(core._accepted_sessions), copy.deepcopy(core._current_session))
    assert core.unlock_session(wrong_request, T0, "2468", assertion) == "AUTHORIZATION_DENIED"
    assert (core._accepted_sessions, core._current_session) == before
    current = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert current is not None and current.state == "LOCKED"


def test_unlock_with_canonical_scope_still_succeeds() -> None:
    core, canonical_request, assertion = _locked_unlock_context()
    assert core.unlock_session(canonical_request, T0, "2468", assertion) == "UNLOCKED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("consumed_challenge_fingerprint_sha256", "d" * 64),
        ("consumed_claim_fingerprint_sha256", "d" * 64),
        ("accepted_pre_fingerprint_sha256", "d" * 64),
        ("current_post_fingerprint_sha256", "d" * 64),
        ("consumed_authority_fingerprint_sha256", "d" * 64),
    ],
)
def test_m03_view_must_be_exact_projection_of_preexisting_evidence(field: str, value: str) -> None:
    core = CoreSecurityModel()
    evidence = _TrustedAuthorityHarness._M03_EVIDENCE
    genuine = project_m03_view(evidence)
    mapping = {
        "accepted_pre_fingerprint_sha256": "pre_state_fingerprint_sha256",
        "current_post_fingerprint_sha256": "post_state_fingerprint_sha256",
    }
    view_field = mapping.get(field, field)
    wrong_view = replace(genuine, **{view_field: value})
    recomputed_binding = bootstrap_binding(wrong_view)
    assert valid_m03_bridge(wrong_view, recomputed_binding)
    assert not _TrustedAuthorityHarness.seed_m03(core, wrong_view)
    assert core.establish_initial_security(wrong_view, "2468")[0] == "AUTHORIZATION_DENIED"


def test_m03_evidence_without_explicit_challenge_cannot_be_projected_as_authority() -> None:
    core = CoreSecurityModel()
    malformed_evidence = replace(
        _TrustedAuthorityHarness._M03_EVIDENCE,
        consumed_challenge_fingerprint_sha256=cast(str, None),
    )
    malformed_view = project_m03_view(malformed_evidence)
    assert not valid_m03_bridge(malformed_view, bootstrap_binding(malformed_view))
    assert not _TrustedAuthorityHarness.seed_m03(core, malformed_view)


def _issue_specific_live_admin(
    core: CoreSecurityModel,
    operation: str,
    *,
    actor_device: str = DEVICE,
    grant_id: str = LIVE_GRANT,
    policy_scope: str = FP_C,
    current_revision: int,
) -> tuple[AuthorizationRequest, AuthenticationProof]:
    assert _TrustedAuthorityHarness.derive_entitlement(core, operation, "LIVE")
    req = request(
        operation,
        "LIVE",
        device_installation_id=actor_device,
        grant_id=grant_id,
        policy_scope=policy_scope,
        current_revision=current_revision,
    )
    identity = core._identity(ACCOUNT, OPERATOR)
    session = core._session(ACCOUNT, OPERATOR, actor_device)
    assert identity is not None and session is not None
    assertion = make_assertion(
        req,
        device=actor_device,
        security_generation=identity.security_generation,
        session_generation=session.session_generation,
    )
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    status, proof = core.issue_authentication_proof(
        req,
        T0,
        utc_text(T0 + timedelta(seconds=60)),
        raw_pin="2468",
        assertion=assertion,
    )
    assert status == "AUTHENTICATED" and proof is not None
    return req, proof


def test_second_active_grant_for_same_installation_policy_scope_is_denied() -> None:
    core, req, proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    assert core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)[0] == "ACTIVE"
    other_grant = "lgrant_01890f3a-2b4c-7abc-adef-0123456789ab"
    second_req, second_proof = _issue_specific_live_admin(
        core,
        "GRANT_LIVE_ACCESS",
        grant_id=other_grant,
        current_revision=0,
    )
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert (
        core.transition_live_grant(second_proof, second_req, T0, other_grant, FP_C)[0]
        == "AUTHORIZATION_DENIED"
    )
    assert (core._accepted_grants, core._current_grant) == before


def test_existing_grant_id_cannot_change_parent_device_or_policy_scope() -> None:
    core, req, proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    assert core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)[0] == "ACTIVE"
    core._accept_device(make_device(device=DEVICE_2))
    core._accept_pin(make_pin(device=DEVICE_2))
    core._accept_session(make_session(device=DEVICE_2))
    device_req, device_proof = _issue_specific_live_admin(
        core,
        "SUSPEND_LIVE_ACCESS",
        actor_device=DEVICE_2,
        current_revision=1,
    )
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert (
        core.transition_live_grant(device_proof, device_req, T0, LIVE_GRANT, FP_C)[0]
        == "AUTHORIZATION_DENIED"
    )
    policy_q = "d" * 64
    policy_req, policy_proof = _issue_specific_live_admin(
        core,
        "SUSPEND_LIVE_ACCESS",
        policy_scope=policy_q,
        current_revision=1,
    )
    assert (
        core.transition_live_grant(policy_proof, policy_req, T0, LIVE_GRANT, policy_q)[0]
        == "AUTHORIZATION_DENIED"
    )
    assert (core._accepted_grants, core._current_grant) == before


@pytest.mark.parametrize("operation", ["SUSPEND_LIVE_ACCESS", "REVOKE_LIVE_ACCESS"])
def test_suspend_and_revoke_from_absent_are_denied_without_mutation(operation: str) -> None:
    core, _, _ = initialized(platform=False)
    req, proof = _issue_specific_live_admin(core, operation, current_revision=0)
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)[0] == "AUTHORIZATION_DENIED"
    assert (core._accepted_grants, core._current_grant) == before


def _active_live_grant_fixture() -> tuple[CoreSecurityModel, LiveAccessGrantSecurityProjection]:
    core, req, proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    status, grant = core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)
    assert status == "ACTIVE" and grant is not None
    assert core.validate_live_grant(grant) == "GRANT_ACTIVE"
    return core, grant


def test_legal_device_revocation_fences_current_active_live_grant_end_to_end() -> None:
    core, active = _active_live_grant_fixture()
    assert _TrustedAuthorityHarness.derive_entitlement(core, "REVOKE_DEVICE")
    req = request("REVOKE_DEVICE", current_revision=1)
    assertion = make_assertion(req)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    status, proof = core.issue_authentication_proof(
        req,
        T0,
        utc_text(T0 + timedelta(seconds=60)),
        raw_pin="2468",
        assertion=assertion,
    )
    assert status == "AUTHENTICATED" and proof is not None
    assert core.transition_device(proof, req, T0, DEVICE) == "REVOKED"
    current_device = core._device(ACCOUNT, DEVICE)
    assert current_device is not None and current_device.state == "REVOKED"
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"


@pytest.mark.parametrize("state", ["REPLACED", "ENROLLED_UNTRUSTED"])
def test_nontrusted_current_device_fences_active_live_grant(state: str) -> None:
    core, active = _active_live_grant_fixture()
    before_grants = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    core._accept_device(make_device(state=state, revision=2))
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    assert (core._accepted_grants, core._current_grant) == before_grants


def test_revoked_current_operator_fences_active_live_grant() -> None:
    core, active = _active_live_grant_fixture()
    before_grants = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    core._accept_identity(make_identity(state="REVOKED", revision=2))
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    assert (core._accepted_grants, core._current_grant) == before_grants


@pytest.mark.parametrize("parent", ["identity", "device"])
def test_stale_live_grant_security_generation_is_fenced(parent: str) -> None:
    core, active = _active_live_grant_fixture()
    before_grants = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    if parent == "identity":
        core._accept_identity(make_identity(revision=2, generation=2))
    else:
        core._accept_device(make_device(revision=2, generation=2))
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    assert (core._accepted_grants, core._current_grant) == before_grants


def test_coherent_current_security_epoch_keeps_exact_current_active_grant_valid() -> None:
    core, active = _active_live_grant_fixture()
    identity = core._identity(ACCOUNT, OPERATOR)
    device = core._device(ACCOUNT, DEVICE)
    assert identity is not None and identity.state == "ACTIVE"
    assert device is not None and device.state == "TRUSTED"
    assert identity.security_generation == device.security_generation == active.security_generation
    assert core.validate_live_grant(active) == "GRANT_ACTIVE"


def _seed_second_actor_device(core: CoreSecurityModel) -> None:
    assert core._accept_device(make_device(device=DEVICE_2))
    assert core._accept_pin(make_pin(device=DEVICE_2))
    assert core._accept_session(make_session(device=DEVICE_2))


def _issue_device_operation_from_second_actor(
    core: CoreSecurityModel, operation: str, target: str, current_revision: int
) -> tuple[AuthorizationRequest, AuthenticationProof]:
    _seed_second_actor_device(core)
    assert _TrustedAuthorityHarness.derive_entitlement(core, operation)
    req = request(
        operation,
        device_installation_id=DEVICE_2,
        target_device=target,
        current_revision=current_revision,
    )
    assertion = make_assertion(req, device=DEVICE_2)
    assert _TrustedAuthorityHarness.seed_platform(core, assertion)
    status, proof = core.issue_authentication_proof(
        req,
        T0,
        utc_text(T0 + timedelta(seconds=60)),
        raw_pin="2468",
        assertion=assertion,
    )
    assert status == "AUTHENTICATED" and proof is not None
    return req, proof


def test_revoked_device_is_terminal_and_cannot_resurrect_active_grant() -> None:
    core, active = _active_live_grant_fixture()
    _seed_second_actor_device(core)
    revoke_req, revoke_proof = _issue_device_operation_from_second_actor(
        core, "REVOKE_DEVICE", DEVICE, 1
    )
    assert core.transition_device(revoke_proof, revoke_req, T0, DEVICE) == "REVOKED"
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    trust_req, trust_proof = _issue_device_operation_from_second_actor(
        core, "TRUST_DEVICE", DEVICE, 2
    )
    assert core.transition_device(trust_proof, trust_req, T0, DEVICE) == "AUTHORIZATION_DENIED"
    current = core._device(ACCOUNT, DEVICE)
    assert current is not None and current.state == "REVOKED"
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"


def test_replaced_device_is_terminal_and_cannot_resurrect_active_grant() -> None:
    core, active = _active_live_grant_fixture()
    assert core._accept_device(make_device(state="REPLACED", revision=2))
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    trust_req, trust_proof = _issue_device_operation_from_second_actor(
        core, "TRUST_DEVICE", DEVICE, 2
    )
    assert core.transition_device(trust_proof, trust_req, T0, DEVICE) == "AUTHORIZATION_DENIED"
    current = core._device(ACCOUNT, DEVICE)
    assert current is not None and current.state == "REPLACED"
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"


def test_revoked_operator_is_terminal_and_cannot_resurrect_active_grant() -> None:
    core, active = _active_live_grant_fixture()
    assert core._accept_identity(make_identity(state="REVOKED", revision=2))
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"
    assert not core._accept_identity(make_identity(state="ACTIVE", revision=3))
    current = core._identity(ACCOUNT, OPERATOR)
    assert current is not None and current.state == "REVOKED"
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"


@pytest.mark.parametrize("parent", ["identity", "device"])
def test_security_generation_epoch_cannot_roll_back_and_resurrect_grant(parent: str) -> None:
    core, active = _active_live_grant_fixture()
    if parent == "identity":
        assert core._accept_identity(make_identity(revision=2, generation=2))
        assert core.validate_live_grant(active) != "GRANT_ACTIVE"
        assert not core._accept_identity(make_identity(revision=3, generation=1))
        identity_current = core._identity(ACCOUNT, OPERATOR)
        assert identity_current is not None
        current_generation = identity_current.security_generation
    else:
        assert core._accept_device(make_device(revision=2, generation=2))
        assert core.validate_live_grant(active) != "GRANT_ACTIVE"
        assert not core._accept_device(make_device(revision=3, generation=1))
        device_current = core._device(ACCOUNT, DEVICE)
        assert device_current is not None
        current_generation = device_current.security_generation
    assert current_generation == 2
    assert core.validate_live_grant(active) != "GRANT_ACTIVE"


@pytest.mark.parametrize(
    "epoch", ["identity_revision", "trust_revision", "platform_enrollment_revision"]
)
def test_proof_bound_identity_device_epoch_rollback_cannot_resurrect(epoch: str) -> None:
    core, req, old_proof = issued()
    assert core.authorize(old_proof, req, T0) == "AUTHORIZED"
    if epoch == "identity_revision":
        assert core._accept_identity(make_identity(revision=2))
        assert core.authorize(old_proof, req, T0) == "PROOF_STALE"
        assert not core._accept_identity(make_identity(revision=1))
        current = core._identity(ACCOUNT, OPERATOR)
        assert current is not None and current.identity_revision == 2
    elif epoch == "trust_revision":
        assert core._accept_device(make_device(revision=2))
        assert core.authorize(old_proof, req, T0) == "PROOF_STALE"
        assert not core._accept_device(make_device(revision=1))
        current_device = core._device(ACCOUNT, DEVICE)
        assert current_device is not None and current_device.trust_revision == 2
    else:
        assert core._accept_device(make_device(revision=2, enrollment=2))
        assert core.authorize(old_proof, req, T0) == "PROOF_STALE"
        assert not core._accept_device(make_device(revision=3, enrollment=1))
        current_device = core._device(ACCOUNT, DEVICE)
        assert current_device is not None and current_device.platform_enrollment_revision == 2
    assert core.authorize(old_proof, req, T0) == "PROOF_STALE"


def test_pin_revision_rollback_cannot_resurrect_old_proof_or_verifier() -> None:
    core, req, old_proof = issued("CHANGE_PIN")
    old_record = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert old_record is not None
    assert core.transition_pin(old_proof, req, T0, "8642") == "PIN_CHANGED"
    assert core.authorize(old_proof, req, T0) == "PROOF_STALE"
    assert not core._accept_pin(old_record)
    current = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert (
        current is not None
        and current.pin_revision == 2
        and current.verifier != old_record.verifier
    )
    assert core.authorize(old_proof, req, T0) == "PROOF_STALE"


def test_session_generation_rollback_cannot_resurrect_locked_out_proof() -> None:
    core, req, old_proof = issued("LOCK_SESSION")
    old_session = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert old_session is not None and old_session.session_generation == 1
    assert core.transition_session(old_proof, req, T0) == "LOCKED"
    assert core.authorize(old_proof, req, T0) == "PROOF_STALE"
    assert not core._accept_session(old_session)
    current = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert current is not None and current.state == "LOCKED" and current.session_generation == 2
    assert core.authorize(old_proof, req, T0) == "PROOF_STALE"


def test_same_epoch_authority_content_changes_are_denied_but_exact_current_is_idempotent() -> None:
    core, _, _ = initialized()
    identity = core._identity(ACCOUNT, OPERATOR)
    device = core._device(ACCOUNT, DEVICE)
    pin = core._pin(ACCOUNT, OPERATOR, DEVICE)
    session = core._session(ACCOUNT, OPERATOR, DEVICE)
    assert identity and device and pin and session
    assert (
        core._accept_identity(identity)
        and core._accept_device(device)
        and core._accept_pin(pin)
        and core._accept_session(session)
    )
    assert not core._accept_identity(make_identity(state="REVOKED", revision=1))
    assert not core._accept_device(make_device(state="ENROLLED_UNTRUSTED", revision=1))
    changed_pin = replace(pin, verifier="d" * 64, content_fingerprint_sha256="")
    changed_pin = replace(
        changed_pin,
        content_fingerprint_sha256=fingerprint(content(changed_pin, "content_fingerprint_sha256")),
    )
    assert not core._accept_pin(changed_pin)
    assert not core._accept_session(make_session(state="LOCKED", generation=1))


def test_failed_pin_attempt_same_revision_remains_a_legal_state_update() -> None:
    core, _, _ = initialized()
    before = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert before is not None and before.pin_revision == 1
    assert (
        core.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, "wrong", T0) == "AUTHENTICATION_FAILED"
    )
    after = core._pin(ACCOUNT, OPERATOR, DEVICE)
    assert after is not None and after.pin_revision == 1 and after.failed_attempts == 1
    assert after.verifier == before.verifier and after.salt_reference == before.salt_reference


GRANT_2 = "lgrant_01890f3a-2b4c-7abc-adef-0123456789ab"


def _revoked_g1_history() -> tuple[
    CoreSecurityModel, tuple[LiveAccessGrantSecurityProjection, ...]
]:
    core, grant_req, grant_proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    status, active = core.transition_live_grant(grant_proof, grant_req, T0, LIVE_GRANT, FP_C)
    assert status == "ACTIVE" and active is not None
    core, suspend_req, suspend_proof = _issue_live_admin("SUSPEND_LIVE_ACCESS", core)
    status, suspended = core.transition_live_grant(suspend_proof, suspend_req, T0, LIVE_GRANT, FP_C)
    assert status == "SUSPENDED" and suspended is not None
    core, revoke_req, revoke_proof = _issue_live_admin("REVOKE_LIVE_ACCESS", core)
    status, revoked = core.transition_live_grant(revoke_proof, revoke_req, T0, LIVE_GRANT, FP_C)
    assert status == "REVOKED" and revoked is not None
    return core, (active, suspended, revoked)


def test_revoked_scope_accepts_fresh_distinct_grant_identity_at_revision_one() -> None:
    core, g1_history = _revoked_g1_history()
    before_g1 = tuple(
        grant
        for grant in core._accepted_grants.values()
        if grant.live_access_grant_id == LIVE_GRANT
    )
    req, proof = _issue_specific_live_admin(
        core, "GRANT_LIVE_ACCESS", grant_id=GRANT_2, current_revision=0
    )
    status, g2 = core.transition_live_grant(proof, req, T0, GRANT_2, FP_C)
    assert status == "ACTIVE" and g2 is not None and g2.grant_revision == 1
    assert core.validate_live_grant(g2) == "GRANT_ACTIVE"
    assert (
        tuple(
            grant
            for grant in core._accepted_grants.values()
            if grant.live_access_grant_id == LIVE_GRANT
        )
        == before_g1
    )
    assert all(core.validate_live_grant(old) != "GRANT_ACTIVE" for old in g1_history)


def test_revoked_historical_grant_id_cannot_be_reused_without_mutation() -> None:
    core, _ = _revoked_g1_history()
    req, proof = _issue_specific_live_admin(
        core, "GRANT_LIVE_ACCESS", grant_id=LIVE_GRANT, current_revision=0
    )
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert core.transition_live_grant(proof, req, T0, LIVE_GRANT, FP_C)[0] == "AUTHORIZATION_DENIED"
    assert (core._accepted_grants, core._current_grant) == before


def test_suspended_scope_blocks_fresh_second_grant() -> None:
    core, grant_req, grant_proof = _issue_live_admin("GRANT_LIVE_ACCESS")
    assert core.transition_live_grant(grant_proof, grant_req, T0, LIVE_GRANT, FP_C)[0] == "ACTIVE"
    core, suspend_req, suspend_proof = _issue_live_admin("SUSPEND_LIVE_ACCESS", core)
    assert (
        core.transition_live_grant(suspend_proof, suspend_req, T0, LIVE_GRANT, FP_C)[0]
        == "SUSPENDED"
    )
    req, proof = _issue_specific_live_admin(
        core, "GRANT_LIVE_ACCESS", grant_id=GRANT_2, current_revision=0
    )
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert core.transition_live_grant(proof, req, T0, GRANT_2, FP_C)[0] == "AUTHORIZATION_DENIED"
    assert (core._accepted_grants, core._current_grant) == before


def test_never_seen_rule_survives_multiple_revoked_current_grant_identities() -> None:
    core, _ = _revoked_g1_history()
    req, proof = _issue_specific_live_admin(
        core, "GRANT_LIVE_ACCESS", grant_id=GRANT_2, current_revision=0
    )
    assert core.transition_live_grant(proof, req, T0, GRANT_2, FP_C)[0] == "ACTIVE"
    req, proof = _issue_specific_live_admin(
        core, "REVOKE_LIVE_ACCESS", grant_id=GRANT_2, current_revision=1
    )
    assert core.transition_live_grant(proof, req, T0, GRANT_2, FP_C)[0] == "REVOKED"
    reuse_req, reuse_proof = _issue_specific_live_admin(
        core, "GRANT_LIVE_ACCESS", grant_id=LIVE_GRANT, current_revision=0
    )
    before = (copy.deepcopy(core._accepted_grants), copy.deepcopy(core._current_grant))
    assert (
        core.transition_live_grant(reuse_proof, reuse_req, T0, LIVE_GRANT, FP_C)[0]
        == "AUTHORIZATION_DENIED"
    )
    assert (core._accepted_grants, core._current_grant) == before


def _source_closure_content_hash(record: dict[str, Any], definition: dict[str, Any]) -> str:
    payload = {field: record[field] for field in definition["input_fields"]}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("UTF-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("schema_name", ["SessionSecurityState", "SecretMetadataProjection"])
def test_source_owned_terminal_fingerprint_is_schema_minus_terminal(schema_name: str) -> None:
    protocol = thaw(EXPECTED_PROTOCOL)
    schema = protocol["executable_boundary_schemas"][schema_name]
    definition = schema["terminal_fingerprint"]
    assert (
        schema["exact_fields_source_pointer"] if "exact_fields_source_pointer" in schema else True
    )
    assert set(definition["input_fields"]) == set(schema["exact_fields"]) - {
        "content_fingerprint_sha256"
    }
    assert definition == protocol["executable_boundary_terminal_fingerprints"][schema_name]
    assert definition["authority_boundary"].startswith("INTEGRITY_ONLY")


def test_session_and_secret_metadata_mutations_change_content_fingerprint() -> None:
    protocol = thaw(EXPECTED_PROTOCOL)
    for schema_name in ["SessionSecurityState", "SecretMetadataProjection"]:
        schema = protocol["executable_boundary_schemas"][schema_name]
        definition = schema["terminal_fingerprint"]
        record = {field: f"value-{index}" for index, field in enumerate(definition["input_fields"])}
        if "permitted_operations" in record:
            record["permitted_operations"] = ["READ", "ROTATE"]
        baseline = _source_closure_content_hash(record, definition)
        for field in definition["input_fields"]:
            mutated = copy.deepcopy(record)
            mutated[field] = [mutated[field], "mutation"]
            assert _source_closure_content_hash(mutated, definition) != baseline


def test_secret_metadata_fingerprint_input_contains_metadata_only() -> None:
    definition = thaw(EXPECTED_PROTOCOL)["executable_boundary_terminal_fingerprints"][
        "SecretMetadataProjection"
    ]
    forbidden = {
        "raw_secret",
        "api_secret",
        "password",
        "private_key",
        "pin",
        "biometric_material",
        "bearer_token",
        "session_token",
        "decrypted_credential",
        "encrypted_secret_payload",
    }
    assert forbidden.isdisjoint(definition["input_fields"])


def _valid_permitted_operations(value: Any) -> bool:
    protocol = thaw(EXPECTED_PROTOCOL)
    registry = protocol["registries"]["secret_use_operation_registry"]
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(type(item) is str and item in registry for item in value)
        and len(value) == len(set(value))
        and list(value) == [item for item in registry if item in value]
    )


@pytest.mark.parametrize(
    "value",
    [
        ["PRIVATE_DATA"],
        ["ORDER_ENTRY"],
        ["PRIVATE_DATA", "ORDER_ENTRY"],
    ],
)
def test_secret_use_operations_accept_only_canonical_permission_sets(value: list[str]) -> None:
    assert _valid_permitted_operations(value)


@pytest.mark.parametrize(
    "value",
    [
        [],
        ["UNKNOWN"],
        ["PRIVATE_DATA", "PRIVATE_DATA"],
        ["ORDER_ENTRY", "PRIVATE_DATA"],
        "PRIVATE_DATA",
        [1],
        [True],
    ],
)
def test_secret_use_operations_reject_noncanonical_carriers(value: Any) -> None:
    assert not _valid_permitted_operations(value)


def test_secret_use_registry_is_distinct_from_admin_and_credential_permissions() -> None:
    protocol = thaw(EXPECTED_PROTOCOL)
    secret_use = protocol["registries"]["secret_use_operation_registry"]
    assert secret_use == ["PRIVATE_DATA", "ORDER_ENTRY"]
    assert set(secret_use) != set(protocol["operation_policy_registry"])
    upstream = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
    assert set(secret_use) != set(upstream["credential_profile_contract"]["permission_registry"])
    schema = protocol["executable_boundary_schemas"]["SecretMetadataProjection"]["field_schemas"][
        "permitted_operations"
    ]
    assert schema["items_source_pointer"] == "/registries/secret_use_operation_registry"
    assert schema["min_items"] == 1 and schema["unique"] is True
    assert schema["canonical_order"] == "REGISTRY_ORDER"


def test_secret_fingerprint_permutation_is_not_repaired() -> None:
    canonical = make_secret()
    reordered = replace(canonical, permitted_operations=("ORDER_ENTRY", "PRIVATE_DATA"))
    repaired_hash = replace(
        reordered,
        content_fingerprint_sha256=fingerprint(
            {
                key: value
                for key, value in reordered.__dict__.items()
                if key != "content_fingerprint_sha256"
            }
        ),
    )
    assert valid_secret(canonical)
    assert not _valid_permitted_operations(repaired_hash.permitted_operations)
