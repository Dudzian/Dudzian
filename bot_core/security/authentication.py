"""Core-owned M0.10 authentication semantic authority.

This slice issues PIN-only and combined PIN-plus-biometric proofs by consuming
pre-existing external-platform biometric assertion membership.  It does not
authorize requests or execute security transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, NoReturn, Protocol, cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.current_projection_authority import (
    _accept_pin_projection,
    _validate_pin_successor,
)
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityAuthority,
    InitialSecurityAuthoritySnapshot,
    InitialSecurityError,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
)

_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_SOURCE = "CoreHost"
_MAX_FAILED_ATTEMPTS = 3
_LOCKOUT_SECONDS = 300


class AuthenticationError(RuntimeError):
    """Controlled fail-closed error whose graph never retains a raw PIN."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise AuthenticationError(reason)


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    account_id: str
    operator_id: str
    device_installation_id: str
    environment: str
    operation: str
    scope_fingerprint_sha256: str
    mutation_fingerprint_sha256: str
    causation_id: str
    correlation_id: str
    declared_intent: str | None = None


@dataclass(frozen=True, slots=True)
class AuthenticationProof:
    account_id: str
    operator_id: str
    device_installation_id: str
    factor_set: tuple[str, ...]
    issued_at_utc: str
    expires_at_utc: str
    identity_revision: int
    device_trust_revision: int
    pin_revision: int
    platform_enrollment_revision: int
    security_generation: int
    session_generation: int
    environment: str
    operation: str
    scope_fingerprint_sha256: str
    mutation_fingerprint_sha256: str
    causation_id: str
    correlation_id: str
    proof_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class CoreIssuedAuthenticationProofBinding:
    proof_fingerprint_sha256: str
    complete_proof_content_fingerprint_sha256: str
    authority_source: str
    account_id: str
    operator_id: str
    device_installation_id: str
    identity_revision: int
    device_trust_revision: int
    pin_revision: int
    platform_enrollment_revision: int
    security_generation: int
    session_generation: int


@dataclass(frozen=True, slots=True)
class PlatformBiometricAssertion:
    account_id: str
    device_installation_id: str
    platform_authenticator_source: object
    platform_enrollment_revision: int
    challenge_fingerprint_sha256: str
    outcome: str
    verified_at_utc: str
    expires_at_utc: str
    assertion_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class CoreAcceptedPlatformBiometricAssertionBinding:
    assertion_fingerprint_sha256: str
    complete_assertion_content_fingerprint_sha256: str
    authority_source: str
    account_id: str
    device_installation_id: str
    platform_enrollment_revision: int
    challenge_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class OperationPolicy:
    factor_policy: str
    freshness_seconds: int
    authorization_scope: str
    environments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DownstreamOperationDefinition:
    """Core-accepted policy declaration owned by a downstream milestone."""

    owner_milestone: str
    operation: str
    factor_policy: str
    freshness_seconds: int
    authorization_scope: str
    environments: tuple[str, ...]
    target_scope_contract: tuple[str, ...]
    mutation_binding_contract: tuple[str, ...]
    declared_intent: str
    owner_artifact: str
    owner_json_pointer: str
    owner_contract_fingerprint_sha256: str
    declaration_fingerprint_sha256: str
    dependency_fingerprint_sha256: str
    definition_revision: int
    content_fingerprint_sha256: str


OPERATION_POLICY_REGISTRY = MappingProxyType(
    {
        "TRUST_DEVICE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "trust_device", ("PAPER", "TESTNET")
        ),
        "REVOKE_DEVICE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "revoke_device", ("PAPER", "TESTNET")
        ),
        "SETUP_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "setup_pin", ("PAPER", "TESTNET")),
        "CHANGE_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "change_pin", ("PAPER", "TESTNET")),
        "RESET_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "reset_pin", ("PAPER", "TESTNET")),
        "LOCK_SESSION": OperationPolicy("PIN", 60, "lock_session", ("PAPER", "TESTNET")),
        "UNLOCK_SESSION": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "unlock_session", ("PAPER", "TESTNET")
        ),
        "LOGOUT_SESSION": OperationPolicy("PIN", 60, "logout_session", ("PAPER", "TESTNET")),
        "ROTATE_SECRET_REFERENCE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "rotate_secret_reference", ("PAPER", "TESTNET")
        ),
        "REBIND_SECRET_REFERENCE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "rebind_secret_reference", ("PAPER", "TESTNET")
        ),
        "ACTIVATE_CREDENTIAL_PROFILE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "activate_credential_profile", ("PAPER", "TESTNET")
        ),
        "DEACTIVATE_CREDENTIAL_PROFILE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "deactivate_credential_profile", ("PAPER", "TESTNET")
        ),
        "CHANGE_RISK_POLICY": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_risk_policy", ("PAPER", "TESTNET")
        ),
        "CHANGE_KILL_SWITCH": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_kill_switch", ("PAPER", "TESTNET")
        ),
        "CHANGE_PRODUCT_CAPABILITIES": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_product_capabilities", ("PAPER", "TESTNET")
        ),
        "GRANT_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "grant_live_access", ("LIVE",)
        ),
        "SUSPEND_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "suspend_live_access", ("LIVE",)
        ),
        "REVOKE_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "revoke_live_access", ("LIVE",)
        ),
    }
)

OPERATION_OWNERSHIP = MappingProxyType(
    {
        "TRUST_DEVICE": "M0.10_OWNED_TRANSITION",
        "REVOKE_DEVICE": "M0.10_OWNED_TRANSITION",
        "SETUP_PIN": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_PIN": "M0.10_OWNED_TRANSITION",
        "RESET_PIN": "M0.10_OWNED_TRANSITION",
        "LOCK_SESSION": "M0.10_OWNED_TRANSITION",
        "UNLOCK_SESSION": "M0.10_OWNED_TRANSITION",
        "LOGOUT_SESSION": "M0.10_OWNED_TRANSITION",
        "ROTATE_SECRET_REFERENCE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "REBIND_SECRET_REFERENCE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "ACTIVATE_CREDENTIAL_PROFILE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "DEACTIVATE_CREDENTIAL_PROFILE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_RISK_POLICY": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_KILL_SWITCH": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_PRODUCT_CAPABILITIES": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "GRANT_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
        "SUSPEND_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
        "REVOKE_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
    }
)

_DOWNSTREAM_OPERATION_RE = re.compile(r"^(M0\.[0-9]+)/([A-Z][A-Z0-9_]*)$")


def downstream_operation_definition_fingerprint(definition: DownstreamOperationDefinition) -> str:
    return _fingerprint_without(definition, "content_fingerprint_sha256")


_ARCHITECTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "cryptohunter_product_architecture"
)
# Closed, code-owned mapping.  Neither callers nor plugins can add owners/artifacts.
_CANONICAL_DOWNSTREAM_OWNER_ARTIFACTS = MappingProxyType(
    {"M0.12": "audit_observability_alerts_and_updater.json"}
)


def _resolve_json_pointer(document: object, pointer: str) -> object:
    if not pointer.startswith("/"):
        _deny("CONTRACT_INCONSISTENT")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and token.isdecimal():
            index = int(token)
            if index >= len(value):
                _deny("CONTRACT_INCONSISTENT")
            value = value[index]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            _deny("CONTRACT_INCONSISTENT")
    return value


def _validated_artifact_identity(document: object, expected_owner: str) -> tuple[str, str]:
    """Validate actual top-level identity selected by the closed owner mapping."""
    if not isinstance(document, dict):
        _deny("CONTRACT_INCONSISTENT")
    actual_owner = document.get("m0_element")
    schema_version = document.get("schema_version")
    if (
        not isinstance(actual_owner, str)
        or actual_owner != expected_owner
        or not isinstance(schema_version, str)
        or not schema_version
    ):
        _deny("CONTRACT_INCONSISTENT")
    return actual_owner, schema_version


def _canonical_declaration(definition: DownstreamOperationDefinition) -> dict[str, object]:
    expected_artifact = _CANONICAL_DOWNSTREAM_OWNER_ARTIFACTS.get(definition.owner_milestone)
    if expected_artifact is None or definition.owner_artifact != expected_artifact:
        _deny("CONTRACT_INCONSISTENT")
    path = _ARCHITECTURE_ROOT / expected_artifact
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError, TypeError):
        _deny("CONTRACT_INCONSISTENT")
    actual_owner, schema_version = _validated_artifact_identity(
        document, definition.owner_milestone
    )
    declaration = _resolve_json_pointer(document, definition.owner_json_pointer)
    if not isinstance(declaration, dict):
        _deny("CONTRACT_INCONSISTENT")
    contract_fingerprint = cast(
        str,
        canonical_json_sha256(
            {
                "m0_element": actual_owner,
                "schema_version": schema_version,
            }
        ),
    )
    declaration_fingerprint = cast(str, canonical_json_sha256(declaration))
    semantic = {
        "owner_milestone": definition.owner_milestone,
        "operation": definition.operation,
        "factor_policy": definition.factor_policy,
        "freshness_seconds": definition.freshness_seconds,
        "authorization_scope": definition.authorization_scope,
        "environments": list(definition.environments),
        "target_scope_contract": list(definition.target_scope_contract),
        "mutation_binding_contract": list(definition.mutation_binding_contract),
        "declared_intent": definition.declared_intent,
        "definition_revision": definition.definition_revision,
    }
    if any(declaration.get(key) != value for key, value in semantic.items()):
        _deny("CONTRACT_INCONSISTENT")
    if (
        definition.owner_contract_fingerprint_sha256 != contract_fingerprint
        or definition.declaration_fingerprint_sha256 != declaration_fingerprint
        or definition.dependency_fingerprint_sha256 != declaration_fingerprint
    ):
        _deny("CONTRACT_INCONSISTENT")
    return declaration


def _valid_downstream_operation_definition(definition: object) -> bool:
    if not isinstance(definition, DownstreamOperationDefinition):
        return False
    match = _DOWNSTREAM_OPERATION_RE.fullmatch(definition.operation)
    return bool(
        match
        and match.group(1) == definition.owner_milestone
        and definition.operation not in OPERATION_POLICY_REGISTRY
        and definition.factor_policy in {"PIN", "PIN_AND_BIOMETRIC"}
        and isinstance(definition.freshness_seconds, int)
        and not isinstance(definition.freshness_seconds, bool)
        and definition.freshness_seconds > 0
        and isinstance(definition.authorization_scope, str)
        and bool(definition.authorization_scope)
        and isinstance(definition.environments, tuple)
        and bool(definition.environments)
        and set(definition.environments) <= {"PAPER", "TESTNET", "LIVE"}
        and len(set(definition.environments)) == len(definition.environments)
        and isinstance(definition.target_scope_contract, tuple)
        and all(isinstance(field, str) and field for field in definition.target_scope_contract)
        and len(set(definition.target_scope_contract)) == len(definition.target_scope_contract)
        and isinstance(definition.mutation_binding_contract, tuple)
        and all(isinstance(field, str) and field for field in definition.mutation_binding_contract)
        and isinstance(definition.declared_intent, str)
        and bool(definition.declared_intent)
        and len(set(definition.mutation_binding_contract))
        == len(definition.mutation_binding_contract)
        and all(
            _SHA_RE.fullmatch(value)
            for value in (
                definition.owner_contract_fingerprint_sha256,
                definition.declaration_fingerprint_sha256,
                definition.dependency_fingerprint_sha256,
                definition.content_fingerprint_sha256,
            )
        )
        and isinstance(definition.definition_revision, int)
        and not isinstance(definition.definition_revision, bool)
        and definition.definition_revision >= 1
        and downstream_operation_definition_fingerprint(definition)
        == definition.content_fingerprint_sha256
    )


def _definition_from_declaration(
    artifact: str,
    pointer: str,
    artifact_owner: str,
    schema_version: str,
    declaration: dict[str, object],
) -> DownstreamOperationDefinition:
    """Convert validated JSON-domain declaration or fail with controlled taxonomy."""
    required_strings = (
        "owner_milestone",
        "operation",
        "factor_policy",
        "authorization_scope",
        "declared_intent",
    )
    required_lists = (
        "environments",
        "target_scope_contract",
        "mutation_binding_contract",
    )
    try:
        if (
            any(
                not isinstance(declaration[name], str) or not declaration[name]
                for name in required_strings
            )
            or any(
                not isinstance(declaration[name], list)
                or not declaration[name]
                or any(not isinstance(value, str) or not value for value in declaration[name])
                for name in required_lists
            )
            or not isinstance(declaration["freshness_seconds"], int)
            or isinstance(declaration["freshness_seconds"], bool)
            or not isinstance(declaration["definition_revision"], int)
            or isinstance(declaration["definition_revision"], bool)
        ):
            _deny("CONTRACT_INCONSISTENT")
        declaration_fingerprint = cast(str, canonical_json_sha256(declaration))
        item = DownstreamOperationDefinition(
            owner_milestone=declaration["owner_milestone"],
            operation=declaration["operation"],
            factor_policy=declaration["factor_policy"],
            freshness_seconds=declaration["freshness_seconds"],
            authorization_scope=declaration["authorization_scope"],
            environments=tuple(declaration["environments"]),
            target_scope_contract=tuple(declaration["target_scope_contract"]),
            mutation_binding_contract=tuple(declaration["mutation_binding_contract"]),
            declared_intent=declaration["declared_intent"],
            owner_artifact=artifact,
            owner_json_pointer=pointer,
            owner_contract_fingerprint_sha256=cast(
                str,
                canonical_json_sha256(
                    {
                        "m0_element": artifact_owner,
                        "schema_version": schema_version,
                    }
                ),
            ),
            declaration_fingerprint_sha256=declaration_fingerprint,
            dependency_fingerprint_sha256=declaration_fingerprint,
            definition_revision=declaration["definition_revision"],
            content_fingerprint_sha256="",
        )
    except (KeyError, TypeError, ValueError):
        _deny("CONTRACT_INCONSISTENT")
    return replace(
        item, content_fingerprint_sha256=downstream_operation_definition_fingerprint(item)
    )


def _architecture_downstream_definitions() -> tuple[DownstreamOperationDefinition, ...]:
    """Validate and return exact contiguous canonical histories through declared current."""
    result: list[DownstreamOperationDefinition] = []
    for owner, artifact in _CANONICAL_DOWNSTREAM_OWNER_ARTIFACTS.items():
        path = _ARCHITECTURE_ROOT / artifact
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            artifact_owner, schema_version = _validated_artifact_identity(document, owner)
            declarations = document["downstream_operation_declarations"]
            currents = document["current_downstream_operation_definition_revisions"]
        except (OSError, ValueError, TypeError, KeyError):
            _deny("CONTRACT_INCONSISTENT")
        if not isinstance(declarations, list) or not isinstance(currents, dict):
            _deny("CONTRACT_INCONSISTENT")
        grouped: dict[str, list[tuple[int, int, dict[str, object]]]] = {}
        for index, declaration in enumerate(declarations):
            if not isinstance(declaration, dict) or declaration.get("owner_milestone") != owner:
                _deny("CONTRACT_INCONSISTENT")
            operation = declaration.get("operation")
            revision = declaration.get("definition_revision")
            if (
                not isinstance(operation, str)
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision <= 0
            ):
                _deny("CONTRACT_INCONSISTENT")
            grouped.setdefault(operation, []).append((revision, index, declaration))
        if set(currents) != set(grouped):
            _deny("CONTRACT_INCONSISTENT")
        for operation in sorted(grouped):
            current = currents[operation]
            history = sorted(grouped[operation], key=lambda row: row[0])
            revisions = [row[0] for row in history]
            if (
                not isinstance(current, int)
                or isinstance(current, bool)
                or current <= 0
                or revisions != list(range(1, current + 1))
            ):
                _deny("CONTRACT_INCONSISTENT")
            for _, index, declaration in history:
                definition = _definition_from_declaration(
                    artifact,
                    f"/downstream_operation_declarations/{index}",
                    artifact_owner,
                    schema_version,
                    declaration,
                )
                if not _valid_downstream_operation_definition(definition):
                    _deny("CONTRACT_INCONSISTENT")
                result.append(definition)
    return tuple(result)


def _seed_trusted_downstream_operation_definition(
    authority: AuthenticationAuthority,
    definition: DownstreamOperationDefinition,
    *,
    current: bool = True,
) -> None:
    """Accept only an independently resolved canonical architecture declaration."""
    if not isinstance(
        authority, AuthenticationAuthority
    ) or not _valid_downstream_operation_definition(definition):
        _deny("CONTRACT_INCONSISTENT")
    _canonical_declaration(definition)
    with authority._state.lock:  # noqa: SLF001
        before = authority._state.snapshot  # noqa: SLF001
        accepted = dict(before.accepted_downstream_operation_definitions)
        designations = dict(before.current_downstream_operation_definitions)
        existing = accepted.get(definition.content_fingerprint_sha256)
        if existing is not None and existing != definition:
            _deny("CONTRACT_INCONSISTENT")
        current_fingerprint = designations.get(definition.operation)
        current_definition = accepted.get(current_fingerprint) if current_fingerprint else None
        if current and current_definition is not None:
            if not isinstance(current_definition, DownstreamOperationDefinition):
                _deny("CONTRACT_INCONSISTENT")
            if definition.definition_revision < current_definition.definition_revision:
                _deny("CONTRACT_INCONSISTENT")
            if definition.definition_revision == current_definition.definition_revision:
                if definition.content_fingerprint_sha256 != current_fingerprint:
                    _deny("CONTRACT_INCONSISTENT")
                return
            if definition.definition_revision != current_definition.definition_revision + 1:
                _deny("CONTRACT_INCONSISTENT")
        elif current and definition.definition_revision != 1:
            _deny("CONTRACT_INCONSISTENT")
        accepted[definition.content_fingerprint_sha256] = definition
        if current:
            designations[definition.operation] = definition.content_fingerprint_sha256
        authority._state.snapshot = replace(  # noqa: SLF001
            before,
            accepted_downstream_operation_definitions=MappingProxyType(accepted),
            current_downstream_operation_definitions=MappingProxyType(designations),
        )


class PinVerifierComparator(Protocol):
    """Production KDF/secure-store comparison boundary; it grants no authority."""

    def compare(self, raw_pin: str, record: PinVerifierRecord) -> bool: ...


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_utc(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def _parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.endswith("Z"):
        _deny("CONTRACT_INCONSISTENT")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _deny("CONTRACT_INCONSISTENT")
    if parsed.tzinfo != timezone.utc:
        _deny("CONTRACT_INCONSISTENT")
    return parsed


def _valid_canonical_utc_text(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo == timezone.utc and _utc_text(parsed) == value


def _fingerprint_without(value: Any, field: str) -> str:
    return cast(str, canonical_json_sha256({k: v for k, v in asdict(value).items() if k != field}))


def authentication_proof_fingerprint(proof: AuthenticationProof) -> str:
    return _fingerprint_without(proof, "proof_fingerprint_sha256")


def complete_authentication_proof_fingerprint(proof: AuthenticationProof) -> str:
    return cast(str, canonical_json_sha256(asdict(proof)))


def platform_biometric_assertion_fingerprint(assertion: PlatformBiometricAssertion) -> str:
    return _fingerprint_without(assertion, "assertion_fingerprint_sha256")


def complete_platform_biometric_assertion_fingerprint(
    assertion: PlatformBiometricAssertion,
) -> str:
    return cast(str, canonical_json_sha256(asdict(assertion)))


def core_expected_biometric_challenge(
    request: AuthorizationRequest,
    platform_enrollment_revision: int,
    security_generation: int,
    session_generation: int,
) -> str:
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-BIOMETRIC-CHALLENGE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                request.operation,
                request.scope_fingerprint_sha256,
                request.mutation_fingerprint_sha256,
                request.causation_id,
                request.correlation_id,
                platform_enrollment_revision,
                security_generation,
                session_generation,
            ]
        ),
    )


def canonical_scope_fingerprint(request: AuthorizationRequest) -> str:
    policy = OPERATION_POLICY_REGISTRY.get(request.operation)
    scope = policy.authorization_scope if policy is not None else "unsupported"
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-SCOPE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                request.operation,
                scope,
            ]
        ),
    )


def downstream_scope_fingerprint(
    definition: DownstreamOperationDefinition,
    request: AuthorizationRequest,
    target_scope: dict[str, object],
) -> str:
    """Derive the exact downstream target scope, including accepted-definition identity."""
    if not _valid_downstream_operation_definition(definition) or tuple(target_scope) != (
        definition.target_scope_contract
    ):
        _deny("MALFORMED_UNTRUSTED_CONTEXT")
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-DOWNSTREAM-SCOPE-V1",
                definition.owner_milestone,
                definition.operation,
                definition.declared_intent,
                definition.content_fingerprint_sha256,
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                target_scope,
            ]
        ),
    )


def downstream_mutation_fingerprint(
    definition: DownstreamOperationDefinition,
    request: AuthorizationRequest,
    target_scope: dict[str, object],
    mutation: dict[str, object],
) -> str:
    """Derive exact downstream mutation intent and its target/revision edge."""
    if (
        not _valid_downstream_operation_definition(definition)
        or tuple(target_scope) != definition.target_scope_contract
        or tuple(mutation) != definition.mutation_binding_contract
        or mutation.get("intent") != definition.declared_intent
    ):
        _deny("MALFORMED_UNTRUSTED_CONTEXT")
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-DOWNSTREAM-MUTATION-V1",
                definition.owner_milestone,
                definition.operation,
                definition.declared_intent,
                definition.content_fingerprint_sha256,
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                target_scope,
                mutation,
                request.causation_id,
                request.correlation_id,
            ]
        ),
    )


def session_mutation_fingerprint(
    request: AuthorizationRequest,
    target_state: str,
    current_generation: int,
    next_generation: int,
) -> str:
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-SESSION",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.operation,
                target_state,
                current_generation,
                next_generation,
                policy.authorization_scope,
            ]
        ),
    )


def device_mutation_fingerprint(
    request: AuthorizationRequest,
    target_device_id: str,
    current_revision: int,
    next_revision: int,
) -> str:
    """Bind a device transition to its exact target and revision edge."""
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-DEVICE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.operation,
                target_device_id,
                current_revision,
                next_revision,
                policy.authorization_scope,
            ]
        ),
    )


def pin_mutation_fingerprint(
    request: AuthorizationRequest,
    current_revision: int,
    next_revision: int,
) -> str:
    """Bind a PIN replacement to its exact current-to-next revision edge."""
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-PIN-INTENT",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.operation,
                current_revision,
                next_revision,
                policy.authorization_scope,
            ]
        ),
    )


class AuthenticationAuthority:
    """Issues and resolves genuine proofs over the shared initial-security state."""

    def __init__(
        self,
        security: InitialSecurityAuthority,
        comparator: PinVerifierComparator,
    ) -> None:
        self._state = security._state  # noqa: SLF001 -- trusted owner over the exact shared plane
        self._comparator = comparator
        self._runtime_sessions = security._runtime_sessions  # noqa: SLF001
        # Validate and build the complete ephemeral authority in shadow state.
        # No live state changes until every canonical artifact has closed successfully.
        definitions = _architecture_downstream_definitions()
        accepted = {item.content_fingerprint_sha256: item for item in definitions}
        expected_current = {item.operation: item for item in definitions}
        current = {
            operation: item.content_fingerprint_sha256
            for operation, item in expected_current.items()
        }
        if (
            len(accepted) != len(definitions)
            or set(current) != set(expected_current)
            or any(
                accepted.get(fingerprint) != expected_current[operation]
                for operation, fingerprint in current.items()
            )
        ):
            _deny("CONTRACT_INCONSISTENT")
        with self._state.lock:
            before = self._state.snapshot
            self._state.snapshot = replace(
                before,
                accepted_authentication_proofs=MappingProxyType({}),
                accepted_authentication_proof_bindings=MappingProxyType({}),
                authentication_proof_operation_definitions=MappingProxyType({}),
                accepted_downstream_operation_definitions=MappingProxyType(accepted),
                current_downstream_operation_definitions=MappingProxyType(current),
            )

    @property
    def snapshot(self) -> InitialSecurityAuthoritySnapshot:
        return self._state.snapshot

    def derive_platform_biometric_challenge(self, request: object) -> str:
        """Derive the challenge solely from current Core-owned semantic authority."""
        if not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        with self._state.lock:
            self._validate_operation_request_semantics_locked(self._state.snapshot, request)
            identity, device, session = self._resolve_biometric_context(
                self._state.snapshot, request
            )
            return core_expected_biometric_challenge(
                request,
                device.platform_enrollment_revision,
                identity.security_generation,
                session.session_generation,
            )

    def verify_current_pin(
        self,
        account_id: object,
        operator_id: object,
        device_installation_id: object,
        raw_pin: object,
        now_utc: object,
    ) -> str:
        """Verify the current genuine PIN record without granting authority."""
        if (
            not self._canonical_id(account_id, "acct_")
            or not self._canonical_id(operator_id, "op_")
            or not self._canonical_id(device_installation_id, "dev_")
            or not isinstance(raw_pin, str)
            or not _valid_utc(now_utc)
        ):
            return "MALFORMED_UNTRUSTED_CONTEXT"
        scope = (account_id, operator_id, device_installation_id)
        with self._state.lock:
            before = self._state.snapshot
            try:
                pin = cast(
                    PinVerifierRecord,
                    self._resolve_current_projection(
                        before.accepted_pins,
                        before.current_pins,
                        scope,
                        PinVerifierRecord,
                        scope,
                        lambda value: (
                            value.account_id,
                            value.operator_id,
                            value.device_installation_id,
                        ),
                        lambda _value: True,
                        "AUTHENTICATION_FAILED",
                    ),
                )
                self._validate_pin(pin)
            except AuthenticationError as error:
                return error.reason

            current_time = cast(datetime, now_utc)
            lockout = _parse_utc(pin.lockout_until_utc)
            if lockout is not None and current_time < lockout:
                return "PIN_LOCKED"
            try:
                matched = self._comparator.compare(raw_pin, pin)
            except Exception:
                # Do not retain or expose a dependency exception that may contain the PIN.
                return "PIN_VERIFIER_DEPENDENCY_FAILURE"
            if type(matched) is not bool:
                return "PIN_VERIFIER_DEPENDENCY_FAILURE"
            if not matched:
                self._publish_pin_failure(before, pin, current_time)
                return (
                    "PIN_LOCKED"
                    if pin.failed_attempts + 1 >= _MAX_FAILED_ATTEMPTS
                    else "AUTHENTICATION_FAILED"
                )

            if pin.failed_attempts or pin.lockout_until_utc is not None:
                updated = self._updated_pin(pin, 0, None)
                try:
                    if not _accept_pin_projection(self._state, updated):
                        return "CONTRACT_INCONSISTENT"
                except InitialSecurityError:
                    return "CONTRACT_INCONSISTENT"
            return "PIN_ACCEPTED"

    def verify_platform_assertion(self, assertion: object, request: object, now_utc: object) -> str:
        """Consume pre-existing external-platform membership as biometric evidence."""
        if not _valid_utc(now_utc) or not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        if not self._valid_assertion_shape(assertion):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        candidate = cast(PlatformBiometricAssertion, assertion)
        current_time = cast(datetime, now_utc)
        with self._state.lock:
            snapshot = self._state.snapshot
            self._validate_operation_request_semantics_locked(snapshot, request)
            identity, device, session = self._resolve_biometric_context(snapshot, request)
            expected_challenge = core_expected_biometric_challenge(
                request,
                device.platform_enrollment_revision,
                identity.security_generation,
                session.session_generation,
            )
            try:
                exact = CoreAcceptedPlatformBiometricAssertionBinding(
                    candidate.assertion_fingerprint_sha256,
                    complete_platform_biometric_assertion_fingerprint(candidate),
                    "external_platform_authenticator",
                    candidate.account_id,
                    candidate.device_installation_id,
                    candidate.platform_enrollment_revision,
                    candidate.challenge_fingerprint_sha256,
                )
            except (TypeError, ValueError):
                _deny("MALFORMED_UNTRUSTED_CONTEXT")
            binding = snapshot.accepted_platform_biometric_assertion_bindings.get(
                candidate.assertion_fingerprint_sha256
            )
            if binding != exact:
                _deny("AUTHENTICATION_FAILED")
            if candidate.outcome == "UNAVAILABLE":
                _deny("FACTOR_UNAVAILABLE")
            if candidate.outcome != "SUCCESS":
                _deny("AUTHENTICATION_FAILED")
            if (
                candidate.account_id,
                candidate.device_installation_id,
                candidate.platform_enrollment_revision,
                candidate.challenge_fingerprint_sha256,
            ) != (
                request.account_id,
                request.device_installation_id,
                device.platform_enrollment_revision,
                expected_challenge,
            ):
                _deny("AUTHENTICATION_FAILED")
            start = _parse_utc(candidate.verified_at_utc)
            end = _parse_utc(candidate.expires_at_utc)
            if start is None or end is None or not start <= current_time <= end:
                _deny("AUTHENTICATION_FAILED")
            return "BIOMETRIC_ACCEPTED"

    @staticmethod
    def _valid_assertion_shape(assertion: object) -> bool:
        if not isinstance(assertion, PlatformBiometricAssertion):
            return False
        try:
            return bool(
                isinstance(assertion.account_id, str)
                and assertion.account_id.startswith("acct_")
                and _ID_RE.fullmatch(assertion.account_id)
                and isinstance(assertion.device_installation_id, str)
                and assertion.device_installation_id.startswith("dev_")
                and _ID_RE.fullmatch(assertion.device_installation_id)
                and assertion.outcome in {"SUCCESS", "FAILED", "CANCELLED", "UNAVAILABLE"}
                and isinstance(assertion.platform_enrollment_revision, int)
                and not isinstance(assertion.platform_enrollment_revision, bool)
                and assertion.platform_enrollment_revision >= 1
                and isinstance(assertion.challenge_fingerprint_sha256, str)
                and _SHA_RE.fullmatch(assertion.challenge_fingerprint_sha256)
                and _valid_canonical_utc_text(assertion.verified_at_utc)
                and _valid_canonical_utc_text(assertion.expires_at_utc)
                and isinstance(assertion.assertion_fingerprint_sha256, str)
                and _SHA_RE.fullmatch(assertion.assertion_fingerprint_sha256)
                and platform_biometric_assertion_fingerprint(assertion)
                == assertion.assertion_fingerprint_sha256
            )
        except (TypeError, ValueError):
            return False

    def issue_authentication_proof(
        self,
        request: object,
        raw_pin: object,
        now: object,
        *,
        platform_assertion: object = None,
    ) -> AuthenticationProof:
        if not _valid_utc(now):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        if not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        current_time = cast(datetime, now)
        with self._state.lock:
            before = self._state.snapshot
            policy, ownership, definition_fingerprint = (
                self._validate_operation_request_semantics_locked(before, request)
            )
            if request.environment not in policy.environments:
                _deny("AUTHORIZATION_DENIED")
            pin_only = policy.factor_policy == "PIN" and request.operation in {
                "LOCK_SESSION",
                "LOGOUT_SESSION",
            }
            combined = policy.factor_policy == "PIN_AND_BIOMETRIC"
            if ownership == "DOWNSTREAM_OWNED_PRIVILEGED_OPERATION":
                pin_only = policy.factor_policy == "PIN"
            if not pin_only and not combined:
                _deny("OPERATION_UNSUPPORTED")
            if combined and platform_assertion is None:
                _deny("AUTHENTICATION_FAILED")
            if not isinstance(raw_pin, str):
                _deny("AUTHENTICATION_FAILED" if combined else "MALFORMED_UNTRUSTED_CONTEXT")
            identity, device, pin, session = self._resolve_current_family(before, request)
            runtime = self._resolve_runtime(request, session)
            if (
                ownership == "M0.10_OWNED_TRANSITION"
                and request.scope_fingerprint_sha256 != canonical_scope_fingerprint(request)
            ):
                _deny("AUTHORIZATION_DENIED")
            if request.operation in {"LOCK_SESSION", "LOGOUT_SESSION"}:
                target_state = {
                    "LOCK_SESSION": "LOCKED",
                    "LOGOUT_SESSION": "LOGGED_OUT",
                }[request.operation]
                expected_mutation = session_mutation_fingerprint(
                    request,
                    target_state,
                    session.session_generation,
                    session.session_generation + 1,
                )
                if request.mutation_fingerprint_sha256 != expected_mutation:
                    _deny("AUTHORIZATION_DENIED")

            lockout = _parse_utc(pin.lockout_until_utc)
            if lockout is not None and current_time < lockout:
                _deny("PIN_LOCKED")
            dependency_failed = False
            try:
                matched = self._comparator.compare(raw_pin, pin)
            except Exception:
                # Leave the dependency exception's possibly sensitive graph behind.
                dependency_failed = True
                matched = False
            if dependency_failed:
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            if type(matched) is not bool:
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            if not matched:
                self._publish_pin_failure(before, pin, current_time)
                _deny(
                    "PIN_LOCKED"
                    if pin.failed_attempts + 1 >= _MAX_FAILED_ATTEMPTS
                    else "AUTHENTICATION_FAILED"
                )

            if combined:
                try:
                    self.verify_platform_assertion(platform_assertion, request, current_time)
                except AuthenticationError as error:
                    if error.reason in {
                        "AUTHENTICATION_FAILED",
                        "FACTOR_UNAVAILABLE",
                        "MALFORMED_UNTRUSTED_CONTEXT",
                    }:
                        _deny("AUTHENTICATION_FAILED")
                    raise

            effective_pin = pin
            if pin.failed_attempts or pin.lockout_until_utc is not None:
                effective_pin = self._updated_pin(pin, 0, None)
            proof = self._build_proof(
                request,
                policy,
                current_time,
                identity,
                device,
                effective_pin,
                session,
            )
            binding = self._binding(proof)
            self._final_runtime_fence(request, session, runtime)
            accepted_pins = dict(before.accepted_pins)
            current_pins = dict(before.current_pins)
            if effective_pin != pin:
                try:
                    valid_successor = _validate_pin_successor(before, effective_pin)
                except InitialSecurityError:
                    _deny("CONTRACT_INCONSISTENT")
                if not valid_successor:
                    _deny("CONTRACT_INCONSISTENT")
                accepted_pins[effective_pin.content_fingerprint_sha256] = effective_pin
                current_pins[(pin.account_id, pin.operator_id, pin.device_installation_id)] = (
                    effective_pin.content_fingerprint_sha256
                )
            proofs = dict(before.accepted_authentication_proofs)
            bindings = dict(before.accepted_authentication_proof_bindings)
            proof_definitions = dict(before.authentication_proof_operation_definitions)
            proofs[proof.proof_fingerprint_sha256] = proof
            bindings[proof.proof_fingerprint_sha256] = binding
            proof_definitions[proof.proof_fingerprint_sha256] = definition_fingerprint
            self._state.snapshot = replace(
                before,
                accepted_pins=MappingProxyType(accepted_pins),
                current_pins=MappingProxyType(current_pins),
                accepted_authentication_proofs=MappingProxyType(proofs),
                accepted_authentication_proof_bindings=MappingProxyType(bindings),
                authentication_proof_operation_definitions=MappingProxyType(proof_definitions),
            )
            return proof

    def resolve_accepted_proof(self, candidate: object) -> AuthenticationProof:
        with self._state.lock:
            snapshot = self._state.snapshot
            accepted_proof = self._resolve_accepted_proof_membership(candidate, snapshot)
        request = AuthorizationRequest(
            accepted_proof.account_id,
            accepted_proof.operator_id,
            accepted_proof.device_installation_id,
            accepted_proof.environment,
            accepted_proof.operation,
            accepted_proof.scope_fingerprint_sha256,
            accepted_proof.mutation_fingerprint_sha256,
            accepted_proof.causation_id,
            accepted_proof.correlation_id,
        )
        identity, device, pin, session = self._resolve_current_family(snapshot, request)
        if (
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
        ) != (
            accepted_proof.identity_revision,
            accepted_proof.device_trust_revision,
            accepted_proof.pin_revision,
            accepted_proof.platform_enrollment_revision,
            accepted_proof.security_generation,
            accepted_proof.session_generation,
        ):
            _deny("PROOF_STALE")
        return accepted_proof

    def _resolve_accepted_proof_membership(
        self,
        candidate: object,
        snapshot: InitialSecurityAuthoritySnapshot,
    ) -> AuthenticationProof:
        """Resolve only genuine Core-issued proof membership and its exact binding."""
        if not isinstance(candidate, AuthenticationProof):
            _deny("AUTHENTICATION_REQUIRED")
        self._validate_authentication_proof_structure(candidate)
        recomputed = authentication_proof_fingerprint(candidate)
        if candidate.proof_fingerprint_sha256 != recomputed:
            _deny("AUTHENTICATION_REQUIRED")
        accepted = snapshot.accepted_authentication_proofs.get(recomputed)
        binding = snapshot.accepted_authentication_proof_bindings.get(recomputed)
        if accepted != candidate or not isinstance(binding, CoreIssuedAuthenticationProofBinding):
            _deny("AUTHENTICATION_REQUIRED")
        if binding != self._binding(candidate):
            _deny("CONTRACT_INCONSISTENT")
        _, _, current_definition = self._resolve_operation(snapshot, candidate.operation)
        bound_definition = snapshot.authentication_proof_operation_definitions.get(recomputed)
        if candidate.operation in OPERATION_POLICY_REGISTRY:
            # Compatibility contract: built-ins bind directly to the immutable
            # Core policy fingerprint; they have no downstream definition.
            if bound_definition not in (None, current_definition):
                _deny("CONTRACT_INCONSISTENT")
        else:
            if not isinstance(bound_definition, str):
                _deny("CONTRACT_INCONSISTENT")
            definition = snapshot.accepted_downstream_operation_definitions.get(bound_definition)
            if not isinstance(definition, DownstreamOperationDefinition):
                _deny("CONTRACT_INCONSISTENT")
            if definition.operation != candidate.operation or definition.definition_revision < 1:
                _deny("CONTRACT_INCONSISTENT")
            _canonical_declaration(definition)
            if bound_definition != current_definition:
                _deny("PROOF_STALE")
        return candidate

    @staticmethod
    def _validate_authentication_proof_structure(proof: AuthenticationProof) -> None:
        positive_epochs = (
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        )
        if (
            not all(
                isinstance(value, str)
                for value in (
                    proof.account_id,
                    proof.operator_id,
                    proof.device_installation_id,
                    proof.environment,
                    proof.operation,
                    proof.scope_fingerprint_sha256,
                    proof.mutation_fingerprint_sha256,
                    proof.causation_id,
                    proof.correlation_id,
                    proof.proof_fingerprint_sha256,
                )
            )
            or not _ID_RE.fullmatch(proof.account_id)
            or not proof.account_id.startswith("acct_")
            or not _ID_RE.fullmatch(proof.operator_id)
            or not proof.operator_id.startswith("op_")
            or not _ID_RE.fullmatch(proof.device_installation_id)
            or not proof.device_installation_id.startswith("dev_")
            or not isinstance(proof.factor_set, tuple)
            or proof.factor_set not in (("PIN",), ("BIOMETRIC",), ("PIN", "BIOMETRIC"))
            or not _valid_canonical_utc_text(proof.issued_at_utc)
            or not _valid_canonical_utc_text(proof.expires_at_utc)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in positive_epochs
            )
            or proof.environment not in {"PAPER", "TESTNET", "LIVE"}
            or not _SHA_RE.fullmatch(proof.scope_fingerprint_sha256)
            or not _SHA_RE.fullmatch(proof.mutation_fingerprint_sha256)
            or not _SHA_RE.fullmatch(proof.proof_fingerprint_sha256)
            or not proof.causation_id
            or not proof.correlation_id
        ):
            _deny("AUTHENTICATION_REQUIRED")

    @staticmethod
    def _resolve_operation(
        snapshot: InitialSecurityAuthoritySnapshot, operation: str
    ) -> tuple[OperationPolicy, str, str]:
        policy = OPERATION_POLICY_REGISTRY.get(operation)
        if policy is not None:
            owner = OPERATION_OWNERSHIP.get(operation)
            if owner is None:
                _deny("CONTRACT_INCONSISTENT")
            return policy, owner, cast(str, canonical_json_sha256([operation, asdict(policy)]))
        fingerprint = snapshot.current_downstream_operation_definitions.get(operation)
        if not isinstance(fingerprint, str):
            _deny("OPERATION_UNSUPPORTED")
        definition = snapshot.accepted_downstream_operation_definitions.get(fingerprint)
        if (
            not _valid_downstream_operation_definition(definition)
            or definition.content_fingerprint_sha256 != fingerprint
            or definition.operation != operation
        ):
            _deny("CONTRACT_INCONSISTENT")
        _canonical_declaration(definition)
        return (
            OperationPolicy(
                definition.factor_policy,
                definition.freshness_seconds,
                definition.authorization_scope,
                definition.environments,
            ),
            "DOWNSTREAM_OWNED_PRIVILEGED_OPERATION",
            fingerprint,
        )

    @classmethod
    def _validate_operation_request_semantics_locked(
        cls,
        snapshot: InitialSecurityAuthoritySnapshot,
        request: AuthorizationRequest,
    ) -> tuple[OperationPolicy, str, str]:
        """Give every request field one deterministic meaning from current Core authority."""
        policy, ownership, fingerprint = cls._resolve_operation(snapshot, request.operation)
        if ownership == "DOWNSTREAM_OWNED_PRIVILEGED_OPERATION":
            definition = snapshot.accepted_downstream_operation_definitions.get(fingerprint)
            if (
                not isinstance(definition, DownstreamOperationDefinition)
                or request.declared_intent != definition.declared_intent
            ):
                _deny("AUTHORIZATION_DENIED")
        elif request.declared_intent is not None:
            _deny("AUTHORIZATION_DENIED")
        return policy, ownership, fingerprint

    def _resolve_runtime(
        self, request: AuthorizationRequest, session: SessionSecurityState
    ) -> RuntimeSession:
        dependency_failed = False
        try:
            runtime = self._runtime_sessions.resolve_current(
                request.account_id, request.device_installation_id
            )
        except Exception:
            dependency_failed = True
            runtime = None
        if dependency_failed or not self._runtime_matches(runtime, request, session):
            _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
        return cast(RuntimeSession, runtime)

    @staticmethod
    def _runtime_matches(
        runtime: object, request: AuthorizationRequest, session: SessionSecurityState
    ) -> bool:
        return (
            isinstance(runtime, RuntimeSession)
            and type(runtime) is RuntimeSession
            and not runtime.closed
            and runtime.device_installation_id == request.device_installation_id
            and runtime.runtime_session_id == session.runtime_session_id
        )

    def _final_runtime_fence(
        self,
        request: AuthorizationRequest,
        session: SessionSecurityState,
        originally_resolved: RuntimeSession,
    ) -> None:
        dependency_failed = False
        try:
            current = self._runtime_sessions.resolve_current(
                request.account_id, request.device_installation_id
            )
        except Exception:
            dependency_failed = True
            current = None
        if (
            dependency_failed
            or current is not originally_resolved
            or not self._runtime_matches(originally_resolved, request, session)
            or not self._runtime_matches(current, request, session)
        ):
            _deny("RUNTIME_SESSION_AUTHORITY_DENIED")

    @staticmethod
    def _validate_request_structure(request: AuthorizationRequest) -> None:
        if (
            not all(
                isinstance(value, str)
                for value in (
                    request.account_id,
                    request.operator_id,
                    request.device_installation_id,
                    request.environment,
                    request.operation,
                    request.scope_fingerprint_sha256,
                    request.mutation_fingerprint_sha256,
                    request.causation_id,
                    request.correlation_id,
                )
            )
            or not _ID_RE.fullmatch(request.account_id)
            or not request.account_id.startswith("acct_")
            or not _ID_RE.fullmatch(request.operator_id)
            or not request.operator_id.startswith("op_")
            or not _ID_RE.fullmatch(request.device_installation_id)
            or not request.device_installation_id.startswith("dev_")
            or request.environment not in {"PAPER", "TESTNET", "LIVE"}
            or not _SHA_RE.fullmatch(request.scope_fingerprint_sha256)
            or not _SHA_RE.fullmatch(request.mutation_fingerprint_sha256)
            or not (
                request.declared_intent is None
                or (isinstance(request.declared_intent, str) and request.declared_intent)
            )
            or not isinstance(request.causation_id, str)
            or not request.causation_id
            or not isinstance(request.correlation_id, str)
            or not request.correlation_id
        ):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")

    def _resolve_current_family(
        self, snapshot: InitialSecurityAuthoritySnapshot, request: AuthorizationRequest
    ) -> tuple[
        OperatorIdentitySecurityProjection,
        DeviceTrustProjection,
        PinVerifierRecord,
        SessionSecurityState,
    ]:
        scope_i = (request.account_id, request.operator_id)
        scope_d = (request.account_id, request.device_installation_id)
        scope_p = (*scope_i, request.device_installation_id)

        identity = cast(
            OperatorIdentitySecurityProjection,
            self._resolve_current_projection(
                snapshot.accepted_identities,
                snapshot.current_identities,
                scope_i,
                OperatorIdentitySecurityProjection,
                scope_i,
                lambda value: (value.account_id, value.operator_id),
                self._identity_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
            ),
        )
        device = cast(
            DeviceTrustProjection,
            self._resolve_current_projection(
                snapshot.accepted_devices,
                snapshot.current_devices,
                scope_d,
                DeviceTrustProjection,
                scope_d,
                lambda value: (value.account_id, value.device_installation_id),
                self._device_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
            ),
        )
        pin = cast(
            PinVerifierRecord,
            self._resolve_current_projection(
                snapshot.accepted_pins,
                snapshot.current_pins,
                scope_p,
                PinVerifierRecord,
                scope_p,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                lambda _value: True,
                "CONTRACT_INCONSISTENT",
            ),
        )
        session = cast(
            SessionSecurityState,
            self._resolve_current_projection(
                snapshot.accepted_sessions,
                snapshot.current_sessions,
                scope_p,
                SessionSecurityState,
                scope_p,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                self._session_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
            ),
        )
        if not (
            isinstance(identity, OperatorIdentitySecurityProjection)
            and isinstance(device, DeviceTrustProjection)
            and isinstance(pin, PinVerifierRecord)
            and isinstance(session, SessionSecurityState)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in (
                    identity.identity_revision,
                    identity.security_generation,
                    device.trust_revision,
                    device.platform_enrollment_revision,
                    device.security_generation,
                    session.session_generation,
                    session.security_generation,
                )
            )
        ):
            _deny("CONTRACT_INCONSISTENT")
        if (
            (identity.account_id, identity.operator_id) != scope_i
            or (device.account_id, device.device_installation_id) != scope_d
            or (pin.account_id, pin.operator_id, pin.device_installation_id) != scope_p
            or (session.account_id, session.operator_id, session.device_installation_id) != scope_p
        ):
            _deny("CONTRACT_INCONSISTENT")
        if identity.state != "ACTIVE":
            _deny("IDENTITY_INVALID")
        if device.state != "TRUSTED":
            _deny("DEVICE_NOT_TRUSTED")
        if session.state != "UNLOCKED":
            _deny("AUTHENTICATION_REQUIRED")
        self._validate_pin(pin)
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    pin.security_generation,
                    session.security_generation,
                }
            )
            != 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        return identity, device, pin, session

    def _resolve_biometric_context(
        self, snapshot: InitialSecurityAuthoritySnapshot, request: AuthorizationRequest
    ) -> tuple[
        OperatorIdentitySecurityProjection,
        DeviceTrustProjection,
        SessionSecurityState,
    ]:
        identity_scope = (request.account_id, request.operator_id)
        device_scope = (request.account_id, request.device_installation_id)
        session_scope = (*identity_scope, request.device_installation_id)
        identity = cast(
            OperatorIdentitySecurityProjection,
            self._resolve_current_projection(
                snapshot.accepted_identities,
                snapshot.current_identities,
                identity_scope,
                OperatorIdentitySecurityProjection,
                identity_scope,
                lambda value: (value.account_id, value.operator_id),
                self._identity_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        device = cast(
            DeviceTrustProjection,
            self._resolve_current_projection(
                snapshot.accepted_devices,
                snapshot.current_devices,
                device_scope,
                DeviceTrustProjection,
                device_scope,
                lambda value: (value.account_id, value.device_installation_id),
                self._device_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        session = cast(
            SessionSecurityState,
            self._resolve_current_projection(
                snapshot.accepted_sessions,
                snapshot.current_sessions,
                session_scope,
                SessionSecurityState,
                session_scope,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                self._session_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    session.security_generation,
                }
            )
            != 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        return identity, device, session

    @staticmethod
    def _resolve_current_projection(
        accepted: object,
        current: object,
        designation_scope: object,
        expected_type: type[object],
        expected_payload_scope: object,
        payload_scope: Callable[[Any], object],
        intrinsic_validator: Callable[[Any], bool],
        missing_reason: str,
    ) -> object:
        if not isinstance(accepted, MappingProxyType) or not isinstance(current, MappingProxyType):
            _deny("CONTRACT_INCONSISTENT")
        fingerprint = current.get(designation_scope)
        if fingerprint is None:
            _deny(missing_reason)
        if not isinstance(fingerprint, str):
            _deny("CONTRACT_INCONSISTENT")
        value = accepted.get(fingerprint)
        if value is None:
            _deny(missing_reason)
        if not isinstance(value, expected_type):
            _deny("CONTRACT_INCONSISTENT")
        terminal_fingerprint = getattr(value, "content_fingerprint_sha256", None)
        if not isinstance(terminal_fingerprint, str) or not _SHA_RE.fullmatch(terminal_fingerprint):
            _deny("CONTRACT_INCONSISTENT")
        try:
            recomputed_fingerprint = _fingerprint_without(value, "content_fingerprint_sha256")
            valid = (
                terminal_fingerprint == fingerprint == recomputed_fingerprint
                and payload_scope(value) == expected_payload_scope
                and intrinsic_validator(value)
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            _deny("CONTRACT_INCONSISTENT")
        return value

    @staticmethod
    def _canonical_id(value: object, prefix: str) -> bool:
        return isinstance(value, str) and value.startswith(prefix) and bool(_ID_RE.fullmatch(value))

    @staticmethod
    def _positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    @classmethod
    def _identity_intrinsically_valid(cls, value: OperatorIdentitySecurityProjection) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.operator_id, "op_")
            and value.state in {"ACTIVE", "REVOKED"}
            and cls._positive_int(value.identity_revision)
            and cls._positive_int(value.security_generation)
        )

    @classmethod
    def _device_intrinsically_valid(cls, value: DeviceTrustProjection) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.device_installation_id, "dev_")
            and value.state in {"ENROLLED_UNTRUSTED", "TRUSTED", "REVOKED", "REPLACED"}
            and cls._positive_int(value.trust_revision)
            and cls._positive_int(value.security_generation)
            and cls._positive_int(value.platform_enrollment_revision)
        )

    @classmethod
    def _session_intrinsically_valid(cls, value: SessionSecurityState) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.operator_id, "op_")
            and cls._canonical_id(value.device_installation_id, "dev_")
            and cls._canonical_id(value.runtime_session_id, "run_")
            and value.state in {"LOCKED", "UNLOCKED", "LOGGED_OUT"}
            and cls._positive_int(value.session_generation)
            and cls._positive_int(value.security_generation)
        )

    @staticmethod
    def _validate_pin(pin: PinVerifierRecord) -> None:
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
        locator = (
            pin.salt_reference[len("secure-store://") :]
            if isinstance(pin.salt_reference, str)
            and pin.salt_reference.startswith("secure-store://")
            else ""
        )
        if (
            not isinstance(pin.algorithm_id, str)
            or not pin.algorithm_id
            or isinstance(pin.parameter_policy_version, bool)
            or not isinstance(pin.parameter_policy_version, int)
            or pin.parameter_policy_version < 1
            or not isinstance(pin.salt_reference, str)
            or not locator
            or any(char.isspace() or char in "?#=" for char in locator)
            or any(marker in locator.lower() for marker in forbidden)
            or not isinstance(pin.verifier, str)
            or not _SHA_RE.fullmatch(pin.verifier)
            or not isinstance(pin.pin_revision, int)
            or isinstance(pin.pin_revision, bool)
            or pin.pin_revision < 1
            or not isinstance(pin.failed_attempts, int)
            or isinstance(pin.failed_attempts, bool)
            or pin.failed_attempts < 0
            or not isinstance(pin.security_generation, int)
            or isinstance(pin.security_generation, bool)
            or pin.security_generation < 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        _parse_utc(pin.lockout_until_utc)

    @staticmethod
    def _updated_pin(
        pin: PinVerifierRecord, failures: int, lockout: str | None
    ) -> PinVerifierRecord:
        updated = replace(
            pin, failed_attempts=failures, lockout_until_utc=lockout, content_fingerprint_sha256=""
        )
        return replace(
            updated,
            content_fingerprint_sha256=_fingerprint_without(updated, "content_fingerprint_sha256"),
        )

    def _publish_pin_failure(
        self, before: InitialSecurityAuthoritySnapshot, pin: PinVerifierRecord, now: datetime
    ) -> None:
        failures = pin.failed_attempts + 1
        lockout = (
            _utc_text(now + timedelta(seconds=_LOCKOUT_SECONDS))
            if failures >= _MAX_FAILED_ATTEMPTS
            else None
        )
        updated = self._updated_pin(pin, failures, lockout)
        try:
            accepted = _accept_pin_projection(self._state, updated)
        except InitialSecurityError:
            _deny("CONTRACT_INCONSISTENT")
        if not accepted:
            _deny("CONTRACT_INCONSISTENT")

    @staticmethod
    def _build_proof(
        request: AuthorizationRequest,
        policy: OperationPolicy,
        now: datetime,
        identity: OperatorIdentitySecurityProjection,
        device: DeviceTrustProjection,
        pin: PinVerifierRecord,
        session: SessionSecurityState,
    ) -> AuthenticationProof:
        values = dict(
            account_id=request.account_id,
            operator_id=request.operator_id,
            device_installation_id=request.device_installation_id,
            factor_set=(
                ("PIN", "BIOMETRIC") if policy.factor_policy == "PIN_AND_BIOMETRIC" else ("PIN",)
            ),
            issued_at_utc=_utc_text(now),
            expires_at_utc=_utc_text(now + timedelta(seconds=policy.freshness_seconds)),
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
        return AuthenticationProof(
            request.account_id,
            request.operator_id,
            request.device_installation_id,
            (("PIN", "BIOMETRIC") if policy.factor_policy == "PIN_AND_BIOMETRIC" else ("PIN",)),
            _utc_text(now),
            _utc_text(now + timedelta(seconds=policy.freshness_seconds)),
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
            request.environment,
            request.operation,
            request.scope_fingerprint_sha256,
            request.mutation_fingerprint_sha256,
            request.causation_id,
            request.correlation_id,
            cast(str, canonical_json_sha256(values)),
        )

    @staticmethod
    def _binding(proof: AuthenticationProof) -> CoreIssuedAuthenticationProofBinding:
        return CoreIssuedAuthenticationProofBinding(
            proof.proof_fingerprint_sha256,
            complete_authentication_proof_fingerprint(proof),
            _AUTHORITY_SOURCE,
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


__all__ = [
    "AuthenticationAuthority",
    "AuthenticationError",
    "AuthenticationProof",
    "AuthorizationRequest",
    "CoreIssuedAuthenticationProofBinding",
    "OPERATION_OWNERSHIP",
    "OPERATION_POLICY_REGISTRY",
    "DownstreamOperationDefinition",
    "downstream_operation_definition_fingerprint",
    "downstream_scope_fingerprint",
    "downstream_mutation_fingerprint",
    "PinVerifierComparator",
    "authentication_proof_fingerprint",
    "canonical_scope_fingerprint",
    "complete_authentication_proof_fingerprint",
    "pin_mutation_fingerprint",
    "session_mutation_fingerprint",
]
