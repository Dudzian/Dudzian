"""Durable atomic owner for the one-shot M0.10 initial-security publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import sqlite3
from typing import Any, NoReturn, cast

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    BootstrapTransitionResult,
    CoreCurrentBootstrapState,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningBoundary,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
    state_content_fingerprint,
)
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityAuthority,
    InitialSecurityError,
    InitialSecurityEstablishmentResult,
    InitialSecurityState,
    M03BootstrapAuthorityView,
    M03InitialSecurityBridge,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
    _PreparedInitialSecurity,
    _validate_bundle,
    _validate_initial_security_projection_bundle,
)

from .fingerprints import canonical_json_sha256
from .first_run_bootstrap import (
    DurableFirstRunBootstrapRegistry,
    _bootstrap_state_record,
)
from .lifecycle_records import persistence_record
from .records import PersistenceRecord
from .state_store import SQLiteStateStore, StateStoreError, StateStoreSnapshot


class DurableInitialSecurityError(RuntimeError):
    """Controlled fail-closed durable initial-security error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise DurableInitialSecurityError(reason)


@dataclass(frozen=True, slots=True)
class DurableInitialSecurityFamily:
    """Durable facts only; RuntimeSession history is never a live handle."""

    identity: OperatorIdentitySecurityProjection
    device: DeviceTrustProjection
    pin: PinVerifierRecord
    session: SessionSecurityState
    initial: InitialSecurityState
    terminal_bootstrap: CoreCurrentBootstrapState
    bootstrap_view: M03BootstrapAuthorityView
    runtime_session_history: PersistenceRecord


def _history(name: str, value: object) -> PersistenceRecord:
    upstream = asdict(value)  # type: ignore[call-overload]
    payload = {
        "fact_kind": name,
        "upstream_payload": upstream,
        "upstream_payload_fingerprint_sha256": canonical_json_sha256(upstream),
    }
    fields = {
        "OperatorIdentity revisions": ("operator_id", "identity_revision", "security_generation"),
        "DeviceTrust/security revisions": (
            "device_installation_id",
            "trust_revision",
            "security_generation",
            "platform_enrollment_revision",
        ),
        "PinVerifierRecord accepted revisions": (
            "operator_id",
            "pin_revision",
            "security_generation",
        ),
        "SessionSecurityState revision history": (
            "runtime_session_id",
            "session_generation",
            "security_generation",
        ),
        "InitialSecurityState accepted history": (
            "account_id",
            "device_installation_id",
            "security_generation",
            "session_generation",
        ),
    }[name]
    entry_identity = ":".join(str(upstream[field]) for field in fields)
    return persistence_record(name, f"immutable:{name}:{entry_identity}", payload)


def _designation(
    name: str, scope: str, reference: str, revision: int, generation: int
) -> PersistenceRecord:
    payload: dict[str, object] = {
        "scope_key": scope,
        "current_reference": reference,
        "current_revision": revision,
        "current_generation": generation,
    }
    payload["content_fingerprint_sha256"] = canonical_json_sha256(payload)
    key = f"current:{scope}:{reference}:{revision}:{generation}"
    return persistence_record(name, key, payload)


def _records(bundle: _PreparedInitialSecurity, terminal: CoreCurrentBootstrapState):
    identity, device, pin = bundle.identity, bundle.device, bundle.pin
    session, initial = bundle.session, bundle.initial
    current = (
        _bootstrap_state_record(terminal),
        _designation(
            "OperatorIdentity current designation/state",
            f"{identity.account_id}:{identity.operator_id}",
            identity.content_fingerprint_sha256,
            identity.identity_revision,
            identity.security_generation,
        ),
        _designation(
            "DeviceTrust current designation",
            f"{device.account_id}:{device.device_installation_id}",
            device.content_fingerprint_sha256,
            device.trust_revision,
            device.security_generation,
        ),
        _designation(
            "PinVerifierRecord current designation",
            f"{pin.account_id}:{pin.operator_id}:{pin.device_installation_id}",
            pin.content_fingerprint_sha256,
            pin.pin_revision,
            pin.security_generation,
        ),
        persistence_record(
            "SessionSecurityState current generation/state",
            "direct:SessionSecurityState current generation/state:"
            f"{session.account_id}:{session.operator_id}:{session.device_installation_id}:"
            f"{session.runtime_session_id}:{session.session_generation}:{session.security_generation}",
            asdict(session),
        ),
        persistence_record(
            "InitialSecurityState current state",
            f"direct:InitialSecurityState current state:{initial.account_id}:"
            f"{initial.device_installation_id}",
            asdict(initial),
        ),
    )
    history = (
        _history("OperatorIdentity revisions", identity),
        _history("DeviceTrust/security revisions", device),
        _history("PinVerifierRecord accepted revisions", pin),
        _history("SessionSecurityState revision history", session),
        _history("InitialSecurityState accepted history", initial),
    )
    return current, history


class DurableInitialSecurityRegistry:
    """Reconstruct a complete M0.10 family from one verified snapshot."""

    _NAMES = frozenset(
        {
            "OperatorIdentity current designation/state",
            "OperatorIdentity revisions",
            "DeviceTrust current designation",
            "DeviceTrust/security revisions",
            "PinVerifierRecord current designation",
            "PinVerifierRecord accepted revisions",
            "SessionSecurityState current generation/state",
            "SessionSecurityState revision history",
            "InitialSecurityState current state",
            "InitialSecurityState accepted history",
        }
    )

    def __init__(self, store: SQLiteStateStore, provisioning: ProvisioningBoundary) -> None:
        self._store = store
        self._provisioning = provisioning
        self._bootstrap = DurableFirstRunBootstrapRegistry(store)

    @staticmethod
    def _one(records: tuple[PersistenceRecord, ...], name: str) -> PersistenceRecord:
        matches = tuple(item for item in records if item.representation_name == name)
        if len(matches) != 1:
            _deny("CONTRACT_INCONSISTENT")
        return matches[0]

    @staticmethod
    def _upstream(record: PersistenceRecord) -> Mapping[str, object]:
        value = record.payload.get("upstream_payload")
        if not isinstance(value, Mapping):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current(self, account_id: str, device_id: str) -> DurableInitialSecurityFamily:
        try:
            snapshot = self._store.read_verified_snapshot()
            if snapshot is None:
                _deny("AUTHORIZATION_DENIED")
            terminal = self._bootstrap.terminal_state(account_id, device_id)
            bootstrap_states = self._bootstrap._family()
            if len(bootstrap_states) != 3 or bootstrap_states[-1] != terminal:
                _deny("CONTRACT_INCONSISTENT")
            bootstrap_pre, bootstrap_post = bootstrap_states[-3], bootstrap_states[-2]
            current = snapshot.current_records
            history = snapshot.immutable_history
            present = {
                item.representation_name
                for item in (*current, *history)
                if item.representation_name in self._NAMES
            }
            if present != self._NAMES:
                _deny("CONTRACT_INCONSISTENT")
            identity = OperatorIdentitySecurityProjection(
                **cast(
                    dict[str, Any],
                    dict(self._upstream(self._one(history, "OperatorIdentity revisions"))),
                )
            )
            device = DeviceTrustProjection(
                **cast(
                    dict[str, Any],
                    dict(self._upstream(self._one(history, "DeviceTrust/security revisions"))),
                )
            )
            pin = PinVerifierRecord(
                **cast(
                    dict[str, Any],
                    dict(
                        self._upstream(self._one(history, "PinVerifierRecord accepted revisions"))
                    ),
                )
            )
            session = SessionSecurityState(
                **cast(
                    dict[str, Any],
                    dict(
                        self._upstream(self._one(history, "SessionSecurityState revision history"))
                    ),
                )
            )
            initial = InitialSecurityState(
                **cast(
                    dict[str, Any],
                    dict(
                        self._upstream(self._one(history, "InitialSecurityState accepted history"))
                    ),
                )
            )
            direct_session = SessionSecurityState(
                **cast(
                    dict[str, Any],
                    dict(
                        self._one(current, "SessionSecurityState current generation/state").payload
                    ),
                )
            )
            direct_initial = InitialSecurityState(
                **cast(
                    dict[str, Any],
                    dict(self._one(current, "InitialSecurityState current state").payload),
                )
            )
            if direct_session != session or direct_initial != initial:
                _deny("CONTRACT_INCONSISTENT")
            for name, value in (
                ("OperatorIdentity current designation/state", identity),
                ("DeviceTrust current designation", device),
                ("PinVerifierRecord current designation", pin),
            ):
                designation = self._one(current, name).payload
                if name.startswith("OperatorIdentity"):
                    scope = f"{identity.account_id}:{identity.operator_id}"
                    revision, generation = identity.identity_revision, identity.security_generation
                elif name.startswith("DeviceTrust"):
                    scope = f"{device.account_id}:{device.device_installation_id}"
                    revision, generation = device.trust_revision, device.security_generation
                else:
                    scope = f"{pin.account_id}:{pin.operator_id}:{pin.device_installation_id}"
                    revision, generation = pin.pin_revision, pin.security_generation
                expected_designation: dict[str, object] = {
                    "scope_key": scope,
                    "current_reference": value.content_fingerprint_sha256,  # type: ignore[attr-defined]
                    "current_revision": revision,
                    "current_generation": generation,
                }
                expected_designation["content_fingerprint_sha256"] = canonical_json_sha256(
                    expected_designation
                )
                if designation != expected_designation:
                    _deny("CONTRACT_INCONSISTENT")
            claim = self._provisioning.resolve_accepted_claim(
                initial.bootstrap_claim_fingerprint_sha256
            )
            membership = self._provisioning.resolve_membership(
                initial.bootstrap_claim_fingerprint_sha256
            )
            if (
                type(claim) is not FirstRunBootstrapClaim
                or type(membership) is not ProvisioningMembershipBinding
            ):
                _deny("AUTHORIZATION_DENIED")
            consumed = terminal.consumed_authorities
            if len(consumed) != 1:
                _deny("CONTRACT_INCONSISTENT")
            authority = consumed[0]
            if (
                claim_content_fingerprint(claim) != claim.claim_fingerprint_sha256
                or claim.claim_fingerprint_sha256 != initial.bootstrap_claim_fingerprint_sha256
                or membership.authority_source != AUTHORITY_SOURCE
                or membership.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
                or membership.complete_claim_content_fingerprint_sha256
                != claim.claim_fingerprint_sha256
                or membership.provisioning_context_fingerprint_sha256
                != claim.provisioning_context_fingerprint_sha256
                or authority.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
                or authority.challenge_fingerprint_sha256 != claim.challenge_fingerprint_sha256
                or (
                    authority.account_id,
                    authority.device_installation_id,
                    authority.bootstrap_generation,
                    authority.bootstrap_revision,
                )
                != (
                    claim.account_id,
                    claim.device_installation_id,
                    claim.bootstrap_generation,
                    claim.bootstrap_revision,
                )
                or (terminal.expected_generation, terminal.expected_revision)
                != (claim.bootstrap_generation, claim.bootstrap_revision)
                or terminal.intended_operator_id != claim.intended_operator_id
                or (claim.account_id, claim.device_installation_id) != (account_id, device_id)
                or (terminal.account_id, terminal.device_installation_id)
                != (claim.account_id, claim.device_installation_id)
                or bootstrap_post.consumed_authorities != (authority,)
            ):
                _deny("CONTRACT_INCONSISTENT")
            runtime_matches = tuple(
                item
                for item in history
                if item.representation_name == "RuntimeSession canonical identity/history"
                and item.record_key == session.runtime_session_id
            )
            if len(runtime_matches) != 1:
                _deny("CONTRACT_INCONSISTENT")
            runtime_payload = self._upstream(runtime_matches[0])
            if runtime_payload != {
                "runtime_session_id": session.runtime_session_id,
                "device_installation_id": device_id,
            }:
                _deny("CONTRACT_INCONSISTENT")
            view = M03BootstrapAuthorityView(
                claim.claim_fingerprint_sha256,
                claim.account_id,
                claim.device_installation_id,
                claim.intended_operator_id,
                claim.bootstrap_generation,
                claim.bootstrap_revision,
                INITIAL_SECURITY_ESTABLISHMENT_ONLY,
                bootstrap_pre.state_fingerprint_sha256,
                bootstrap_post.state_fingerprint_sha256,
                canonical_json_sha256(asdict(authority)),
                authority.claim_fingerprint_sha256,
                authority.challenge_fingerprint_sha256,
            )
            _validate_initial_security_projection_bundle(
                view, identity, device, pin, session, initial
            )
            if (
                identity.state != "ACTIVE"
                or identity.identity_revision != 1
                or device.state != "TRUSTED"
                or (device.trust_revision, device.platform_enrollment_revision) != (1, 1)
                or pin.pin_revision != 1
                or pin.failed_attempts != 0
                or pin.lockout_until_utc is not None
                or session.state != "UNLOCKED"
                or initial.state != "ESTABLISHED"
            ):
                _deny("CONTRACT_INCONSISTENT")
            return DurableInitialSecurityFamily(
                identity,
                device,
                pin,
                session,
                initial,
                terminal,
                view,
                runtime_matches[0],
            )
        except DurableInitialSecurityError:
            raise
        except InitialSecurityError as exc:
            raise DurableInitialSecurityError("CONTRACT_INCONSISTENT") from exc
        except (
            FirstRunBootstrapError,
            StateStoreError,
            sqlite3.Error,
            KeyError,
            LookupError,
            TypeError,
            ValueError,
        ) as exc:
            raise DurableInitialSecurityError("CONTRACT_INCONSISTENT") from exc

    def assert_absent(self, snapshot: StateStoreSnapshot) -> None:
        if any(
            item.representation_name in self._NAMES
            for item in (*snapshot.current_records, *snapshot.immutable_history)
        ):
            _deny("ALREADY_ESTABLISHED")


class DurableInitialSecurityCoordinator:
    """Prepare, atomically commit, verify, then publish semantic authority."""

    def __init__(
        self,
        store: SQLiteStateStore,
        provisioning: ProvisioningBoundary,
        bridge: M03InitialSecurityBridge,
        authority: InitialSecurityAuthority,
        *,
        before_commit: Callable[[], None] | None = None,
        after_commit: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._provisioning = provisioning
        self._bridge = bridge
        self._authority = authority
        self._registry = DurableInitialSecurityRegistry(store, provisioning)
        self._before_commit = before_commit
        self._after_commit = after_commit
        self._recovery_runtime_session: RuntimeSession | None = None

    def _revalidate_prepared_authority(self, prepared: _PreparedInitialSecurity) -> RuntimeSession:
        """Fence non-SQLite authority; callers invoke this again after race hooks."""
        self._bridge.resolve(prepared.view)
        current = self._authority._runtime_sessions.resolve_current(
            prepared.view.account_id, prepared.view.device_installation_id
        )
        if (
            current is not prepared.runtime_session
            or not isinstance(current, RuntimeSession)
            or current.closed
            or prepared.runtime_session.closed
            or current.runtime_session_id != prepared.runtime_session.runtime_session_id
            or current.device_installation_id != prepared.view.device_installation_id
        ):
            _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
        return current

    def establish(
        self, transition: BootstrapTransitionResult, raw_pin: object
    ) -> InitialSecurityEstablishmentResult:
        try:
            view = self._bridge.accept(transition)
            with self._authority._state.lock:
                prepared_bundle = self._authority._prepare_initial_security(view, raw_pin)
                snapshot = self._store.read_verified_snapshot()
                if snapshot is None:
                    _deny("AUTHORIZATION_DENIED")
                metadata = snapshot.metadata
                if (metadata.account_id, metadata.device_installation_id) != (
                    view.account_id,
                    view.device_installation_id,
                ):
                    _deny("AUTHORIZATION_DENIED")
                self._registry.assert_absent(snapshot)
                predecessor_ref = DurableFirstRunBootstrapRegistry(
                    self._store
                ).current_state_reference(view.account_id, view.device_installation_id)
                predecessor = DurableFirstRunBootstrapRegistry(self._store).resolve_accepted_state(
                    predecessor_ref
                )
                if (
                    predecessor.state_fingerprint_sha256 != view.post_state_fingerprint_sha256
                    or predecessor.initial_security_lifecycle != "PRE_INITIAL_SECURITY"
                    or predecessor.first_operator_presence != "ABSENT"
                    or predecessor.startup_readiness != "SETUP_REQUIRED"
                    or len(predecessor.consumed_authorities) != 1
                ):
                    _deny("STALE_DURABLE_PRE")
                # Exact durable RuntimeSession evidence must predate this mutation.
                runtime = prepared_bundle.runtime_session
                matches = tuple(
                    item
                    for item in snapshot.immutable_history
                    if item.representation_name == "RuntimeSession canonical identity/history"
                    and item.record_key == runtime.runtime_session_id
                )
                if len(matches) != 1:
                    _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
                terminal = replace(
                    predecessor,
                    initial_security_lifecycle="INITIAL_SECURITY_COMPLETED",
                    first_operator_presence="PRESENT",
                    state_revision=predecessor.state_revision + 1,
                    state_fingerprint_sha256="0" * 64,
                )
                terminal = replace(
                    terminal,
                    state_fingerprint_sha256=state_content_fingerprint(terminal),
                )
                current, history = _records(prepared_bundle, terminal)
                self._revalidate_prepared_authority(prepared_bundle)
                target = replace(
                    metadata,
                    protected_freshness_generation=metadata.protected_freshness_generation + 1,
                )
                derived = self._store.derive_prepared_metadata(
                    target,
                    current_records=current,
                    immutable_history=history,
                    expected_current_generation=metadata.protected_freshness_generation,
                )
                if self._before_commit is not None:
                    self._before_commit()
                # Final transaction-time fence for authority outside SQLite's CAS.
                self._revalidate_prepared_authority(prepared_bundle)
                self._store.commit_prepared_state(
                    derived,
                    current_records=current,
                    immutable_history=history,
                    expected_current_generation=metadata.protected_freshness_generation,
                )
                self._recovery_runtime_session = prepared_bundle.runtime_session
                if self._after_commit is not None:
                    self._after_commit()
                verified = self._registry.resolve_current(
                    view.account_id, view.device_installation_id
                )
                post_commit_runtime = self._authority._runtime_sessions.resolve_current(
                    view.account_id, view.device_installation_id
                )
                if (
                    (
                        verified.identity,
                        verified.device,
                        verified.pin,
                        verified.session,
                        verified.initial,
                    )
                    != (
                        prepared_bundle.identity,
                        prepared_bundle.device,
                        prepared_bundle.pin,
                        prepared_bundle.session,
                        prepared_bundle.initial,
                    )
                    or post_commit_runtime is not prepared_bundle.runtime_session
                    or post_commit_runtime.closed
                    or post_commit_runtime.runtime_session_id
                    != prepared_bundle.runtime_session.runtime_session_id
                    or post_commit_runtime.device_installation_id != view.device_installation_id
                    or prepared_bundle.runtime_session.closed
                ):
                    _deny("CONTRACT_INCONSISTENT")
                return self._authority._install_prepared_initial_security(prepared_bundle)
        except DurableInitialSecurityError:
            raise
        except InitialSecurityError as exc:
            raise DurableInitialSecurityError("AUTHORIZATION_DENIED") from exc
        except (FirstRunBootstrapError, StateStoreError, sqlite3.Error) as exc:
            raise DurableInitialSecurityError("STALE_DURABLE_PRE") from exc
        except Exception as exc:
            raise DurableInitialSecurityError("CONTRACT_INCONSISTENT") from exc

    def rehydrate(self, account_id: str, device_id: str) -> InitialSecurityEstablishmentResult:
        try:
            family = self._registry.resolve_current(account_id, device_id)
            retained = self._recovery_runtime_session
            current = self._authority._runtime_sessions.resolve_current(account_id, device_id)
            if (
                retained is None
                or current is not retained
                or current.closed
                or current.runtime_session_id != family.session.runtime_session_id
                or current.device_installation_id != device_id
            ):
                _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
            prepared = _PreparedInitialSecurity(
                family.bootstrap_view,
                current,
                family.identity,
                family.device,
                family.pin,
                family.session,
                family.initial,
            )
            _validate_bundle(
                prepared.view,
                current,
                prepared.identity,
                prepared.device,
                prepared.pin,
                prepared.session,
                prepared.initial,
            )
            return self._authority._install_prepared_initial_security(
                prepared, _durably_verified=True
            )
        except DurableInitialSecurityError:
            raise
        except InitialSecurityError as exc:
            raise DurableInitialSecurityError("AUTHORIZATION_DENIED") from exc
        except (
            StateStoreError,
            sqlite3.Error,
            KeyError,
            LookupError,
            TypeError,
            ValueError,
        ) as exc:
            raise DurableInitialSecurityError("CONTRACT_INCONSISTENT") from exc
        except Exception as exc:
            raise DurableInitialSecurityError("CONTRACT_INCONSISTENT") from exc


__all__ = [
    "DurableInitialSecurityCoordinator",
    "DurableInitialSecurityError",
    "DurableInitialSecurityFamily",
    "DurableInitialSecurityRegistry",
]
