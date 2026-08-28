"""Process-local membership for evidence derived from verified durable state.

Evidence in this module is an integrity carrier, not domain, restore, M0.3, or
LIVE authority.  Membership and current designation exist only in one registry
instance and are deliberately never persisted.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import asdict, dataclass
from threading import RLock
from typing import cast

from .durable_observation import DurableStateObservation, observe_verified_durable_state
from .fingerprints import canonical_json_sha256
from .state_store import SQLiteStateStore

_CANONICAL_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DURABILITY_STATE = "DURABLE_COMMITTED"

EvidenceScope = tuple[str, str, str]


def _validate_canonical_id(value: object, *, prefix: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or _CANONICAL_ID_RE.fullmatch(value) is None
        or not value.startswith(f"{prefix}_")
    ):
        raise ValueError(f"{field_name} must be a canonical M0.2 {prefix} identifier")


def _validate_positive_integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive non-boolean integer")


def _validate_sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be lowercase SHA-256 hexadecimal")


def _fingerprint_projection(
    *,
    account_id: str,
    device_installation_id: str,
    state_store_identity_fingerprint_sha256: str,
    generation: int,
    state_fingerprint_sha256: str,
    transaction_fingerprint_sha256: str,
    durability_state: str,
    evidence_revision: int,
) -> dict[str, object]:
    return {
        "account_id": account_id,
        "device_installation_id": device_installation_id,
        "state_store_identity_fingerprint_sha256": (state_store_identity_fingerprint_sha256),
        "generation": generation,
        "state_fingerprint_sha256": state_fingerprint_sha256,
        "transaction_fingerprint_sha256": transaction_fingerprint_sha256,
        "durability_state": durability_state,
        "evidence_revision": evidence_revision,
    }


@dataclass(frozen=True, slots=True)
class LocalDurableStateEvidence:
    """Exact immutable, derived/rebuildable local durable evidence payload."""

    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    durability_state: str
    evidence_revision: int
    evidence_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _validate_canonical_id(self.account_id, prefix="acct", field_name="account_id")
        _validate_canonical_id(
            self.device_installation_id,
            prefix="dev",
            field_name="device_installation_id",
        )
        _validate_positive_integer(self.generation, field_name="generation")
        _validate_positive_integer(self.evidence_revision, field_name="evidence_revision")
        for field_name in (
            "state_store_identity_fingerprint_sha256",
            "state_fingerprint_sha256",
            "transaction_fingerprint_sha256",
            "evidence_fingerprint_sha256",
        ):
            _validate_sha256(getattr(self, field_name), field_name=field_name)
        if self.durability_state != _DURABILITY_STATE:
            raise ValueError("durability_state must be DURABLE_COMMITTED")
        if self.evidence_fingerprint_sha256 != self._expected_fingerprint():
            raise ValueError("evidence_fingerprint_sha256 does not match the exact payload")

    def _expected_fingerprint(self) -> str:
        return cast(
            str,
            canonical_json_sha256(
                _fingerprint_projection(
                    account_id=self.account_id,
                    device_installation_id=self.device_installation_id,
                    state_store_identity_fingerprint_sha256=(
                        self.state_store_identity_fingerprint_sha256
                    ),
                    generation=self.generation,
                    state_fingerprint_sha256=self.state_fingerprint_sha256,
                    transaction_fingerprint_sha256=self.transaction_fingerprint_sha256,
                    durability_state=self.durability_state,
                    evidence_revision=self.evidence_revision,
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a fresh mapping containing exactly the nine payload fields."""

        return asdict(self)


def _intrinsically_valid(evidence: object) -> bool:
    if not isinstance(evidence, LocalDurableStateEvidence):
        return False
    try:
        evidence.__post_init__()
    except (AttributeError, TypeError, ValueError):
        return False
    return True


class LocalDurableEvidenceRegistry:
    """One process-local accepted/current evidence registry."""

    def __init__(self) -> None:
        self._accepted: dict[str, LocalDurableStateEvidence] = {}
        self._current: dict[EvidenceScope, str] = {}
        self._revision = 0
        self._lock = RLock()

    def publish_verified_state(self, store: SQLiteStateStore) -> str | None:
        """Verify a StateStore afresh, then publish evidence all-or-nothing."""

        if not isinstance(store, SQLiteStateStore):
            raise TypeError("store must be a SQLiteStateStore")
        observation = observe_verified_durable_state(store)
        if observation is None:
            return None
        with self._lock:
            revision = self._revision + 1
            evidence = self._derive_evidence(observation, revision)
            reference = secrets.token_urlsafe(32)
            while reference in self._accepted:
                reference = secrets.token_urlsafe(32)
            scope = self._scope(evidence)
            self._accepted[reference] = evidence
            self._current[scope] = reference
            self._revision = revision
            return reference

    def verify_current(self, scope: tuple[str, str, str], ref: object) -> bool:
        """Verify membership and current designation for the exact requested scope."""

        return self.resolve_current(scope, ref) is not None

    def resolve_current(
        self, scope: EvidenceScope, ref: object
    ) -> LocalDurableStateEvidence | None:
        """Resolve only accepted, intrinsically valid, exactly-current evidence."""

        if not isinstance(ref, str) or not isinstance(scope, tuple) or len(scope) != 3:
            return None
        with self._lock:
            evidence = self._accepted.get(ref)
            if not _intrinsically_valid(evidence):
                return None
            assert isinstance(evidence, LocalDurableStateEvidence)
            if self._scope(evidence) != scope or self._current.get(scope) != ref:
                return None
            return evidence

    @staticmethod
    def _scope(evidence: LocalDurableStateEvidence) -> EvidenceScope:
        return (
            evidence.account_id,
            evidence.device_installation_id,
            evidence.state_store_identity_fingerprint_sha256,
        )

    @staticmethod
    def _derive_evidence(
        observation: DurableStateObservation, revision: int
    ) -> LocalDurableStateEvidence:
        projection = _fingerprint_projection(
            account_id=observation.account_id,
            device_installation_id=observation.device_installation_id,
            state_store_identity_fingerprint_sha256=(
                observation.state_store_identity_fingerprint_sha256
            ),
            generation=observation.protected_freshness_generation,
            state_fingerprint_sha256=observation.state_fingerprint_sha256,
            transaction_fingerprint_sha256=observation.transaction_fingerprint_sha256,
            durability_state=_DURABILITY_STATE,
            evidence_revision=revision,
        )
        return LocalDurableStateEvidence(
            account_id=observation.account_id,
            device_installation_id=observation.device_installation_id,
            state_store_identity_fingerprint_sha256=(
                observation.state_store_identity_fingerprint_sha256
            ),
            generation=observation.protected_freshness_generation,
            state_fingerprint_sha256=observation.state_fingerprint_sha256,
            transaction_fingerprint_sha256=observation.transaction_fingerprint_sha256,
            durability_state=_DURABILITY_STATE,
            evidence_revision=revision,
            evidence_fingerprint_sha256=canonical_json_sha256(projection),
        )
