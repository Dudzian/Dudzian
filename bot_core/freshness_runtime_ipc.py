"""Jednorazowy transport runtime do istniejącej operacji verifiera Linux.

Oczekiwany proces musi pochodzić z niezależnie zatwierdzonej obserwacji usługi
i wydania. Ten adapter wiąże z nim rzeczywiste połączenie; nie kwalifikuje
instalacji, nie wykonuje CAS i nie zastępuje trwałej ochrony replay.
"""

from __future__ import annotations

import hashlib
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from enum import Enum

from bot_core.freshness_semantic_verifier import (
    DOCUMENT_AUTHENTICATION_DOMAIN,
    DOCUMENT_DIGEST_DOMAIN,
    ORIGINAL_DECISION_DOMAIN,
    PREPARATION_DOMAIN,
    PROPOSER_AUTHENTICATION_FIELDS,
    PROPOSER_DOMAIN,
    _decode_ipc_bytes,
    _digest,
    _encode_ipc_bytes,
    _integer,
    _object,
    _string,
)
from bot_core.freshness_service_identity import CGROUP
from bot_core.freshness_verifier_service import (
    AUTHORITY_IDENTITY,
    IPC_GROUP,
    RUNTIME_USER,
    VERIFIER_SOCKET_PATH,
    VERIFIER_USER,
)
from bot_core.linux_service_ipc import PinnedReadOnlyPeer
from bot_core.postgresql_freshness_authority import (
    DOCUMENT_FIELDS,
    DOCUMENT_PAYLOAD_FIELDS,
    PREPARATION_FIELDS,
    RECEIPT_FIELDS,
    canonical_json_bytes,
    parse_canonical_json,
)
from bot_core.production_signer_ipc import (
    EXCHANGE_TIMEOUT_SECONDS,
    ProductionSignerIPCError,
    _expect_eof,
    _read_proc,
    _receive_exact,
    _remaining,
)
from deployment.linux_physical_ipc import ExpectedPeerProcess, _pinned_verifier_socket

MAX_FRAME_BYTES = 4 * 1024 * 1024


class FreshnessIPCOutcome(str, Enum):
    PREPARED = "PREPARED"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


@dataclass(frozen=True, slots=True)
class FreshnessIPCResult:
    outcome: FreshnessIPCOutcome
    preparation_id: str | None = None
    binding: bytes | None = None
    reason: str = "VERIFIER_RESPONSE"


def _require_runtime_identity(expected: ExpectedPeerProcess) -> None:
    """Stałe nazwy kont; bez przyjmowania UID, grup ani ścieżki od użytkownika."""
    if sys.platform != "linux":
        raise PermissionError("NATIVE_LINUX_AF_UNIX_REQUIRED")
    import grp
    import pwd

    runtime = pwd.getpwnam(RUNTIME_USER)
    verifier = pwd.getpwnam(VERIFIER_USER)
    ipc_gid = grp.getgrnam(IPC_GROUP).gr_gid
    if (
        runtime.pw_uid <= 0
        or verifier.pw_uid <= 0
        or runtime.pw_uid == verifier.pw_uid
        or min(runtime.pw_gid, ipc_gid) <= 0
        or (expected.uid, expected.gid) != (verifier.pw_uid, ipc_gid)
        or os.getresuid() != (runtime.pw_uid,) * 3
        or os.getresgid() != (runtime.pw_gid,) * 3
        or not set(os.getgroups()) <= {runtime.pw_gid, ipc_gid}
        or ipc_gid not in {os.getegid(), *os.getgroups()}
    ):
        raise PermissionError("EXACT_FRESHNESS_RUNTIME_IDENTITY_REQUIRED")


def _check_server(server: PinnedReadOnlyPeer, expected: ExpectedPeerProcess) -> None:
    process = server.process
    if (
        process["pid"] != expected.pid
        or process["uid"] != expected.uid
        or process["gid"] != expected.gid
        or process["start_time_ticks"] != expected.start_time_ticks
        or process["exe_sha256"] != expected.executable_sha256
        or process["cgroup"] != CGROUP
        or not set(process["groups"]) <= {expected.gid}
    ):
        # None dla /proc/PID/exe nigdy nie stanowi dowodu programu.
        raise PermissionError("APPROVED_VERIFIER_PROCESS_REQUIRED")
    server.recheck()


def _server_command(pid: int) -> bytes:
    directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        raw = _read_proc(directory, "cmdline")
        if not isinstance(raw, bytes) or not raw.endswith(b"\x00"):
            raise ValueError("VERIFIER_COMMAND_REQUIRED")
        return raw
    finally:
        os.close(directory)


def _decode_response(raw: bytes) -> FreshnessIPCResult:
    value = parse_canonical_json(raw)
    if type(value) is not dict or type(value.get("outcome")) is not str:
        raise ValueError("EXACT_VERIFIER_RESPONSE_REQUIRED")
    outcome = FreshnessIPCOutcome(value["outcome"])
    if outcome is not FreshnessIPCOutcome.PREPARED:
        if set(value) != {"outcome"}:
            raise ValueError("EXACT_VERIFIER_RESPONSE_REQUIRED")
        return FreshnessIPCResult(outcome)
    if set(value) != {"outcome", "preparation_id", "binding"}:
        raise ValueError("EXACT_PREPARATION_RESPONSE_REQUIRED")
    binding = _decode_ipc_bytes(value["binding"], "binding")
    parsed = parse_canonical_json(binding)
    if (
        type(parsed) is not dict
        or set(parsed) != PREPARATION_FIELDS
        or type(value["preparation_id"]) is not str
        or parsed["preparation_id"] != value["preparation_id"]
        or type(parsed["schema_version"]) is not int
        or parsed["schema_version"] != 1
        or parsed["security_profile"] != "PRODUCTION_LOCAL"
        or parsed["environment"] != "PRODUCTION"
        or parsed["operation_type"] != "FULL_AUTHORITATIVE_DOCUMENT"
        or parsed["verifier_authority_identity"] != AUTHORITY_IDENTITY
        or type(parsed["verifier_authority_version"]) is not int
        or parsed["verifier_authority_version"] != 1
    ):
        raise ValueError("EXACT_PREPARATION_BINDING_REQUIRED")
    for key, minimum in (
        ("expected_predecessor_generation", 0),
        ("proposer_key_version", 1),
        ("proposer_lifecycle_generation_observed_for_key_binding_only", 1),
        ("finalization_key_version", 1),
        ("finalization_lifecycle_generation_observed_for_key_binding_only", 1),
    ):
        _integer(parsed[key], key, minimum)
    for key in (
        "expected_predecessor_document_digest",
        "expected_predecessor_complete_semantic_head_digest",
        "proposed_document_digest",
        "original_decision_identity",
        "proposer_public_key_material_identity",
        "proposer_authentication_digest",
        "finalization_public_key_material_identity",
        "authoritative_document_authentication_digest",
        "receipt_canonical_digest",
        "preparation_id",
    ):
        _digest(parsed[key], key)
    for key in (
        "trust_domain",
        "authority_id",
        "proposer_identity",
        "proposer_credential_role_identity",
        "finalization_credential_role_identity",
        "receipt_id",
        "finalization_request_id",
    ):
        _string(parsed[key], key)
    digest = hashlib.sha256(
        PREPARATION_DOMAIN
        + canonical_json_bytes(
            {key: item for key, item in parsed.items() if key != "preparation_id"}
        )
    ).hexdigest()
    if value["preparation_id"] != digest:
        raise ValueError("PREPARATION_IDENTITY_MISMATCH")
    return FreshnessIPCResult(outcome, digest, binding)


def _check_candidate_binding(binding: bytes, inputs: tuple[bytes, bytes, bytes]) -> None:
    """Powiąż PREPARED z niezmiennymi bajtami tego jednego żądania.

    Wewnętrzny hash przygotowania nie dowodzi, którego żądania dotyczy
    odpowiedź. Odtwarzamy istniejące preimages, bez zastępowania weryfikacji
    podpisów lub resolvera poświadczeń po stronie autorytetu.
    """
    prepared = _object(parse_canonical_json(binding), PREPARATION_FIELDS, "preparation")
    proposer = _object(
        parse_canonical_json(inputs[0]), PROPOSER_AUTHENTICATION_FIELDS, "proposer authentication"
    )
    document = _object(parse_canonical_json(inputs[1]), DOCUMENT_FIELDS, "document")
    payload = _object(document["payload"], DOCUMENT_PAYLOAD_FIELDS, "document payload")
    receipt = _object(parse_canonical_json(inputs[2]), RECEIPT_FIELDS, "receipt")
    document_digest = hashlib.sha256(
        DOCUMENT_DIGEST_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    if (
        document["document_digest"] != document_digest
        or proposer["signed_or_authenticated_document_digest"] != document_digest
    ):
        raise ValueError("REQUEST_DOCUMENT_IDENTITY_MISMATCH")
    original_material = {
        key: receipt[key]
        for key in (
            "environment",
            "trust_domain",
            "authority_id",
            "exact_predecessor_generation",
            "exact_predecessor_document_digest",
            "accepted_generation",
            "accepted_document_digest",
            "complete_semantic_head_digest",
            "finalization_request_id",
        )
    }
    expected = {
        "environment": payload["environment"],
        "trust_domain": payload["trust_domain"],
        "authority_id": payload["authority_id"],
        "expected_predecessor_generation": payload["predecessor_generation"],
        "expected_predecessor_document_digest": payload["predecessor_document_digest"],
        "proposed_document_digest": document_digest,
        "proposer_identity": proposer["proposer_identity"],
        "proposer_key_version": proposer["proposer_key_version"],
        "proposer_authentication_digest": hashlib.sha256(
            PROPOSER_DOMAIN
            + canonical_json_bytes(
                {
                    key: item
                    for key, item in proposer.items()
                    if key != "authentication_tag_or_signature"
                }
            )
        ).hexdigest(),
        "finalization_key_version": payload["freshness_authority_key_version"],
        "authoritative_document_authentication_digest": hashlib.sha256(
            DOCUMENT_AUTHENTICATION_DOMAIN
            + canonical_json_bytes(
                {
                    key: item
                    for key, item in document.items()
                    if key != "authentication_tag_or_signature"
                }
            )
        ).hexdigest(),
        # Kontrakt verifiera przechowuje hash dokładnego kanonicznego receipt
        # (łącznie z podpisem), bez dodawania nowej domeny protokołu.
        "receipt_canonical_digest": hashlib.sha256(inputs[2]).hexdigest(),
        "receipt_id": receipt["receipt_id"],
        "finalization_request_id": payload["finalization_request_id"],
        "original_decision_identity": hashlib.sha256(
            ORIGINAL_DECISION_DOMAIN + canonical_json_bytes(original_material)
        ).hexdigest(),
    }
    if any(
        type(prepared[key]) is not type(item) or prepared[key] != item
        for key, item in expected.items()
    ):
        raise ValueError("PREPARATION_FOR_ANOTHER_CANDIDATE")


class RuntimeFreshnessVerifierClient:
    """Jedno połączenie i jedno żądanie; nigdy reconnect ani automatyczny retry.

    ``expected`` to zaufane wejście konstrukcyjne z istniejącego modelu
    obserwacji systemd/wydania, nie dane od klienta IPC. Sam hash interpretera
    bez niezależnego powiązania MainPID, cgroup i aplikacji nie wystarcza.
    Nie ma publicznego wyboru socketu, konta ani transportu.
    """

    def __init__(self, expected: ExpectedPeerProcess) -> None:
        if type(expected) is not ExpectedPeerProcess:
            raise TypeError("EXACT_APPROVED_VERIFIER_PROCESS_REQUIRED")
        self._expected = expected

    def verify_exact_candidate(
        self,
        proposer_authentication: bytes,
        authoritative_document: bytes,
        finalization_receipt: bytes,
    ) -> FreshnessIPCResult:
        if sys.platform != "linux" or not hasattr(socket, "AF_UNIX"):
            return FreshnessIPCResult(
                FreshnessIPCOutcome.UNAVAILABLE, reason="NATIVE_LINUX_AF_UNIX_REQUIRED"
            )
        started = False
        deadline = time.monotonic() + EXCHANGE_TIMEOUT_SECONDS
        try:
            _require_runtime_identity(self._expected)
            inputs = (proposer_authentication, authoritative_document, finalization_receipt)
            # Ograniczenie przed kodowaniem zapobiega nieograniczonej alokacji.
            if any(
                type(item) is not bytes or not 0 < len(item) <= MAX_FRAME_BYTES for item in inputs
            ):
                return FreshnessIPCResult(FreshnessIPCOutcome.INVALID, reason="INVALID_LOCAL_INPUT")
            request = canonical_json_bytes(
                dict(
                    zip(
                        (
                            "proposer_authentication",
                            "authoritative_document",
                            "finalization_receipt",
                        ),
                        map(_encode_ipc_bytes, inputs),
                        strict=True,
                    )
                )
            )
            if not 2 <= len(request) <= MAX_FRAME_BYTES:
                return FreshnessIPCResult(FreshnessIPCOutcome.INVALID, reason="REQUEST_TOO_LARGE")
            frame = struct.pack("!I", len(request)) + request
            with (
                _pinned_verifier_socket(VERIFIER_SOCKET_PATH, self._expected) as (address, recheck),
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer,
            ):
                _remaining(peer, deadline)
                peer.connect(address)
                # Nazwa klasy jest historyczna; kontrola wiąże dowolnego
                # partnera AF_UNIX, bez wywoływania READONLY_QUALIFY.
                with PinnedReadOnlyPeer(
                    peer, uid=self._expected.uid, gid=self._expected.gid
                ) as server:
                    _check_server(server, self._expected)
                    command = _server_command(self._expected.pid)
                    recheck()
                    _require_runtime_identity(self._expected)
                    _remaining(peer, deadline)
                    # Ustaw przed pierwszym send: częściowy zapis też może
                    # dotrzeć do authority. Żaden późniejszy błąd nie dowodzi rollbacku.
                    started = True
                    peer.sendall(frame)
                    peer.shutdown(socket.SHUT_WR)
                    length = struct.unpack("!I", _receive_exact(peer, 4, deadline))[0]
                    if not 2 <= length <= MAX_FRAME_BYTES:
                        raise ValueError("BOUNDED_VERIFIER_RESPONSE_REQUIRED")
                    raw = _receive_exact(peer, length, deadline)
                    _expect_eof(peer, deadline)
                    result = _decode_response(raw)
                    if result.outcome is FreshnessIPCOutcome.PREPARED:
                        if result.binding is None:
                            raise ValueError("PREPARATION_BINDING_REQUIRED")
                        _check_candidate_binding(result.binding, inputs)
                    _check_server(server, self._expected)
                    if _server_command(self._expected.pid) != command:
                        raise PermissionError("VERIFIER_COMMAND_CHANGED")
                    recheck()
                    _require_runtime_identity(self._expected)
                    _remaining(peer, deadline)
            return result
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RecursionError,
            ProductionSignerIPCError,
        ):
            return FreshnessIPCResult(
                FreshnessIPCOutcome.OUTCOME_UNKNOWN if started else FreshnessIPCOutcome.UNAVAILABLE,
                reason="AMBIGUOUS_EXCHANGE" if started else "TRANSPORT_OR_IDENTITY_UNAVAILABLE",
            )
