"""Unprivileged harness regressions; these tests never establish physical evidence.

Only the separate Ubuntu qualification script may execute temporary systemd
units. Fixtures here model public observations and synthetic protocol bytes.
"""

from __future__ import annotations

import base64
import errno
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.local_signing_custody import LocalSigningCustodyError, _authority_secret_scope
from bot_core.root_proof_issuer_substrate import SecurityProfile, SecurityProfileIdentity

from deployment import stage41b_role_worker as worker
from scripts import qualify_stage41b_signer_isolation as harness


pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only Stage 41B protocol and process regressions"
)


def _passing_boundaries():
    return {
        name: {"status": "PASS", "reason": "SYNTHETIC_UNIT_FIXTURE"}
        for name in harness.REQUIRED_BOUNDARIES
    }


def _unused_unit():
    return {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "FragmentPath": "",
        "DropInPaths": "",
        "Transient": "no",
        "InvocationID": "",
        "MainPID": "0",
        "ControlGroup": "",
    }


def _owned_unit():
    unit_id = "cryptohunter-root-proof-signer.service"
    return {
        "Id": unit_id,
        "LoadState": "loaded",
        "ActiveState": "active",
        "FragmentPath": "/run/systemd/transient/" + unit_id,
        "DropInPaths": "",
        "Transient": "yes",
        "Description": "Stage41B TEST ONLY " + "a" * 32 + " root_signer",
        "InvocationID": "b" * 32,
        "MainPID": "1234",
        "ControlGroup": "/system.slice/" + unit_id,
    }


def test_test_only_pass_requires_real_execution_and_complete_cleanup():
    boundaries = _passing_boundaries()
    assert (
        harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True)
        == "STAGE41B_TEST_ONLY_PASS"
    )
    assert harness.aggregate(boundaries, physical_executed=False, cleanup_ok=True) == "BLOCKED"
    assert (
        harness.aggregate(boundaries, physical_executed=True, cleanup_ok=False)
        != "STAGE41B_TEST_ONLY_PASS"
    )


@pytest.mark.parametrize("boundary", harness.REQUIRED_BOUNDARIES)
@pytest.mark.parametrize("status", ["FAIL", "BLOCKED"])
def test_any_required_boundary_vetoes_test_only_pass(boundary, status):
    boundaries = _passing_boundaries()
    boundaries[boundary]["status"] = status
    assert harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True) == status


@pytest.mark.parametrize("boundary", harness.REQUIRED_BOUNDARIES)
def test_missing_required_boundary_never_qualifies(boundary):
    boundaries = _passing_boundaries()
    boundaries.pop(boundary)
    assert harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True) == "BLOCKED"


@pytest.mark.parametrize("status", [None, True, 1, "pass", "", [], {"PASS": True}])
def test_malformed_status_fails_closed(status):
    boundaries = _passing_boundaries()
    boundaries["own_signature"]["status"] = status
    assert harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True) == "BLOCKED"


def test_unexpected_boundary_cannot_extend_qualification_contract():
    boundaries = _passing_boundaries()
    boundaries["production_private_control"] = {"status": "PASS", "reason": "UNTRUSTED"}
    assert harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True) == "BLOCKED"


@pytest.mark.parametrize("value", [None, True, [], "PASS", {"status": "PASS"}])
def test_malformed_boundary_observation_fails_closed(value):
    boundaries = _passing_boundaries()
    boundaries["own_signature"] = value
    assert harness.aggregate(boundaries, physical_executed=True, cleanup_ok=True) == "BLOCKED"


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_physical_execution_flags_are_exact_booleans(value):
    boundaries = _passing_boundaries()
    assert harness.aggregate(boundaries, physical_executed=value, cleanup_ok=True) == "BLOCKED"
    assert (
        harness.aggregate(boundaries, physical_executed=True, cleanup_ok=value)
        != "STAGE41B_TEST_ONLY_PASS"
    )


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EPERM])
def test_permission_denial_has_distinct_evidence(error_number):
    result = harness.classify_denial(error_number, service_live=True, host_path_exists=True)
    assert result == {"status": "PASS", "reason": "PERMISSION_DENIED"}


def test_namespace_absence_requires_positive_host_witness():
    absent = harness.classify_denial(errno.ENOENT, service_live=True, host_path_exists=False)
    isolated = harness.classify_denial(errno.ENOENT, service_live=True, host_path_exists=True)
    assert absent == {"status": "BLOCKED", "reason": "TARGET_MISSING"}
    assert isolated == {"status": "PASS", "reason": "NAMESPACE_PATH_ABSENT"}


@pytest.mark.parametrize(
    "error_number", [None, errno.EACCES, errno.EPERM, errno.ENOENT, errno.ECONNREFUSED]
)
def test_dead_native_service_is_never_isolation_evidence(error_number):
    result = harness.classify_denial(error_number, service_live=False, host_path_exists=True)
    assert result["status"] == "BLOCKED"


def test_successful_foreign_access_is_a_breach():
    result = harness.classify_denial(None, service_live=True, host_path_exists=True)
    assert result == {"status": "FAIL", "reason": "ACCESS_BREACH"}


@pytest.mark.parametrize("error_number", [errno.ECONNREFUSED, errno.ETIMEDOUT, errno.EIO])
def test_failed_service_or_unknown_errno_is_not_a_permission_denial(error_number):
    assert (
        harness.classify_denial(error_number, service_live=True, host_path_exists=True)["status"]
        == "BLOCKED"
    )


def test_only_exact_uninstalled_inactive_unit_is_unused():
    assert harness.unit_is_unused(_unused_unit()) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("LoadState", "loaded"),
        ("LoadState", "masked"),
        ("DropInPaths", "/etc/systemd/system/production.service.d/override.conf"),
        ("LoadState", "error"),
        ("ActiveState", "active"),
        ("ActiveState", "activating"),
        ("FragmentPath", "/etc/systemd/system/production.service"),
        ("Transient", "yes"),
        ("InvocationID", "b" * 32),
        ("MainPID", "1234"),
        ("MainPID", 0),
        ("ControlGroup", "/system.slice/production.service"),
    ],
    ids=[
        "installed",
        "masked",
        "drop-in",
        "load-error",
        "active",
        "activating",
        "fragment-path",
        "transient",
        "invocation",
        "running-pid",
        "nontext-pid",
        "cgroup",
    ],
)
def test_installed_masked_running_or_ambiguous_units_block_execution(field, value):
    unit = _unused_unit()
    unit[field] = value
    assert harness.unit_is_unused(unit) is False


@pytest.mark.parametrize("field", tuple(_unused_unit()))
def test_missing_unit_property_does_not_establish_safe_reuse(field):
    unit = _unused_unit()
    unit.pop(field)
    assert harness.unit_is_unused(unit) is False


def test_cleanup_unit_ownership_requires_original_pinned_invocation():
    pin = _owned_unit()
    assert harness.owns_unit(deepcopy(pin), pin) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("Id", "cryptohunter-history-attestation-signer.service"),
        ("Transient", "no"),
        ("DropInPaths", "/etc/systemd/system/production.service.d/override.conf"),
        ("FragmentPath", "/etc/systemd/system/replacement.service"),
        ("Description", "production"),
        ("InvocationID", "c" * 32),
        ("MainPID", "1235"),
        ("ControlGroup", "/system.slice/replacement.service"),
    ],
    ids=[
        "foreign-role",
        "persistent",
        "drop-in",
        "foreign-fragment",
        "description",
        "invocation",
        "pid",
        "cgroup",
    ],
)
def test_cleanup_refuses_replaced_or_production_unit(field, value):
    pin = _owned_unit()
    unit = deepcopy(pin)
    unit[field] = value
    assert harness.owns_unit(unit, pin) is False


def test_invocation_receipt_before_a_pid_cannot_establish_live_signer_identity():
    receipt = _owned_unit()
    receipt.update(MainPID="0", ActiveState="activating")
    assert harness.owns_invocation(deepcopy(receipt), receipt) is True
    assert harness.owns_unit(deepcopy(receipt), receipt) is False
    running = deepcopy(receipt)
    running.update(MainPID="1234", ActiveState="active")
    assert harness.owns_invocation(running, receipt) is True


def test_completed_original_invocation_remains_owned_without_accepting_replacement_pid():
    pin = _owned_unit()
    ended = deepcopy(pin)
    ended.update(MainPID="0", ActiveState="failed")
    assert harness.owns_invocation(ended, pin) is True
    changed_pid = deepcopy(pin)
    changed_pid["MainPID"] = "5678"
    assert harness.owns_invocation(changed_pid, pin) is False
    reused_pid = deepcopy(pin)
    reused_pid["InvocationID"] = "c" * 32
    assert harness.owns_invocation(reused_pid, pin) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("MainPID", None),
        ("MainPID", 0),
        ("MainPID", "-1"),
        ("InvocationID", ""),
        ("InvocationID", "b" * 31),
        ("Description", "Stage41B TEST ONLY " + "c" * 32 + " root_signer"),
        ("DropInPaths", "/etc/systemd/system/production.service.d/override.conf"),
        ("FragmentPath", "/etc/systemd/system/production.service"),
    ],
    ids=[
        "missing-pid",
        "nontext-pid",
        "negative-pid",
        "missing-invocation",
        "short-invocation",
        "foreign-run",
        "drop-in",
        "foreign-fragment",
    ],
)
def test_partial_invocation_receipt_cannot_adopt_ambiguous_or_replaced_unit(field, value):
    pin = _owned_unit()
    pin["MainPID"] = "0"
    observed = deepcopy(pin)
    observed[field] = value
    assert harness.owns_invocation(observed, pin) is False


@pytest.mark.parametrize(
    "uid,gid", [(0, 0), (1, 0), (0, 1), (1000, 1000), (True, 0), (0, False), (0.0, 0)]
)
def test_test_endpoint_authorization_uses_exact_root_peer_credentials(uid, gid):
    assert worker.authorized_peer(uid, gid) is (
        type(uid) is int and type(gid) is int and uid == gid == 0
    )


@pytest.mark.parametrize("operation", ["TEST_STATUS", "TEST_SIGN"])
def test_test_requests_accept_only_fixed_no_argument_operations(operation):
    assert worker.validate_request({"operation": operation, "params": {}}) == (operation, {})


@pytest.mark.parametrize(
    "operation",
    [
        "SIGN",
        "BIND",
        "ENROLL",
        "ENROLLMENT",
        "PROVISION",
        "DIAGNOSTIC",
        "QUALIFY",
        "TEST_SIGN_ANYTHING",
        None,
        True,
        [],
    ],
)
def test_test_ipc_rejects_production_and_unknown_operations(operation):
    with pytest.raises((ValueError, RuntimeError)):
        worker.validate_request({"operation": operation, "params": {}})


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        True,
        {"operation": "TEST_SIGN"},
        {"operation": "TEST_SIGN", "params": {}, "extra": "untrusted"},
        {"operation": "TEST_SIGN", "params": {"payload": "caller payload"}},
        {"operation": "TEST_SIGN", "params": {"identity": "foreign"}},
        {"operation": "TEST_SIGN", "params": {"role": "history_signer"}},
        {"operation": "TEST_SIGN", "params": []},
        {"operation": "TEST_STATUS", "params": {"debug": True}},
    ],
)
def test_test_ipc_rejects_malformed_or_caller_selected_signing_inputs(value):
    with pytest.raises((ValueError, RuntimeError)):
        worker.validate_request(value)


@pytest.mark.parametrize("material", [b"raw", b"hex", b"base64"])
def test_private_material_detector_covers_raw_hex_and_base64(material):
    secret = bytes(range(32))
    encoded = {b"raw": secret, b"hex": secret.hex().encode(), b"base64": base64.b64encode(secret)}[
        material
    ]
    assert worker.private_material_exposed(b"prefix " + encoded + b" suffix", (secret,)) is True
    assert worker.private_material_exposed(b"public observations only", (secret,)) is False


def test_fixed_test_payload_separates_both_roles_and_run_ids():
    root_payload = worker.payload_for("a" * 32, "root_signer")
    history_payload = worker.payload_for("a" * 32, "history_signer")
    next_payload = worker.payload_for("b" * 32, "root_signer")
    assert root_payload != history_payload != next_payload
    assert b"TEST" in root_payload.upper()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e999}',
        b"\xff",
        b"{",
        b"{} trailing",
    ],
    ids=[
        "duplicate-field",
        "nested-duplicate",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "float-overflow",
        "invalid-utf8",
        "truncated-json",
        "trailing-data",
    ],
)
def test_json_decoder_rejects_ambiguous_or_nonfinite_requests(raw):
    with pytest.raises((ValueError, UnicodeError)):
        worker.safe_json_loads(raw)


@pytest.mark.parametrize(
    "raw",
    [None, "{}", bytearray(b"{}"), b" " * (worker.MAX_REQUEST_BYTES + 1)],
    ids=["none", "text", "bytearray", "one-byte-over-limit"],
)
def test_json_decoder_accepts_only_bounded_exact_bytes(raw):
    with pytest.raises(ValueError):
        worker.safe_json_loads(raw)


def test_artifact_scan_request_requires_canonical_bounded_base64():
    encoded = base64.b64encode(b"public fixture").decode()
    request = {"operation": "TEST_SCAN", "params": {"artifacts_b64": [encoded]}}
    assert worker.validate_request(request) == ("TEST_SCAN", request["params"])
    for value in [None, "", [None], ["?"], ["YQ==\n"], ["YQ="], ["x" * 131073], [""] * 65]:
        with pytest.raises(ValueError):
            worker.validate_request({"operation": "TEST_SCAN", "params": {"artifacts_b64": value}})


@pytest.fixture
def unit_role_factory(monkeypatch, tmp_path):
    """Fake only daemon/native edges; never run physical qualification here."""
    made = []
    records = {}
    for key in (
        "HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "PYTHON_KEYRING_BACKEND",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = 10000 + len(made)
            self.stdin = io.BytesIO() if kwargs.get("stdin") == subprocess.PIPE else None
            self.exitcode = None
            self.terminated = False
            self.killed = False
            made.append(self)

        def poll(self):
            return self.exitcode

        def terminate(self):
            self.terminated = True
            self.exitcode = 0

        def wait(self, timeout):
            if self.exitcode is None:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return self.exitcode

        def kill(self):
            self.killed = True
            self.exitcode = -9

    class FakeBackend:
        def get_password(self, service, reference):
            return records.get((service, reference))

    class FakeStorage:
        MASTER_KEY_SLOT = "unit-fake-master"

        def __init__(self, *, service_name, **kwargs):
            self.service_name = service_name
            self._service_name = service_name
            self._backend = FakeBackend()

        def _read_hwid_digest(self):
            return "a" * 64

        def _store_master_key(self, master, digest):
            assert len(master) == 32 and digest == "a" * 64
            records[self.service_name, self.MASTER_KEY_SLOT] = (
                "unit-fake-native-master:" + base64.b64encode(master).decode()
            )

        def set_secret(self, reference, value):
            records[self.service_name, reference] = value

    class FakeReader:
        def __init__(self, *, security, role, **kwargs):
            self.service_namespace = worker._canonical_service(security, role)
            self.prefix = _authority_secret_scope(security, role).reference_prefix
            self.references = {
                reference for service, reference in records if service == self.service_namespace
            }

        def read_existing_authority_secret(self, reference):
            if (self.service_namespace, reference) in records:
                return records[self.service_namespace, reference]
            if not reference.startswith(self.prefix):
                raise LocalSigningCustodyError("outside synthetic scope")
            return None

    class FakeProvider:
        def __init__(self, directory, **kwargs):
            if not directory.is_relative_to(tmp_path):
                raise PermissionError(errno.EACCES, "unit foreign store denial")
            self.directory = directory

        def _sign(self, payload):
            instance = next(item for item in instances if item.state == self.directory)
            seed = base64.b64decode(
                records[
                    instance.reader.service_namespace,
                    instance.metadata.protected_private_material_reference,
                ]
            )
            return Ed25519PrivateKey.from_private_bytes(seed).sign(payload)

        sign_root_proof = _sign
        sign_history_head = _sign

        def public_key(self, identity):
            raise LocalSigningCustodyError("unknown synthetic identity")

    keyring = SimpleNamespace(set_keyring=lambda value: None)
    native = SimpleNamespace(Keyring=FakeBackend)
    original_import = worker.importlib.import_module

    def fake_import(name):
        return {"keyring": keyring, "keyring.backends.SecretService": native}.get(
            name
        ) or original_import(name)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)
    monkeypatch.setattr(worker, "_program", lambda name: "/usr/bin/" + name, raising=False)
    monkeypatch.setattr(worker.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(worker, "KeyringSecretStorage", FakeStorage)
    monkeypatch.setattr(worker, "NativeKeyringSigningSecretReader", FakeReader)
    monkeypatch.setattr(worker, "LocalRootProofSigningProvider", FakeProvider)
    monkeypatch.setattr(worker, "LocalHistoryAttestationSigningProvider", FakeProvider)
    monkeypatch.setattr(worker.RoleWorker, "_bus_owner", lambda self: ":1.42")
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: self.name == "bus" or real_exists(self))
    instances = []

    def factory(role="root_signer"):
        instance = object.__new__(worker.RoleWorker)
        state = tmp_path / (role + str(len(instances)))
        state.mkdir(mode=0o700)
        instance.role, instance.run_id, instance.state = role, "a" * 32, state
        instance.socket_path = state / "control.sock"
        instance.hidden_foreign = state / "hidden-foreign"
        instance.provider_role, _token, instance.unit = worker._ROLE[role]
        instance.security = SecurityProfileIdentity(
            SecurityProfile.PRODUCTION_LOCAL, "stage41b.test-only." + "a" * 32
        )
        instance.bus_path = state / "bus"
        instance.index_path = state / "secret-index.json"
        instance.metadata_path = state / worker._RECORD_FILENAME[instance.provider_role]
        instance.processes, instance.material, instance.lock = [], (), threading.Lock()
        instances.append(instance)
        return instance

    return factory, made, records


@pytest.mark.parametrize("role", ["root_signer", "history_signer"])
def test_unit_fake_daemon_lifecycle_uses_original_metadata_and_fixed_test_signature(
    unit_role_factory, role
):
    factory, processes, _records = unit_role_factory
    instance = factory(role)
    instance.start()
    assert instance.healthy() is True
    proof = instance.sign()
    payload = base64.b64decode(proof["payload_b64"], validate=True)
    signature = base64.b64decode(proof["signature_b64"], validate=True)
    assert payload == worker.payload_for(instance.run_id, role)
    Ed25519PrivateKey.from_private_bytes(instance.material[1]).public_key().verify(
        signature, payload
    )
    assert worker.private_material_exposed(json.dumps(proof).encode(), instance.material) is False
    assert (instance.state / "home").is_dir()
    assert "--session" in processes[0].command
    assert "--unlock" in processes[1].command
    assert not any(
        "password" in item or instance.material[0].decode() in item
        for process in processes
        for item in process.command
    )
    instance.close()
    assert all(process.terminated for process in processes)


def test_dead_native_process_prevents_test_signing(unit_role_factory):
    factory, processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    processes[1].exitcode = 1
    assert instance.healthy() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        instance.sign()


def test_native_secret_mismatch_prevents_liveness(unit_role_factory):
    factory, _processes, records = unit_role_factory
    instance = factory()
    instance.start()
    records[
        instance.reader.service_namespace, instance.metadata.protected_private_material_reference
    ] = "mismatched synthetic material"
    assert instance.healthy() is False


def test_scanner_reports_leak_without_echoing_private_material(unit_role_factory):
    factory, _processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    assert (
        instance.dispatch(
            "TEST_SCAN", {"artifacts_b64": [base64.b64encode(b"public report").decode()]}
        )["status"]
        == "PASS"
    )
    result = instance.dispatch(
        "TEST_SCAN", {"artifacts_b64": [base64.b64encode(instance.material[1]).decode()]}
    )
    assert result == {"status": "FAIL", "reason": "PRIVATE_MATERIAL_EXPOSED"}
    assert worker.private_material_exposed(json.dumps(result).encode(), instance.material) is False


@contextmanager
def _unit_socket_server(monkeypatch, *, output=None, failure=None, authorize=True):
    """A fake worker behind the real bounded test transport, without systemd."""
    calls = []
    material = (os.urandom(32),)

    def dispatch(operation, params):
        calls.append((operation, params))
        if failure is not None:
            raise failure
        if output == "private":
            return {"native_backend_error": base64.b64encode(material[0]).decode()}
        return (
            output if output is not None else {"status": "PASS", "reason": "UNIT_PUBLIC_RESPONSE"}
        )

    monkeypatch.setattr(worker, "authorized_peer", lambda uid, gid: authorize)
    with tempfile.TemporaryDirectory(prefix="s41b-unit-") as directory:
        path = Path(directory) / "s"
        with worker._Server(str(path), worker._Handler) as server:
            server.worker = SimpleNamespace(
                lock=threading.Lock(), material=material, dispatch=dispatch
            )
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            try:
                yield path, calls, material
            finally:
                server.shutdown()
                thread.join(timeout=2)
                assert not thread.is_alive()


def _socket_raw(path, raw):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(3)
        peer.connect(str(path))
        peer.sendall(raw)
        peer.shutdown(socket.SHUT_WR)
        return peer.recv(65536)


def test_harness_and_worker_agree_on_the_exact_test_request_protocol(monkeypatch):
    with _unit_socket_server(monkeypatch) as (path, calls, _material):
        response, transcript = harness._request(path, "TEST_STATUS")
        assert response == {"status": "PASS", "reason": "UNIT_PUBLIC_RESPONSE"}
        assert calls == [("TEST_STATUS", {})]
        assert b'"params":{}' in transcript


def test_harness_socket_requires_pinned_server_pid_uid_and_gid(monkeypatch):
    own_peer = (os.getpid(), os.geteuid(), os.getegid())
    with _unit_socket_server(monkeypatch) as (path, calls, _material):
        response, _transcript = harness._request(path, "TEST_STATUS", expected_peer=own_peer)
        assert response["status"] == "PASS"
        assert calls == [("TEST_STATUS", {})]
    with _unit_socket_server(monkeypatch) as (path, calls, _material):
        with pytest.raises(harness.QualificationBlocked, match="TEST_SOCKET_PEER_CHANGED"):
            harness._request(path, "TEST_SIGN", expected_peer=(own_peer[0] + 1, *own_peer[1:]))
        assert calls == []


def test_unprivileged_socket_request_is_rejected_before_dispatch(monkeypatch):
    with _unit_socket_server(monkeypatch, authorize=False) as (path, calls, _material):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(3)
            peer.connect(str(path))
            try:
                peer.sendall(b'{"operation":"TEST_SIGN","params":{}}\n')
            except BrokenPipeError:
                # The worker rejects SO_PEERCRED before reading request bytes.
                # Its fixed denial may therefore precede this send attempt.
                pass
            raw = peer.recv(65536)
        assert json.loads(raw) == {"status": "FAIL", "reason": "UNAUTHORIZED_PEER"}
        assert calls == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"operation":"SIGN","params":{}}\n',
        b'{"operation":"TEST_SIGN","params":{"payload":"foreign"}}\n',
        b'{"operation":"TEST_STATUS","operation":"TEST_SIGN","params":{}}\n',
        b'{"operation":"TEST_STATUS","params":{}}',
    ],
    ids=["production-sign", "caller-payload", "duplicate-operation", "missing-newline"],
)
def test_socket_rejects_unauthorized_operations_or_malformed_requests(monkeypatch, raw):
    with _unit_socket_server(monkeypatch) as (path, calls, _material):
        value = json.loads(_socket_raw(path, raw))
        assert value == {
            "status": "FAIL",
            "reason": "TEST_REQUEST_REJECTED",
        }
        assert calls == []


def test_socket_redacts_native_exception_text_and_private_response_material(monkeypatch):
    private_message = base64.b64encode(os.urandom(32)).decode()
    with _unit_socket_server(monkeypatch, failure=RuntimeError(private_message)) as (
        path,
        _calls,
        _material,
    ):
        raw = _socket_raw(path, b'{"operation":"TEST_STATUS","params":{}}\n')
        assert private_message.encode() not in raw
        assert json.loads(raw) == {"status": "BLOCKED", "reason": "TEST_SERVICE_UNAVAILABLE"}
    with _unit_socket_server(monkeypatch, output="private") as (path, _calls, material):
        raw = _socket_raw(path, b'{"operation":"TEST_STATUS","params":{}}\n')
        assert worker.private_material_exposed(raw, material) is False
        assert json.loads(raw) == {"status": "FAIL", "reason": "PRIVATE_MATERIAL_EXPOSED"}


def test_original_custody_binding_rejects_foreign_public_status_before_access(unit_role_factory):
    factory, _processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    with pytest.raises(ValueError, match="foreign"):
        instance._foreign(None)
    with pytest.raises(ValueError, match="same-run"):
        instance._foreign(
            {"role": "history_signer", "run_id": "b" * 32, "native_service_live": True}
        )


def test_path_probe_distinguishes_foreign_readability_and_missing_target(tmp_path):
    public = tmp_path / "readable"
    public.write_bytes(b"public probe fixture")
    assert worker._path_probe(public) == {"status": "FAIL", "reason": "ISOLATION_BREACH"}
    assert worker._path_probe(tmp_path / "absent") == {
        "status": "BLOCKED",
        "reason": "MOUNT_PATH_ABSENT",
    }


def test_signature_proofs_are_bound_to_role_run_and_own_public_key():
    keys = {role: Ed25519PrivateKey.generate() for role in harness.ROLES}
    run_id = "a" * 32
    statuses = {
        role: {"public_key_b64": base64.b64encode(key.public_key().public_bytes_raw()).decode()}
        for role, key in keys.items()
    }
    proofs = {}
    for role, key in keys.items():
        payload = worker.payload_for(run_id, role)
        proofs[role] = {
            "public_key_b64": statuses[role]["public_key_b64"],
            "payload_b64": base64.b64encode(payload).decode(),
            "signature_b64": base64.b64encode(key.sign(payload)).decode(),
        }
    harness._verify_signatures(statuses, proofs, run_id)
    foreign = deepcopy(proofs)
    foreign["root_signer"] = proofs["history_signer"]
    with pytest.raises(harness.QualificationBlocked):
        harness._verify_signatures(statuses, foreign, run_id)
    with pytest.raises(harness.QualificationBlocked):
        harness._verify_signatures(statuses, proofs, "b" * 32)
    altered = deepcopy(proofs)
    signature = bytearray(base64.b64decode(altered["root_signer"]["signature_b64"], validate=True))
    signature[0] ^= 1
    altered["root_signer"]["signature_b64"] = base64.b64encode(signature).decode()
    with pytest.raises(harness.QualificationFailed, match="OWN_KEY_SIGNATURE_INVALID") as failure:
        harness._verify_signatures(statuses, altered, run_id)
    assert failure.value.boundary == "own_signature"


def test_same_private_key_cannot_satisfy_cross_role_separation():
    shared_key = Ed25519PrivateKey.generate()
    public_key_b64 = base64.b64encode(shared_key.public_key().public_bytes_raw()).decode()
    run_id = "a" * 32
    statuses = {role: {"public_key_b64": public_key_b64} for role in harness.ROLES}
    proofs = {}
    for role in harness.ROLES:
        payload = worker.payload_for(run_id, role)
        proofs[role] = {
            "public_key_b64": public_key_b64,
            "payload_b64": base64.b64encode(payload).decode(),
            "signature_b64": base64.b64encode(shared_key.sign(payload)).decode(),
        }
    with pytest.raises(
        harness.QualificationFailed, match="CROSS_ROLE_SIGNATURE_COLLISION"
    ) as failure:
        harness._verify_signatures(statuses, proofs, run_id)
    assert failure.value.boundary == "private_material_separation"


def _foreign_public(instance):
    other = "history_signer" if instance.role == "root_signer" else "root_signer"
    role, token, _unit = worker._ROLE[other]
    state = Path(f"/run/ch-stage41b-{instance.run_id}-{token}")
    identity = worker._canonical_signing_identity(instance.security, role, "b" * 32)
    return {
        "role": other,
        "run_id": instance.run_id,
        "state": str(state),
        "socket": str(state / "control.sock"),
        "bus": str(state / "bus"),
        "metadata_path": str(state / worker._RECORD_FILENAME[role]),
        "private_index_path": str(state / "secret-index.json"),
        "keyring_path": str(state / "home/.local/share/keyrings"),
        "credential_id": identity.credential_id,
        "secret_reference": identity.protected_private_material_reference,
        "service_namespace": worker._canonical_service(instance.security, role),
        "native_service_live": True,
    }


def _fake_probe_edges(monkeypatch, instance, *, allow_foreign_bus=False):
    class ProbeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def settimeout(self, timeout):
            assert timeout > 0

        def connect(self, path):
            if path == str(instance.socket_path) or (allow_foreign_bus and path.endswith("/bus")):
                return None
            raise PermissionError(errno.EACCES, "unit foreign socket denied")

        def sendall(self, raw):
            assert json.loads(raw)["operation"] == "TEST_SIGN"

        def recv(self, size):
            return b'{"status":"FAIL","reason":"UNAUTHORIZED_PEER"}\n'

    def path_probe(path):
        if path == instance.hidden_foreign:
            return {"status": "BLOCKED", "reason": "MOUNT_PATH_ABSENT"}
        return {"status": "PASS", "reason": "PERMISSION_DENIED"}

    monkeypatch.setattr(worker.socket, "socket", lambda *args, **kwargs: ProbeSocket())
    monkeypatch.setattr(worker, "_path_probe", path_probe)


def test_unit_probe_exercises_original_identity_binding_and_distinct_denials(
    unit_role_factory, monkeypatch
):
    factory, _processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    foreign = _foreign_public(instance)
    assert instance._foreign(foreign) == foreign
    _fake_probe_edges(monkeypatch, instance)
    result = instance.dispatch("TEST_PROBE", {"foreign": foreign})["boundaries"]
    assert all(value["status"] == "PASS" for value in result.values())
    assert result["foreign_mount"]["reason"] == "NAMESPACE_PATH_ABSENT"
    assert result["foreign_store_dac"]["reason"] == "PERMISSION_DENIED"
    assert result["foreign_dbus"]["reason"] == "PERMISSION_DENIED"
    assert result["identity_binding"]["reason"] == "ROLE_AND_CREDENTIAL_BINDING_REJECTED"
    public = instance.dispatch("TEST_STATUS", {})
    assert public["native_service_live"] is True
    assert public["role"] == instance.role
    assert worker.private_material_exposed(json.dumps(public).encode(), instance.material) is False


def test_unit_probe_foreign_bus_or_secret_access_is_a_visible_breach(
    unit_role_factory, monkeypatch
):
    factory, _processes, records = unit_role_factory
    instance = factory()
    instance.start()
    foreign = _foreign_public(instance)
    records[foreign["service_namespace"], foreign["secret_reference"]] = (
        "unit fake foreign material"
    )
    _fake_probe_edges(monkeypatch, instance, allow_foreign_bus=True)
    result = instance.probe(foreign)["boundaries"]
    assert result["foreign_dbus"]["status"] == "FAIL"
    assert result["foreign_keyring"]["status"] == "FAIL"
    assert result["foreign_custody"]["status"] == "FAIL"


def test_unit_probe_stops_when_own_native_service_is_dead(unit_role_factory, monkeypatch):
    factory, processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    foreign = _foreign_public(instance)
    processes[1].exitcode = 1
    monkeypatch.setattr(
        worker,
        "_path_probe",
        lambda path: pytest.fail("a dead service must not produce denial evidence"),
    )
    assert instance.probe(foreign) == {
        "boundaries": {"own_liveness": {"status": "BLOCKED", "reason": "SERVICE_UNAVAILABLE"}}
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_service_live", False),
        ("state", "/etc/cryptohunter"),
        ("credential_id", "foreign"),
        ("secret_reference", "caller:reference"),
        ("service_namespace", "production"),
        ("run_id", "b" * 32),
    ],
    ids=[
        "dead-service",
        "foreign-state",
        "invalid-credential",
        "foreign-reference",
        "foreign-namespace",
        "foreign-run",
    ],
)
def test_foreign_fact_validation_rejects_caller_selected_paths_roles_and_identity(
    unit_role_factory, field, value
):
    factory, _processes, _records = unit_role_factory
    instance = factory()
    instance.start()
    foreign = _foreign_public(instance)
    foreign[field] = value
    with pytest.raises((ValueError, LocalSigningCustodyError)):
        instance._foreign(foreign)


def test_role_constructor_requires_exact_owned_private_runtime(monkeypatch):
    run_id = "a" * 32
    state = Path(f"/run/ch-stage41b-{run_id}-root")
    hidden = Path(f"/run/cryptohunter-stage41b-hidden-{run_id}/history_signer/witness")
    metadata = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=1000)
    monkeypatch.setattr(Path, "lstat", lambda self: metadata)
    monkeypatch.setattr(worker.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(worker.os, "getegid", lambda: 1000)
    instance = worker.RoleWorker(
        role="root_signer",
        run_id=run_id,
        state=state,
        socket_path=state / "control.sock",
        hidden_foreign=hidden,
    )
    assert instance.material == ()
    metadata.st_mode = stat.S_IFDIR | 0o755
    with pytest.raises(ValueError, match="isolated"):
        worker.RoleWorker(
            role="root_signer",
            run_id=run_id,
            state=state,
            socket_path=state / "control.sock",
            hidden_foreign=hidden,
        )


def test_bus_liveness_requires_native_owner_process_pid_and_uid(monkeypatch):
    instance = object.__new__(worker.RoleWorker)
    instance.processes = [SimpleNamespace(pid=123), SimpleNamespace(pid=456)]
    monkeypatch.setattr(worker, "_program", lambda name: "/usr/bin/" + name, raising=False)
    replies = [
        SimpleNamespace(returncode=0, stdout=b'string ":1.42"'),
        SimpleNamespace(returncode=0, stdout=b"uint32 456"),
        SimpleNamespace(returncode=0, stdout=f"uint32 {os.geteuid()}".encode()),
    ]
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: replies.pop(0))
    assert instance._bus_owner() == ":1.42"
    replies.extend(
        [
            SimpleNamespace(returncode=0, stdout=b'string ":1.42"'),
            SimpleNamespace(returncode=0, stdout=b"uint32 999"),
        ]
    )
    assert instance._bus_owner() is None
    replies.append(SimpleNamespace(returncode=1, stdout=b"service absent"))
    assert instance._bus_owner() is None
    replies.extend(
        [
            SimpleNamespace(returncode=0, stdout=b'string ":1.42"'),
            SimpleNamespace(returncode=0, stdout=b"uint32 456"),
            SimpleNamespace(returncode=0, stdout=f"uint32 {os.geteuid() + 1}".encode()),
        ]
    )
    assert instance._bus_owner() is None


def test_native_program_selection_checks_executable_and_every_ancestor(monkeypatch):
    seen = []

    def metadata(path):
        seen.append(str(path))
        kind = stat.S_IFREG if str(path) == "/usr/bin/dbus-send" else stat.S_IFDIR
        return SimpleNamespace(st_mode=kind | 0o755, st_uid=0)

    monkeypatch.setattr(Path, "lstat", metadata)
    assert worker._program("dbus-send") == "/usr/bin/dbus-send"
    assert seen == ["/usr/bin/dbus-send", "/usr/bin", "/usr", "/"]
    for name in ("sh", "/tmp/dbus-send", "../dbus-send"):
        with pytest.raises(ValueError, match="fixed native"):
            worker._program(name)


@pytest.mark.parametrize(
    "invalid", ["owner", "writable-file", "writable-parent", "symlink", "not-executable"]
)
def test_native_program_refuses_untrusted_file_or_ancestry(monkeypatch, invalid):
    def metadata(path):
        selected = str(path) == "/usr/bin/dbus-send"
        mode, owner = (stat.S_IFREG if selected else stat.S_IFDIR) | 0o755, 0
        if invalid == "owner" and selected:
            owner = 1000
        elif invalid == "writable-file" and selected:
            mode |= 0o002
        elif invalid == "writable-parent" and str(path) == "/usr/bin":
            mode |= 0o020
        elif invalid == "symlink" and selected:
            mode = stat.S_IFLNK | 0o777
        elif invalid == "not-executable" and selected:
            mode = stat.S_IFREG | 0o644
        return SimpleNamespace(st_mode=mode, st_uid=owner)

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(RuntimeError, match="protected native executable"):
        worker._program("dbus-send")


def test_native_daemon_environment_has_only_fixed_os_and_own_session_settings(monkeypatch):
    own = {
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/unit-own/bus",
        "XDG_DATA_HOME": "/run/unit-own/data",
        "XDG_CONFIG_HOME": "/run/unit-own/config",
        "XDG_RUNTIME_DIR": "/run/unit-own",
    }
    for key, value in own.items():
        monkeypatch.setenv(key, value)
    for key in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "DBUS_SYSTEM_BUS_ADDRESS",
        "GNOME_KEYRING_CONTROL",
    ):
        monkeypatch.setenv(key, "/unit-untrusted/override")
    monkeypatch.setenv("PATH", "/unit-untrusted/bin")
    assert worker._native_environment() == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **own,
    }
