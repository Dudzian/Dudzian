"""Synthetic Stage 41B worker. Never installed as a production signer service.

Only the separate physical harness starts this module. Its isolated GNOME Secret
Service contains fresh random test material; original custody readers/providers
are used unchanged. No production installation file is opened.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import errno
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    LocalRootProofSigningProvider,
    LocalSigningCustodyError,
    LocalSigningMetadata,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
    _canonical_signing_identity,
    _decode_record,
    _encode_record,
    _RECORD_FILENAME,
    _SEMANTIC_ROLE,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from bot_core.security.keyring_storage import KeyringSecretStorage

MAX_REQUEST_BYTES = 4 * 1024 * 1024
_ROLE = {
    "root_signer": (
        ProviderRole.ROOT_PROOF_SIGNING,
        "root",
        "cryptohunter-root-proof-signer.service",
    ),
    "history_signer": (
        ProviderRole.HISTORY_ATTESTATION_SIGNING,
        "history",
        "cryptohunter-history-attestation-signer.service",
    ),
}
_OPERATIONS = frozenset({"TEST_STATUS", "TEST_SIGN", "TEST_PROBE", "TEST_SCAN"})


def _program(name: str) -> str:
    """Use only protected OS executables, never a caller-controlled PATH."""
    if name not in {"dbus-daemon", "dbus-send", "gnome-keyring-daemon"}:
        raise ValueError("fixed native program required")
    selected = Path("/usr/bin") / name
    for candidate in (selected, *selected.parents):
        info = candidate.lstat()
        kind = stat.S_ISREG if candidate == selected else stat.S_ISDIR
        if (
            not kind(info.st_mode)
            or info.st_uid != 0
            or info.st_mode & 0o022
            or (candidate == selected and not info.st_mode & 0o111)
        ):
            raise RuntimeError("protected native executable required")
    return str(selected)


def _native_environment() -> dict[str, str]:
    """The private native session receives no inherited loader or bus overrides."""
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    for name in (
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    ):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def safe_json_loads(raw: bytes) -> object:
    """Reject duplicate fields and nonfinite values before request validation."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    def nonfinite(_value: str) -> None:
        raise ValueError("nonfinite value")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite value")
        return result

    if type(raw) is not bytes or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("bounded request required")
    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float
    )


def authorized_peer(uid: int, gid: int) -> bool:
    return type(uid) is int and type(gid) is int and uid == 0 and gid == 0


def validate_request(value: object) -> tuple[str, dict[str, Any]]:
    if type(value) is not dict or set(value) != {"operation", "params"}:
        raise ValueError("exact request required")
    operation, params = value["operation"], value["params"]
    if type(operation) is not str or operation not in _OPERATIONS or type(params) is not dict:
        raise ValueError("test operation required")
    expected = {
        "TEST_STATUS": set(),
        "TEST_SIGN": set(),
        "TEST_PROBE": {"foreign"},
        "TEST_SCAN": {"artifacts_b64"},
    }
    if set(params) != expected[operation]:
        raise ValueError("exact test parameters required")
    if operation == "TEST_PROBE" and type(params["foreign"]) is not dict:
        raise ValueError("foreign public status required")
    if operation == "TEST_SCAN":
        artifacts = params["artifacts_b64"]
        if type(artifacts) is not list or len(artifacts) > 64:
            raise ValueError("bounded artifact list required")
        for artifact in artifacts:
            if type(artifact) is not str or len(artifact) > 131072:
                raise ValueError("bounded encoded artifact required")
            decoded = base64.b64decode(artifact, validate=True)
            if base64.b64encode(decoded).decode("ascii") != artifact:
                raise ValueError("canonical artifact encoding required")
    return operation, params


def private_material_exposed(data: bytes, material: tuple[bytes, ...]) -> bool:
    return any(
        form in data
        for value in material
        if value
        for form in (value, value.hex().encode("ascii"), base64.b64encode(value))
    )


def classify_access_error(error: OSError) -> dict[str, str]:
    if error.errno in {errno.EACCES, errno.EPERM}:
        return {"status": "PASS", "reason": "PERMISSION_DENIED"}
    if error.errno == errno.ENOENT:
        return {"status": "BLOCKED", "reason": "MOUNT_PATH_ABSENT"}
    if error.errno in {errno.ECONNREFUSED, errno.ECONNRESET, errno.ETIMEDOUT}:
        return {"status": "BLOCKED", "reason": "SERVICE_UNAVAILABLE"}
    return {"status": "BLOCKED", "reason": "UNCLASSIFIED_OS_ERROR"}


def payload_for(run_id: str, role: str) -> bytes:
    if (
        type(run_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
        or type(role) is not str
        or role not in _ROLE
    ):
        raise ValueError("exact synthetic run and role required")
    return (
        b"CryptoHunter Stage41B TEST ONLY\x00"
        + run_id.encode("ascii")
        + b"\x00"
        + role.encode("ascii")
    )


def _boundary(ok: bool, reason: str) -> dict[str, str]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason if ok else "ISOLATION_BREACH"}


def _path_probe(path: Path) -> dict[str, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        return classify_access_error(exc)
    else:
        os.close(descriptor)
        return _boundary(False, "")


class RoleWorker:
    role: str
    run_id: str
    state: Path
    socket_path: Path
    hidden_foreign: Path
    provider_role: ProviderRole
    unit: str
    security: SecurityProfileIdentity
    bus_path: Path
    index_path: Path
    metadata_path: Path

    def __init__(
        self, *, role: str, run_id: str, state: Path, socket_path: Path, hidden_foreign: Path
    ) -> None:
        if sys.platform != "linux":
            raise RuntimeError("physical Linux worker required")
        payload_for(run_id, role)
        provider_role, token, unit = _ROLE[role]
        expected_state = Path(f"/run/ch-stage41b-{run_id}-{token}")
        other = "history_signer" if role == "root_signer" else "root_signer"
        expected_hidden = Path(f"/run/cryptohunter-stage41b-hidden-{run_id}/{other}/witness")
        info = state.lstat()
        if (
            state != expected_state
            or socket_path != state / "control.sock"
            or hidden_foreign != expected_hidden
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or os.geteuid() == 0
            or os.getegid() == 0
        ):
            raise ValueError("isolated synthetic runtime directory required")
        self.role, self.run_id, self.state = role, run_id, state
        self.socket_path, self.hidden_foreign = socket_path, hidden_foreign
        self.provider_role, self.unit = provider_role, unit
        self.security = SecurityProfileIdentity(
            SecurityProfile.PRODUCTION_LOCAL, f"stage41b.test-only.{run_id}"
        )
        self.bus_path = state / "bus"
        self.index_path = state / "secret-index.json"
        self.metadata_path = state / _RECORD_FILENAME[provider_role]
        self.processes: list[subprocess.Popen[bytes]] = []
        self.material: tuple[bytes, ...] = ()
        self.lock = threading.Lock()

    def start(self) -> None:
        home = self.state / "home"
        home.mkdir(mode=0o700)
        os.environ.update(
            {
                "XDG_DATA_HOME": str(home / ".local/share"),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_RUNTIME_DIR": str(self.state),
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path={self.bus_path}",
                "PYTHON_KEYRING_BACKEND": "keyring.backends.SecretService.Keyring",
            }
        )
        daemon = subprocess.Popen(
            [
                _program("dbus-daemon"),
                "--session",
                "--nofork",
                f"--address=unix:path={self.bus_path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_native_environment(),
        )
        self.processes.append(daemon)
        deadline = time.monotonic() + 15
        while not self.bus_path.exists():
            if daemon.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("test bus unavailable")
            time.sleep(0.05)
        password = secrets.token_urlsafe(48).encode("ascii")
        native = subprocess.Popen(
            [
                _program("gnome-keyring-daemon"),
                "--foreground",
                "--unlock",
                "--components=secrets",
                f"--control-directory={self.state / 'keyring-control'}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_native_environment(),
        )
        self.processes.append(native)
        if native.stdin is None:
            raise RuntimeError("test keyring password pipe unavailable")
        native.stdin.write(password)
        native.stdin.close()
        self.material = (password,)
        while not self._bus_owner():
            if native.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("native test service unavailable")
            time.sleep(0.1)
        keyring = importlib.import_module("keyring")
        native_module = importlib.import_module("keyring.backends.SecretService")
        keyring.set_keyring(native_module.Keyring())
        canonical = _canonical_signing_identity(
            self.security, self.provider_role, secrets.token_hex(16)
        )
        seed, master = secrets.token_bytes(32), secrets.token_bytes(32)
        self.material += (seed, master)
        public_key = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
        self.metadata = LocalSigningMetadata(
            self.security.profile,
            self.security.trust_domain,
            self.provider_role,
            _SEMANTIC_ROLE[self.provider_role],
            canonical.credential_id,
            canonical.provider_namespace,
            canonical.key_handle,
            1,
            canonical.lifecycle_namespace,
            1,
            SigningKeyLifecycle.ACTIVE,
            public_key,
            canonical.protected_private_material_reference,
        )
        # This explicit synthetic ceremony initializes ONLY this fresh private
        # bus. Production provisioning/admission ports are deliberately unused.
        storage = KeyringSecretStorage(
            service_name=_canonical_service(self.security, self.provider_role),
            index_path=self.index_path,
            custody_hwid=True,
            custody_systemd_unit=self.unit,
        )
        if type(storage._backend) is not native_module.Keyring:
            raise RuntimeError("exact native Secret Service backend required")
        storage._store_master_key(master, storage._read_hwid_digest())
        storage.set_secret(
            canonical.protected_private_material_reference, base64.b64encode(seed).decode("ascii")
        )
        for slot in (storage.MASTER_KEY_SLOT, canonical.protected_private_material_reference):
            native_record = storage._backend.get_password(storage._service_name, slot)
            if type(native_record) is not str or not native_record:
                raise RuntimeError("synthetic native credential record unavailable")
            self.material += (native_record.encode("utf-8"),)
        self.metadata_path.write_bytes(_encode_record(self.metadata))
        self.reader = NativeKeyringSigningSecretReader(
            index_path=self.index_path, security=self.security, role=self.provider_role
        )
        provider_class = (
            LocalRootProofSigningProvider
            if self.role == "root_signer"
            else LocalHistoryAttestationSigningProvider
        )
        self.provider = provider_class(
            self.state, keyring_index_path=self.index_path, security=self.security
        )
        self.backend = storage._backend
        if not self.healthy():
            raise RuntimeError("original native custody unavailable")

    def _bus_owner(self) -> str | None:
        if sys.platform != "linux":
            raise RuntimeError("physical Linux worker required")
        result = subprocess.run(
            [
                _program("dbus-send"),
                "--session",
                "--print-reply",
                "--dest=org.freedesktop.DBus",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus.GetNameOwner",
                "string:org.freedesktop.secrets",
            ],
            capture_output=True,
            env=_native_environment(),
            timeout=5,
            check=False,
        )
        match = re.search(rb'string "(:[0-9]+\.[0-9]+)"', result.stdout)
        if result.returncode != 0 or match is None or len(self.processes) != 2:
            return None
        owner = match.group(1).decode("ascii")
        pid_result = subprocess.run(
            [
                _program("dbus-send"),
                "--session",
                "--print-reply",
                "--dest=org.freedesktop.DBus",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus.GetConnectionUnixProcessID",
                f"string:{owner}",
            ],
            capture_output=True,
            env=_native_environment(),
            timeout=5,
            check=False,
        )
        pid_match = re.search(rb"uint32 ([0-9]+)", pid_result.stdout)
        if pid_result.returncode != 0 or pid_match is None:
            return None
        pid = int(pid_match.group(1))
        if pid != self.processes[1].pid:
            return None
        uid_result = subprocess.run(
            [
                _program("dbus-send"),
                "--session",
                "--print-reply",
                "--dest=org.freedesktop.DBus",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus.GetConnectionUnixUser",
                f"string:{owner}",
            ],
            capture_output=True,
            env=_native_environment(),
            timeout=5,
            check=False,
        )
        uid_match = re.search(rb"uint32 ([0-9]+)", uid_result.stdout)
        # GNOME may be nondumpable, so same-UID /proc/<pid>/exe access is not a
        # usable worker proof. The privileged orchestrator independently checks
        # the tracked child's executable, UID and systemd cgroup instead.
        if (
            uid_result.returncode != 0
            or uid_match is None
            or int(uid_match.group(1)) != os.geteuid()
        ):
            return None
        return owner

    def healthy(self) -> bool:
        if any(process.poll() is not None for process in self.processes) or not self._bus_owner():
            return False
        secret = self.reader.read_existing_authority_secret(
            self.metadata.protected_private_material_reference
        )
        return type(secret) is str and secrets.compare_digest(
            secret.encode("ascii"), base64.b64encode(self.material[1])
        )

    def status(self) -> dict[str, Any]:
        if sys.platform != "linux":
            raise RuntimeError("physical Linux worker required")
        no_new_privs = "NoNewPrivs:\t1" in Path("/proc/self/status").read_text()
        return {
            "role": self.role,
            "run_id": self.run_id,
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "pid": os.getpid(),
            "state": str(self.state),
            "socket": str(self.socket_path),
            "bus": str(self.bus_path),
            "namespaces": {
                name: os.stat(f"/proc/self/ns/{name}").st_ino for name in ("mnt", "ipc", "net")
            },
            "no_new_privs": no_new_privs,
            "native_service_live": self.healthy(),
            "bus_owner": self._bus_owner(),
            "bus_pid": self.processes[0].pid,
            "keyring_pid": self.processes[1].pid,
            "public_key_b64": base64.b64encode(self.metadata.public_key).decode("ascii"),
            "credential_id": self.metadata.credential_id,
            "secret_reference": self.metadata.protected_private_material_reference,
            "service_namespace": self.reader.service_namespace,
            "metadata_path": str(self.metadata_path),
            "private_index_path": str(self.index_path),
            "keyring_path": str(self.state / "home/.local/share/keyrings"),
            "hidden_foreign": str(self.hidden_foreign),
        }

    def sign(self) -> dict[str, str]:
        payload = payload_for(self.run_id, self.role)
        if not self.healthy():
            raise RuntimeError("native service unavailable")
        if self.role == "root_signer":
            signature = self.provider.sign_root_proof(payload)
        else:
            signature = self.provider.sign_history_head(payload)
        Ed25519PrivateKey.from_private_bytes(self.material[1]).public_key().verify(
            signature, payload
        )
        return {
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "signature_b64": base64.b64encode(signature).decode("ascii"),
            "public_key_b64": base64.b64encode(self.metadata.public_key).decode("ascii"),
        }

    def _foreign(self, value: object) -> dict[str, Any]:
        if type(value) is not dict:
            raise ValueError("foreign public status required")
        other = "history_signer" if self.role == "root_signer" else "root_signer"
        other_role, token, _unit = _ROLE[other]
        state = Path(f"/run/ch-stage41b-{self.run_id}-{token}")
        expected = {
            "role": other,
            "run_id": self.run_id,
            "state": str(state),
            "socket": str(state / "control.sock"),
            "bus": str(state / "bus"),
            "metadata_path": str(state / _RECORD_FILENAME[other_role]),
            "private_index_path": str(state / "secret-index.json"),
            "keyring_path": str(state / "home/.local/share/keyrings"),
        }
        if (
            any(value.get(key) != item for key, item in expected.items())
            or value.get("native_service_live") is not True
        ):
            raise ValueError("live same-run foreign fixture required")
        for key in ("secret_reference", "service_namespace", "credential_id"):
            if type(value.get(key)) is not str or not value[key] or len(value[key]) > 512:
                raise ValueError("bounded foreign identity required")
        canonical = _canonical_signing_identity(
            self.security, other_role, _foreign_unique(value["credential_id"], other_role)
        )
        if value["secret_reference"] != canonical.protected_private_material_reference or value[
            "service_namespace"
        ] != _canonical_service(self.security, other_role):
            raise ValueError("canonical foreign identity required")
        return value

    def probe(self, value: object) -> dict[str, Any]:
        if sys.platform != "linux":
            raise RuntimeError("physical Linux worker required")
        foreign = self._foreign(value)
        if not self.healthy():
            return {
                "boundaries": {
                    "own_liveness": {"status": "BLOCKED", "reason": "SERVICE_UNAVAILABLE"}
                }
            }
        boundaries = {
            "foreign_store_dac": _path_probe(Path(foreign["private_index_path"])),
            "foreign_keyring_files": _path_probe(Path(foreign["keyring_path"])),
            "foreign_mount": _path_probe(self.hidden_foreign),
        }
        # ENOENT can establish only this explicit synthetic namespace witness;
        # the orchestrator must separately prove the host witness and both live
        # services before accepting this boundary.
        if boundaries["foreign_mount"]["reason"] == "MOUNT_PATH_ABSENT":
            boundaries["foreign_mount"] = {"status": "PASS", "reason": "NAMESPACE_PATH_ABSENT"}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            try:
                connection.connect(foreign["bus"])
            except OSError as exc:
                boundaries["foreign_dbus"] = classify_access_error(exc)
            else:
                boundaries["foreign_dbus"] = _boundary(False, "")
        foreign_role = _ROLE[foreign["role"]][0]
        foreign_reader = NativeKeyringSigningSecretReader(
            index_path=self.index_path, security=self.security, role=foreign_role
        )
        absent = foreign_reader.read_existing_authority_secret(foreign["secret_reference"]) is None
        native_absent = (
            self.backend.get_password(foreign["service_namespace"], foreign["secret_reference"])
            is None
        )
        boundaries["foreign_keyring"] = _boundary(
            absent and native_absent, "FOREIGN_SECRET_ABSENT_FROM_OWN_NATIVE_BUS"
        )
        rejected = False
        try:
            self.reader.read_existing_authority_secret(foreign["secret_reference"])
        except LocalSigningCustodyError:
            rejected = True
        unknown_identity = False
        try:
            self.provider.public_key(foreign["credential_id"])
        except LocalSigningCustodyError:
            unknown_identity = True
        boundaries["foreign_custody"] = _boundary(
            rejected and absent and unknown_identity, "FOREIGN_REFERENCE_AND_IDENTITY_REJECTED"
        )
        foreign_provider_class = (
            LocalHistoryAttestationSigningProvider
            if self.role == "root_signer"
            else LocalRootProofSigningProvider
        )
        provider_denied = False
        try:
            foreign_provider_class(
                Path(foreign["state"]),
                keyring_index_path=Path(foreign["private_index_path"]),
                security=self.security,
            )
        except (LocalSigningCustodyError, PermissionError):
            provider_denied = True
        boundaries["foreign_custody_store"] = _boundary(
            provider_denied, "FOREIGN_ORIGINAL_PROVIDER_STORE_REJECTED"
        )
        altered = replace(self.metadata, credential_id=foreign["credential_id"])
        binding_rejected = False
        try:
            _decode_record(
                _encode_record(altered),
                expected_security=self.security,
                expected_role=self.provider_role,
            )
        except LocalSigningCustodyError:
            binding_rejected = True
        wrong_role_rejected = False
        try:
            _decode_record(
                _encode_record(self.metadata),
                expected_security=self.security,
                expected_role=foreign_role,
            )
        except LocalSigningCustodyError:
            wrong_role_rejected = True
        boundaries["identity_binding"] = _boundary(
            binding_rejected and wrong_role_rejected, "ROLE_AND_CREDENTIAL_BINDING_REJECTED"
        )
        # An actual socket request from this unprivileged service UID exercises
        # SO_PEERCRED independently of the root orchestration client.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(str(self.socket_path))
            connection.sendall(b'{"operation":"TEST_SIGN","params":{}}\n')
            denied = safe_json_loads(connection.recv(4096)) == {
                "status": "FAIL",
                "reason": "UNAUTHORIZED_PEER",
            }
        boundaries["unauthorized_ipc"] = _boundary(denied, "UNPRIVILEGED_PEER_REJECTED")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            try:
                connection.connect(foreign["socket"])
            except OSError as exc:
                boundaries["foreign_ipc"] = classify_access_error(exc)
            else:
                boundaries["foreign_ipc"] = _boundary(False, "")
        boundaries["own_liveness"] = {
            "status": "PASS" if self.healthy() else "BLOCKED",
            "reason": "ORIGINAL_CUSTODY_READ_AND_BUS_OWNER",
        }
        return {"boundaries": boundaries}

    def dispatch(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        if operation == "TEST_STATUS":
            return self.status()
        if operation == "TEST_SIGN":
            return self.sign()
        if operation == "TEST_PROBE":
            return self.probe(params["foreign"])
        artifacts = [base64.b64decode(value, validate=True) for value in params["artifacts_b64"]]
        exposed = any(private_material_exposed(value, self.material) for value in artifacts)
        return {
            "status": "FAIL" if exposed else "PASS",
            "reason": "PRIVATE_MATERIAL_EXPOSED" if exposed else "NO_PRIVATE_MATERIAL_IN_ARTIFACTS",
        }

    def close(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for process in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


def _canonical_service(security: SecurityProfileIdentity, role: ProviderRole) -> str:
    from bot_core.local_signing_custody import _authority_secret_scope

    value = _authority_secret_scope(security, role).service_name
    if type(value) is not str:
        raise ValueError("exact canonical service required")
    return value


def _foreign_unique(identity: str, role: ProviderRole) -> str:
    from bot_core.local_signing_custody import _provisioning_unique_from_credential_id

    value = _provisioning_unique_from_credential_id(identity, role)
    if type(value) is not str:
        raise ValueError("exact canonical identity required")
    return value


if sys.platform == "linux":

    class _Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
        worker: RoleWorker

else:

    class _Server(socketserver.ThreadingTCPServer):
        worker: RoleWorker


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError("physical Linux worker required")
        server = self.server
        if not isinstance(server, _Server):
            return
        self.request.settimeout(10)
        try:
            _pid, uid, gid = struct.unpack(
                "3i",
                self.request.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                ),
            )
        except OSError:
            return
        if not authorized_peer(uid, gid):
            self.wfile.write(b'{"status":"FAIL","reason":"UNAUTHORIZED_PEER"}\n')
            return
        try:
            request = self.rfile.readline(MAX_REQUEST_BYTES + 2)
            if not request.endswith(b"\n") or len(request) > MAX_REQUEST_BYTES:
                raise ValueError("bounded newline request required")
            operation, params = validate_request(safe_json_loads(request))
        except OSError:
            output = b'{"status":"BLOCKED","reason":"TEST_TRANSPORT_UNAVAILABLE"}'
        except Exception:
            output = b'{"status":"FAIL","reason":"TEST_REQUEST_REJECTED"}'
        else:
            try:
                with server.worker.lock:
                    result = server.worker.dispatch(operation, params)
                    output = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                    if private_material_exposed(output, server.worker.material):
                        output = b'{"status":"FAIL","reason":"PRIVATE_MATERIAL_EXPOSED"}'
            except Exception:
                # Native error strings and tracebacks are private data too.
                output = b'{"status":"BLOCKED","reason":"TEST_SERVICE_UNAVAILABLE"}'
        try:
            self.wfile.write(output + b"\n")
        except OSError:
            return


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "linux":
        print('{"status":"BLOCKED","reason":"PHYSICAL_LINUX_WORKER_REQUIRED"}')
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=tuple(_ROLE))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--hidden-foreign", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    worker: RoleWorker | None = None
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda _number, _frame: stop.set())
    try:
        worker = RoleWorker(
            role=args.role,
            run_id=args.run_id,
            state=args.state,
            socket_path=args.socket,
            hidden_foreign=args.hidden_foreign,
        )
        worker.start()
        with _Server(str(worker.socket_path), _Handler) as server:
            server.worker = worker
            server.timeout = 0.25
            os.chmod(worker.socket_path, 0o600)
            while not stop.is_set():
                server.handle_request()
        return 0
    except Exception:
        print('{"status":"BLOCKED","reason":"SYNTHETIC_ROLE_START_FAILED"}')
        return 2
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
