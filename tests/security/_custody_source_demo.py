"""Closed TEST-only mutation process; no production custody/admission capability.

One disposable namespace supports one synthetic metadata transition. The process
owns its generated fixture credential and derives every byte it authenticates.
Its synthetic PostgreSQL close and same-UID files are never qualified authority.
Crash barriers are test-process arguments, never production/request flags.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import re
import stat
import sys

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from bot_core import local_signing_custody as custody
from bot_core import local_signing_lifecycle_evidence as lifecycle
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _checked(path, mode, *, directory=False, pin=None):
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != mode
        or (pin is not None and _identity(path) != tuple(pin))
    ):
        raise ValueError("test source path/identity/permissions changed")
    return info


def _read(path, mode=0o600, *, pin=None):
    info = _checked(path, mode, pin=pin)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != (
            info.st_dev,
            info.st_ino,
        ):
            raise ValueError("test source file replaced during open")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024 or _identity(path) != (info.st_dev, info.st_ino):
            raise ValueError("test source file changed/oversized")
        return raw
    finally:
        os.close(descriptor)


def _write(path, raw, mode=0o600):
    # Explicit file fsync -> atomic rename -> directory fsync, only TEST data.
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _sync(path):
    for target, flags in ((path, os.O_RDONLY), (path.parent, os.O_RDONLY | os.O_DIRECTORY)):
        descriptor = os.open(target, flags | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _lineage_bytes(journal, raw):
    # Only the worker writes this separate domain after its own durable phase.
    metadata = journal.parent.parent / "metadata"
    return b"CryptoHunter/TestCustodySourceLineage/V1\0" + canonical_json_bytes(
        {
            "journal_sha256": hashlib.sha256(raw).hexdigest(),
            "journal_inode": list(_identity(journal)),
            "metadata_sha256": hashlib.sha256(_read(metadata)).hexdigest(),
            "metadata_inode": list(_identity(metadata)),
        }
    )


def _retain_lineage(journal, raw, key):
    _write(journal.with_name("lineage"), key.sign(_lineage_bytes(journal, raw)))


def _check_lineage(config, journal, raw):
    if _read(journal) != raw:
        raise ValueError("test source journal bytes changed")
    Ed25519PublicKey.from_public_bytes(config["public_key"]).verify(
        _read(journal.with_name("lineage")), _lineage_bytes(journal, raw)
    )


def _event_lineage_bytes(event, raw, policy):
    return b"CryptoHunter/TestCustodySourceEventLineage/V1\0" + canonical_json_bytes(
        {
            "event_sha256": hashlib.sha256(raw).hexdigest(),
            "event_inode": list(_identity(event)),
            "policy_sha256": hashlib.sha256(policy).hexdigest(),
            "policy_inode": list(_identity(event.with_name("public-policy"))),
        }
    )


def _check_event_lineage(config, event, raw, policy):
    Ed25519PublicKey.from_public_bytes(config["public_key"]).verify(
        _read(event.with_name("lineage")), _event_lineage_bytes(event, raw, policy)
    )


def _before(role):
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "test.source-demo")
    provider = ProviderRole(role)
    token = "a" if provider is ProviderRole.ROOT_PROOF_SIGNING else "b"
    identity = custody._canonical_signing_identity(security, provider, token * 32)
    return custody.LocalSigningMetadata(
        security.profile,
        security.trust_domain,
        provider,
        custody._SEMANTIC_ROLE[provider],
        identity.credential_id,
        identity.provider_namespace,
        identity.key_handle,
        1,
        identity.lifecycle_namespace,
        1,
        custody.SigningKeyLifecycle.ACTIVE,
        bytes(range(32)) if token == "a" else bytes(reversed(range(32))),
        identity.protected_private_material_reference,
    )


def _initialize(root, role, completion_public_key=None):
    if completion_public_key is not None and (
        type(completion_public_key) is not bytes or len(completion_public_key) != 32
    ):
        raise ValueError("exact public-only TEST completion key required")
    _checked(root, 0o700, directory=True)
    for name in ("credential", "journal", "events"):
        (root / name).mkdir(mode=0o700)
    # The parent never generates or receives this fixture private key/seed.
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    _write(root / "credential" / "seed", seed, 0o400)
    _write(root / "credential" / "lifecycle", b"ACTIVE:1:1")
    _write(root / "metadata", custody._encode_record(_before(role)))
    _write(root / "lock", b"")
    return {
        "root": str(root),
        "role": role,
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "directories": {
            str(path): _identity(path) for path in [root, *root.iterdir()] if path.is_dir()
        },
        "key_inode": _identity(root / "credential" / "seed"),
        "lock_inode": _identity(root / "lock"),
        "metadata_inode": _identity(root / "metadata"),
        "public_key": key.public_key().public_bytes_raw(),
        **({"completion_public_key": completion_public_key} if completion_public_key is not None else {}),
    }


def _recheck(config):
    completion_public = config.get("completion_public_key")
    if completion_public is not None and (
        type(completion_public) is not bytes or len(completion_public) != 32
    ):
        raise ValueError("invalid TEST completion public-only key")
    for path, pin in config["directories"].items():
        _checked(Path(path), 0o700, directory=True, pin=pin)
    root = Path(config["root"])
    _checked(root / "lock", 0o600, pin=config["lock_inode"])
    if _read(root / "credential" / "lifecycle") != b"ACTIVE:1:1":
        raise ValueError("test source credential rotated/revoked")
    key = Ed25519PrivateKey.from_private_bytes(
        _read(root / "credential" / "seed", 0o400, pin=config["key_inode"])
    )
    if key.public_key().public_bytes_raw() != config["public_key"]:
        raise ValueError("test source credential material changed")
    return key


def _barrier(stage, crash_at, pause_at, control):
    if stage == crash_at:
        os._exit(73)
    if stage == pause_at:
        control.send(("barrier", stage))
        if not control.poll(10) or control.recv() != "resume":
            raise ValueError("test barrier timeout")


def _operation(root, before, request):
    after = replace(
        before,
        lifecycle_generation=2,
        lifecycle_state=custody.SigningKeyLifecycle(request["target_state"]),
    )
    descriptor = canonical_json_bytes(
        {
            "protocol_version": 2,
            "operation_id": request["operation_id"],
            "fence_key": "synthetic-test-close-only",
            "semantic_role": before.provider_role.value,
            "credential_identity": asdict(before.credential_identity()),
            "public_key_hex": before.public_key.hex(),
            "key_version": 1,
            "lifecycle_generation": 1,
            "predecessor_state": "ACTIVE",
            "target_state": after.lifecycle_state.value,
            "custody_scope": {
                "directory": str(root),
                "security_profile": "PRODUCTION_LOCAL",
                "trust_domain": before.trust_domain,
            },
            "database_identity": {
                "system_identifier": "100",
                "database_oid": 200,
                "database_name": "synthetic-test-database",
                "cluster_name": "",
                "connection_host": "/run/postgresql",
                "connection_port": 5432,
                "server_address": None,
                "server_port": None,
                "instance_id": "synthetic-test-installation",
            },
        }
    )
    return lifecycle._OperationRecord(request["operation_id"], descriptor, before, after, "INTENT")


def _execute(config, request, crash_at, pause_at, control):
    import fcntl

    from tests.security._authenticated_custody_source_fixtures import build_source_case

    if (
        type(request) is not dict
        or set(request) != {"operation_id", "target_state"}
        or type(request["operation_id"]) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", request["operation_id"]) is None
        or request["target_state"] not in {"VERIFY_ONLY", "REVOKED"}
    ):
        raise ValueError("closed TEST operation required; signing arbitrary data is unavailable")
    _recheck(config)
    root = Path(config["root"])
    lock = os.open(root / "lock", os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        key = _recheck(config)
        _barrier("locked", crash_at, pause_at, control)
        record = _operation(root, _before(config["role"]), request)
        intent = lifecycle._document(record)
        complete = lifecycle._document(replace(record, state="COMPLETE"))
        journal, event_path = root / "journal" / "operation", root / "events" / "original"
        if journal.exists() or journal.is_symlink():
            retained_journal = _read(journal)
            if retained_journal not in (intent, complete):
                raise ValueError("operation/descriptor/journal conflict")
            _check_lineage(config, journal, retained_journal)
        else:
            if journal.with_name("lineage").exists():
                raise ValueError("orphan test execution lineage requires resolution")
            if _read(root / "metadata", pin=config["metadata_inode"]) != custody._encode_record(
                record.before
            ):
                raise ValueError("exact predecessor missing")
            _write(journal, intent)
            _retain_lineage(journal, intent, key)
        current = _read(root / "metadata")
        after = custody._encode_record(record.after)
        if current not in (custody._encode_record(record.before), after):
            raise ValueError("metadata conflict")
        if _read(journal) == complete and current != after:
            raise ValueError("COMPLETE metadata changed")
        if event_path.exists() or event_path.is_symlink():
            if _read(journal) != complete or current != after:
                raise ValueError("retained event has no exact completed operation")
            from bot_core.authenticated_custody_mutation_source import (
                parse_custody_mutation_execution,
                parse_source_verification_policy,
                verify_custody_mutation_execution,
            )

            retained = _read(event_path)
            policy_bytes = _read(root / "events" / "public-policy")
            _check_event_lineage(config, event_path, retained, policy_bytes)
            event = parse_custody_mutation_execution(retained)
            policy = parse_source_verification_policy(policy_bytes)
            if (
                event.payload["originals"]["operation_descriptor_b64"]
                != base64.b64encode(record.descriptor).decode("ascii")
                or event.payload["originals"]["complete_journal_b64"]
                != base64.b64encode(complete).decode("ascii")
                or event.payload["executor"]["public_key_b64"]
                != base64.b64encode(config["public_key"]).decode("ascii")
            ):
                raise ValueError("retained event differs from controlled operation/source")
            verify_custody_mutation_execution(event, policy_loader=lambda: policy)
            completion_public = config.get("completion_public_key")
            if completion_public is not None and (
                policy.document["completion_authority"]["public_key_b64"]
                != base64.b64encode(completion_public).decode("ascii")
            ):
                raise ValueError("retained TEST CCA public key differs from initial binding")
            _recheck(config)
            return {"event": retained, "recovered": True}
        if _read(journal) == intent:
            _write(root / "metadata", after)
            _retain_lineage(journal, intent, key)
            _barrier("before_complete", crash_at, pause_at, control)
            _recheck(config)
            if _read(journal) != intent or _read(root / "metadata") != after:
                raise ValueError("operation bytes changed before COMPLETE")
            _check_lineage(config, journal, intent)
            _write(journal, complete)
            _retain_lineage(journal, complete, key)
        _barrier("after_complete", crash_at, pause_at, control)
        key = _recheck(config)
        if _read(journal) != complete or _read(root / "metadata") != after:
            raise ValueError("operation bytes changed before event")
        _check_lineage(config, journal, complete)
        # Recovery after COMPLETE owns/repeats the durable sync steps itself;
        # a reread of public metadata alone never produces a new source event.
        _sync(root / "metadata")
        _sync(journal)
        # This fixture builder is called only on bytes the controlled process
        # derived and wrote. A request never carries payload, digest or key.
        case = build_source_case(
            record.descriptor,
            complete,
            signing_key=key,
            completion_public_key=config.get("completion_public_key"),
        )
        _recheck(config)
        policy_bytes = canonical_json_bytes(case.policy_document)
        _write(root / "events" / "public-policy", policy_bytes)
        _write(event_path, case.envelope)
        _write(
            event_path.with_name("lineage"),
            key.sign(_event_lineage_bytes(event_path, case.envelope, policy_bytes)),
        )
        _barrier("after_event", crash_at, pause_at, control)
        _recheck(config)
        _check_lineage(config, journal, complete)
        retained = _read(event_path)
        _check_event_lineage(config, event_path, retained, _read(root / "events" / "public-policy"))
        return {"event": retained, "recovered": False}
    finally:
        os.close(lock)


def _worker(control, root, role, config, request, crash_at, pause_at, completion_public_key=None):
    try:
        if sys.platform != "linux":
            raise ValueError("the closed TEST process requires Linux")
        if config is None:
            result = _initialize(Path(root), role, completion_public_key)
        else:
            result = _execute(config, request, crash_at, pause_at, control)
            result.update(
                status="NOT_QUALIFIED",
                production_emission="BLOCKED",
                close="SYNTHETIC_TEST_CLAIM_NOT_QUALIFIED_POSTGRESQL_CLOSE",
            )
        control.send(("ok", result))
    except (OSError, ValueError, TypeError, InvalidSignature) as error:
        control.send(("refused", type(error).__name__, str(error)))
    finally:
        control.close()


def start_process(
    *, root=None, role=None, config=None, request=None, crash_at=None,
    pause_at=None, completion_public_key=None
):
    """Test orchestration only; no helper is installed as a service or runtime."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(
        target=_worker,
        args=(child, root, role, config, request, crash_at, pause_at, completion_public_key),
    )
    process.start()
    child.close()
    return process, parent
