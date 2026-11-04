"""Jawne modele NSS i credentials, bez zmiany rzeczywistych UID lub grup.

Te testy sprawdzają decyzję klienta przed transportem. Nie są dowodem
kernelowej izolacji kont ani działania właściwej usługi runtime.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bot_core import freshness_runtime_ipc as ipc

RUNTIME_UID = 12001
VERIFIER_UID = 12002
RUNTIME_GID = 12003
IPC_GID = 12004
FOREIGN_ID = 12099


def _expected():
    return ipc.ExpectedPeerProcess(4242, VERIFIER_UID, IPC_GID, 31337, "a" * 64)


@pytest.fixture
def principal_model(monkeypatch):
    """Model wyłącznie lookupów i odczytów credentials; bez setuid/chown."""
    model = SimpleNamespace(
        runtime_uid=RUNTIME_UID,
        runtime_gid=RUNTIME_GID,
        verifier_uid=VERIFIER_UID,
        ipc_gid=IPC_GID,
        uids=(RUNTIME_UID,) * 3,
        gids=(RUNTIME_GID,) * 3,
        egid=RUNTIME_GID,
        groups=[RUNTIME_GID, IPC_GID],
        missing=None,
        expected=_expected(),
        lookups=[],
    )

    def account(name):
        model.lookups.append(("account", name))
        if name == model.missing:
            raise KeyError(name)
        if name == ipc.RUNTIME_USER:
            return SimpleNamespace(pw_uid=model.runtime_uid, pw_gid=model.runtime_gid)
        if name == ipc.VERIFIER_USER:
            return SimpleNamespace(pw_uid=model.verifier_uid, pw_gid=IPC_GID)
        raise AssertionError("Nieoczekiwany wybór konta przez klienta")

    def group(name):
        model.lookups.append(("group", name))
        if name == model.missing:
            raise KeyError(name)
        assert name == ipc.IPC_GROUP
        return SimpleNamespace(gr_gid=model.ipc_gid)

    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=account))
    monkeypatch.setitem(sys.modules, "grp", SimpleNamespace(getgrnam=group))
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        ipc,
        "os",
        SimpleNamespace(
            getresuid=lambda: model.uids,
            getresgid=lambda: model.gids,
            getegid=lambda: model.egid,
            getgroups=lambda: list(model.groups),
        ),
    )
    return model


def _forbid_transport(monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("niedozwolony efekt uboczny")
        pytest.fail("Odmowa musi nastąpić przed transportem i kodowaniem żądania")

    monkeypatch.setattr(
        ipc, "socket", SimpleNamespace(AF_UNIX=object(), SOCK_STREAM=object(), socket=forbidden)
    )
    monkeypatch.setattr(ipc, "_pinned_verifier_socket", forbidden)
    monkeypatch.setattr(ipc, "PinnedReadOnlyPeer", forbidden)
    monkeypatch.setattr(ipc, "_encode_ipc_bytes", forbidden)
    return calls


@pytest.mark.parametrize("groups", [[IPC_GID], [RUNTIME_GID, IPC_GID]])
def test_model_exact_runtime_with_supplementary_ipc_is_allowed(principal_model, groups):
    principal_model.groups = groups

    ipc._require_runtime_identity(principal_model.expected)

    assert principal_model.lookups == [
        ("account", ipc.RUNTIME_USER),
        ("account", ipc.VERIFIER_USER),
        ("group", ipc.IPC_GROUP),
    ]


def test_model_primary_ipc_group_needs_no_supplementary_group(principal_model):
    principal_model.runtime_gid = IPC_GID
    principal_model.gids = (IPC_GID,) * 3
    principal_model.egid = IPC_GID
    principal_model.groups = []

    ipc._require_runtime_identity(principal_model.expected)


DENIED_CREDENTIALS = [
    pytest.param("runtime_uid", 0, id="nss-runtime-root"),
    pytest.param("verifier_uid", 0, id="nss-verifier-root"),
    pytest.param("runtime_uid", VERIFIER_UID, id="same-runtime-and-verifier-uid"),
    pytest.param("runtime_gid", 0, id="nss-runtime-root-group"),
    pytest.param("ipc_gid", 0, id="nss-ipc-root-group"),
    pytest.param("uids", (0, 0, 0), id="actual-root-model"),
    pytest.param("uids", (FOREIGN_ID,) * 3, id="foreign-uid"),
    pytest.param("uids", (FOREIGN_ID, RUNTIME_UID, RUNTIME_UID), id="foreign-real-uid"),
    pytest.param("uids", (RUNTIME_UID, FOREIGN_ID, RUNTIME_UID), id="foreign-effective-uid"),
    pytest.param("uids", (RUNTIME_UID, RUNTIME_UID, FOREIGN_ID), id="foreign-saved-uid"),
    pytest.param("gids", (FOREIGN_ID,) * 3, id="foreign-gid"),
    pytest.param("gids", (FOREIGN_ID, RUNTIME_GID, RUNTIME_GID), id="foreign-real-gid"),
    pytest.param("gids", (RUNTIME_GID, FOREIGN_ID, RUNTIME_GID), id="foreign-effective-gid"),
    pytest.param("gids", (RUNTIME_GID, RUNTIME_GID, FOREIGN_ID), id="foreign-saved-gid"),
    pytest.param("groups", [RUNTIME_GID, IPC_GID, FOREIGN_ID], id="foreign-supplementary-group"),
    pytest.param("groups", [RUNTIME_GID, IPC_GID, 0], id="root-supplementary-group"),
    pytest.param("groups", [RUNTIME_GID], id="missing-ipc-membership"),
    pytest.param("groups", [], id="no-ipc-membership"),
    pytest.param("expected", replace(_expected(), uid=FOREIGN_ID), id="expected-uid-mismatch"),
    pytest.param("expected", replace(_expected(), gid=FOREIGN_ID), id="expected-gid-mismatch"),
]


@pytest.mark.parametrize(("field", "value"), DENIED_CREDENTIALS)
def test_model_unapproved_credentials_are_denied_before_transport(
    monkeypatch, principal_model, field, value
):
    setattr(principal_model, field, value)
    calls = _forbid_transport(monkeypatch)

    with pytest.raises(PermissionError, match="EXACT_FRESHNESS_RUNTIME_IDENTITY_REQUIRED"):
        ipc._require_runtime_identity(principal_model.expected)
    result = ipc.RuntimeFreshnessVerifierClient(principal_model.expected).verify_exact_candidate(
        b"proposer", b"document", b"receipt"
    )

    assert result == ipc.FreshnessIPCResult(
        ipc.FreshnessIPCOutcome.UNAVAILABLE, reason="TRANSPORT_OR_IDENTITY_UNAVAILABLE"
    )
    assert calls == []


@pytest.mark.parametrize("missing", [ipc.RUNTIME_USER, ipc.VERIFIER_USER, ipc.IPC_GROUP])
def test_model_missing_nss_identity_is_unavailable_without_transport(
    monkeypatch, principal_model, missing
):
    principal_model.missing = missing
    calls = _forbid_transport(monkeypatch)

    with pytest.raises(KeyError):
        ipc._require_runtime_identity(principal_model.expected)
    result = ipc.RuntimeFreshnessVerifierClient(principal_model.expected).verify_exact_candidate(
        b"proposer", b"document", b"receipt"
    )

    assert result == ipc.FreshnessIPCResult(
        ipc.FreshnessIPCOutcome.UNAVAILABLE, reason="TRANSPORT_OR_IDENTITY_UNAVAILABLE"
    )
    assert calls == []


@pytest.mark.parametrize("platform", ["win32", "darwin", "freebsd14"])
def test_model_unsupported_platform_does_not_query_nss_or_transport(monkeypatch, platform):
    calls = _forbid_transport(monkeypatch)
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform=platform))

    def forbidden_nss(_name):
        pytest.fail("Nieobsługiwany system nie może odpytywać NSS")

    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=forbidden_nss))
    monkeypatch.setitem(sys.modules, "grp", SimpleNamespace(getgrnam=forbidden_nss))

    result = ipc.RuntimeFreshnessVerifierClient(_expected()).verify_exact_candidate(
        b"proposer", b"document", b"receipt"
    )

    assert result == ipc.FreshnessIPCResult(
        ipc.FreshnessIPCOutcome.UNAVAILABLE, reason="NATIVE_LINUX_AF_UNIX_REQUIRED"
    )
    assert calls == []


def test_model_linux_without_af_unix_does_not_query_nss_or_transport(monkeypatch):
    calls = _forbid_transport(monkeypatch)
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delattr(ipc.socket, "AF_UNIX")

    def forbidden_nss(_name):
        pytest.fail("Brak AF_UNIX musi być rozpoznany przed NSS")

    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=forbidden_nss))
    monkeypatch.setitem(sys.modules, "grp", SimpleNamespace(getgrnam=forbidden_nss))

    result = ipc.RuntimeFreshnessVerifierClient(_expected()).verify_exact_candidate(
        b"proposer", b"document", b"receipt"
    )

    assert result == ipc.FreshnessIPCResult(
        ipc.FreshnessIPCOutcome.UNAVAILABLE, reason="NATIVE_LINUX_AF_UNIX_REQUIRED"
    )
    assert calls == []


@pytest.mark.parametrize("changes", [{"exe_sha256": None}, {"groups": [IPC_GID, FOREIGN_ID]}])
def test_model_incomplete_or_unisolated_server_identity_is_denied(changes):
    expected = _expected()
    record = {
        "pid": expected.pid,
        "uid": expected.uid,
        "gid": expected.gid,
        "start_time_ticks": expected.start_time_ticks,
        "exe_sha256": expected.executable_sha256,
        "cgroup": ipc.CGROUP,
        "groups": [IPC_GID],
        **changes,
    }
    with pytest.raises(PermissionError, match="APPROVED_VERIFIER_PROCESS_REQUIRED"):
        ipc._check_server(SimpleNamespace(process=record), expected)
