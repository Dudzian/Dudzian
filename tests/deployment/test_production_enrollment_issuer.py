"""TEST_ONLY installed-boundary fixtures; genuine provenance guards stay active."""

from __future__ import annotations

import copy
import os
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

import deployment.production_enrollment_issuer as issuer
from bot_core.licensing.pdsa_enrollment_challenge import PDSAChallengeStore
from bot_core.licensing.production_tpm_custody import ProductionTPMChallengeStore
from tests.licensing.test_pdsa_enrollment_challenge import harness as challenge_harness

installed_configuration_boundary = issuer._installed_service_configuration


@pytest.fixture
def installed(monkeypatch, tmp_path):
    return challenge_harness.__wrapped__(monkeypatch, tmp_path)


def test_factory_pair_shares_exact_opaque_context(installed):
    value = installed.issuer
    assert issuer.require_production_enrollment_issuer(value, context=installed.context) is value
    assert issuer.require_production_pdsa_store(value.pdsa_store, issuer=value) is value
    assert (
        issuer.require_production_tpm_store(
            value.tpm_store, issuer=value, pdsa_store=value.pdsa_store, context=value.trust
        )
        is value
    )
    for store in (value.pdsa_store, value.tpm_store):
        with store._connect() as database:
            assert database.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert database.execute("PRAGMA synchronous").fetchone()[0] == 2


@pytest.mark.parametrize("argument", ["path", "directory", "backend", "factory", "connection"])
def test_production_factory_cannot_accept_transport_configuration(argument):
    with pytest.raises(TypeError):
        issuer.open_installed_production_enrollment_issuer(**{argument: object()})


def test_constructor_copied_and_unregistered_contexts_have_no_authority(installed):
    with pytest.raises(TypeError):
        issuer.ProductionEnrollmentIssuerContext()
    for value in (
        object(),
        object.__new__(issuer.ProductionEnrollmentIssuerContext),
        copy.copy(installed.issuer),
        copy.deepcopy(installed.issuer),
    ):
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_REQUIRED"):
            issuer.require_production_enrollment_issuer(value)
    with pytest.raises(TypeError, match="immutable"):
        installed.issuer.trust = installed.context


@pytest.mark.parametrize("store_kind", ["pdsa", "tpm"])
def test_same_canonical_path_constructor_and_copied_object_do_not_register(installed, store_kind):
    genuine = installed.issuer.pdsa_store if store_kind == "pdsa" else installed.issuer.tpm_store
    constructor = PDSAChallengeStore if store_kind == "pdsa" else ProductionTPMChallengeStore
    guard = (
        issuer.require_production_pdsa_store
        if store_kind == "pdsa"
        else issuer.require_production_tpm_store
    )
    copied = object.__new__(constructor)
    object.__setattr__(copied, "_path", genuine._path)
    for value in (constructor(genuine._path), copied):
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
            guard(value)
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="immutable"):
            copier(genuine)


def test_genuine_contexts_cannot_be_mixed_even_for_same_canonical_files(installed):
    second = installed.reopen()
    assert second is not installed.issuer
    assert second.pdsa_store.path == installed.store.path
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        issuer.require_production_tpm_store(second.tpm_store, pdsa_store=installed.store)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        issuer.require_production_pdsa_store(installed.store, issuer=second)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        issuer.require_production_pdsa_store(installed.store, context=object())


def test_close_invalidates_old_stores_and_capabilities_before_safe_reopen(installed):
    raw = installed.store.issue(installed.context, installed.sign)
    verified = installed.store.verify_issued(raw, installed.context)
    old = installed.issuer
    old.close()
    old.close()
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        installed.store.verify_issued(raw, installed.context)
    from bot_core.licensing.pdsa_enrollment_challenge import require_verified_issued_challenge

    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        require_verified_issued_challenge(verified)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_REQUIRED"):
        _ = old.trust
    restarted = installed.reopen()
    assert restarted.pdsa_store.verify_issued(raw, restarted.trust)


@pytest.mark.parametrize("source", ["pdsa", "tpm", "directory"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer permission boundary")
def test_source_permission_change_invalidates_entire_pair(installed, source):
    path = {
        "pdsa": installed.store.path,
        "tpm": installed.issuer.tpm_store._path,
        "directory": installed.store.path.parent,
    }[source]
    path.chmod(0o755 if source == "directory" else 0o644)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        issuer.require_production_enrollment_issuer(installed.issuer)


@pytest.mark.parametrize("source", ["pdsa", "tpm"])
def test_database_replaced_by_exact_copy_at_original_path_invalidates_pair(installed, source):
    store = installed.store if source == "pdsa" else installed.issuer.tpm_store
    replacement = store._path.with_suffix(".replacement")
    with sqlite3.connect(store._path) as database, sqlite3.connect(replacement) as target:
        database.backup(target)
    replacement.chmod(0o600)
    os.replace(replacement, store._path)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        issuer.require_production_enrollment_issuer(installed.issuer)


def test_directory_replacement_invalidates_pair(installed):
    directory = installed.store.path.parent
    directory.rename(directory.with_name("previous-issuer"))
    directory.mkdir(mode=0o700)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        issuer.require_production_enrollment_issuer(installed.issuer)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer link boundary")
def test_bootstrap_refuses_linked_database(installed, link_kind):
    path = installed.store.path
    target = path.with_name("saved-database.sqlite3")
    path.rename(target)
    if link_kind == "symlink":
        path.symlink_to(target)
    else:
        os.link(target, path)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        installed.reopen()


def test_structural_guard_does_not_renew_expired_trust_for_receipt_retry(installed, monkeypatch):
    def expired(value):
        raise ValueError("TEST_ONLY_EXPIRED_TRUST")

    monkeypatch.setattr(issuer, "require_current_production_trust_context", expired)
    assert issuer.require_production_pdsa_store(installed.store) is installed.issuer
    with pytest.raises(ValueError, match="TEST_ONLY_EXPIRED_TRUST"):
        installed.reopen()


def test_installed_bootstrap_rejects_wrong_platform(monkeypatch):
    monkeypatch.setattr(issuer.sys, "platform", "win32")
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
        issuer.open_installed_production_enrollment_issuer()


@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer principal boundary")
def test_installed_bootstrap_rejects_missing_service_principal(monkeypatch):
    monkeypatch.setattr(issuer.sys, "platform", "linux")
    import pwd

    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", missing)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
        issuer.open_installed_production_enrollment_issuer()


@pytest.mark.parametrize("uid,gid", [(0, 0), (123456, 123456), (234567, 123456)])
@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer principal boundary")
def test_installed_bootstrap_rejects_root_or_wrong_os_identity(monkeypatch, uid, gid):
    import pwd

    monkeypatch.setattr(issuer.sys, "platform", "linux")
    monkeypatch.setattr(issuer.os, "geteuid", lambda: 234567)
    monkeypatch.setattr(issuer.os, "getegid", lambda: 234567)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=uid, pw_gid=gid))
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
        issuer.open_installed_production_enrollment_issuer()


@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer permission boundary")
def test_bootstrap_rejects_writable_or_missing_protected_ancestors(tmp_path):
    tmp_path.chmod(0o777)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
        issuer._require_protected_ancestors(tmp_path)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
        issuer._require_protected_ancestors(tmp_path / "missing")


@pytest.mark.skipif(os.name != "posix", reason="POSIX installed issuer principal boundary")
@pytest.mark.parametrize("wrong_owner", [None, "directory", "pdsa", "tpm"])
def test_installed_configuration_binds_fixed_principal_paths_and_source_owners(
    installed, monkeypatch, wrong_owner
):
    """Emulate OS ownership/NSS only; canonical lookup and trust guard stay active."""
    import pwd

    state = installed.store.path.parent
    package = state.parent / "TEST_ONLY_TRUST_PACKAGE"
    principal = SimpleNamespace(pw_uid=234567, pw_gid=234568)
    metadata_boundary = issuer._source_identity
    inspected_ancestors = []
    loaded_packages = []

    def owned_source(path, *, directory):
        source = metadata_boundary(path, directory=directory)
        role = "directory" if directory else "pdsa" if path == installed.store.path else "tpm"
        return replace(
            source,
            uid=principal.pw_uid + (1 if role == wrong_owner else 0),
            gid=principal.pw_gid,
        )

    def load_package(path):
        loaded_packages.append(path)
        return installed.context

    monkeypatch.setattr(issuer.sys, "platform", "linux")
    monkeypatch.setattr(pwd, "getpwnam", lambda name: principal)
    monkeypatch.setattr(issuer.os, "geteuid", lambda: principal.pw_uid)
    monkeypatch.setattr(issuer.os, "getegid", lambda: principal.pw_gid)
    monkeypatch.setattr(issuer, "STATE_DIRECTORY", state)
    monkeypatch.setattr(issuer, "TRUST_PACKAGE_DIRECTORY", package)
    monkeypatch.setattr(issuer, "_source_identity", owned_source)
    monkeypatch.setattr(issuer, "_require_protected_ancestors", inspected_ancestors.append)
    monkeypatch.setattr(issuer, "load_production_trust", load_package)
    monkeypatch.setenv("PDSA_STORE_PATH", "/caller-selected/forbidden.sqlite3")
    monkeypatch.setenv("PDSA_TRUST_PACKAGE", "/caller-selected/forbidden-trust")
    if wrong_owner is not None:
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
            installed_configuration_boundary()
        assert loaded_packages == []
    else:
        configuration = installed_configuration_boundary()
        assert configuration.state_directory == state
        assert configuration.trust is installed.context
        assert inspected_ancestors == [state.parent, package]
        assert loaded_packages == [package]
