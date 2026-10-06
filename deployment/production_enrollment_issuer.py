"""Trusted off-host PDSA bootstrap and non-transferable store provenance.

The public factory has no path, backend, connection or configuration arguments.
Its fixed service principal and protected installed paths are deployment inputs,
never enrollment transport inputs. Constructors and copied SQLite data provide
mechanics only. This registry is process-local; a restart must bootstrap again
from the protected canonical files, whose terminal state remains durable.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from weakref import ReferenceType, WeakKeyDictionary, ref

from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    ProductionTrustContext,
    load_production_trust,
    require_current_production_trust_context,
)

if TYPE_CHECKING:
    from bot_core.licensing.pdsa_enrollment_challenge import PDSAChallengeStore
    from bot_core.licensing.production_tpm_custody import ProductionTPMChallengeStore

ISSUER_SERVICE_USER = "cryptohunter-pdsa-enrollment"
STATE_DIRECTORY = Path("/var/lib/cryptohunter/pdsa-enrollment")
TRUST_PACKAGE_DIRECTORY = Path("/etc/cryptohunter/production-trust") / CEREMONY_ID
PDSA_DATABASE_NAME = "pdsa-challenges.sqlite3"
TPM_DATABASE_NAME = "tpm-challenges.sqlite3"


class ProductionEnrollmentIssuerError(ValueError):
    """The issuer's deployment or process-local provenance failed closed."""


@dataclass(frozen=True, slots=True)
class _InstalledIssuerConfiguration:
    state_directory: Path
    trust: ProductionTrustContext


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    path: Path
    device: int
    inode: int
    uid: int
    gid: int
    mode: int
    directory: bool


def _source_identity(path: Path, *, directory: bool) -> _SourceIdentity:
    try:
        metadata = path.lstat()
        kind = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
        expected_mode = 0o700 if directory else 0o600
        if (
            not path.is_absolute()
            or path.resolve(strict=True) != path
            or not kind
            or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != expected_mode)
            or (not directory and metadata.st_nlink != 1)
        ):
            raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_SOURCE_CHANGED")
        return _SourceIdentity(
            path,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            stat.S_IMODE(metadata.st_mode),
            directory,
        )
    except OSError as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_SOURCE_CHANGED") from exc


def _require_protected_ancestors(path: Path) -> None:
    """Installed code/trust ancestors are root owned and never caller writable."""
    try:
        for ancestor in (path, *path.parents):
            metadata = ancestor.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != 0
                or metadata.st_mode & 0o022
            ):
                raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")
    except OSError as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED") from exc


def _installed_service_configuration() -> _InstalledIssuerConfiguration:
    """The only production configuration boundary; no environment overrides."""
    if sys.platform != "linux":
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")
    # pwd is a POSIX-only installed service boundary, not a Windows client API.
    import pwd

    try:
        principal = pwd.getpwnam(ISSUER_SERVICE_USER)
        if (
            principal.pw_uid == 0
            or os.geteuid() != principal.pw_uid
            or os.getegid() != principal.pw_gid
        ):
            raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")
        _require_protected_ancestors(STATE_DIRECTORY.parent)
        _require_protected_ancestors(TRUST_PACKAGE_DIRECTORY)
        state = _source_identity(STATE_DIRECTORY, directory=True)
        if (state.uid, state.gid) != (principal.pw_uid, principal.pw_gid):
            raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")
        for name in (PDSA_DATABASE_NAME, TPM_DATABASE_NAME):
            database = STATE_DIRECTORY / name
            if database.exists() or database.is_symlink():
                source = _source_identity(database, directory=False)
                if (source.uid, source.gid) != (principal.pw_uid, principal.pw_gid):
                    raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")
        trust = require_current_production_trust_context(
            load_production_trust(TRUST_PACKAGE_DIRECTORY)
        )
        return _InstalledIssuerConfiguration(STATE_DIRECTORY, trust)
    except (KeyError, OSError) as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED") from exc


class ProductionEnrollmentIssuerContext:
    """Factory-issued opaque authority for one exact pair of retained stores."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("use open_installed_production_enrollment_issuer()")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("production enrollment issuer context is immutable")

    @property
    def trust(self) -> ProductionTrustContext:
        return _snapshot(self).configuration.trust

    @property
    def pdsa_store(self) -> PDSAChallengeStore:
        return _snapshot(self).pdsa_store

    @property
    def tpm_store(self) -> ProductionTPMChallengeStore:
        return _snapshot(self).tpm_store

    def close(self) -> None:
        """Invalidate this process bundle and all capabilities bound to it."""
        snapshot = _ISSUERS.pop(self, None)
        if snapshot is not None:
            _STORE_ISSUERS.pop(snapshot.pdsa_store, None)
            _STORE_ISSUERS.pop(snapshot.tpm_store, None)


@dataclass(frozen=True, slots=True)
class _IssuerSnapshot:
    configuration: _InstalledIssuerConfiguration
    pdsa_store: PDSAChallengeStore
    tpm_store: ProductionTPMChallengeStore
    directory: _SourceIdentity
    pdsa_source: _SourceIdentity
    tpm_source: _SourceIdentity


_ISSUERS: WeakKeyDictionary[ProductionEnrollmentIssuerContext, _IssuerSnapshot] = (
    WeakKeyDictionary()
)
_STORE_ISSUERS: WeakKeyDictionary[object, ReferenceType[ProductionEnrollmentIssuerContext]] = (
    WeakKeyDictionary()
)


def _snapshot(value: object) -> _IssuerSnapshot:
    if type(value) is not ProductionEnrollmentIssuerContext:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_CONTEXT_REQUIRED")
    snapshot = _ISSUERS.get(value)
    if snapshot is None:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_CONTEXT_REQUIRED")
    return snapshot


def require_production_enrollment_issuer(
    value: object, *, context: object | None = None
) -> ProductionEnrollmentIssuerContext:
    """Structural provenance only; retrieving old receipts does not renew trust."""
    snapshot = _snapshot(value)
    if context is not None and context is not snapshot.configuration.trust:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_CONTEXT_MISMATCH")
    if (
        snapshot.pdsa_store.path != snapshot.pdsa_source.path
        or snapshot.tpm_store._path != snapshot.tpm_source.path
        or _source_identity(snapshot.directory.path, directory=True) != snapshot.directory
        or _source_identity(snapshot.pdsa_source.path, directory=False) != snapshot.pdsa_source
        or _source_identity(snapshot.tpm_source.path, directory=False) != snapshot.tpm_source
    ):
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_SOURCE_CHANGED")
    # Exact type was established by _snapshot before any caller property access.
    return cast(ProductionEnrollmentIssuerContext, value)


def _store_issuer(value: object) -> ProductionEnrollmentIssuerContext:
    try:
        reference = _STORE_ISSUERS.get(value)
    except TypeError:
        reference = None
    issuer = reference() if reference is not None else None
    if issuer is None:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_STORE_REQUIRED")
    return require_production_enrollment_issuer(issuer)


def require_production_pdsa_store(
    value: object,
    *,
    context: object | None = None,
    issuer: ProductionEnrollmentIssuerContext | None = None,
) -> ProductionEnrollmentIssuerContext:
    from bot_core.licensing.pdsa_enrollment_challenge import PDSAChallengeStore

    if type(value) is not PDSAChallengeStore:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_STORE_REQUIRED")
    source = _store_issuer(value)
    snapshot = _snapshot(source)
    if value is not snapshot.pdsa_store or (issuer is not None and issuer is not source):
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_CONTEXT_MISMATCH")
    return require_production_enrollment_issuer(source, context=context)


def require_production_tpm_store(
    value: object,
    *,
    context: object | None = None,
    issuer: ProductionEnrollmentIssuerContext | None = None,
    pdsa_store: object | None = None,
) -> ProductionEnrollmentIssuerContext:
    from bot_core.licensing.production_tpm_custody import ProductionTPMChallengeStore

    if type(value) is not ProductionTPMChallengeStore:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_STORE_REQUIRED")
    source = _store_issuer(value)
    snapshot = _snapshot(source)
    if value is not snapshot.tpm_store or (issuer is not None and issuer is not source):
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_CONTEXT_MISMATCH")
    require_production_enrollment_issuer(source, context=context)
    if pdsa_store is not None:
        require_production_pdsa_store(pdsa_store, context=context, issuer=source)
    return source


def _prepare_database(path: Path, directory: _SourceIdentity) -> None:
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
    except FileExistsError:
        descriptor = None
    except OSError as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_SOURCE_CHANGED") from exc
    if descriptor is not None:
        os.close(descriptor)
    source = _source_identity(path, directory=False)
    if (source.uid, source.gid) != (directory.uid, directory.gid):
        raise ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_SOURCE_CHANGED")


def open_installed_production_enrollment_issuer() -> ProductionEnrollmentIssuerContext:
    """Bootstrap only fixed protected service state; never accept caller paths."""
    from bot_core.licensing.pdsa_enrollment_challenge import PDSAChallengeStore
    from bot_core.licensing.production_tpm_custody import ProductionTPMChallengeStore

    configuration = _installed_service_configuration()
    require_current_production_trust_context(configuration.trust)
    # Production requires an already provisioned private directory. TEST_ONLY
    # fixtures may replace the installed configuration boundary with an ephemeral
    # directory; no production registrar or injectable factory is exposed.
    configuration.state_directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    directory = _source_identity(configuration.state_directory, directory=True)
    pdsa_path = directory.path / PDSA_DATABASE_NAME
    tpm_path = directory.path / TPM_DATABASE_NAME
    _prepare_database(pdsa_path, directory)
    _prepare_database(tpm_path, directory)
    pdsa_store = PDSAChallengeStore(pdsa_path)
    tpm_store = ProductionTPMChallengeStore(tpm_path)
    require_current_production_trust_context(configuration.trust)
    result = object.__new__(ProductionEnrollmentIssuerContext)
    snapshot = _IssuerSnapshot(
        configuration,
        pdsa_store,
        tpm_store,
        directory,
        _source_identity(pdsa_path, directory=False),
        _source_identity(tpm_path, directory=False),
    )
    # Registration occurs only after trusted bootstrap and both exact stores are
    # ready. A constructor, copied token or reconstructed object never registers.
    _ISSUERS[result] = snapshot
    _STORE_ISSUERS[pdsa_store] = ref(result)
    _STORE_ISSUERS[tpm_store] = ref(result)
    return require_production_enrollment_issuer(result)
