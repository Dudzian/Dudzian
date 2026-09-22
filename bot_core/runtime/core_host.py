"""CoreHost single-instance ownership and minimal startup lifecycle.

The process lock is a topology gate.  It is intentionally independent from
SQLite and from every durable authority represented in the StateStore.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Protocol, TypeVar, cast

from bot_core.persistence.runtime_session_history import RuntimeSessionPublicationResult
from .core_host_catalog_runtime import (
    CatalogRuntimeDeploymentConfiguration,
    CatalogRuntimeStartupResult,
    CoreHostCatalogRuntimeLifecycle,
)

from .core_host_recovery_types import (
    CoreHostRecoveryClassification,
    CoreHostRecoveryResult,
    CoreHostStartupDisposition,
    StartupSubsystemRecoveryClassification,
    StartupSubsystemRecoveryResult,
)
from .runtime_session import RuntimeSession, create_runtime_session

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class CoreHostAlreadyRunningError(RuntimeError):
    """Raised when another CoreHost owns the canonical local scope."""


@dataclass(frozen=True, slots=True)
class CoreHostScope:
    """Resolved M0.3 DeviceInstallation/local-state-store lock identity."""

    account_id: str
    device_installation_id: str
    state_store_path: Path

    def __post_init__(self) -> None:
        if not self.account_id or not self.device_installation_id:
            raise ValueError("CoreHost scope identifiers must not be empty")
        object.__setattr__(self, "state_store_path", self.state_store_path.resolve())

    @property
    def lock_path(self) -> Path:
        """Return a stable locator; existence of this file is not ownership."""

        identity = f"{self.account_id}\0{self.device_installation_id}".encode()
        suffix = hashlib.sha256(identity).hexdigest()[:16]
        return self.state_store_path.with_name(
            f"{self.state_store_path.name}.{suffix}.corehost.lock"
        )


class CoreHostProcessLock:
    """Live, process-local capability backed by an OS-owned file lock."""

    def __init__(self, scope: CoreHostScope) -> None:
        self._scope = scope
        self._file: BinaryIO | None = None

    @property
    def held(self) -> bool:
        return self._file is not None

    def acquire(self) -> None:
        if self.held:
            return
        path = self._scope.lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = path.open("a+b")
        try:
            if os.name == "nt":
                lock_file.seek(0)
                if lock_file.read(1) == b"":
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    lock_file.fileno(),
                    getattr(msvcrt, "LK_NBLCK"),
                    1,
                )
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise CoreHostAlreadyRunningError(
                "another CoreHost already owns this DeviceInstallation/state-store scope"
            ) from exc
        self._file = lock_file

    def release(self) -> None:
        lock_file = self._file
        if lock_file is None:
            return
        self._file = None
        try:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    lock_file.fileno(),
                    getattr(msvcrt, "LK_UNLCK"),
                    1,
                )
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    close = release

    def __enter__(self) -> CoreHostProcessLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class Closeable(Protocol):
    def close(self) -> None: ...


SessionT = TypeVar("SessionT", bound=Closeable)
StoreT = TypeVar("StoreT", bound=Closeable)


class StartupRecovery(Protocol):
    def recover(self) -> StartupSubsystemRecoveryResult: ...


class RuntimeSessionPublisher(Protocol):
    def publish_current_session(
        self, session: RuntimeSession
    ) -> RuntimeSessionPublicationResult: ...


class RuntimeSessionPublicationOwner(StartupRecovery, Protocol):
    def runtime_session_history_publisher(self) -> RuntimeSessionPublisher: ...


class CoreHost:
    """Own P1A/P1B sequencing and the P1C RuntimeSession boundary.

    Initialized startup publishes immutable RuntimeSession history and derives
    only SETUP_REQUIRED or PROCEED_TO_LATER_STARTUP_GATES.  Later readiness
    gates and final READY remain outside this class.
    """

    def __init__(
        self,
        scope: CoreHostScope,
        *,
        runtime_session_factory: Callable[[], SessionT] | None = None,
        state_store_factory: Callable[[], StoreT],
        startup_recovery_factory: Callable[[CoreHostScope, StoreT], StartupRecovery],
        runtime_session_publication_hook: Callable[[str, RuntimeSession], None] | None = None,
        lock_factory: Callable[[CoreHostScope], CoreHostProcessLock] = CoreHostProcessLock,
        catalog_runtime_configuration: CatalogRuntimeDeploymentConfiguration | None = None,
    ) -> None:
        self._scope = scope
        self._runtime_session_factory = runtime_session_factory or cast(
            Callable[[], SessionT],
            lambda: create_runtime_session(scope.device_installation_id),
        )
        self._state_store_factory = state_store_factory
        self._lock_factory = lock_factory
        self._startup_recovery_factory = startup_recovery_factory
        self._runtime_session_publication_hook = runtime_session_publication_hook
        if (
            catalog_runtime_configuration is not None
            and type(catalog_runtime_configuration)
            is not CatalogRuntimeDeploymentConfiguration
        ):
            raise TypeError("exact CatalogRuntimeDeploymentConfiguration required")
        self._catalog_runtime_configuration = catalog_runtime_configuration
        self._catalog_runtime: CoreHostCatalogRuntimeLifecycle | None = None
        self._lock: CoreHostProcessLock | None = None
        self._runtime_session: SessionT | None = None
        self._state_store: StoreT | None = None
        self._startup_recovery_result: CoreHostRecoveryResult | None = None
        self._startup_disposition: CoreHostStartupDisposition | None = None

    @property
    def owns_process_lock(self) -> bool:
        return self._lock is not None and self._lock.held

    @property
    def startup_recovery_result(self) -> CoreHostRecoveryResult | None:
        return self._startup_recovery_result

    @property
    def startup_disposition(self) -> CoreHostStartupDisposition | None:
        return self._startup_disposition

    @property
    def catalog_runtime_startup_result(self) -> CatalogRuntimeStartupResult | None:
        return None if self._catalog_runtime is None else self._catalog_runtime.result

    def fetch_source_catalog_once(self) -> object | None:
        """Invoke the frozen release binding only through the owned ready lifecycle."""
        if self._catalog_runtime is None:
            return None
        return self._catalog_runtime.fetch_catalog_once()

    def start(self) -> None:
        if self.owns_process_lock:
            return
        process_lock = self._lock_factory(self._scope)
        process_lock.acquire()
        self._lock = process_lock
        session: Any = None
        store: Any = None
        try:
            # M0.3 order 4 precedes order 5: RuntimeSession, then mutable store.
            session = self._runtime_session_factory()
            self._runtime_session = session
            store = self._state_store_factory()
            self._state_store = store
            recovery = self._startup_recovery_factory(self._scope, store)
            subsystem = recovery.recover()
            if type(subsystem) is not StartupSubsystemRecoveryResult:
                raise RuntimeError("startup recovery returned an invalid result type")
            if (
                self._lock is not process_lock
                or not process_lock.held
                or self._runtime_session is not session
                or self._state_store is not store
            ):
                raise RuntimeError("CoreHost topology changed during startup recovery")
            if (
                subsystem.classification
                is StartupSubsystemRecoveryClassification.EMPTY_UNINITIALIZED
            ):
                classification = CoreHostRecoveryClassification.EMPTY_UNINITIALIZED
            elif (
                subsystem.classification
                is StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
            ):
                classification = CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
            else:
                raise RuntimeError("startup recovery returned an invalid classification")
            recovery_result = CoreHostRecoveryResult(classification)
            self._startup_recovery_result = recovery_result
            if classification is CoreHostRecoveryClassification.EMPTY_UNINITIALIZED:
                self._startup_disposition = CoreHostStartupDisposition.SETUP_REQUIRED
            else:
                catalog_runtime = CoreHostCatalogRuntimeLifecycle(
                    self._catalog_runtime_configuration
                )
                catalog_runtime.start_after_recovery()
                self._catalog_runtime = catalog_runtime
                if not isinstance(session, RuntimeSession):
                    raise RuntimeError("initialized CoreHost requires a RuntimeSession")
                if session.device_installation_id != self._scope.device_installation_id:
                    raise RuntimeError("RuntimeSession device binding mismatch")
                hook = self._runtime_session_publication_hook
                if hook is not None:
                    hook("before", session)
                self._assert_runtime_session_publication_topology(
                    process_lock,
                    session,
                    store,
                    recovery_result,
                )
                owner = cast(RuntimeSessionPublicationOwner, recovery)
                publisher = owner.runtime_session_history_publisher()
                publication = publisher.publish_current_session(session)
                if type(publication) is not RuntimeSessionPublicationResult:
                    raise RuntimeError("RuntimeSession publisher returned an invalid result type")
                if hook is not None:
                    hook("after", session)
                if (
                    self._lock is not process_lock
                    or not process_lock.held
                    or self._runtime_session is not session
                    or self._state_store is not store
                    or self._startup_recovery_result is not recovery_result
                ):
                    raise RuntimeError(
                        "CoreHost topology changed during RuntimeSession publication"
                    )
                self._startup_disposition = (
                    CoreHostStartupDisposition.PROCEED_TO_LATER_STARTUP_GATES
                )
        except BaseException:
            self._cleanup_failed_start_attempt(process_lock, session, store)
            raise

    def _assert_runtime_session_publication_topology(
        self,
        process_lock: CoreHostProcessLock,
        session: RuntimeSession,
        store: StoreT,
        recovery_result: CoreHostRecoveryResult,
    ) -> None:
        if (
            recovery_result.classification
            is not CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
            or self._lock is not process_lock
            or not process_lock.held
            or self._runtime_session is not session
            or self._state_store is not store
            or self._startup_recovery_result is not recovery_result
        ):
            raise RuntimeError("CoreHost topology changed before RuntimeSession publication")

    def _cleanup_failed_start_attempt(
        self,
        process_lock: CoreHostProcessLock,
        session: Closeable | None,
        store: Closeable | None,
    ) -> None:
        """Release only the exact resources created by this failed start attempt."""

        self._startup_recovery_result = None
        self._startup_disposition = None
        self._catalog_runtime = None
        self._state_store = None
        self._runtime_session = None
        self._lock = None
        try:
            if store is not None:
                store.close()
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                process_lock.release()

    def close(self) -> None:
        self._startup_recovery_result = None
        self._startup_disposition = None
        self._catalog_runtime = None
        store, self._state_store = self._state_store, None
        session, self._runtime_session = self._runtime_session, None
        process_lock, self._lock = self._lock, None
        try:
            if store is not None:
                store.close()
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                if process_lock is not None:
                    process_lock.release()

    def __enter__(self) -> CoreHost:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
