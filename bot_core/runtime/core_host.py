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
from typing import Any, BinaryIO, Protocol, TypeVar

from .core_host_recovery_types import (
    CoreHostRecoveryClassification,
    CoreHostRecoveryResult,
    StartupSubsystemRecoveryClassification,
    StartupSubsystemRecoveryResult,
)

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


class CoreHost:
    """Own the lock and enforce the frozen initial CoreHost startup ordering.

    The owned startup prefix reaches recovery-complete and deliberately stops
    before durable RuntimeSession history publication and readiness.
    """

    def __init__(
        self,
        scope: CoreHostScope,
        *,
        runtime_session_factory: Callable[[], SessionT],
        state_store_factory: Callable[[], StoreT],
        startup_recovery_factory: Callable[[CoreHostScope, StoreT], StartupRecovery],
        lock_factory: Callable[[CoreHostScope], CoreHostProcessLock] = CoreHostProcessLock,
    ) -> None:
        self._scope = scope
        self._runtime_session_factory = runtime_session_factory
        self._state_store_factory = state_store_factory
        self._lock_factory = lock_factory
        self._startup_recovery_factory = startup_recovery_factory
        self._lock: CoreHostProcessLock | None = None
        self._runtime_session: SessionT | None = None
        self._state_store: StoreT | None = None
        self._startup_recovery_result: CoreHostRecoveryResult | None = None

    @property
    def owns_process_lock(self) -> bool:
        return self._lock is not None and self._lock.held

    @property
    def startup_recovery_result(self) -> CoreHostRecoveryResult | None:
        return self._startup_recovery_result

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
            self._startup_recovery_result = CoreHostRecoveryResult(classification)
        except BaseException:
            self._cleanup_failed_start_attempt(process_lock, session, store)
            raise

    def _cleanup_failed_start_attempt(
        self,
        process_lock: CoreHostProcessLock,
        session: Closeable | None,
        store: Closeable | None,
    ) -> None:
        """Release only the exact resources created by this failed start attempt."""

        self._startup_recovery_result = None
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
