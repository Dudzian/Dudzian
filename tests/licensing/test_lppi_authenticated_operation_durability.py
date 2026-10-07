"""Hosted ABI checks for durability primitives, never native qualification."""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deployment import windows_production_lppi_operation as operation


@pytest.mark.parametrize("success", [True, False])
def test_windows_replace_uses_write_through_and_propagates_failure(monkeypatch, tmp_path, success):
    source, destination = tmp_path / "reserved.tmp", tmp_path / "reserved.json"
    move = Mock(return_value=int(success))
    library = Mock(return_value=SimpleNamespace(MoveFileExW=move))
    last_error = Mock(return_value=5)
    native_error = Mock(return_value=OSError(5, "denied by Windows"))
    monkeypatch.setattr(operation, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        operation,
        "ctypes",
        SimpleNamespace(
            WinDLL=library,
            c_wchar_p=ctypes.c_wchar_p,
            c_uint32=ctypes.c_uint32,
            c_int=ctypes.c_int,
            get_last_error=last_error,
            WinError=native_error,
        ),
    )
    if success:
        operation._replace_write_through(source, destination)
        last_error.assert_not_called()
        native_error.assert_not_called()
    else:
        with pytest.raises(OSError, match="denied by Windows"):
            operation._replace_write_through(source, destination)
        last_error.assert_called_once_with()
        native_error.assert_called_once_with(5)
    library.assert_called_once_with("kernel32", use_last_error=True)
    move.assert_called_once_with(str(source), str(destination), 0x1 | 0x8)
    assert move.argtypes == [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    assert move.restype is ctypes.c_int


def test_retained_state_open_error_fails_with_stable_error(monkeypatch, tmp_path):
    refused = Mock(side_effect=PermissionError("state is inaccessible"))
    monkeypatch.setattr(
        operation,
        "os",
        SimpleNamespace(open=refused, O_RDONLY=os.O_RDONLY, O_NOFOLLOW=0),
    )
    with pytest.raises(
        operation.LPPIAuthenticatedOperationError,
        match="INVALID_RETAINED_LPPI_AUTHENTICATED_OPERATION",
    ):
        operation._read(Path(tmp_path / "initial-authenticated-operation.json"))


def test_posix_replace_flushes_directory_after_rename(monkeypatch, tmp_path):
    source, destination = tmp_path / "reserved.tmp", tmp_path / "reserved.json"
    replace, open_directory, fsync, close = Mock(), Mock(return_value=31), Mock(), Mock()
    monkeypatch.setattr(
        operation,
        "os",
        SimpleNamespace(
            name="posix",
            replace=replace,
            open=open_directory,
            fsync=fsync,
            close=close,
            O_RDONLY=os.O_RDONLY,
            O_DIRECTORY=getattr(os, "O_DIRECTORY", 0),
        ),
    )
    operation._replace_write_through(source, destination)
    replace.assert_called_once_with(source, destination)
    open_directory.assert_called_once_with(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    fsync.assert_called_once_with(31)
    close.assert_called_once_with(31)


def test_posix_directory_flush_failure_closes_descriptor(monkeypatch, tmp_path):
    close = Mock()
    monkeypatch.setattr(
        operation,
        "os",
        SimpleNamespace(
            name="posix",
            replace=Mock(),
            open=Mock(return_value=31),
            fsync=Mock(side_effect=OSError("directory flush failed")),
            close=close,
            O_RDONLY=os.O_RDONLY,
            O_DIRECTORY=getattr(os, "O_DIRECTORY", 0),
        ),
    )
    with pytest.raises(OSError, match="directory flush failed"):
        operation._replace_write_through(tmp_path / "reserved.tmp", tmp_path / "reserved.json")
    close.assert_called_once_with(31)


def test_missing_signing_owner_rejects_before_authority_or_state_reads(monkeypatch):
    source = Mock(side_effect=AssertionError("authority must not be read"))
    monkeypatch.setattr(operation, "_source", source)
    with pytest.raises(
        operation.LPPIAuthenticatedOperationError,
        match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED",
    ):
        operation.require_reserved_operation_binding(b"{}", object())
    source.assert_not_called()


@pytest.mark.parametrize("fail_inside", [True, False])
def test_signing_owner_reset_prevents_copied_context_replay(monkeypatch, tmp_path, fail_inside):
    monkeypatch.setattr(operation, "_state_path", lambda: tmp_path / "operation.json")
    retained_context = None
    try:
        with operation._locked_signing():
            operation._require_signing_owner()
            retained_context = copy_context()
            if fail_inside:
                raise RuntimeError("signing crashed")
    except RuntimeError as exc:
        assert fail_inside and str(exc) == "signing crashed"
    assert retained_context is not None
    assert operation._SIGNING_OWNER.get() is None
    assert operation._HELD_SIGNING_OWNER is None
    with pytest.raises(
        operation.LPPIAuthenticatedOperationError,
        match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED",
    ):
        retained_context.run(operation._require_signing_owner)


def test_signing_owner_rejects_copied_context_on_another_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(operation, "_state_path", lambda: tmp_path / "operation.json")
    with operation._locked_signing():
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            attempted = executor.submit(context.run, operation._require_signing_owner)
            with pytest.raises(
                operation.LPPIAuthenticatedOperationError,
                match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED",
            ):
                attempted.result()
        operation._require_signing_owner()


def test_signing_owner_rejects_inherited_pid_and_changed_singleton_path(monkeypatch, tmp_path):
    path = tmp_path / "operation.json"
    monkeypatch.setattr(operation, "_state_path", lambda: path)
    with operation._locked_signing():
        with monkeypatch.context() as patch:
            patch.setattr(operation, "os", SimpleNamespace(getpid=lambda: os.getpid() + 1))
            with pytest.raises(
                operation.LPPIAuthenticatedOperationError,
                match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED",
            ):
                operation._require_signing_owner()
        with monkeypatch.context() as patch:
            patch.setattr(operation, "_state_path", lambda: path.with_name("foreign.json"))
            with pytest.raises(
                operation.LPPIAuthenticatedOperationError,
                match="LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED",
            ):
                operation._require_signing_owner()
        operation._require_signing_owner()


def test_concurrent_signing_lock_fails_busy_without_losing_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(operation, "_state_path", lambda: tmp_path / "operation.json")
    with operation._locked_signing():
        with pytest.raises(operation.LPPIAuthenticatedOperationError, match="OPERATION_BUSY"):
            with operation._locked_signing():
                pytest.fail("a second invocation obtained the held signing lock")
        operation._require_signing_owner()
