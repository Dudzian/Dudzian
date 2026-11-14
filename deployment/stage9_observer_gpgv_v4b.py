"""Bounded detached-signature adapter for caller-pinned, local TEST material.

Sealed inputs and an explicit executable digest bind the subprocess to exact
bytes. This is application containment, not an independently accepted operating
system sandbox. Dynamic-loader/runtime closure and host isolation remain open.
Nothing in this module enrolls a key or approves a trust context.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator


class GpgvRefusalV4b(ValueError):
    """Fixed diagnostics, without subprocess output or caller input."""

    def __init__(self, code: str) -> None:
        codes = {
            "GPGV_TOOL_REJECTED",
            "GPGV_STATUS_REJECTED",
            "GPGV_PROCESS_FAILED",
            "GPGV_TIMEOUT",
            "GPGV_OUTPUT_LIMIT",
            "GPGV_INPUT_REJECTED",
            "GPGV_INPUT_CHANGED",
            "GPGV_UNSUPPORTED_ALGORITHM",
        }
        self.code = code if code in codes else "GPGV_INPUT_REJECTED"
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class GpgvToolV4b:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GpgvLimitsV4b:
    timeout_seconds: float = 5
    max_input_bytes: int = 16 * 1024 * 1024
    max_output_bytes: int = 65536


@dataclass(frozen=True, slots=True)
class GpgvEvidenceV4b:
    signing_fingerprint: str
    primary_fingerprint: str
    public_key_algorithm: int
    hash_algorithm: int
    signature_timestamp: int
    signature_expiry: int


_DEFAULT_LIMITS = GpgvLimitsV4b()
_FINGERPRINT = re.compile(r"[0-9A-F]{40}")
_TOOL_BYTES = 32 * 1024 * 1024
# Linux UAPI values; some CPython builds omit these names from fcntl even
# though the running kernel implements seals. The public entry point is Linux
# only, and a failed or incomplete kernel operation always refuses.
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_REQUIRED_SEALS = 0x01 | 0x02 | 0x04 | 0x08


def _require(condition: bool, code: str = "GPGV_INPUT_REJECTED") -> None:
    if not condition:
        raise GpgvRefusalV4b(code)


def _check_time(deadline: float) -> None:
    _require(time.monotonic() < deadline, "GPGV_TIMEOUT")


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


@contextlib.contextmanager
def _sealed(raw: bytes) -> Iterator[int]:
    """Create a seekable descriptor that cannot be rewritten, even by its owner."""
    if sys.platform != "linux":
        raise GpgvRefusalV4b("GPGV_INPUT_REJECTED")
    import fcntl

    descriptor = os.memfd_create("stage9-v4b-test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        position = 0
        while position < len(raw):
            written = os.write(descriptor, raw[position : position + 65536])
            _require(written > 0)
            position += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _REQUIRED_SEALS)
        _require(fcntl.fcntl(descriptor, _F_GET_SEALS) == _REQUIRED_SEALS)
        yield descriptor
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _tool(tool: GpgvToolV4b, deadline: float) -> Iterator[int]:
    _require(type(tool) is GpgvToolV4b, "GPGV_TOOL_REJECTED")
    _require(
        type(tool.path) is str
        and tool.path.startswith("/")
        and len(tool.path) <= 4096
        and "\x00" not in tool.path
        and all(part not in {"", ".", ".."} for part in tool.path.split("/")[1:])
        and type(tool.sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", tool.sha256) is not None,
        "GPGV_TOOL_REJECTED",
    )
    try:
        descriptor = os.open(tool.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise GpgvRefusalV4b("GPGV_TOOL_REJECTED") from exc
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= _TOOL_BYTES
            and bool(before.st_mode & 0o111),
            "GPGV_TOOL_REJECTED",
        )
        raw = bytearray()
        digest = hashlib.sha256()
        while len(raw) < before.st_size:
            _check_time(deadline)
            block = os.read(descriptor, min(65536, before.st_size - len(raw)))
            _require(bool(block), "GPGV_INPUT_CHANGED")
            raw.extend(block)
            digest.update(block)
        _require(bytes(raw[:4]) == b"\x7fELF", "GPGV_TOOL_REJECTED")
        _require(digest.hexdigest() == tool.sha256, "GPGV_TOOL_REJECTED")
        _require(_snapshot(before) == _snapshot(os.fstat(descriptor)), "GPGV_INPUT_CHANGED")
        with _sealed(bytes(raw)) as frozen:
            yield frozen
        _check_time(deadline)
        try:
            current = os.stat(tool.path, follow_symlinks=False)
        except OSError as exc:
            raise GpgvRefusalV4b("GPGV_INPUT_CHANGED") from exc
        _require(
            _snapshot(before) == _snapshot(os.fstat(descriptor))
            and _snapshot(before) == _snapshot(current),
            "GPGV_INPUT_CHANGED",
        )
    finally:
        os.close(descriptor)


def _date(timestamp: str, date: str) -> int:
    _require(re.fullmatch(r"[1-9][0-9]{0,10}", timestamp) is not None, "GPGV_STATUS_REJECTED")
    value = int(timestamp)
    _require(
        datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d") == date,
        "GPGV_STATUS_REJECTED",
    )
    return value


def _parse_status(raw: bytes) -> GpgvEvidenceV4b:
    """Parse only the dedicated machine channel; stderr has no evidential role."""
    _require(bool(raw) and raw.endswith(b"\n"), "GPGV_STATUS_REJECTED")
    _require(all(byte == 10 or 32 <= byte <= 126 for byte in raw), "GPGV_STATUS_REJECTED")
    lines = raw.decode("ascii").splitlines()
    _require(4 <= len(lines) <= 12, "GPGV_STATUS_REJECTED")
    newsig = 0
    good: str | None = None
    valid: list[str] | None = None
    sig_id: list[str] | None = None
    considered: list[str] = []
    for line in lines:
        _require(line.startswith("[GNUPG:] ") and len(line) <= 4096, "GPGV_STATUS_REJECTED")
        name, _, value = line[9:].partition(" ")
        if name == "NEWSIG":
            _require(newsig == 0 and good is None and valid is None, "GPGV_STATUS_REJECTED")
            # Optional signer UID is informational; no identity is derived from it.
            newsig += 1
        elif name == "KEY_CONSIDERED":
            fields = value.split(" ")
            _require(
                newsig == 1
                and len(fields) == 2
                and _FINGERPRINT.fullmatch(fields[0]) is not None
                and fields[1] == "0"
                and len(considered) < 4,
                "GPGV_STATUS_REJECTED",
            )
            considered.append(fields[0])
        elif name == "SIG_ID":
            fields = value.split(" ")
            _require(
                newsig == 1
                and sig_id is None
                and good is None
                and len(fields) == 3
                and re.fullmatch(r"[A-Za-z0-9+/=]{1,64}", fields[0]) is not None,
                "GPGV_STATUS_REJECTED",
            )
            sig_id = fields
        elif name == "GOODSIG":
            keyid, separator, _uid = value.partition(" ")
            _require(
                newsig == 1
                and good is None
                and valid is None
                and bool(separator)
                and re.fullmatch(r"[0-9A-F]{16}", keyid) is not None,
                "GPGV_STATUS_REJECTED",
            )
            good = keyid
        elif name == "VALIDSIG":
            fields = value.split(" ")
            _require(newsig == 1 and good is not None and valid is None, "GPGV_STATUS_REJECTED")
            _require(
                len(fields) == 10
                and _FINGERPRINT.fullmatch(fields[0]) is not None
                and _FINGERPRINT.fullmatch(fields[9]) is not None
                and re.fullmatch(r"0|[1-9][0-9]{0,10}", fields[3]) is not None
                and fields[4:6] == ["4", "0"]
                and fields[8] == "00",
                "GPGV_STATUS_REJECTED",
            )
            _require(
                fields[6] in {"1", "22"} and fields[7] in {"8", "10"}, "GPGV_UNSUPPORTED_ALGORITHM"
            )
            valid = fields
        else:
            raise GpgvRefusalV4b("GPGV_STATUS_REJECTED")
    _require(
        newsig == 1 and good is not None and valid is not None and sig_id is not None,
        "GPGV_STATUS_REJECTED",
    )
    assert valid is not None and sig_id is not None and good is not None
    timestamp = _date(valid[2], valid[1])
    _require(
        good == valid[0][-16:]
        and all(fingerprint == valid[9] for fingerprint in considered)
        and _date(sig_id[2], sig_id[1]) == timestamp
        and (int(valid[3]) == 0 or int(valid[3]) > timestamp),
        "GPGV_STATUS_REJECTED",
    )
    return GpgvEvidenceV4b(
        valid[0], valid[9], int(valid[6]), int(valid[7]), timestamp, int(valid[3])
    )


def _run(
    argv: list[str],
    descriptors: tuple[int, ...],
    status_read: int,
    status_write: int,
    home: str,
    limits: GpgvLimitsV4b,
    deadline: float,
) -> bytes:
    if sys.platform != "linux":
        raise GpgvRefusalV4b("GPGV_INPUT_REJECTED")
    process: subprocess.Popen[bytes] | None = None
    output = bytearray()
    total = 0
    writer_open = True
    try:
        _check_time(deadline)
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=home,
            env={"LC_ALL": "C", "GNUPGHOME": home},
            close_fds=True,
            pass_fds=(*descriptors, status_write),
            start_new_session=True,
        )
        os.close(status_write)
        writer_open = False
        assert process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            for stream, channel in (
                (status_read, "status"),
                (process.stdout.fileno(), "stdout"),
                (process.stderr.fileno(), "stderr"),
            ):
                os.set_blocking(stream, False)
                selector.register(stream, selectors.EVENT_READ, channel)
            while selector.get_map():
                _check_time(deadline)
                for key, _events in selector.select(
                    min(0.05, max(0.0, deadline - time.monotonic()))
                ):
                    block = os.read(key.fd, min(4096, limits.max_output_bytes - total + 1))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(block)
                    _require(total <= limits.max_output_bytes, "GPGV_OUTPUT_LIMIT")
                    if key.data == "status":
                        output.extend(block)
                    elif key.data == "stdout":
                        raise GpgvRefusalV4b("GPGV_STATUS_REJECTED")
        _check_time(deadline)
        try:
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise GpgvRefusalV4b("GPGV_TIMEOUT") from exc
        _check_time(deadline)
        _require(code == 0, "GPGV_PROCESS_FAILED")
        return bytes(output)
    finally:
        if writer_open:
            os.close(status_write)
        # A child may retain pipes after its parent exits. Kill the entire
        # session on both success and refusal; this is not a privilege boundary.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired as exc:
                raise GpgvRefusalV4b("GPGV_PROCESS_FAILED") from exc
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def verify_detached_v4b(
    *,
    release: bytes,
    signature: bytes,
    keyring: bytes,
    tool: GpgvToolV4b,
    limits: GpgvLimitsV4b = _DEFAULT_LIMITS,
) -> GpgvEvidenceV4b:
    """Verify one detached binary signature over exact frozen Release bytes.

    Caller-selected tool/keyring pins are measurements, not independent trust
    approval. No configuration, user environment, key import or key retrieval
    participates. All three output channels share a strict cumulative ceiling.
    """
    try:
        if sys.platform != "linux":
            raise GpgvRefusalV4b("GPGV_INPUT_REJECTED")
        _require(hasattr(os, "memfd_create"))
        _require(type(limits) is GpgvLimitsV4b)
        _require(
            type(limits.timeout_seconds) in (int, float)
            and math.isfinite(limits.timeout_seconds)
            and 0 < limits.timeout_seconds <= 30
            and type(limits.max_input_bytes) is int
            and 1 <= limits.max_input_bytes <= 16 * 1024 * 1024
            and type(limits.max_output_bytes) is int
            and 1 <= limits.max_output_bytes <= 65536,
        )
        _require(all(type(raw) is bytes and bool(raw) for raw in (release, signature, keyring)))
        _require(sum(len(raw) for raw in (release, signature, keyring)) <= limits.max_input_bytes)
        _require(
            len(release) <= 1024 * 1024
            and len(signature) <= 65536
            and len(keyring) <= 8 * 1024 * 1024
        )
        deadline = time.monotonic() + limits.timeout_seconds
        with contextlib.ExitStack() as stack:
            executable = stack.enter_context(_tool(tool, deadline))
            release_fd = stack.enter_context(_sealed(release))
            signature_fd = stack.enter_context(_sealed(signature))
            keyring_fd = stack.enter_context(_sealed(keyring))
            home = stack.enter_context(tempfile.TemporaryDirectory(prefix="stage9-v4b-test-"))
            status_read, status_write = os.pipe2(os.O_CLOEXEC)
            stack.callback(os.close, status_read)
            argv = [
                f"/proc/self/fd/{executable}",
                "--homedir",
                home,
                "--keyring",
                f"/proc/self/fd/{keyring_fd}",
                "--status-fd",
                str(status_write),
                "--",
                f"/proc/self/fd/{signature_fd}",
                f"/proc/self/fd/{release_fd}",
            ]
            raw = _run(
                argv,
                (executable, release_fd, signature_fd, keyring_fd),
                status_read,
                status_write,
                home,
                limits,
                deadline,
            )
            evidence = _parse_status(raw)
            _check_time(deadline)
        return evidence
    except GpgvRefusalV4b:
        raise
    except (OSError, ValueError, OverflowError, MemoryError, subprocess.SubprocessError) as exc:
        raise GpgvRefusalV4b("GPGV_PROCESS_FAILED") from exc
