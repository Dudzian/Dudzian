"""Adversarial process/status tests; all identities and output are synthetic TEST data."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_gpgv_v4b as gpgv

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="Sealed descriptor adapter is Linux only"),
]

TEST_FINGERPRINT = "1234567890ABCDEF1234567890ABCDEF12345678"
TEST_OTHER_SIGNING_FINGERPRINT = "876543210FEDCBA9876543210FEDCBA987654321"
TEST_RELEASE = b"Origin: TEST ONLY\nSHA256:\n"


def _status(*, signing: str = TEST_FINGERPRINT, primary: str = TEST_FINGERPRINT) -> bytes:
    return (
        "[GNUPG:] NEWSIG\n"
        f"[GNUPG:] KEY_CONSIDERED {primary} 0\n"
        "[GNUPG:] SIG_ID dGVzdA 2023-11-14 1700000000\n"
        f"[GNUPG:] GOODSIG {signing[-16:]} TEST ONLY\n"
        f"[GNUPG:] VALIDSIG {signing} 2023-11-14 1700000000 0 4 0 1 8 00 {primary}\n"
    ).encode("ascii")


@dataclass
class ChildHarness:
    """Substitute a real TEST child while retaining the adapter's process machinery.

    The executable pin and sealed descriptor creation still run. Only the final
    Popen command is replaced, preserving its environment, status descriptor,
    cwd, session and pipe settings. Real gpgv is covered by integration tests.
    """

    monkeypatch: pytest.MonkeyPatch
    tool: Any
    commands: list[tuple[str, ...]] = field(default_factory=list)
    launch_options: list[dict[str, Any]] = field(default_factory=list)

    def install(self, body: str) -> None:
        real_popen = subprocess.Popen

        def launch(command: list[str], **kwargs: Any) -> Any:
            self.commands.append(tuple(command))
            self.launch_options.append(kwargs.copy())
            status_fd = command[command.index("--status-fd") + 1]
            source = (
                "import errno, json, os, signal, subprocess, sys, time\n"
                "status_fd = int(sys.argv[1])\n"
                "original_args = sys.argv[2:]\n" + body
            )
            return real_popen([sys.executable, "-c", source, status_fd, *command], **kwargs)

        self.monkeypatch.setattr(gpgv.subprocess, "Popen", launch)

    def verify(self, **kwargs: Any) -> Any:
        arguments = {
            "release": TEST_RELEASE,
            "signature": b"TEST detached signature",
            "keyring": b"TEST public keyring",
            "tool": self.tool,
        }
        arguments.update(kwargs)
        return gpgv.verify_detached_v4b(**arguments)

    def emit(self, status: bytes = b"", *, channel: str = "status", exitcode: int = 0) -> None:
        descriptor = {"status": "status_fd", "stdout": "1", "stderr": "2"}[channel]
        self.install(f"os.write({descriptor}, {status!r})\nsys.exit({exitcode})\n")


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChildHarness:
    executable = shutil.which("true")
    assert executable is not None, "TEST process harness requires the POSIX true executable"
    path = tmp_path / "TEST-gpgv"
    shutil.copyfile(Path(executable).resolve(), path)
    path.chmod(0o700)
    tool = gpgv.GpgvToolV4b(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    return ChildHarness(monkeypatch, tool)


def _refused(harness: ChildHarness, code: str | None = None, **kwargs: Any) -> str:
    with pytest.raises(gpgv.GpgvRefusalV4b) as failure:
        harness.verify(**kwargs)
    if code is not None:
        assert failure.value.code == code
    return str(failure.value.code)


def test_complete_machine_status_produces_only_crypto_evidence(harness: ChildHarness) -> None:
    harness.emit(_status())
    evidence = harness.verify()
    assert evidence.signing_fingerprint == TEST_FINGERPRINT
    assert evidence.primary_fingerprint == TEST_FINGERPRINT
    assert evidence.public_key_algorithm == 1
    assert evidence.hash_algorithm == 8
    assert evidence.signature_timestamp == 1700000000
    assert evidence.signature_expiry == 0
    assert not hasattr(evidence, "source_authenticity")
    assert not hasattr(evidence, "acceptance")


def test_exact_subkey_and_primary_are_distinct_evidence(harness: ChildHarness) -> None:
    harness.emit(_status(signing=TEST_OTHER_SIGNING_FINGERPRINT))
    evidence = harness.verify()
    assert evidence.signing_fingerprint == TEST_OTHER_SIGNING_FINGERPRINT
    assert evidence.primary_fingerprint == TEST_FINGERPRINT


@pytest.mark.parametrize("channel", ["stdout", "stderr"])
def test_human_channels_cannot_forge_machine_evidence(harness: ChildHarness, channel: str) -> None:
    harness.emit(_status(), channel=channel)
    _refused(harness, "GPGV_STATUS_REJECTED")


def test_zero_exit_without_any_evidence_refuses(harness: ChildHarness) -> None:
    harness.emit()
    _refused(harness, "GPGV_STATUS_REJECTED")


@pytest.mark.parametrize("missing", [b"NEWSIG", b"SIG_ID", b"GOODSIG", b"VALIDSIG"])
def test_zero_exit_with_incomplete_evidence_refuses(harness: ChildHarness, missing: bytes) -> None:
    raw = b"\n".join(line for line in _status().splitlines() if missing not in line) + b"\n"
    harness.emit(raw)
    _refused(harness, "GPGV_STATUS_REJECTED")


@pytest.mark.parametrize(
    "extra",
    [
        b"[GNUPG:] BADSIG 1234567812345678 TEST\n",
        b"[GNUPG:] ERRSIG 1234567812345678 1 8 00 1700000000 9\n",
        b"[GNUPG:] NO_PUBKEY 1234567812345678\n",
        b"[GNUPG:] EXPSIG 1234567812345678 TEST\n",
        b"[GNUPG:] EXPKEYSIG 1234567812345678 TEST\n",
        b"[GNUPG:] REVKEYSIG 1234567812345678 TEST\n",
        b"[GNUPG:] KEYREVOKED\n",
        b"[GNUPG:] FAILURE TEST 1\n",
        b"[GNUPG:] ERROR TEST 1\n",
        b"[GNUPG:] NODATA 1\n",
        b"[GNUPG:] TRUST_ULTIMATE 0 pgp\n",
        b"[GNUPG:] UNKNOWN_TEST_STATUS\n",
    ],
)
def test_contradictory_or_unknown_status_never_partially_succeeds(
    harness: ChildHarness, extra: bytes
) -> None:
    harness.emit(_status() + extra)
    _refused(harness, "GPGV_STATUS_REJECTED")


@pytest.mark.parametrize("duplicate", [b"NEWSIG", b"SIG_ID", b"GOODSIG", b"VALIDSIG"])
def test_duplicate_signature_evidence_refuses(harness: ChildHarness, duplicate: bytes) -> None:
    line = next(line for line in _status().splitlines(keepends=True) if duplicate in line)
    harness.emit(_status() + line)
    _refused(harness, "GPGV_STATUS_REJECTED")


def test_two_complete_signatures_refuse(harness: ChildHarness) -> None:
    harness.emit(_status() + _status(signing=TEST_OTHER_SIGNING_FINGERPRINT))
    _refused(harness, "GPGV_STATUS_REJECTED")


@pytest.mark.parametrize(
    "raw",
    [
        _status().replace(b"[GNUPG:] NEWSIG", b"NEWSIG", 1),
        _status().replace(b"[GNUPG:] NEWSIG", b" [GNUPG:] NEWSIG", 1),
        _status().replace(b"[GNUPG:] NEWSIG", b"[GNUPG:] NEWSIG\x00", 1),
        _status().replace(b"\n", b"\r\n"),
        _status()[:-1],
        _status() + b"\xff\n",
        _status().replace(b" 4 0 1 8 00 ", b" 4 0 1 8 01 "),
        _status().replace(b" 4 0 1 8 00 ", b" 5 0 1 8 00 "),
        _status().replace(b" 4 0 1 8 00 ", b" 4 1 1 8 00 "),
        _status().replace(b" 1700000000 0 4 ", b" 1700000000 -1 4 "),
        _status().replace(TEST_FINGERPRINT[-16:].encode() + b" TEST", b"0000000000000000 TEST"),
        _status().replace(
            b"KEY_CONSIDERED " + TEST_FINGERPRINT.encode(),
            b"KEY_CONSIDERED " + TEST_OTHER_SIGNING_FINGERPRINT.encode(),
        ),
        _status().replace(
            b"SIG_ID dGVzdA 2023-11-14 1700000000", b"SIG_ID dGVzdA 2023-11-14 1700000001"
        ),
        _status().replace(TEST_FINGERPRINT.encode(), TEST_FINGERPRINT.lower().encode()),
    ],
)
def test_malformed_or_incoherent_status_refuses(harness: ChildHarness, raw: bytes) -> None:
    harness.emit(raw)
    _refused(harness, "GPGV_STATUS_REJECTED")


@pytest.mark.parametrize("algorithm", [b"1 2", b"17 8", b"99 8", b"1 99", b"0 8"])
def test_unsupported_algorithm_refuses(harness: ChildHarness, algorithm: bytes) -> None:
    harness.emit(_status().replace(b" 1 8 00 ", b" " + algorithm + b" 00 "))
    _refused(harness, "GPGV_UNSUPPORTED_ALGORITHM")


@pytest.mark.parametrize("exitcode", [1, 2, 127])
def test_nonzero_exit_overrides_good_status(harness: ChildHarness, exitcode: int) -> None:
    harness.emit(_status(), exitcode=exitcode)
    _refused(harness, "GPGV_PROCESS_FAILED")


def test_process_crash_overrides_good_status(harness: ChildHarness) -> None:
    harness.install(f"os.write(status_fd, {_status()!r})\nos.kill(os.getpid(), signal.SIGKILL)\n")
    _refused(harness, "GPGV_PROCESS_FAILED")


def test_deadline_interrupts_real_child(harness: ChildHarness) -> None:
    harness.install("time.sleep(30)\n")
    started = time.monotonic()
    _refused(harness, "GPGV_TIMEOUT", limits=gpgv.GpgvLimitsV4b(timeout_seconds=0.1))
    assert time.monotonic() - started < 3


def test_child_cannot_evade_deadline_by_closing_all_output_pipes(harness: ChildHarness) -> None:
    harness.install("os.close(status_fd)\nos.close(1)\nos.close(2)\ntime.sleep(30)\n")
    started = time.monotonic()
    _refused(harness, "GPGV_TIMEOUT", limits=gpgv.GpgvLimitsV4b(timeout_seconds=0.1))
    assert time.monotonic() - started < 3


def test_descendant_holding_pipe_cannot_extend_deadline(harness: ChildHarness) -> None:
    harness.install(
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "pass_fds=(status_fd,))\n"
        "sys.exit(0)\n"
    )
    started = time.monotonic()
    _refused(harness, "GPGV_TIMEOUT", limits=gpgv.GpgvLimitsV4b(timeout_seconds=0.1))
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("descriptor", ["status_fd", "1", "2"])
def test_output_cap_covers_every_real_pipe(harness: ChildHarness, descriptor: str) -> None:
    harness.install(f"while True:\n    os.write({descriptor}, b'x' * 4096)\n")
    _refused(harness, "GPGV_OUTPUT_LIMIT", limits=gpgv.GpgvLimitsV4b(max_output_bytes=1024))


def test_output_cap_is_aggregate_across_channels(harness: ChildHarness) -> None:
    harness.install(f"os.write(2, b'x' * 800)\nos.write(status_fd, {_status()!r})\n")
    _refused(harness, "GPGV_OUTPUT_LIMIT", limits=gpgv.GpgvLimitsV4b(max_output_bytes=1024))


@pytest.mark.parametrize("field", ["release", "signature", "keyring"])
@pytest.mark.parametrize("value", [b"", "TEST string", None, bytearray(b"TEST")])
def test_missing_or_mutable_inputs_refuse(harness: ChildHarness, field: str, value: Any) -> None:
    harness.emit(_status())
    _refused(harness, "GPGV_INPUT_REJECTED", **{field: value})
    assert not harness.commands


def test_input_limit_is_aggregate_and_precedes_execution(harness: ChildHarness) -> None:
    harness.emit(_status())
    _refused(harness, "GPGV_INPUT_REJECTED", limits=gpgv.GpgvLimitsV4b(max_input_bytes=16))
    assert not harness.commands


@pytest.mark.parametrize("field", ["timeout_seconds", "max_input_bytes", "max_output_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, None, float("inf"), float("nan")])
def test_invalid_limits_refuse_before_execution(
    harness: ChildHarness, field: str, value: Any
) -> None:
    harness.emit(_status())
    _refused(harness, "GPGV_INPUT_REJECTED", limits=replace(gpgv.GpgvLimitsV4b(), **{field: value}))
    assert not harness.commands


@pytest.mark.parametrize("mutation", ["missing", "hash", "relative", "symlink", "non-elf"])
def test_unpinned_or_unavailable_tool_refuses(
    harness: ChildHarness, tmp_path: Path, mutation: str
) -> None:
    harness.emit(_status())
    tool = harness.tool
    if mutation == "missing":
        tool = replace(tool, path=str(tmp_path / "absent"))
    elif mutation == "hash":
        tool = replace(tool, sha256="0" * 64)
    elif mutation == "relative":
        tool = replace(tool, path="TEST-gpgv")
    elif mutation == "symlink":
        link = tmp_path / "TEST-symlink"
        link.symlink_to(tool.path)
        tool = replace(tool, path=str(link))
    else:
        Path(tool.path).write_bytes(b"#!/bin/sh\nexit 0\n")
        tool = replace(tool, sha256=hashlib.sha256(Path(tool.path).read_bytes()).hexdigest())
    _refused(harness, "GPGV_TOOL_REJECTED", tool=tool)
    assert not harness.commands


def test_spawn_failure_is_an_explicit_refusal(harness: ChildHarness) -> None:
    def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        raise OSError("TEST spawn failure")

    harness.monkeypatch.setattr(gpgv.subprocess, "Popen", fail_spawn)
    _refused(harness, "GPGV_PROCESS_FAILED")


def test_closed_environment_private_home_and_explicit_no_shell(
    harness: ChildHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_SECRET_MUST_NOT_LEAK", "TEST ONLY")
    harness.install(
        "assert 'TEST_SECRET_MUST_NOT_LEAK' not in os.environ\n"
        "assert os.environ.get('LC_ALL') == 'C'\n"
        "home = original_args[original_args.index('--homedir') + 1]\n"
        "assert os.path.realpath(os.getcwd()) == os.path.realpath(home)\n"
        "assert os.stat(home).st_mode & 0o077 == 0\n"
        f"os.write(status_fd, {_status()!r})\n"
    )
    assert harness.verify().signing_fingerprint == TEST_FINGERPRINT
    command = harness.commands[0]
    options = harness.launch_options[0]
    assert "--keyring" in command
    assert "--homedir" in command
    assert options.get("shell", False) is False
    assert options.get("close_fds", True) is True
    assert options["start_new_session"] is True
    assert "TEST_SECRET_MUST_NOT_LEAK" not in options["env"]
    assert not any("keyserver" in item for item in command)
    assert not Path(command[command.index("--homedir") + 1]).exists()


def test_release_signature_and_keyring_are_sealed_exact_bytes(harness: ChildHarness) -> None:
    harness.install(
        "keyring_path = original_args[original_args.index('--keyring') + 1]\n"
        "paths = [keyring_path, original_args[-2], original_args[-1]]\n"
        "expected = [b'TEST public keyring', b'TEST detached signature', "
        f"{TEST_RELEASE!r}]\n"
        "for path, content in zip(paths, expected):\n"
        "    with open(path, 'rb') as stream:\n"
        "        assert stream.read() == content\n"
        "    try:\n"
        "        fd = os.open(path, os.O_WRONLY)\n"
        "        try:\n"
        "            os.write(fd, b'FORGED TEST')\n"
        "        finally:\n"
        "            os.close(fd)\n"
        "    except OSError as exc:\n"
        "        assert exc.errno in (errno.EPERM, errno.EACCES)\n"
        "    else:\n"
        "        raise AssertionError('sealed TEST input was writable')\n"
        f"os.write(status_fd, {_status()!r})\n"
    )
    assert harness.verify().signing_fingerprint == TEST_FINGERPRINT


def test_executable_descriptor_is_sealed_and_matches_explicit_pin(harness: ChildHarness) -> None:
    harness.install(
        "import hashlib\n"
        "with open(original_args[0], 'rb') as stream:\n"
        f"    assert hashlib.sha256(stream.read()).hexdigest() == {harness.tool.sha256!r}\n"
        "fd = os.open(original_args[0], os.O_WRONLY)\n"
        "try:\n"
        "    try:\n"
        "        os.write(fd, b'FORGED TEST EXECUTABLE')\n"
        "    except OSError as exc:\n"
        "        assert exc.errno == errno.EPERM\n"
        "    else:\n"
        "        raise AssertionError('pinned TEST executable was writable')\n"
        "finally:\n"
        "    os.close(fd)\n"
        f"os.write(status_fd, {_status()!r})\n"
    )
    assert harness.verify().signing_fingerprint == TEST_FINGERPRINT


def test_original_tool_change_during_verification_refuses(harness: ChildHarness) -> None:
    harness.install(
        f"with open({harness.tool.path!r}, 'ab') as stream:\n"
        "    stream.write(b'CHANGED TEST')\n"
        f"os.write(status_fd, {_status()!r})\n"
    )
    _refused(harness, "GPGV_INPUT_CHANGED")


def test_original_tool_removal_during_verification_refuses(harness: ChildHarness) -> None:
    harness.install(f"os.unlink({harness.tool.path!r})\nos.write(status_fd, {_status()!r})\n")
    _refused(harness, "GPGV_INPUT_CHANGED")
