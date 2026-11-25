"""Exercise the Ubuntu operator wrapper with synthetic input and collector bytes."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
    pytest.mark.skipif(sys.platform != "linux", reason="Physical collector targets Ubuntu"),
]
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stage9_v5d7_collect_physical.sh"


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def operator(tmp_path):
    source = tmp_path / "source checkout"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    inputs = tmp_path / "test inputs"
    inputs.mkdir()
    (inputs / "wheelhouse").mkdir()
    (inputs / "origins.json").write_text("{}", encoding="utf-8")
    reference = inputs / "reference.tar.gz"
    reference.write_bytes(b"TEST ONLY reference")
    python = tmp_path / "fake interpreter"
    python.write_text(
        "#!/bin/sh\n"
        'if [ "$4" = -c ]; then printf "%s\\n" "${TEST_PYTHON_VERSION:-cpython 3.12.3 final}"; exit; fi\n'
        'test "$1 $2 $3" = "-I -B -S" || exit 99\n'
        'test "${11}" = --measure-host-abi && test "${12}" = --proposal-output || exit 99\n'
        "rc=${TEST_COLLECTOR_EXIT:-0}\n"
        'test "$rc" != 0 || printf \'{"test_proposal":true}\' > "${13}"\n'
        "printf '{\"test_evidence\":true}\\n'\n"
        'printf "TEST ONLY diagnostic\\n" >&2\n'
        'exit "$rc"\n',
        encoding="utf-8",
    )
    python.chmod(0o700)
    wrapper = scripts / SCRIPT.name
    wrapper.write_text(
        SCRIPT.read_text(encoding="utf-8")
        .replace("python=/usr/bin/python3.12", "python=" + shlex.quote(str(python)))
        .replace(
            "base=/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7",
            "base=" + shlex.quote(str(inputs)),
        )
        .replace(
            "reference=/home/dudzian/stage9-v5d6-audit-PFvqXhAF/stage9-v5d6-full-evidence.tar.gz",
            "reference=" + shlex.quote(str(reference)),
        ),
        encoding="utf-8",
    )
    git(source, "init", "-q")
    git(source, "add", "--", "scripts")
    git(
        source,
        "-c",
        "user.name=Stage 9 Fixture",
        "-c",
        "user.email=stage9-fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "TEST ONLY checkout",
    )
    output = tmp_path / "test outputs"
    output.mkdir()
    return wrapper, git(source, "rev-parse", "HEAD"), output


def run(wrapper, revision, output):
    return subprocess.run(
        ["/bin/bash", str(wrapper), revision, str(output)], capture_output=True, text=True
    )


@pytest.mark.parametrize("collector_exit", [0, 1, 2])
def test_archive_contains_only_verified_evidence_members(operator, monkeypatch, collector_exit):
    wrapper, revision, output = operator
    monkeypatch.setenv("TEST_COLLECTOR_EXIT", str(collector_exit))
    result = run(wrapper, revision, output)
    assert result.returncode == collector_exit, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    archive = Path(values["ARCHIVE"])
    digest = Path(values["SHA256_FILE"])
    assert sorted(archive.parent.iterdir()) == sorted([archive, digest])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest.read_text().split()[0]
    assert values["COLLECTOR_EXIT"] == str(collector_exit)

    contents = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for entry in bundle:
            with bundle.extractfile(entry) as stream:
                contents[entry.name] = stream.read()
    expected_members = {
        "revision.txt",
        "inputs.txt",
        "wheelhouse-evidence.json",
        "stderr.txt",
        "exit-code.txt",
    }
    if collector_exit == 0:
        expected_members.add("v5d7-proposal.json")
    sums = dict(
        line.split(None, 1)[::-1] for line in contents.pop("SHA256SUMS").decode().splitlines()
    )
    assert set(sums) == set(contents) == expected_members
    assert all(sums[name] == hashlib.sha256(data).hexdigest() for name, data in contents.items())
    assert contents["revision.txt"].decode().strip() == revision
    assert contents["exit-code.txt"].decode().strip() == str(collector_exit)


def test_wrapper_refuses_wrong_source_and_python_before_collection(operator, monkeypatch):
    wrapper, revision, output = operator
    assert "V5D7_SOURCE_REVISION_MISMATCH" in run(wrapper, "0" * 40, output).stderr
    monkeypatch.setenv("TEST_PYTHON_VERSION", "cpython 3.12.4 final")
    assert "V5D7_SYSTEM_PYTHON_VERSION_MISMATCH" in run(wrapper, revision, output).stderr
    monkeypatch.setenv("TEST_PYTHON_VERSION", "cpython 3.12.3 candidate")
    assert "V5D7_SYSTEM_PYTHON_VERSION_MISMATCH" in run(wrapper, revision, output).stderr
    with wrapper.open("a", encoding="utf-8") as stream:
        stream.write("\n# Dirty test checkout\n")
    assert "V5D7_SOURCE_WORKTREE_CHANGED" in run(wrapper, revision, output).stderr
    assert not list(output.iterdir())
