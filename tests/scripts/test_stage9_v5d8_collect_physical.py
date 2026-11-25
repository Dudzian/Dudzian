"""Synthetic inputs for the V5d8 operator collector, with no physical activity."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys
import tarfile

import pytest

from scripts import stage9_v5d8_collect_physical as operator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
    pytest.mark.skipif(sys.platform != "linux", reason="Physical collector targets Ubuntu"),
]
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def capture(tmp_path):
    contents = tmp_path / "evidence"
    contents.mkdir()
    return operator.Collection(contents, tmp_path)


def documents(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:gz") as bundle:
        return {item.name: bundle.extractfile(item).read() for item in bundle if item.isfile()}


def test_cache_records_aliases_and_refuses_incomplete_rows():
    data = (
        "2 libs found in cache `/etc/ld.so.cache'\n"
        "\tlibdemo.so.1 (libc6,x86-64) => /lib/libdemo.so.1\n"
        "\tlibdemo.so.1 (libc6,x86-64, hwcap: test) => /lib/glibc-hwcaps/test/libdemo.so.1\n"
    )
    assert [row["path"] for row in operator.parse_cache(data)["libdemo.so.1"]] == [
        "/lib/libdemo.so.1",
        "/lib/glibc-hwcaps/test/libdemo.so.1",
    ]
    with pytest.raises(operator.Refusal, match="LOADER_CACHE_COUNT_MISMATCH"):
        operator.parse_cache(data.replace("2 libs", "3 libs"))


def test_origin_support_and_unresolved_tokens():
    known, unknown = operator.directories(
        ["$ORIGIN/../private", "${ORIGIN}", "/absolute", "", "$ORIGINATOR", "$LIB", "relative"],
        {"/aliases/lib.so", "/fixed/else/lib.so"},
    )
    assert known == {
        Path("/private"),
        Path("/fixed/private"),
        Path("/aliases"),
        Path("/fixed/else"),
        Path("/absolute"),
    }
    assert unknown == {"", "$ORIGINATOR", "$LIB", "relative"}


def test_baseline_refuses_wrong_hash_and_proposal(tmp_path, monkeypatch):
    archive = tmp_path / "synthetic.tar.gz"
    values = {
        "wheelhouse-evidence.json": b'{"native_inventory":{"test":true}}',
        "v5d7-proposal.json": b'{"test":"only"}',
    }
    with tarfile.open(archive, mode="w:gz") as bundle:
        for name, data in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    with pytest.raises(operator.Refusal, match="PHYSICAL_V5D7_ARCHIVE_MISMATCH"):
        operator.read_baseline(archive)
    monkeypatch.setattr(operator, "BASELINE", operator.sha(archive.read_bytes()))
    with pytest.raises(operator.Refusal, match="PHYSICAL_V5D7_PROPOSAL_MISMATCH"):
        operator.read_baseline(archive)
    monkeypatch.setattr(operator, "PROPOSAL", operator.sha(values["v5d7-proposal.json"]))
    assert operator.read_baseline(archive) == {"test": True}


def test_regular_reader_refuses_symlink_fifo_and_oversize(tmp_path):
    path = tmp_path / "regular"
    path.write_bytes(b"sample")
    assert operator.regular(path)[0] == b"sample"
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    fifo = tmp_path / "fifo"
    operator.os.mkfifo(fifo)
    for bad, size in ((alias, 32), (fifo, 32), (path, 2)):
        with pytest.raises(operator.Refusal, match="REGULAR_FILE_REQUIRED"):
            operator.regular(bad, size)


def test_command_and_readelf_inspect_synthetic_elf_without_execution(capture, monkeypatch):
    abi = runpy.run_path(str(REPO / "scripts" / operator.ABI))
    monkeypatch.setattr(operator, "SYSTEM_ROOTS", (capture.scratch,))
    file = capture.scratch / "readelf-test.elf"
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + bytes(9),
        3,
        62,
        1,
        0,
        0,
        0,
        0,
        64,
        0,
        0,
        0,
        0,
        0,
    )
    file.write_bytes(header)
    key = capture.measure(str(file), abi)
    record = capture.measured[key]
    assert record["elf"]["class"] == "ELF64"
    assert record["elf"]["needed"] == ()
    raw = (capture.contents / record["elf"]["raw_static_evidence"]).read_text()
    assert "Advanced Micro Devices X86-64" in raw
    alias = capture.scratch / "test-alias.so"
    alias.symlink_to(file)
    assert capture.measure(str(alias), abi) == key
    assert len(capture.steps) == 1
    file.write_bytes(header + b"changed")
    with pytest.raises(operator.Refusal, match="FILE_CHANGED"):
        capture.measure(str(alias), abi)


def test_graph_follows_transitive_dependencies_and_tracks_missing_data(capture, monkeypatch):
    root = capture.scratch
    libraries = root / "libraries"
    libraries.mkdir()
    root_file = root / "python-stub"
    private = root / "private"
    private.mkdir()
    dependency = private / "libfirst.so.1"
    linkage = private / "libsecond.so.1"
    provider = libraries / "libsecond.so.1.0"
    bzip_real = libraries / "libbz2.so.1.0"
    bzip_alias = libraries / "libbz2.so.1"
    unknown_cache = libraries / "stale-libmissing.so.1"
    for file in (root_file, dependency, provider, bzip_real):
        file.write_bytes(b"\x7fELF" + file.name.encode())
    linkage.symlink_to(provider)
    bzip_alias.symlink_to(bzip_real)
    records = {
        root_file.name: (["libfirst.so.1"], ["$ORIGIN/private"], [], None),
        dependency.name: (["libsecond.so.1"], [], [], dependency.name),
        provider.name: (["libmissing.so.1"], [], [], "libsecond.so.1"),
        bzip_real.name: ([], [], [], bzip_real.name),
    }

    def fixture_static_reader(self, data, digest, row, abi):
        needed, rpath, runpath, soname = records[data[4:].decode()]
        row["elf"] = {
            "class": "ELF64",
            "machine": "Advanced Micro Devices X86-64",
            "needed": needed,
            "rpath": rpath,
            "runpath": runpath,
            "soname": soname,
            "interpreter": None,
        }

    monkeypatch.setattr(operator, "SYSTEM_ROOTS", (root,))
    monkeypatch.setattr(operator, "DEFAULT_DIRS", (libraries,))
    monkeypatch.setattr(operator.Collection, "measure_elf", fixture_static_reader)
    capture.cache = {
        "libmissing.so.1": [{"path": str(unknown_cache), "abi_as_listed": "libc6,x86-64"}]
    }
    baseline = {
        "host_abi_inputs": {
            "paths": [{"path": str(root_file), "resolved_path": str(root_file)}],
            "files_by_resolved_path": {str(root_file): {"sha256": "historical-different-hash"}},
        },
        "dependency_synthesis": {
            "unmatched_needed_names": ["libbz2.so.1", "libwheel-gap.so.1"],
            "bundled_candidates_by_needed_name": {"libbundle.so.1": []},
            "program_interpreters": [],
        },
    }
    capture.collect_graph(baseline, {}, set())
    graph = json.loads((capture.contents / "system-graph.json").read_text())
    assert graph["aliases"][str(bzip_alias)] == str(bzip_real)
    assert graph["aliases"][str(linkage)] == str(provider)
    assert graph["wheel_system_queries_without_candidate"] == ["libwheel-gap.so.1"]
    assert graph["wheel_system_queries"]["libbundle.so.1"] == {}
    assert any("HOST_DEPENDENCY_NO_MEASURED_X86_64_CANDIDATE" in issue for issue in capture.issues)
    assert any("PHYSICAL_BASELINE_CHANGED" in issue for issue in capture.issues)
    assert any("HOST_FILE_MEASUREMENT_FAILED" in issue for issue in capture.issues)
    assert any(
        edge["consumer"] == str(dependency)
        and edge["lookup"] == "libsecond.so.1"
        and edge["candidates"][0]["soname_matches"]
        for edge in graph["edges"]
    )


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        [operator.GIT, "-C", str(source), *args], capture_output=True, check=True, text=True
    ).stdout.strip()


def test_source_uses_exact_commit_even_if_checkout_changes(capture):
    source = capture.scratch / "source-checkout"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    for name in operator.HELPER_PINS:
        (scripts / name).write_bytes((REPO / "scripts" / name).read_bytes())
    entry = scripts / "stage9_observer_kvm_preflight_v5d1.py"
    entry.write_text("import ast\nraise RuntimeError('fixture must only be read')\n")
    (source / "pyproject.toml").write_text(
        '[project]\nrequires-python=">=3.12"\ndependencies=[]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n"
    )
    git(source, "init", "--quiet")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "user.name=Fixture",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "synthetic closure",
    )
    head = git(source, "rev-parse", "HEAD")
    entry.write_text("raise RuntimeError('changed checkout')\n")
    actual, _, _ = capture.collect_source(source, head)
    assert actual == head
    data = json.loads((capture.contents / "source-closure.json").read_bytes())
    assert data["static_edges"][0]["module"] == "ast"
    assert any("CHECKOUT_SOURCE_CHANGED" in issue for issue in capture.issues)


def test_partial_and_success_archives_are_verified_and_no_clobber(capture, capsys):
    capture.command("fixture", [sys.executable, "-I", "-S", "-c", "raise SystemExit(12)"])
    capture.problem("TEST ONLY REFUSAL")
    assert capture.finish() == 2
    output = capture.scratch / "result.tar.gz"
    digest = operator.archive(capture.contents, capture.scratch / "first.tar.gz", output)
    with pytest.raises(FileExistsError):
        operator.archive(capture.contents, capture.scratch / "second.tar.gz", output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert output.stat().st_mode & 0o777 == 0o600
    evidence = documents(output)
    checksums = dict(
        line.split("  ")[::-1] for line in evidence.pop("SHA256SUMS").decode().splitlines()
    )
    assert checksums == {name: hashlib.sha256(raw).hexdigest() for name, raw in evidence.items()}
    assert json.loads(evidence["exit-codes.json"])["fixture"]["exit_code"] == 12
    assert evidence["exit-code.txt"] == b"2\n"
    assert evidence["stderr.txt"] == b"TEST ONLY REFUSAL\n"
    status = json.loads(evidence["collection.json"])
    assert status["authorization"] == "NOT_AUTHORIZED"
    assert status["native_load"] == "NOT_EXECUTED"
    assert not capsys.readouterr().out


def test_wrong_head_keeps_diagnostic_archive(tmp_path, capsys):
    repository = tmp_path / "empty"
    repository.mkdir()
    old = tmp_path / "unused.tar.gz"
    output = tmp_path / "partial.tar.gz"
    assert operator.collect(repository, old, output, operator.REVISION) == 2
    contents = documents(output)
    assert contents["revision.txt"] == b"\n"
    assert b"SOURCE_HEAD_MISMATCH" in contents["stderr.txt"]
    assert json.loads(contents["exit-codes.json"])["git-head"]["exit_code"] != 0
    assert re.fullmatch(r"[0-9a-f]{64}", capsys.readouterr().out.split()[0])


def test_host_contract_uses_synthetic_static_inputs(capture, monkeypatch):
    root = capture.scratch
    closure = runpy.run_path(str(REPO / "scripts" / operator.CLOSURE))
    abi = runpy.run_path(str(REPO / "scripts" / operator.ABI))
    stdlib = root / "python3.12"
    dynload = stdlib / "lib-dynload"
    dynload.mkdir(parents=True)
    binary = dynload / "extension.so"
    binary.write_bytes(b"\x7fELFTEST_STDLIB")
    ignored = stdlib / "dist-packages"
    ignored.mkdir()
    (ignored / "test.py").write_text("raise RuntimeError('not read')")
    header = root / "patchlevel.h"
    header.write_text(
        "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 12\n#define PY_MICRO_VERSION 3\n"
    )
    sysconfig = stdlib / "_sysconfigdata__x86_64-linux-gnu.py"
    sysconfig.write_text(
        "raise RuntimeError('only parse this fixture')\n"
        "build_time_vars = {'VERSION':'3.12','SOABI':'cpython-312-x86_64-linux-gnu',"
        "'MULTIARCH':'x86_64-linux-gnu','Py_DEBUG':0}\n"
    )
    executable = root / "test-python3.12"
    qemu = root / "qemu-img"
    executable.write_bytes(b"\x7fELFTEST_CPYTHON")
    qemu.write_bytes(b"\x7fELFTEST_QEMU")
    monkeypatch.setattr(operator, "PYTHON", str(executable))
    monkeypatch.setattr(operator, "QEMU_IMG", str(qemu))
    monkeypatch.setattr(operator, "HEADER", str(header))
    monkeypatch.setattr(operator, "SYSCONFIG", str(sysconfig))
    monkeypatch.setattr(operator, "SYSTEM_ROOTS", (root,))
    monkeypatch.setattr(sys, "path", [str(dynload), str(root / "python312.zip"), str(stdlib)])
    monkeypatch.setattr(
        operator.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"}
    )
    monkeypatch.setattr(operator.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(operator.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(operator.platform, "system", lambda: "Linux")
    monkeypatch.setattr(operator.os, "confstr", lambda key: "glibc 2.39")
    monkeypatch.setattr(operator.Collection, "measure_elf", lambda *args: None)
    assert capture.collect_host(closure, abi) == {str(executable), str(qemu), str(binary)}
    host = json.loads((capture.contents / "host-candidates.json").read_bytes())
    assert host["python_contract"] == "STATIC_FIELDS_MATCH"
    assert host["abi_inputs"]["qemu_img"]["sha256"] == operator.sha(qemu.read_bytes())
    base = json.loads((capture.contents / "base-runtime.json").read_bytes())
    assert base["roots"] == [str(stdlib)]
    assert base["excluded"] == [str(ignored)]
    assert base["installation_approved"] is False
    assert not capture.issues


def test_missing_physical_abi_inputs_still_report_base_path(capture, monkeypatch):
    old = capture.scratch
    monkeypatch.setattr(operator, "SYSTEM_ROOTS", (old,))
    monkeypatch.setattr(sys, "path", ["relative", str(old / "missing-zip")])
    for item in ("PYTHON", "QEMU_IMG", "HEADER", "SYSCONFIG"):
        monkeypatch.setattr(operator, item, str(old / item.lower()))
    assert capture.collect_host({}, {}) == {operator.PYTHON, operator.QEMU_IMG}
    assert "BASE_RUNTIME_ROOTS_MISSING" in capture.issues
    assert any("BASE_IMPORT_PATH_REJECTED" in issue for issue in capture.issues)
    assert any("HOST_INPUT_UNAVAILABLE" in issue for issue in capture.issues)
    host = json.loads((capture.contents / "host-candidates.json").read_bytes())
    assert host["python_contract"] == "MISSING_INPUT"
    assert host["host"]["soabi"] is None


def test_cache_and_package_provenance_use_print_and_query_only(capture, monkeypatch):
    root = capture.scratch
    cache = root / "ld.so.cache"
    cache.write_bytes(b"test cache")
    fake_owner = root / "fake-owned-file"
    fake_owner.write_bytes(b"file")
    monkeypatch.setattr(operator, "CACHE", cache)
    monkeypatch.setattr(operator, "SYSTEM_ROOTS", (root,))
    for item in ("GIT", "READELF", "LDCONFIG", "DPKG_QUERY"):
        fake_tool = root / item.lower()
        fake_tool.write_bytes(item.encode())
        monkeypatch.setattr(operator, item, str(fake_tool))
    capture.measure(str(fake_owner), {})
    invoked = []

    def fake_command(self, label, args, *, save_stdout):
        invoked.append(args)
        if args == [operator.LDCONFIG, "-p"]:
            output = f"1 lib found in cache {cache}\nlibfake.so (libc6,x86-64) => {fake_owner}\n".encode()
        elif label.startswith("dpkg-owners-"):
            output = f"sample:amd64: {fake_owner}\n".encode()
        elif label == "dpkg-packages":
            output = b"sample:amd64\t10.1\tamd64\tii \n"
        else:
            pytest.fail(f"unexpected command: {args}")
        self.store(save_stdout, output)
        return 0, output

    monkeypatch.setattr(operator.Collection, "command", fake_command)
    capture.load_cache()
    capture.collect_provenance({})
    owners = json.loads((capture.contents / "dpkg-file-owners.json").read_bytes())
    assert owners[str(fake_owner)][str(fake_owner)] == ["sample:amd64"]
    assert capture.cache["libfake.so"][0]["path"] == str(fake_owner)
    assert capture.report["ld_so_cache_sha256"] == operator.sha(cache.read_bytes())
    assert invoked[0] == [operator.LDCONFIG, "-p"]
    assert all(command[0] == operator.DPKG_QUERY for command in invoked[1:])
    assert "sample:amd64" in invoked[-1]
    assert not capture.issues


def test_failed_command_captures_exit_and_stderr(capture):
    code, _ = capture.command("missing-exec", [str(capture.scratch / "absent")])
    assert code == 125
    assert (
        "COLLECTION_COMMAND_FAILED"
        in (capture.contents / "commands/missing-exec.stderr.txt").read_text()
    )


def test_cli_refuses_ordinary_nonisolated_execution(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--repository",
            "/unused",
            "--physical-v5d7",
            "/unused",
            "--output",
            "/unused",
        ],
    )
    assert operator.main() == 2
    assert "UNPRIVILEGED_SYSTEM_CPYTHON_REQUIRED_USE_I_S_B" in capsys.readouterr().err
