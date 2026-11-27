"""Synthetic operator boundary tests. No namespaces or Observer are started."""

from __future__ import annotations

from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d10_launch as launch
from scripts import stage9_v5d11_operator as operator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
    pytest.mark.skipif(sys.platform != "linux", reason="Profile A operator requires Linux"),
]
CONTRACT = vars(contract)
LAUNCH = vars(launch)
SOCKET = "/run/libvirt/libvirt-sock-ro"
SHA = "a" * 64
DATA = "/etc/cryptohunter/observer-v5d1/domain-requirements.json"


def sample_manifest(tmp_path):
    return {
        "profile": operator.MANIFEST,
        "root": str(tmp_path / "physical"),
        "approval": str(tmp_path / "approval"),
        "abi_request": str(tmp_path / "request"),
        "bootstrap": str(tmp_path / "bootstrap"),
        "review_key": str(tmp_path / "key.der"),
        "uid": 1234,
        "gid": 1234,
        "libvirt_socket": SOCKET,
        "data_files": {},
        **{
            f"{name}_sha256": SHA
            for name in (
                "receipt",
                "approval",
                "bootstrap",
                "review_key",
                "apparmor_source",
                "apparmor_compiled",
                "seccomp",
            )
        },
    }


def case(tmp_path, monkeypatch):
    manifest = sample_manifest(tmp_path)
    root = manifest["root"]
    link = "/usr/lib/python3.12/link.so"
    provider = "/usr/lib/x86_64-linux-gnu/loader.so"
    interpreter = f"{root}/venv/bin/python3.12"
    sources = {
        manifest["approval"]: b"approval",
        manifest["abi_request"]: contract.canonical({"library_map": {"/lib64/ld.so": provider}}),
        f"{root}/ENVIRONMENT.json": b"receipt",
        interpreter: b"synthetic executable",
        contract.QEMU_IMG: b"synthetic qemu",
        provider: b"synthetic ELF",
    }
    pins = {name: contract.digest(raw) for name, raw in sources.items()}
    manifest["approval_sha256"] = pins[manifest["approval"]]
    manifest["receipt_sha256"] = pins[f"{root}/ENVIRONMENT.json"]
    base = {
        "excluded": [],
        "files": {provider: pins[provider]},
        "symlinks": {link: {"target": "../x86_64-linux-gnu/loader.so", "resolved": provider}},
    }
    receipt = {
        "directories": {name: 0o700 for name in (".", "venv", "venv/bin")},
        "files": {"venv/bin/python3.12": {"mode": 0o755, "sha256": pins[interpreter]}},
        "host": {"interpreter_sha256": pins[interpreter]},
        "base_runtime": base,
        "external_files": {contract.QEMU_IMG: pins[contract.QEMU_IMG], provider: pins[provider]},
        **{name: SHA for name in contract.APPROVAL_DIGESTS},
        "abi_request_sha256": pins[manifest["abi_request"]],
    }
    approval = {
        **receipt,
        "host_tools": {contract.QEMU_IMG: pins[contract.QEMU_IMG]},
    }
    mocked = {
        **CONTRACT,
        "secure_read": lambda path, **kw: sources[str(path)],
        "loads": lambda raw: approval if raw == b"approval" else contract.loads(raw),
        "validate_approval": lambda doc: None,
    }
    mocked_launch = {**LAUNCH, "verify": lambda *args: (receipt, mocked)}
    chowns = []
    monkeypatch.setattr(os, "chown", lambda *args: chowns.append(args), raising=False)
    original = Path.lstat
    socket = SimpleNamespace(st_dev=1, st_ino=2, st_uid=0, st_gid=3, st_mode=stat.S_IFSOCK | 0o666)
    monkeypatch.setattr(Path, "lstat", lambda p: socket if str(p) == SOCKET else original(p))
    return manifest, receipt, approval, sources, mocked, mocked_launch, socket, chowns


@pytest.mark.parametrize("mask", [0o022, 0o077, 0o777])
def test_stage_exact_copy_and_explicit_modes(tmp_path, monkeypatch, mask):
    manifest, receipt, _, sources, mocked, mocked_launch, _, chowns = case(tmp_path, monkeypatch)
    tree = tmp_path / "tree"
    tree.mkdir()
    data = str(tmp_path / "data")
    sources[data] = b"explicit data"
    manifest["data_files"][DATA] = {"source": data, "sha256": contract.digest(sources[data])}
    previous = os.umask(mask)
    try:
        aliases, socket, actual = operator.stage(tree, manifest, mocked, mocked_launch)
    finally:
        os.umask(previous)
    root = tree / manifest["root"][1:]
    assert actual == receipt and socket == (1, 2, 0, 3, 0o666)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "ENVIRONMENT.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tree / "usr/lib64").stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "venv/bin/python3.12").stat().st_mode) == 0o755
    assert all(args[1:] == (1234, 1234) for args in chowns)
    assert os.readlink(tree / "lib64") == "/usr/lib64"
    assert (
        os.readlink(tree / "usr/lib/python3.12/link.so") == aliases["/usr/lib/python3.12/link.so"]
    )
    assert (tree / DATA[1:]).read_bytes() == b"explicit data"
    assert (tree / operator.AUTHORITY[1:]).read_bytes() == b""
    assert (tree / operator.CLIENT_CONFIG[1:]).read_bytes() == operator.CLIENT_BYTES
    assert not (tree / "etc/ld.so.cache").exists()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("approval", "INPUT_PIN_MISMATCH"),
        ("receipt", "INPUT_PIN_MISMATCH"),
        ("binding", "RECEIPT_APPROVAL_BINDING_REJECTED"),
        ("host-tool", "RECEIPT_APPROVAL_BINDING_REJECTED"),
        ("excluded", "BASE_RUNTIME_EXCLUSIONS_REJECTED"),
        ("interpreter", "INTERPRETER_PIN_REJECTED"),
        ("runtime", "CONFLICTING_RUNTIME_PIN"),
        ("extra", "EXECUTABLE_DATA_REJECTED"),
        ("loader", "VIEW_LAYOUT_REJECTED"),
        ("collision", "VIEW_ENTRY_COLLISION"),
        ("socket", "SOURCE_SOCKET_REJECTED"),
    ],
)
def test_stage_refuses_drift(tmp_path, monkeypatch, change, reason):
    manifest, receipt, approval, sources, mocked, mocked_launch, socket, _ = case(
        tmp_path, monkeypatch
    )
    if change == "approval":
        sources[manifest["approval"]] = b"changed"
    elif change == "receipt":
        sources[f"{manifest['root']}/ENVIRONMENT.json"] = b"changed"
    elif change == "binding":
        approval["abi_request_sha256"] = SHA
    elif change == "host-tool":
        approval["host_tools"][contract.QEMU_IMG] = SHA
    elif change == "excluded":
        receipt["base_runtime"]["excluded"] = ["/alien"]
    elif change == "interpreter":
        receipt["host"]["interpreter_sha256"] = SHA
    elif change == "runtime":
        receipt["external_files"][next(iter(receipt["base_runtime"]["files"]))] = SHA
    elif change == "extra":
        manifest["data_files"][DATA] = {"source": DATA, "sha256": contract.digest(b"\x7fELF")}
        sources[DATA] = b"\x7fELF"
    elif change == "loader":
        receipt["external_files"]["/etc/ld.so.cache"] = SHA
    elif change == "collision":
        receipt["external_files"][operator.AUTHORITY] = SHA
    else:
        socket.st_uid = 1000
    tree = tmp_path / "tree"
    tree.mkdir()
    with pytest.raises(operator.Refusal, match=reason):
        operator.stage(tree, manifest, mocked, mocked_launch)


def test_manifest_exact_shape(tmp_path):
    manifest = sample_manifest(tmp_path)
    assert operator.read_manifest(contract.canonical(manifest), CONTRACT, LAUNCH) == manifest
    for key, value, reason in (
        ("uid", 0, "MANIFEST"),
        ("gid", True, "MANIFEST"),
        ("receipt_sha256", "*", "MANIFEST"),
        ("bootstrap", "local", "MANIFEST"),
        ("profile", "other", "MANIFEST"),
        ("libvirt_socket", "/run/libvirt/libvirt-sock", "MANIFEST"),
        ("data_files", [], "DATA"),
        ("data_files", {"/bin/sh": {"sha256": SHA, "source": "/bin/sh"}}, "DATA"),
        ("data_files", {DATA: {"sha256": SHA, "source": "relative"}}, "DATA"),
    ):
        with pytest.raises(operator.Refusal, match=reason):
            operator.read_manifest(contract.canonical({**manifest, key: value}), CONTRACT, LAUNCH)


@pytest.mark.parametrize(
    ("aliases", "targets", "files", "error"),
    [
        ({"/link": "dest"}, {"/link": "/dest"}, {"/dest"}, None),
        ({"/link": "/other"}, {"/link": "/dest"}, {"/dest", "/other"}, "TARGET_REJECTED"),
        ({"/link": "/link"}, {"/link": "/dest"}, {"/dest"}, "ALIAS_LOOP"),
        ({"/link": "dest"}, {"/link": "/dest"}, set(), "PROVIDER_MISSING"),
        ({}, {"/link": "/dest"}, {"/dest"}, "SET_REJECTED"),
    ],
)
def test_aliases(aliases, targets, files, error):
    if error:
        with pytest.raises(operator.Refusal, match=error):
            operator.validate_aliases(aliases, targets, files)
    else:
        operator.validate_aliases(aliases, targets, files)


def measured():
    mountinfo = (
        b"1 0 0:1 / / ro,nosuid,nodev - tmpfs tmpfs rw\n"
        b"2 1 0:2 / /proc ro,nosuid,nodev,noexec - proc proc rw\n"
        b"3 1 0:3 /null /dev/null ro,nosuid,noexec - devtmpfs devtmpfs rw\n"
        + f"4 1 0:4 {SOCKET} {SOCKET} ro,nosuid,nodev,noexec - tmpfs tmpfs rw\n".encode()
    )
    header = {
        "protocol": operator.PROTOCOL,
        "mountinfo_bytes": len(mountinfo),
        "namespaces": {"mnt": 100, "pid": 200, "net": 300},
        "apparmor_label": operator.APPARMOR_LABEL,
        "socket_identity": [1, 2, 0, 3, 0o666],
    }
    return header, mountinfo


def frame(header, mountinfo):
    return io.BytesIO(contract.canonical(header) + b"\n" + mountinfo)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("extra", "UNAPPROVED_SUBMOUNT"),
        ("writable", "WRITABLE_SUBMOUNT"),
        ("missing", "BOOTSTRAP_MOUNT_SET"),
        ("short", "FRAME_TRUNCATED"),
        ("namespace", "BOOTSTRAP_IDENTITY"),
        ("label", "BOOTSTRAP_FRAME"),
    ],
)
def test_frame_requirements(change, error):
    header, mountinfo = measured()
    if change == "extra":
        mountinfo += b"5 1 0:5 / /usr ro - tmpfs tmpfs rw\n"
    elif change == "writable":
        mountinfo = mountinfo.replace(b"/ / ro", b"/ / rw")
    elif change == "missing":
        mountinfo = b"\n".join(mountinfo.splitlines()[:3]) + b"\n"
    elif change == "namespace":
        header["namespaces"]["pid"] = 0
    elif change == "label":
        header["apparmor_label"] = "unconfined"
    header["mountinfo_bytes"] = len(mountinfo) + (1 if change == "short" else 0)
    with pytest.raises(ValueError, match=error):
        operator.read_frame(frame(header, mountinfo), SOCKET, CONTRACT, LAUNCH)


def test_candidate_matches_existing_launch_schema_and_preserves_native_block(tmp_path):
    header, mountinfo = measured()
    assert operator.read_frame(frame(header, mountinfo), SOCKET, CONTRACT, LAUNCH) == (
        header,
        mountinfo,
    )
    manifest = sample_manifest(tmp_path)
    aliases = {"/lib": "/usr/lib"}
    proposal = operator.candidate(manifest, aliases, header, mountinfo, CONTRACT, LAUNCH)
    doc = contract.loads(proposal)
    assert doc == {
        "profile": launch.LAUNCH_AUTHORIZATION,
        "receipt_sha256": SHA,
        "approval_sha256": SHA,
        "mountinfo_sha256": contract.digest(mountinfo),
        "namespaces": header["namespaces"],
        "apparmor_label": operator.APPARMOR_LABEL,
        "libvirt_socket": SOCKET,
        "libvirt_socket_identity": header["socket_identity"],
        "libvirt_config_sha256": contract.digest(operator.CLIENT_BYTES),
        "abi_request_path": operator.ABI_REQUEST,
        "aliases": aliases,
        "data_paths": [],
    }
    output = tmp_path / "out"
    output.mkdir()
    binding = operator.publish(
        output,
        manifest,
        contract.canonical(manifest),
        {"file_plan_sha256": SHA},
        {"test": True},
        proposal,
        mountinfo,
        CONTRACT,
    )
    record = contract.loads((output / "measurement.json").read_bytes())
    assert record["status"] == "CANDIDATE_ONLY"
    assert record["authorization"] == "NOT_AUTHORIZED"
    assert record["native_load"] == "NOT_TESTED"
    assert record["candidate_sha256"] == binding["candidate_sha256"]
    assert (output / "launch-authorization.candidate.json").read_bytes() == proposal
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())


@pytest.mark.skipif(
    shutil.which("openssl") is None, reason="OpenSSL required for synthetic Ed25519 test"
)
def test_detached_review_is_cryptographic_and_bound_to_this_live_measurement(tmp_path):
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", "private"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=10,
    )
    key = subprocess.run(
        ["openssl", "pkey", "-in", "private", "-pubout", "-outform", "DER"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    binding = {"session": "1" * 64, "candidate_sha256": "2" * 64, "measurement_sha256": "3" * 64}
    review = {"profile": operator.REVIEW, "decision": "ACCEPT_MEASURED_VIEW", **binding}
    (tmp_path / "message").write_bytes(contract.canonical(review))
    signature = subprocess.run(
        ["openssl", "pkeyutl", "-sign", "-inkey", "private", "-rawin", "-in", "message"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    envelope = {"review": review, "signature_hex": signature.hex()}
    encoded = contract.canonical(envelope)
    operator.verify_review(encoded, key, binding, tmp_path, CONTRACT)
    for name in binding:
        with pytest.raises(operator.Refusal, match="REVIEW_SCOPE_REJECTED"):
            operator.verify_review(encoded, key, {**binding, name: SHA}, tmp_path, CONTRACT)
    for changed, changed_key, reason in (
        (review, key, "REVIEW_ENVELOPE_REJECTED"),
        ({**envelope, "signature_hex": "0" * 127}, key, "REVIEW_SIGNATURE_SCHEMA_REJECTED"),
        ({**envelope, "signature_hex": "0" * 128}, key, "REVIEW_SIGNATURE_REJECTED"),
        (envelope, key[1:], "REVIEW_SIGNATURE_SCHEMA_REJECTED"),
    ):
        with pytest.raises(operator.Refusal, match=reason):
            operator.verify_review(
                contract.canonical(changed), changed_key, binding, tmp_path, CONTRACT
            )


@pytest.mark.parametrize("change", [None, "namespace", "socket", "policy", "dead"])
def test_measure_from_staging_to_shutdown_without_dispatch(tmp_path, monkeypatch, change):
    import tempfile

    manifest = sample_manifest(tmp_path)
    manifest_bytes = contract.canonical(manifest)
    header, mountinfo = measured()
    if change == "namespace":
        header["namespaces"]["mnt"] = Path("/proc/self/ns/mnt").stat().st_ino
    checkout = tmp_path / "operator.json"
    output = tmp_path / "out"
    monkeypatch.setattr(
        operator, "operator_inputs", lambda p: (manifest_bytes, manifest, CONTRACT, LAUNCH)
    )
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    monkeypatch.setattr(operator, "pinned", lambda *a, **kw: b"key")
    policies = [{"rule": 1}, {"rule": 2 if change == "policy" else 1}]
    monkeypatch.setattr(operator, "check_policy", lambda *a: policies.pop(0))
    creates = tempfile.TemporaryDirectory
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda **kw: creates(dir=tmp_path))

    def fake_stage(tree, *_):
        auth = tree / operator.AUTHORITY[1:]
        auth.parent.mkdir(parents=True)
        auth.touch()
        identity = (1, 2, 0, 3, 0o666)
        return (
            {},
            (identity if change != "socket" else (2, *identity[1:])),
            {"file_plan_sha256": SHA},
        )

    monkeypatch.setattr(operator, "stage", fake_stage)

    class FakeProcess:
        stdin = io.BytesIO()
        stdout = frame(header, mountinfo)
        stderr = io.BytesIO()
        status = 125 if change == "dead" else None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            assert self.poll() is not None

        def poll(self):
            return self.status

        def wait(self, timeout):
            assert timeout <= 10
            if self.status is None:
                assert self.stdin.closed
                self.status = 0
            return self.status

        def kill(self):
            self.status = -9

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(operator.select, "select", lambda *a: ([process.stdout], [], []))
    if change:
        expected = {
            "namespace": "NAMESPACE_NOT_PRIVATE",
            "socket": "SOCKET_IDENTITY_CHANGED",
            "policy": "KERNEL_POLICY_CHANGED",
            "dead": "LIVE_VIEW_LOST",
        }
        with pytest.raises(operator.Refusal, match=expected[change]):
            operator.measure(checkout, output, False)
        assert not list(output.iterdir())
    else:
        operator.measure(checkout, output, False)
        assert process.stdin.closed and process.status == 0
        assert (
            contract.loads((output / "measurement.json").read_bytes())["authorization"]
            == "NOT_AUTHORIZED"
        )


def test_live_review_reads_one_absolute_file_and_pins_candidate(tmp_path, monkeypatch):
    authority = tmp_path / "candidate.json"
    authority.write_bytes(b"candidate")
    review = tmp_path / "review.json"
    review.write_bytes(b"independent review")
    process = SimpleNamespace(stdout=io.BytesIO(), poll=lambda: None)
    stdin = io.StringIO(str(review) + "\n")
    monkeypatch.setattr(operator.sys, "stdin", stdin)
    monkeypatch.setattr(operator.select, "select", lambda *a: ([stdin], [], []))
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    accepted = []
    monkeypatch.setattr(operator, "verify_review", lambda *a: accepted.append(a))
    binding = {"candidate_sha256": contract.digest(authority.read_bytes())}
    assert (
        operator.await_review(process, authority, b"key", binding, tmp_path, CONTRACT)
        == review.read_bytes()
    )
    assert accepted[0][:3] == (review.read_bytes(), b"key", binding)
    stdin.seek(0)
    with pytest.raises(operator.Refusal, match="INPUT_PIN_MISMATCH"):
        operator.await_review(
            process, authority, b"key", {"candidate_sha256": SHA}, tmp_path, CONTRACT
        )
    monkeypatch.setattr(operator.select, "select", lambda *a: ([process.stdout], [], []))
    with pytest.raises(operator.Refusal, match="LIVE_VIEW_LOST"):
        operator.await_review(process, authority, b"key", binding, tmp_path, CONTRACT)


def test_policy_pins_gate(tmp_path, monkeypatch):
    manifest = sample_manifest(tmp_path)
    manifest["data_files"][DATA] = {"source": str(tmp_path / "input"), "sha256": SHA}
    manifest["seccomp_sha256"] = contract.digest(b"synthetic BPF")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["env"] == operator.COMMAND_ENV and kwargs["cwd"] == "/"
        if args[-1] == "--dump-seccomp":
            return SimpleNamespace(returncode=0, stdout=b"synthetic BPF")
        return SimpleNamespace(returncode=0, stdout=b'{"enforced":true}')

    monkeypatch.setattr(subprocess, "run", run)
    assert operator.check_policy(manifest, Path(manifest["bootstrap"]), CONTRACT)["apparmor"] == {
        "enforced": True
    }
    assert calls[0][-2:] == ["--data-path", DATA]
    manifest["seccomp_sha256"] = SHA
    with pytest.raises(operator.Refusal, match="SECCOMP_POLICY_PIN_MISMATCH"):
        operator.check_policy(manifest, Path(manifest["bootstrap"]), CONTRACT)


def test_build_refuses_relative_target_and_creates_static_worker(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("C compiler unavailable")
    with pytest.raises(operator.Refusal, match="LINUX_ABSOLUTE_BUILD_PATH_REQUIRED"):
        operator.build(Path("relative"))
    output = tmp_path / "bootstrap"
    result = operator.build(output)
    assert result == {
        "bootstrap_sha256": contract.digest(output.read_bytes()),
        "seccomp_sha256": contract.digest(operator.seccomp_bytes(output)),
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o755
