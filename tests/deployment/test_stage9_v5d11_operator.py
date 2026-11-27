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
    runtime = {"apply_tree": lambda *a: None, "verify_tree": lambda *a: None}
    monkeypatch.setattr(operator, "load_runtime", lambda *a: (runtime, b"runtime", {}))

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


def diagnostic_fixture(tmp_path):
    manifest = sample_manifest(tmp_path)
    paths = {
        "raw": "/var/lib/libvirt/images/synthetic.raw",
        "qcow2": "/var/lib/libvirt/images/synthetic.qcow2",
    }
    manifest["scratch"] = str(tmp_path)
    manifest["data_files"] = {
        path: {"source": str(tmp_path / name), "sha256": SHA} for name, path in paths.items()
    }
    raw = contract.canonical(manifest)
    measured = {
        "profile": operator.EVIDENCE,
        "status": "CANDIDATE_ONLY",
        "manifest_sha256": contract.digest(raw),
        "runtime_receipt_sha256": SHA,
        "immutable_images": dict.fromkeys(paths.values(), {}),
    }
    measurement = tmp_path / "measurement.json"
    measurement.write_bytes(contract.canonical(measured))
    spec = {
        "profile": operator.DIAGNOSTIC_SPEC,
        "manifest": str(tmp_path / "inputs.json"),
        "manifest_sha256": contract.digest(raw),
        "measurement": str(measurement),
        "measurement_sha256": contract.digest(measurement.read_bytes()),
        "runtime_receipt_sha256": SHA,
        "payload_sha256": SHA,
        "static_report": str(tmp_path / "report.json"),
        "static_report_sha256": SHA,
        "apparmor_source_sha256": "1" * 64,
        "apparmor_compiled_sha256": "2" * 64,
        "readonly_implementation_sha256": SHA,
        "domain_uuid": "00000000-0000-4000-8000-000000000000",
        "network_name": "synthetic",
        "fixture_socket": str(tmp_path / "disposable.socket"),
        "fixture_pid": 42,
        "fixtures": paths,
    }
    spec_path = tmp_path / "diagnostic.json"
    spec_path.write_bytes(contract.canonical(spec))
    return raw, manifest, spec, spec_path


@pytest.mark.parametrize("change", [None, "manifest", "fixture", "measurement", "runtime"])
def test_diagnostic_input_binding(tmp_path, monkeypatch, change):
    raw, manifest, spec, path = diagnostic_fixture(tmp_path)
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    if change == "manifest":
        spec["manifest_sha256"] = SHA
    elif change == "fixture":
        spec["fixtures"]["qcow2"] = spec["fixtures"]["raw"]
    elif change == "measurement":
        spec["measurement_sha256"] = SHA
    elif change == "runtime":
        spec["runtime_receipt_sha256"] = "0" * 64
    path.write_bytes(contract.canonical(spec))
    if change:
        with pytest.raises(operator.Refusal):
            operator.diagnostic_inputs(path, Path(spec["manifest"]), raw, manifest, CONTRACT)
    else:
        assert operator.diagnostic_inputs(
            path, Path(spec["manifest"]), raw, manifest, CONTRACT
        ) == (
            path.read_bytes(),
            spec,
        )
        diagnostic = operator.diagnostic_policy(manifest, spec)
        assert manifest["apparmor_source_sha256"] == SHA
        assert diagnostic["apparmor_compiled_sha256"] == "2" * 64
        assert operator.apparmor_command(diagnostic, "check", diagnostic=True)[-1] == "--diagnostic"


@pytest.mark.parametrize("expiry", [999, 1001, 4601])
def test_authorization_expiry_and_domain_handoff(tmp_path, monkeypatch, expiry):
    _, manifest, _, _ = diagnostic_fixture(tmp_path)
    path = tmp_path / "approval.json"
    envelope = {"review": {"nonce": SHA, "expires_utc": expiry}, "signature_hex": "placeholder"}
    path.write_bytes(contract.canonical(envelope))
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    monkeypatch.setattr(operator.time, "time", lambda: 1000)
    monkeypatch.setattr(operator, "review_key", lambda *a: b"key")
    calls = []
    monkeypatch.setattr(operator, "verify_review", lambda *a, **kw: calls.append((a, kw)))
    if expiry == 1001:
        assert operator.diagnostic_authority(path, b"spec", manifest, CONTRACT)[1] == SHA
        assert calls[0][0][2] == {
            "spec_sha256": contract.digest(b"spec"),
            "nonce": SHA,
            "expires_utc": expiry,
        }
        assert calls[0][1] == {
            "profile": operator.DIAGNOSTIC_REVIEW,
            "decision": "AUTHORIZE_NATIVE_DIAGNOSTIC_ONLY",
        }
    else:
        with pytest.raises(operator.Refusal, match="AUTHORIZATION_WINDOW"):
            operator.diagnostic_authority(path, b"spec", manifest, CONTRACT)
        assert not calls


def test_dedicated_account_checks_every_uid_gid_and_supplementary_group(tmp_path):
    proc = tmp_path / "42"
    proc.mkdir()
    status = proc / "status"
    status.write_bytes(b"Uid:\t1 2 3 4\nGid:\t6 7 8 9\nGroups:\t10 11\n")
    operator.check_dedicated_account(5, 12, tmp_path)
    for uid, group in ((4, 12), (5, 8), (5, 10)):
        with pytest.raises(operator.Refusal, match="ACCOUNT_IN_USE"):
            operator.check_dedicated_account(uid, group, tmp_path)


def test_stage_diagnostic_uses_only_fixed_carrier_and_data(tmp_path, monkeypatch):
    _, manifest, spec, _ = diagnostic_fixture(tmp_path)
    tree = tmp_path / "test-view"
    (tree / Path(operator.DIAGNOSTIC_SCRIPT).parent.relative_to("/")).mkdir(parents=True)
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    monkeypatch.setattr(operator, "pinned", lambda *a: b"trusted payload")
    calls = []
    monkeypatch.setattr(os, "chown", lambda *a: calls.append(a))
    operator.stage_diagnostic(tree, manifest, spec, CONTRACT)
    payload = tree / operator.DIAGNOSTIC_SCRIPT[1:]
    config = tree / operator.DIAGNOSTIC_CONFIG[1:]
    assert payload.read_bytes() == b"trusted payload"
    assert contract.loads(config.read_bytes())["images"]["raw"]["sha256"] == SHA
    assert calls == [(payload, 0, manifest["gid"]), (config, 0, manifest["gid"])]
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o440 for item in (payload, config))


def test_diagnostic_frame_requires_actual_identity_and_image_check():
    header, mountinfo = measured()
    for expected, diagnostic, images in (
        (header, False, False),
        ({**header, "images_verified": True}, False, True),
        (
            {
                **header,
                "apparmor_label": operator.APPARMOR_LABEL.replace(" (", "-diagnostic ("),
                "host_pid": 99,
                "images_verified": True,
            },
            True,
            True,
        ),
    ):
        assert operator.read_frame(
            frame(expected, mountinfo),
            SOCKET,
            CONTRACT,
            LAUNCH,
            diagnostic=diagnostic,
            images=images,
        ) == (expected, mountinfo)
    with pytest.raises(operator.Refusal, match="FRAME_REJECTED"):
        operator.read_frame(frame(header, mountinfo), SOCKET, CONTRACT, LAUNCH, diagnostic=True)


def test_one_shot_authorization_marker_never_overwrites(tmp_path):
    marker = tmp_path / "diagnostic-used"
    operator.consume_diagnostic_authority(marker, b"signed-fixture")
    with pytest.raises(FileExistsError):
        operator.consume_diagnostic_authority(marker, b"duplicate")
    assert marker.read_bytes() == b"signed-fixture"


@pytest.mark.parametrize("failure", [None, "runtime", "policy", "socket", "worker"])
def test_diagnostic_operator_only_releases_a_reviewed_view(tmp_path, monkeypatch, failure):
    from contextlib import nullcontext

    raw, manifest, spec, spec_path = diagnostic_fixture(tmp_path)
    manifest_path = Path(spec["manifest"])
    tree = tmp_path / "staging"
    output = tmp_path / "run"
    approval = contract.canonical({"review": {"expires_utc": 12345}})
    bootstrap = tmp_path / "bootstrap"
    socket_id = (11, 12, 0, 13, 0o666)
    header, mountinfo = measured()
    header["host_pid"] = 42
    receipt = {"abi_request_sha256": SHA}
    images = SimpleNamespace(arguments=lambda: ["--images"], evidence=lambda: {"fixture": {}})
    stages = (
        tree,
        {},
        socket_id,
        receipt,
        "0" * 64 if failure == "runtime" else SHA,
        b"runtime",
        images,
    )
    process = SimpleNamespace(wait=lambda **kw: 5 if failure == "worker" else 0)
    calls = []

    def capture(*args, staging_root, on_libvirt):
        calls.append("diagnostic")
        assert staging_root == tree
        assert on_libvirt(123) == [SOCKET]
        return {"python-startup": {}}, {"view": {}, "read_results": {}, "connect_paths": [SOCKET]}

    daemon_id = [*socket_id] if failure != "socket" else [1, *socket_id[1:]]
    tools = {
        "v5d11_capture": {"collect": capture},
        "v5d11_qualify": {
            "PROFILE": "evidence",
            "_objects": lambda *a: None,
            "assess": lambda *a, **kw: {"status": "CANDIDATE"},
        },
        "v5d11_libvirt": {
            "measure_host": lambda *a: {"host_socket_identity": daemon_id},
            "collect_fixture_denial": lambda *a: calls.append("denial"),
            "observe_live_connections": lambda *a: [SOCKET],
            "assemble": lambda *a: {"socket": SOCKET},
        },
    }
    monkeypatch.setattr(operator, "operator_inputs", lambda *a: (raw, manifest, CONTRACT, LAUNCH))
    monkeypatch.setattr(operator, "protected", lambda *a, **kw: None)
    monkeypatch.setattr(operator, "diagnostic_inputs", lambda *a: (spec_path.read_bytes(), spec))
    monkeypatch.setattr(operator, "diagnostic_authority", lambda *a: (approval, SHA))
    monkeypatch.setattr(operator, "trusted_bootstrap", lambda *a: bootstrap)
    monkeypatch.setattr(operator, "pinned", lambda *a, **kw: b"{}")
    monkeypatch.setattr(operator, "tool", tools.__getitem__)
    monkeypatch.setattr(operator, "stage_diagnostic", lambda *a: None)
    monkeypatch.setattr(operator, "staged", lambda *a: nullcontext(stages))
    monkeypatch.setattr(operator, "check_dedicated_account", lambda *a: None)
    monkeypatch.setattr(
        operator, "isolated_worker", lambda *a, **kw: nullcontext((process, header, mountinfo))
    )
    monkeypatch.setattr(operator.time, "time", lambda: 12344)
    policies = [{"kernel": 1}, {"kernel": 2 if failure == "policy" else 1}, {"kernel": 1}]
    monkeypatch.setattr(operator, "check_policy", lambda *a, **kw: policies.pop(0))
    marker = manifest_path.parent / f"diagnostic-used-{SHA}"
    if failure:
        reasons = {
            "runtime": "RUNTIME_PIN_MISMATCH",
            "policy": "KERNEL_POLICY_CHANGED",
            "socket": "SOCKET_IDENTITY_CHANGED",
            "worker": "EXECUTION_FAILED",
        }
        with pytest.raises(operator.Refusal, match=reasons[failure]):
            operator.diagnose(manifest_path, spec_path, tmp_path / "authorization", output)
        assert calls == (["denial", "diagnostic"] if failure == "worker" else [])
        assert marker.exists() == (failure in {"socket", "worker"})
        assert not (output / "native-evidence.json").exists()
    else:
        operator.diagnose(manifest_path, spec_path, tmp_path / "authorization", output)
        assert calls == ["denial", "diagnostic"] and marker.read_bytes() == approval
        evidence = (output / "native-evidence.json").read_bytes()
        record = contract.loads((output / "diagnostic-run.json").read_bytes())
        assert record["evidence_sha256"] == contract.digest(evidence)
        assert contract.loads(evidence)["view"]["purpose"] == "diagnostic"
