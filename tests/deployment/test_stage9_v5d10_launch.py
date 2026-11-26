"""Synthetic import/receipt and kernel-boundary checks, without native loading."""

from __future__ import annotations

import errno
from importlib import machinery
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import stage9_v5d4_closure as closure
from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d10_launch as launch

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="Linux Observer preflight profile"),
]


def put(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o600)


@pytest.fixture
def sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "sealed"
    root.mkdir(mode=0o700)
    base = tmp_path / "base"
    put(base / "statistics.py", b"answer = 42\n")
    put(
        root / "packages/scripts/preflight.py", b"raise AssertionError('entrypoint must not run')\n"
    )
    put(root / "packages/cryptography/__init__.py", b"value = 0\n")
    put(root / "packages/cryptography/child.py", b"value = 42\n")
    put(root / "contract.py", Path(contract.__file__).read_bytes())
    put(root / "runtime-audit.py", Path(closure.__file__).read_bytes())
    plan = contract.inventory(root)
    receipt = {
        "profile": contract.RECEIPT,
        "root": str(root),
        "entrypoint": contract.ENTRYPOINT,
        "host": {**contract.HOST, "interpreter_sha256": "f" * 64},
        "base_runtime": closure.base_runtime_inventory([base]),
        "external_files": {contract.QEMU_IMG: contract.digest(b"synthetic qemu")},
        "authorization": "NOT_AUTHORIZED",
        "native_load": "NOT_EXECUTED",
        **plan,
        **{name: "a" * 64 for name in contract.APPROVAL_DIGESTS | {"approval_sha256"}},
    }
    receipt["file_plan_sha256"] = contract.plan_sha256(plan)
    original = launch._helper

    def measured_helper(directory, name, manifest):
        loaded = original(directory, name, manifest)
        if name == "contract.py":
            read = loaded["secure_read"]
            loaded["secure_read"] = (
                lambda path, **kwargs: b"synthetic qemu"
                if str(path) == contract.QEMU_IMG
                else read(path, **kwargs)
            )
        return loaded

    monkeypatch.setattr(launch, "_helper", measured_helper)
    return root, receipt


def seal(root: Path, receipt: dict[str, Any]) -> str:
    data = contract.canonical(receipt) + b"\n"
    put(root / "ENVIRONMENT.json", data)
    return contract.digest(data)


def test_verify_exact_seal_is_static(sealed) -> None:
    root, receipt = sealed
    digest = seal(root, receipt)
    assert launch.verify(root, digest, receipt["approval_sha256"])[0] == receipt


@pytest.mark.parametrize(
    "mutate,reason",
    [
        ("wrong_receipt_digest", "INDEPENDENT_RECEIPT_PIN_MISMATCH"),
        ("wrong_approval_digest", "INDEPENDENT_APPROVAL_PIN_MISMATCH"),
        ("wrong_profile", "RECEIPT_PROFILE_REJECTED"),
        ("renamed_profile", "PROFILE_SCHEMA_REJECTED"),
        ("modified_helper", "HELPER_PIN_MISMATCH"),
        ("moved_root", "RECEIPT_ROOT_MISMATCH"),
        ("changed_file", "RECEIPT_PLAN_MISMATCH"),
        ("foreign_library", "RECEIPT_PLAN_MISMATCH"),
        ("changed_mode", "RECEIPT_PLAN_MISMATCH"),
        ("changed_external", "EXTERNAL_PIN_MISMATCH"),
        ("changed_base", "BASE_RUNTIME_MISMATCH"),
    ],
)
def test_verify_refuses_inconsistent_evidence(sealed, mutate: str, reason: str) -> None:
    root, receipt = sealed
    if mutate == "wrong_profile":
        receipt["profile"] = "stage9-v5d7-prepared-environment-v1"
    elif mutate == "renamed_profile":
        receipt["wheel_shared_members"] = []
    elif mutate == "modified_helper":
        put(root / "contract.py", b"unreviewed helper")
    elif mutate == "moved_root":
        receipt["root"] = str(root.with_name("other"))
    elif mutate == "changed_file":
        put(root / "packages/cryptography/child.py", b"value = 0\n")
    elif mutate == "foreign_library":
        put(root / "packages/foreign.so", b"synthetic foreign library")
        put(root / "packages/zstandard.py", b"raise AssertionError('B code must not run')\n")
    elif mutate == "changed_mode":
        (root / "packages/cryptography/child.py").chmod(0o644)
    elif mutate == "changed_external":
        receipt["external_files"][contract.QEMU_IMG] = "0" * 64
    elif mutate == "changed_base":
        put(Path(receipt["base_runtime"]["roots"][0]) / "statistics.py", b"value = 43\n")
    digest = seal(root, receipt)
    supplied_approval = receipt["approval_sha256"]
    if mutate == "wrong_receipt_digest":
        digest = "0" * 64
    elif mutate == "wrong_approval_digest":
        supplied_approval = "0" * 64
    with pytest.raises(ValueError, match=reason):
        launch.verify(root, digest, supplied_approval)


def test_only_exact_import_paths_even_with_ambient_contamination(sealed, tmp_path: Path) -> None:
    root, receipt = sealed
    finder = launch.ApprovedImports(root, receipt, vars(contract))
    ambient = tmp_path / "ambient"
    put(ambient / "cryptography.py", b"raise AssertionError('ambient import')\n")
    put(ambient / "statistics.py", b"raise AssertionError('ambient import')\n")
    package = finder.find_spec("cryptography")
    assert package.origin == str(root / "packages/cryptography/__init__.py")
    child = finder.find_spec("cryptography.child", package.submodule_search_locations)
    assert child.origin == str(root / "packages/cryptography/child.py")
    assert child.loader.get_code("cryptography.child").co_consts[0] == 42
    assert finder.find_spec("statistics").origin == str(
        Path(receipt["base_runtime"]["roots"][0]) / "statistics.py"
    )
    assert finder.find_spec("sys").origin == "built-in"
    assert finder.find_spec("scripts").submodule_search_locations == [
        str(root / "packages/scripts")
    ]
    with pytest.raises(launch.Refusal, match="IMPORT_PARENT_PATH_REJECTED"):
        finder.find_spec("cryptography.child", [str(ambient)])
    for name in ("zstandard", "PySide6", "libvirt_qemu", "sitecustomize", "ctypes"):
        with pytest.raises(ModuleNotFoundError, match="Observer A import refused"):
            finder.find_spec(name)
    put(root / "packages/cryptography/unplanned.py", b"value = 0\n")
    with pytest.raises(launch.Refusal, match="IMPORT_FILE_NOT_PINNED"):
        finder.find_spec("cryptography.unplanned", package.submodule_search_locations)
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("defusedxml")
    put(Path(child.origin), b"value = 43\n")
    with pytest.raises(launch.Refusal, match="IMPORT_PIN_MISMATCH"):
        child.loader.get_code("cryptography.child")


def test_startup_rejects_preloaded_modules_finders_and_foreign_roots(
    sealed, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt = sealed
    finder = launch.ApprovedImports(root, receipt, vars(contract))
    base = receipt["base_runtime"]["roots"][0]
    module = SimpleNamespace(__spec__=SimpleNamespace(origin=str(Path(base) / "statistics.py")))
    fixture = SimpleNamespace(
        path=[base],
        meta_path=[machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder],
        modules={"statistics": module, "__main__": None},
        stdlib_module_names=sys.stdlib_module_names,
    )
    monkeypatch.setattr(launch, "sys", fixture)
    finder.validate_startup()
    fixture.path.append("/foreign")
    with pytest.raises(launch.Refusal, match="STARTUP_IMPORT_ROOT_REJECTED"):
        finder.validate_startup()
    fixture.path.pop()
    fixture.modules["zstandard"] = object()
    with pytest.raises(launch.Refusal, match="PRELOADED_MODULE_REJECTED"):
        finder.validate_startup()
    del fixture.modules["zstandard"]
    fixture.meta_path.append(object())
    with pytest.raises(launch.Refusal, match="STARTUP_FINDERS_REJECTED"):
        finder.validate_startup()


def test_audit_denies_native_fallback_and_arbitrary_subprocess(sealed) -> None:
    root, receipt = sealed
    finder = launch.ApprovedImports(root, receipt, vars(contract))
    with pytest.raises(launch.Refusal, match="AUDIT_IMPORT_DENIED"):
        launch._audit(finder, "", "import", ("_ctypes", "/fake/ctypes.so"))
    with pytest.raises(launch.Refusal, match="IMPORT_FILE_NOT_PINNED"):
        launch._audit(finder, "", "import", ("math", "/fake/math.so"))
    with pytest.raises(launch.Refusal, match="DYNAMIC_EXECUTION_REJECTED"):
        launch._audit(finder, "", "ctypes.dlopen", ("libanything.so",))
    with pytest.raises(launch.Refusal, match="QEMU_INVOCATION_REJECTED"):
        launch._audit(finder, "", "subprocess.Popen", ("/bin/sh", ["/bin/sh"], None, {}))


BASE_MOUNTS = b"1 0 1:2 / / ro,nosuid,nodev - tmpfs tmpfs ro\n2 1 0:3 / /proc ro,nosuid,nodev,noexec - proc proc ro\n"


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"bogus line", "MOUNT_INFO_REJECTED"),
        (BASE_MOUNTS + b"3 1 0:3 / /usr rw - tmpfs tmpfs rw\n", "UNAPPROVED_SUBMOUNT"),
        (BASE_MOUNTS.replace(b"- proc", b"shared:10 - proc"), "MOUNT_PROPAGATION_REJECTED"),
        (BASE_MOUNTS.replace(b"ro,nosuid,nodev", b"rw,nosuid,nodev"), "WRITABLE_SUBMOUNT"),
        (BASE_MOUNTS.replace(b"nodev,noexec", b"nodev"), "PROC_VIEW_REQUIRED"),
    ],
)
def test_mount_view_is_closed(raw: bytes, reason: str) -> None:
    launch._mount_view(BASE_MOUNTS, "/run/libvirt/libvirt-sock-ro")
    with pytest.raises(launch.Refusal, match=reason):
        launch._mount_view(raw, "/run/libvirt/libvirt-sock-ro")


def test_kernel_observations_must_match_independent_launch_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = {
        "/proc/self/status": b"NoNewPrivs:\t1\nSeccomp:\t2\nCapEff:\t0000000000000000\nCapPrm:\t0000000000000000\n",
        "/proc/self/mountinfo": BASE_MOUNTS,
        "/proc/self/attr/apparmor/current": b"cryptohunter-observer-preflight-test (enforce)\n",
    }
    auth = {
        "mountinfo_sha256": contract.digest(BASE_MOUNTS),
        "namespaces": {"mnt": 7, "pid": 7, "net": 7},
        "libvirt_socket": "/run/libvirt/libvirt-sock-ro",
        "apparmor_label": fake["/proc/self/attr/apparmor/current"].strip().decode(),
    }
    monkeypatch.setattr(launch, "_proc", lambda path, limit=0: fake[path])
    monkeypatch.setattr(
        launch,
        "os",
        SimpleNamespace(
            stat=lambda _: SimpleNamespace(st_ino=7),
            statvfs=lambda _: SimpleNamespace(f_flag=1),
            ST_RDONLY=1,
        ),
    )
    launch._kernel_boundary(auth)
    with pytest.raises(launch.Refusal, match="NAMESPACE_ID_MISMATCH"):
        launch._kernel_boundary({**auth, "namespaces": {"mnt": 8, "pid": 7, "net": 7}})
    with pytest.raises(launch.Refusal, match="APPARMOR_CONFINEMENT_REQUIRED"):
        launch._kernel_boundary({**auth, "apparmor_label": "unconfined"})
    with pytest.raises(launch.Refusal, match="MOUNT_VIEW_MISMATCH"):
        launch._kernel_boundary({**auth, "mountinfo_sha256": "0" * 64})
    fake["/proc/self/status"] = b"NoNewPrivs:\t0\n"
    with pytest.raises(launch.Refusal, match="KERNEL_CONFINEMENT_REQUIRED"):
        launch._kernel_boundary(auth)


def test_start_rejects_profile_relabel_and_unapproved_launch(
    sealed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt = sealed
    authority = tmp_path / "launch-auth.json"
    put(authority, json.dumps({"profile": "stage9-v5d7-prepared-environment-v1"}).encode())
    monkeypatch.setattr(
        launch, "sys", SimpleNamespace(executable=str(root / "venv/bin/python3.12"))
    )
    monkeypatch.setattr(
        launch.os.path,
        "samefile",
        lambda left, right: left == "/proc/self/exe" and right == root / "venv/bin/python3.12",
    )
    with pytest.raises(launch.Refusal, match="LAUNCH_AUTHORIZATION_PIN_MISMATCH"):
        launch.start(root, receipt, vars(contract), authority, "0" * 64, "a" * 64)
    with pytest.raises(launch.Refusal, match="LAUNCH_AUTHORIZATION_REJECTED"):
        launch.start(
            root,
            receipt,
            vars(contract),
            authority,
            contract.digest(authority.read_bytes()),
            "a" * 64,
        )


def test_inherited_descriptors_must_be_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = ["0", "1", "2", "11"]
    null = Path("/dev/null").stat()

    def fake_fstat(fd):
        if fd == 11:
            raise OSError(errno.EBADF, "closed")
        return null if fd == 0 else SimpleNamespace(st_mode=stat.S_IFIFO)

    fake = SimpleNamespace(fstat=fake_fstat, listdir=lambda _: entries)
    monkeypatch.setattr(launch, "os", fake)
    launch._file_descriptors()
    fake.fstat = lambda _: null
    with pytest.raises(launch.Refusal, match="UNEXPECTED_INHERITED_FD"):
        launch._file_descriptors()


def test_json_and_plan_schemas_refuse_ambiguous_data(tmp_path: Path) -> None:
    with pytest.raises(contract.Refusal, match="DUPLICATE_JSON_KEY"):
        contract.loads(b'{"profile": 1, "profile": 2}')
    with pytest.raises(contract.Refusal, match="INVALID_JSON_CONSTANT"):
        contract.loads(b'{"mode": NaN}')
    with pytest.raises(contract.Refusal, match="JSON_OBJECT_REQUIRED"):
        contract.loads(b"[]")
    with pytest.raises(contract.Refusal, match="FILE_PLAN_PARENT_MISSING"):
        contract.plan_sha256(
            {
                "files": {"child/model.py": {"sha256": "0" * 64, "mode": 0o600}},
                "directories": {".": 0o700},
            }
        )
    file = tmp_path / "file"
    put(file, b"abc")
    link = tmp_path / "link"
    link.symlink_to(file)
    with pytest.raises(OSError):
        contract.secure_read(link)
    with pytest.raises(contract.Refusal, match="SECURE_READ_OBJECT_REJECTED"):
        contract.secure_read(file, limit=2)


def test_cli_verifies_only_and_requires_separate_launch_authorization(
    sealed, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, receipt = sealed
    receipt_sha = seal(root, receipt)
    monkeypatch.setattr(
        launch,
        "sys",
        SimpleNamespace(
            platform="linux", flags=SimpleNamespace(isolated=1, no_site=1), dont_write_bytecode=True
        ),
    )
    monkeypatch.setattr(launch.os, "geteuid", lambda: 1000)
    inputs = [
        "--root",
        str(root),
        "--receipt-sha256",
        receipt_sha,
        "--approval-sha256",
        receipt["approval_sha256"],
    ]
    assert launch.main(["verify", *inputs]) == 0
    assert json.loads(capsys.readouterr().out)["native_load"] == "NOT_EXECUTED"
    assert launch.main(["preflight", *inputs]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "SEPARATE_LAUNCH_AUTHORIZATION_REQUIRED"


def test_view_refuses_loader_cache_and_executable_paths_as_data(
    sealed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt = sealed
    authority = tmp_path / "launch-auth.json"
    abi_request = tmp_path / "abi-request.json"
    auth = {"aliases": {}, "data_paths": ["/usr/lib/anything.so"]}
    with pytest.raises(launch.Refusal, match="VIEW_DATA_PATH_REJECTED"):
        launch._view(auth, root, receipt, vars(contract), authority, abi_request, {})
    auth["data_paths"] = []
    with pytest.raises(launch.Refusal, match="VIEW_ABI_ALIAS_SET_REJECTED"):
        launch._view(
            auth, root, receipt, vars(contract), authority, abi_request, {"/lib": "/usr/lib"}
        )
    monkeypatch.setattr(launch.os.path, "lexists", lambda path: path == "/etc/ld.so.cache")
    with pytest.raises(launch.Refusal, match="UNQUALIFIED_LOADER_PATH"):
        launch._view(auth, root, receipt, vars(contract), authority, abi_request, {})


def test_alias_targets_are_bound_to_approved_abi_and_base_runtime() -> None:
    module = "/usr/lib/x86_64-linux-gnu/libsample.so.0"
    loader = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
    table = {
        "library_map": {
            "libsample.so.0": module + ".1",
            "libexact.so.1": "/usr/lib/x86_64-linux-gnu/libexact.so.1",
            "/lib64/ld-linux-x86-64.so.2": loader,
        }
    }
    assert launch._abi_link_targets(
        table, {"/usr/lib/python3.12/sample.py": {"resolved": "/etc/python3.12/sample.py"}}
    ) == {
        "/lib": "/usr/lib",
        "/lib64": "/usr/lib64",
        "/usr/lib/python3.12/sample.py": "/etc/python3.12/sample.py",
        "/usr/lib64/ld-linux-x86-64.so.2": loader,
        module: module + ".1",
    }
    with pytest.raises(launch.Refusal, match="ABI_ALIAS_CONFLICT"):
        launch._abi_link_targets(
            table, {module: {"resolved": "/usr/lib/x86_64-linux-gnu/foreign.so"}}
        )
