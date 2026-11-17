"""Syntetyczny staging; brak native load, libvirt, VM i podpisów operatora."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import zipfile

import pytest

from scripts import stage9_v5d4_environment as environment_module
from scripts import stage9_v5d4_launch as launch_module
from scripts import stage9_v5d4_abi as abi_module
from scripts import stage9_v5d4_dependencies as dependency_module

env = vars(environment_module)
launch = vars(launch_module)


@pytest.mark.parametrize(
    "reader", [environment_module.read, abi_module.read_bytes, dependency_module.read_regular]
)
def test_readers_allow_read_access_time_update(tmp_path: Path, reader) -> None:
    path = tmp_path / "fresh-input"
    path.write_bytes(b"approved bytes")
    now = time.time()
    os.utime(path, (now - 7200, now - 3600))
    assert reader(path) == b"approved bytes"


@pytest.mark.parametrize(
    "reader", [environment_module.read, abi_module.read_bytes, dependency_module.read_regular]
)
def test_readers_detect_same_size_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"approved")
    original = os.fdopen

    class MutatingReader:
        def __init__(self, descriptor, mode):
            self.stream = original(descriptor, mode)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, limit):
            data = self.stream.read(limit)
            path.write_bytes(b"tampered")
            return data

    monkeypatch.setattr(os, "fdopen", MutatingReader)
    with pytest.raises(ValueError, match="CHANGED"):
        reader(path)


def wheel(files: dict[str, bytes]) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for path, value in files.items():
            archive.writestr(path, value)
    return data.getvalue()


@pytest.mark.parametrize("name", ["../x", "/x", "x/../../y", "x\\y", "x//y"])
def test_wheel_path_escape(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        env["extract_wheel"](wheel({name: b"bad"}), tmp_path)
    assert not (tmp_path.parent / "x").exists()


@pytest.mark.parametrize(
    "name",
    [
        "evil.pth",
        "sitecustomize.py",
        "usercustomize.py",
        "x.pyc",
        "x/__pycache__/x.pyc",
        "x.data/scripts/run",
    ],
)
def test_startup_and_script_refusal(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        env["extract_wheel"](wheel({name: b"raise RuntimeError('must not run')"}), tmp_path)


def test_wheel_data_relocation_and_collision(tmp_path: Path) -> None:
    env["extract_wheel"](wheel({"p.data/platlib/package/__init__.py": b"safe = True\n"}), tmp_path)
    assert (tmp_path / "package/__init__.py").read_bytes() == b"safe = True\n"
    with pytest.raises(FileExistsError):
        env["extract_wheel"](wheel({"package/__init__.py": b"replace"}), tmp_path)


def test_runtime_unexpected_nested_receipt_is_not_ignored(tmp_path: Path) -> None:
    (tmp_path / "packages").mkdir()
    (tmp_path / "ENVIRONMENT.json").write_text("{}")
    (tmp_path / "packages/ENVIRONMENT.json").write_text("unapproved")
    with pytest.raises(ValueError, match="RUNTIME_CONTENT_MISMATCH"):
        launch["verify_tree"](tmp_path, {})


def test_runtime_unexpected_fifo_refuses_before_open(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "module.py")
    with pytest.raises(ValueError, match="NONREGULAR_RUNTIME"):
        launch["verify_tree"](tmp_path, {})


def test_runtime_hash_missing_symlink_and_extra_refuse(tmp_path: Path) -> None:
    file = tmp_path / "test.py"
    file.write_bytes(b"pass\n")
    expected = {"test.py": hashlib.sha256(file.read_bytes()).hexdigest()}
    launch["verify_tree"](tmp_path, expected)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError):
        launch["verify_tree"](tmp_path, expected)
    file.unlink()
    file.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        launch["verify_tree"](tmp_path, expected)


def test_native_load_requires_explicit_gate() -> None:
    assert launch["main"]([]) == 2
    assert launch["main"](["--authorize-native-load"]) == 2


def test_native_gate_checks_receipt_and_never_invokes_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = launch["main"].__globals__
    calls: list[str] = []
    finder_list: list[object] = []
    monkeypatch.setitem(namespace, "__file__", str(tmp_path / "launch.py"))
    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1, no_site=1),
            dont_write_bytecode=True,
            executable=sys.executable,
            meta_path=finder_list,
        ),
    )
    monkeypatch.setitem(namespace, "importlib", SimpleNamespace(import_module=calls.append))
    monkeypatch.setitem(
        namespace,
        "runpy",
        SimpleNamespace(
            run_path=lambda path: {"base_runtime_inventory": lambda roots: {"roots": []}}
        ),
    )
    receipt = {
        "root": str(tmp_path),
        "files": {},
        "host": env["host_contract"](),
        "external_files": {},
        "base_runtime": {"roots": []},
    }
    raw = json.dumps(receipt).encode()
    (tmp_path / "ENVIRONMENT.json").write_bytes(raw)
    assert launch["main"](["--authorize-native-load", "--receipt-sha256", "0" * 64]) == 2
    assert calls == [] and finder_list == []
    assert launch["main"](["--authorize-native-load", "--receipt-sha256", env["sha"](raw)]) == 0
    assert calls == ["libvirt"]
    assert len(finder_list) == 1


def test_fixed_finder_keeps_sys_path_and_resolves_package_initializer(tmp_path: Path) -> None:
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample/__init__.py").write_bytes(b"raise RuntimeError('do not execute')")
    before = sys.path.copy()
    finder = launch["FixedRootFinder"](tmp_path)
    spec = finder.find_spec("sample")
    assert spec.origin == str(tmp_path / "sample/__init__.py")
    assert finder.find_spec("sample.child", path=[str(tmp_path / "sample")]) is None
    assert sys.path == before


def test_build_refuses_without_isolated_startup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="REQUIRE_ISOLATED_NO_SITE"):
        env["build"]({}, {}, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.fixture
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Zastąp wyłącznie granicę hosta/ELF; pliki i staging pozostają rzeczywiste."""
    namespace = env["build"].__globals__
    host = {
        "python_full_version": "3.12.3",
        "implementation": "CPython",
        "machine": "x86_64",
        "system": "Linux",
        "soabi": "cpython-312-x86_64-linux-gnu",
        "interpreter_sha256": "f" * 64,
        "distribution": "ubuntu",
        "distribution_version": "24.04",
        "glibc": "glibc 2.39",
    }
    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(flags=SimpleNamespace(isolated=1, no_site=1), platform="linux", path=[]),
    )
    monkeypatch.setitem(namespace, "host_contract", lambda: host)
    output = tmp_path / "new"
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "controller.py").write_bytes(b"raise RuntimeError('must not execute')\n")
    binding = tmp_path / "libvirt.py"
    binding.write_bytes(b"raise RuntimeError('must not import')\n")
    extension = tmp_path / "libvirtmod.so"
    extension.write_bytes(b"SYNTHETIC ELF DATA")
    inputs = {
        "repository": str(repository),
        "wheelhouse": str(tmp_path),
        "origins": str(tmp_path / "origins.json"),
        "proposal": str(tmp_path / "proposal.json"),
        "abi_request": str(tmp_path / "abi.json"),
        "abi_expectations": str(tmp_path / "expected.json"),
    }
    proposal = {"packages": []}
    closure = {
        "static_graph_closed": True,
        "sources": [
            {
                "path": "controller.py",
                "sha256": env["sha"]((repository / "controller.py").read_bytes()),
            }
        ],
    }
    request = {
        "interpreter": str(output / "venv/bin/python3.12"),
        "import_roots": [str(output / "packages")],
        "required_imports": {
            name: "synthetic" for name in ("libvirt", "libvirtmod", "cryptography", "defusedxml")
        },
        "libvirt_module": str(output / "packages/libvirt.py"),
        "libvirt_extension": str(output / "packages/libvirtmod.cpython-312-x86_64-linux-gnu.so"),
        "additional_extensions": ["/usr/bin/qemu-img"],
    }
    for key, value in (
        ("origins", {}),
        ("proposal", proposal),
        ("abi_request", request),
        ("abi_expectations", {"sha256": {}}),
    ):
        Path(inputs[key]).write_text(json.dumps(value))
    approval = {
        "base_runtime": {"roots": [], "files": {}, "native": []},
        "host_tools": {"/usr/bin/qemu-img": env["sha"](b"SYNTHETIC QEMU NOT EXECUTED")},
        "host": host,
        "proposal_sha256": "a" * 64,
        "wheel_native": {},
        "dynamic_review_sha256": env["sha"](json.dumps(closure).encode()),
        "code": {"controller.py": closure["sources"][0]["sha256"]},
        "abi_request_sha256": env["sha"](Path(inputs["abi_request"]).read_bytes()),
        "abi_expectations_sha256": env["sha"](Path(inputs["abi_expectations"]).read_bytes()),
        "bindings": {
            "libvirt.py": {"path": str(binding), "sha256": env["sha"](binding.read_bytes())},
            "libvirtmod.cpython-312-x86_64-linux-gnu.so": {
                "path": str(extension),
                "sha256": env["sha"](extension.read_bytes()),
            },
        },
    }

    def fake_helper(name):
        if name.endswith("dependencies.py"):
            return {
                "verify": lambda *args: {"status": "LOCAL_CONSISTENCY_VERIFIED"},
                "canonical": lambda value: json.dumps(value).encode(),
            }
        if name.endswith("closure.py"):
            return {
                "inventory": lambda *args: closure,
                "base_runtime_inventory": lambda *args: {"roots": [], "files": {}, "native": []},
            }
        return {
            "diagnose": lambda *args: {
                "status": "STATIC_ABI_MATCH",
                "elf_measurements": {
                    request["libvirt_extension"]: approval["bindings"][
                        "libvirtmod.cpython-312-x86_64-linux-gnu.so"
                    ]["sha256"],
                    "/usr/bin/qemu-img": approval["host_tools"]["/usr/bin/qemu-img"],
                },
            }
        }

    class SyntheticVenv:
        def __init__(self, **kwargs):
            assert kwargs == {"system_site_packages": False, "with_pip": False, "symlinks": False}

        def create(self, path):
            (path / "bin").mkdir(parents=True)
            (path / "bin/python3.12").write_bytes(b"SYNTHETIC INTERPRETER NOT EXECUTABLE")

    monkeypatch.setitem(namespace, "helper", fake_helper)
    original_read = namespace["read"]
    monkeypatch.setitem(
        namespace,
        "read",
        lambda path: b"SYNTHETIC QEMU NOT EXECUTED"
        if str(path) == "/usr/bin/qemu-img"
        else original_read(path),
    )
    monkeypatch.setitem(namespace, "venv", SimpleNamespace(EnvBuilder=SyntheticVenv))
    return inputs, approval, output, request


def test_staging_copies_without_loading_code(staging) -> None:
    inputs, approval, output, _ = staging
    result = env["build"](inputs, approval, output)
    assert result["authorization"] == "NOT_AUTHORIZED"
    assert result["native_load"] == "NOT_EXECUTED"
    assert (output / "packages/controller.py").is_file()
    assert (output / "ENVIRONMENT.json").is_file()
    launch["verify_tree"](output, result["files"])


@pytest.mark.parametrize(
    "field",
    ["code", "dynamic_review_sha256", "abi_request_sha256", "abi_expectations_sha256", "host"],
)
def test_staging_missing_evidence_refuses_before_output(staging, field) -> None:
    inputs, approval, output, _ = staging
    approval[field] = {} if field in {"host", "code"} else "0" * 64
    with pytest.raises(ValueError):
        env["build"](inputs, approval, output)
    assert not output.exists()


def test_abi_failure_never_seals_environment(staging, monkeypatch) -> None:
    inputs, approval, output, _ = staging
    namespace = env["build"].__globals__
    original = namespace["helper"]
    monkeypatch.setitem(
        namespace,
        "helper",
        lambda name: {"diagnose": lambda *args: {"status": "REFUSED"}}
        if name.endswith("abi.py")
        else original(name),
    )
    with pytest.raises(ValueError, match="ABI_REFUSED"):
        env["build"](inputs, approval, output)
    assert not (output / "ENVIRONMENT.json").exists()


def test_base_runtime_and_import_path_evidence_is_required(staging, monkeypatch) -> None:
    inputs, approval, output, _ = staging
    approval["base_runtime"]["files"] = {"/unknown/base.pyc": "0" * 64}
    with pytest.raises(ValueError, match="BASE_RUNTIME_APPROVAL_MISMATCH"):
        env["build"](inputs, approval, output)
    approval["base_runtime"]["files"] = {}
    monkeypatch.setitem(
        env["build"].__globals__,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1, no_site=1),
            platform="linux",
            path=[inputs["repository"]],
        ),
    )
    with pytest.raises(ValueError, match="BASE_IMPORT_PATH_UNCOVERED"):
        env["build"](inputs, approval, output)
    assert not output.exists()


def test_qemu_tool_requires_exact_independent_hash(staging) -> None:
    inputs, approval, output, _ = staging
    approval["host_tools"]["/usr/bin/qemu-img"] = "0" * 64
    with pytest.raises(ValueError, match="HOST_TOOL_SHA256_MISMATCH"):
        env["build"](inputs, approval, output)
    assert not output.exists()


def test_missing_native_abi_evidence_leaves_unsealed_staging(staging) -> None:
    inputs, approval, output, request = staging
    request["additional_extensions"] = []
    Path(inputs["abi_request"]).write_text(json.dumps(request))
    approval["abi_request_sha256"] = env["sha"](Path(inputs["abi_request"]).read_bytes())
    with pytest.raises(ValueError, match="NATIVE_ABI_INVENTORY_INCOMPLETE"):
        env["build"](inputs, approval, output)
    assert not (output / "ENVIRONMENT.json").exists()
