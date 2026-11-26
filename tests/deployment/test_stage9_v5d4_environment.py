"""Syntetyczny staging; brak native load, libvirt, VM i podpisów operatora."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
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
@pytest.mark.skipif(
    os.name != "posix", reason="Test deskryptora POSIX; Windows blokuje zapis natywnym uchwytem."
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
            # ZipInfo(path) normalizuje separator Windows przed zapisem. Fixture
            # musi zachować w lokalnym i centralnym nagłówku badane surowe bajty.
            info = zipfile.ZipInfo()
            info.filename = path
            info.orig_filename = path
            archive.writestr(info, value)
    return data.getvalue()


@pytest.mark.parametrize("name", ["../x", "/x", "x/../../y", "x\\y", "x//y"])
def test_wheel_path_escape(tmp_path: Path, name: str) -> None:
    data = wheel({name: b"bad"})
    assert name.encode("utf-8") in data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.infolist()[0].orig_filename == name
    with pytest.raises(ValueError):
        env["extract_wheel"](data, tmp_path)
    assert not (tmp_path.parent / "x").exists()


@pytest.mark.parametrize(
    "name",
    [
        "evil.pth",
        "sitecustomize.py",
        "usercustomize.py",
        "sitecustomize/__init__.py",
        "usercustomize/__init__.py",
        "sitecustomize.abi3.so",
        "usercustomize.cpython-312-x86_64-linux-gnu.so",
        "package/sitecustomize/module.py",
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
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env["extract_wheel"](wheel({"package/__init__.py": b"replace"}), tmp_path)


def test_extract_validates_all_members_before_writing(tmp_path: Path) -> None:
    data = wheel({"safe.py": b"raise RuntimeError('must not execute')", "unsafe.pth": b"import x"})
    with pytest.raises(ValueError, match="STARTUP_OR_BYTECODE_REJECTED"):
        env["extract_wheel"](data, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "entries",
    [
        {"a.data/platlib/pkg.py": b"same", "pkg.py": b"same"},
        {"pkg": b"file", "pkg/module.py": b"source"},
        {"pkg/module.py": b"source", "pkg": b"file"},
    ],
)
def test_extract_refuses_internal_relocation_and_ancestor_collisions(
    tmp_path: Path, entries
) -> None:
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env["extract_wheel"](wheel(entries), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_extract_refuses_symlink_destination_before_writing(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "packages"
    destination.mkdir()
    (destination / "package").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="MATERIALIZATION_SYMLINK"):
        env["extract_wheel"](wheel({"package/module.py": b"source"}), destination)
    assert list(outside.iterdir()) == []


def test_materialized_snapshot_checks_inventory_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_bytes(b"source")
    expected = {"module.py": env["sha"](b"source")}
    env["verify_materialized"](tmp_path, expected)
    path.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="MATERIALIZED_INVENTORY_MISMATCH"):
        env["verify_materialized"](tmp_path, expected)
    path.unlink()
    with pytest.raises(ValueError, match="MATERIALIZED_INVENTORY_MISMATCH"):
        env["verify_materialized"](tmp_path, expected)


def test_empty_unapproved_cache_directory_refuses_in_snapshot(tmp_path: Path) -> None:
    (tmp_path / "__pycache__").mkdir()
    with pytest.raises(ValueError, match="STARTUP_OR_BYTECODE_REJECTED"):
        env["verify_materialized"](tmp_path, {})


def test_runtime_unexpected_nested_receipt_is_not_ignored(tmp_path: Path) -> None:
    (tmp_path / "packages").mkdir()
    (tmp_path / "ENVIRONMENT.json").write_text("{}")
    (tmp_path / "packages/ENVIRONMENT.json").write_text("unapproved")
    with pytest.raises(ValueError, match="RUNTIME_CONTENT_MISMATCH"):
        launch["verify_tree"](tmp_path, {})


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO wymaga natywnego API POSIX.")
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


@pytest.mark.parametrize("host_platform", ["win32", "darwin"])
def test_unsupported_build_host_refuses_before_linux_apis(tmp_path, monkeypatch, host_platform):
    namespace = env["build"].__globals__

    def forbidden(*args, **kwargs):
        pytest.fail("Nieobsługiwany host nie może odczytywać API ani materiału Linux.")

    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(flags=SimpleNamespace(isolated=1, no_site=1), platform=host_platform),
    )
    monkeypatch.setitem(namespace, "os", SimpleNamespace(geteuid=forbidden, confstr=forbidden))
    monkeypatch.setitem(namespace, "platform", SimpleNamespace(freedesktop_os_release=forbidden))
    monkeypatch.setitem(namespace, "helper", forbidden)
    with pytest.raises(ValueError, match="UBUNTU_LINUX_HOST_REQUIRED"):
        env["host_contract"]()
    with pytest.raises(ValueError, match="UBUNTU_LINUX_HOST_REQUIRED"):
        env["build"]({}, {}, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("host_platform", ["win32", "darwin"])
def test_unsupported_launcher_host_refuses_before_evidence_or_imports(
    monkeypatch, capsys, host_platform
):
    namespace = launch["main"].__globals__

    def forbidden(*args, **kwargs):
        pytest.fail("Nieobsługiwany host nie może czytać os-release ani ładować runtime.")

    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1, no_site=1),
            dont_write_bytecode=True,
            platform=host_platform,
        ),
    )
    monkeypatch.setitem(namespace, "os", SimpleNamespace(confstr=forbidden))
    monkeypatch.setitem(namespace, "platform", SimpleNamespace(freedesktop_os_release=forbidden))
    monkeypatch.setitem(namespace, "runpy", SimpleNamespace(run_path=forbidden))
    monkeypatch.setitem(namespace, "importlib", SimpleNamespace(import_module=forbidden))
    assert launch["main"](["--authorize-native-load", "--receipt-sha256", "0" * 64]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["reason"] == "UBUNTU_LINUX_HOST_REQUIRED"
    assert "Traceback" not in captured.err


def test_build_missing_uid_capability_is_controlled_refusal(tmp_path, monkeypatch):
    namespace = env["build"].__globals__
    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(flags=SimpleNamespace(isolated=1, no_site=1), platform="linux"),
    )
    monkeypatch.setitem(namespace, "os", SimpleNamespace())
    with pytest.raises(ValueError, match="LINUX_HOST_CAPABILITY_MISSING"):
        env["build"]({}, {}, tmp_path / "new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("missing", ["os_release", "confstr"])
def test_host_contract_missing_linux_capability_refuses_before_calls(monkeypatch, missing):
    namespace = env["host_contract"].__globals__

    def forbidden(*args, **kwargs):
        pytest.fail("Niekompletny zestaw API nie może być użyty do pomiaru hosta.")

    monkeypatch.setitem(namespace, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setitem(
        namespace,
        "platform",
        SimpleNamespace(
            **({} if missing == "os_release" else {"freedesktop_os_release": forbidden})
        ),
    )
    monkeypatch.setitem(
        namespace, "os", SimpleNamespace(**({} if missing == "confstr" else {"confstr": forbidden}))
    )
    with pytest.raises(ValueError, match="LINUX_HOST_CAPABILITY_MISSING"):
        env["host_contract"]()


@pytest.mark.parametrize("missing", ["os_release", "confstr"])
def test_launcher_missing_linux_capability_refuses_before_evidence(monkeypatch, capsys, missing):
    namespace = launch["main"].__globals__

    def forbidden(*args, **kwargs):
        pytest.fail("Brak możliwości pomiaru hosta musi zatrzymać launcher przed odczytami.")

    monkeypatch.setitem(
        namespace,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1, no_site=1), dont_write_bytecode=True, platform="linux"
        ),
    )
    monkeypatch.setitem(
        namespace,
        "platform",
        SimpleNamespace(
            **({} if missing == "os_release" else {"freedesktop_os_release": forbidden})
        ),
    )
    monkeypatch.setitem(
        namespace, "os", SimpleNamespace(**({} if missing == "confstr" else {"confstr": forbidden}))
    )
    assert launch["main"](["--authorize-native-load", "--receipt-sha256", "0" * 64]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["reason"] == "LINUX_HOST_CAPABILITY_MISSING"
    assert "Traceback" not in captured.err


@pytest.mark.skipif(
    sys.platform == "linux", reason="Odmowa wymaga rzeczywistego Windows lub macOS."
)
@pytest.mark.parametrize("tool", ["builder", "launcher"])
def test_native_unsupported_host_cli_refuses_without_traceback(tmp_path, tool):
    if tool == "builder":
        inputs = tmp_path / "inputs.json"
        inputs.write_bytes(b"{}")
        approval = tmp_path / "approval.json"
        approval.write_bytes(b"{}")
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            env["__file__"],
            "--inputs",
            str(inputs),
            "--approval",
            str(approval),
            "--approval-sha256",
            env["sha"](b"{}"),
            "--output",
            str(tmp_path / "new"),
        ]
    else:
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            launch["__file__"],
            "--authorize-native-load",
            "--receipt-sha256",
            "0" * 64,
        ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "REFUSED"
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "new").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Natywna bramka uruchomienia wymaga Linux.")
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
            platform=sys.platform,
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
        "profile": "stage9-v5d4-prepared-environment-v1",
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
    for profile in (None, "stage9-v5d10-observer-preflight-receipt-projection-v1"):
        candidate = {**receipt, "profile": profile}
        if profile is None:
            del candidate["profile"]
        candidate_raw = json.dumps(candidate).encode()
        (tmp_path / "ENVIRONMENT.json").write_bytes(candidate_raw)
        assert (
            launch["main"](
                ["--authorize-native-load", "--receipt-sha256", env["sha"](candidate_raw)]
            )
            == 2
        )
        assert calls == [] and finder_list == []
    (tmp_path / "ENVIRONMENT.json").write_bytes(raw)
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
    if sys.platform != "linux":
        pytest.skip("Model rootless staging bada kontrakt Linux; inne hosty odmawiają.")
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


@pytest.mark.parametrize("field", ["abi_request", "abi_expectations"])
def test_build_parses_exact_pinned_abi_bytes_once(staging, monkeypatch, field) -> None:
    inputs, approval, output, _ = staging
    namespace = env["build"].__globals__
    original_read = namespace["read"]
    target = Path(inputs[field])
    pinned = original_read(target)
    reads: list[Path] = []

    def mutate_after_pinned_read(path):
        raw = original_read(path)
        if path == target:
            reads.append(path)
            target.write_bytes(b'{"changed_after_pinned_read":"must not be parsed"}')
        return raw

    monkeypatch.setitem(namespace, "read", mutate_after_pinned_read)
    result = env["build"](inputs, approval, output)
    assert reads == [target]
    assert target.read_bytes() != pinned
    assert result["authorization"] == "NOT_AUTHORIZED"
    launch["verify_tree"](output, result["files"])


def test_cli_parses_exact_pinned_approval_bytes_once(tmp_path, monkeypatch, capsys) -> None:
    namespace = env["main"].__globals__
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_bytes(b'{"source":"TEST ONLY"}')
    approval_path = tmp_path / "approval.json"
    pinned = b'{"expectation":"TEST ONLY pinned approval"}'
    approval_path.write_bytes(pinned)
    output = tmp_path / "new"
    original_read = namespace["read"]
    reads: list[Path] = []
    observed: list[dict] = []

    def mutate_after_pinned_read(path):
        raw = original_read(path)
        if path == approval_path:
            reads.append(path)
            approval_path.write_bytes(b'{"expectation":"replaced approval must not be parsed"}')
        return raw

    def capture_build(inputs, approval, destination):
        assert inputs == {"source": "TEST ONLY"}
        assert destination == output
        observed.append(approval)
        return {"files": {}}

    monkeypatch.setitem(namespace, "read", mutate_after_pinned_read)
    monkeypatch.setitem(namespace, "build", capture_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            env["__file__"],
            "--inputs",
            str(inputs_path),
            "--approval",
            str(approval_path),
            "--approval-sha256",
            env["sha"](pinned),
            "--output",
            str(output),
        ],
    )
    assert env["main"]() == 0
    assert reads == [approval_path]
    assert observed == [{"expectation": "TEST ONLY pinned approval"}]
    assert json.loads(capsys.readouterr().out)["authorization"] == "NOT_AUTHORIZED"


@pytest.fixture
def reviewed_wheel_staging(staging, monkeypatch: pytest.MonkeyPatch):
    """TEST ONLY: rzeczywisty verifier i pliki; zastąpione wyłącznie host/ELF.

    Oczekiwania pochodzą z syntetycznej fixture. Nie są aprobatą dystrybucji
    upstream ani dowodem niezależnego recenzenta produkcyjnego.
    """
    from tests.deployment.test_stage9_v5d4_dependencies import (
        cache_material,
        pin_test_cache,
        wheel as dependency_wheel,
    )

    inputs, approval, output, _ = staging
    repository = Path(inputs["repository"])
    project = repository / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["pyside6-essentials==6.10.3"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    directory = output.parent / "wheelhouse"
    directory.mkdir()
    inputs["wheelhouse"] = str(directory)
    namespace = env["build"].__globals__
    original_helper = namespace["helper"]
    closure = original_helper("stage9_v5d4_closure.py")["inventory"](repository)
    approval["dynamic_review_sha256"] = dependency_module.digest(
        dependency_module.canonical(closure)
    )
    monkeypatch.setitem(
        namespace,
        "helper",
        lambda name: vars(dependency_module)
        if name.endswith("dependencies.py")
        else original_helper(name),
    )

    def prepare(extra=None):
        files = {**cache_material(), **(extra or {})}
        path, origin = dependency_wheel(
            directory,
            "pyside6_essentials",
            "6.10.3",
            tag="cp39-abi3-manylinux_2_34_x86_64",
            files=files,
        )
        pin_test_cache(monkeypatch, path, files)
        origins = {path.name: origin}
        Path(inputs["origins"]).write_text(json.dumps(origins), encoding="utf-8")
        proposal = dependency_module.propose(project, directory, origins)
        Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
        approval["proposal_sha256"] = dependency_module.digest(
            dependency_module.canonical(proposal)
        )
        approval["wheel_omissions"] = dependency_module.omission_expectation(proposal)
        return path, proposal

    path, proposal = prepare()
    return inputs, approval, output, path, proposal, prepare


def test_real_verification_to_target_omits_only_reviewed_cache(reviewed_wheel_staging) -> None:
    inputs, approval, output, path, proposal, _ = reviewed_wheel_staging
    original = path.read_bytes()
    result = env["build"](inputs, approval, output)
    package = proposal["packages"][0]
    for member in package["omitted_members"]:
        assert not (output / "packages" / member["member"]).exists()
        assert (output / "packages" / member["source"]).is_file()
    assert not list((output / "packages").rglob("*.pyc"))
    assert not list((output / "packages").rglob("__pycache__"))
    record = "pyside6_essentials-6.10.3.dist-info/RECORD"
    with zipfile.ZipFile(path) as archive:
        assert (output / "packages" / record).read_bytes() == archive.read(record)
        assert all(member["member"] in archive.namelist() for member in package["omitted_members"])
    assert path.read_bytes() == original
    assert result["wheel_omissions"] == approval["wheel_omissions"]["members"]
    assert result["wheel_archives"][0]["sha256"] == dependency_module.digest(original)
    assert result["wheel_record_semantics"] == (
        "UNCHANGED_UPSTREAM_ARCHIVE_RECORD_NOT_INSTALLED_INVENTORY"
    )
    assert result["authorization"] == "NOT_AUTHORIZED"
    assert result["native_load"] == "NOT_EXECUTED"
    launch["verify_tree"](output, result["files"])


@pytest.mark.parametrize("change", ["missing", "incomplete", "extra", "hash", "contract"])
def test_actual_build_requires_complete_explicit_omissions(reviewed_wheel_staging, change) -> None:
    inputs, approval, output, _, _, _ = reviewed_wheel_staging
    if change == "missing":
        approval.pop("wheel_omissions")
    elif change == "incomplete":
        approval["wheel_omissions"]["members"].pop()
    elif change == "extra":
        approval["wheel_omissions"]["members"].append(
            copy.deepcopy(approval["wheel_omissions"]["members"][0])
        )
    elif change == "hash":
        approval["wheel_omissions"]["members"][0]["sha256"] = "0" * 64
    else:
        approval["wheel_omissions"].pop("installation_contract_exception")
    with pytest.raises(ValueError, match="UNAPPROVED_WHEEL_OMISSIONS"):
        env["build"](inputs, approval, output)
    assert not output.exists()


@pytest.mark.parametrize("collision", ["controller.py", "libvirt.py", "libvirt.py/child.py"])
def test_actual_build_checks_external_collisions_before_output(reviewed_wheel_staging, collision):
    inputs, approval, output, _, _, prepare = reviewed_wheel_staging
    prepare({collision: b"raise RuntimeError('must not execute')\n"})
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env["build"](inputs, approval, output)
    assert not output.exists()


def test_actual_build_rejects_colliding_wheels_before_output(reviewed_wheel_staging) -> None:
    from tests.deployment.test_stage9_v5d4_dependencies import wheel as dependency_wheel

    inputs, approval, output, _, proposal, _ = reviewed_wheel_staging
    directory = Path(inputs["wheelhouse"])
    path, origin = dependency_wheel(
        directory, "neighbor", files={"PySide6/support/__init__.py": b"collision"}
    )
    origins = json.loads(Path(inputs["origins"]).read_text(encoding="utf-8"))
    origins[path.name] = origin
    Path(inputs["origins"]).write_text(json.dumps(origins), encoding="utf-8")
    project = Path(inputs["repository"]) / "pyproject.toml"
    project.write_text(
        project.read_text(encoding="utf-8").replace(
            'dependencies=["pyside6-essentials==6.10.3"]',
            'dependencies=["pyside6-essentials==6.10.3", "neighbor"]',
        ),
        encoding="utf-8",
    )
    # Syntetyczna próba przedłożenia własnego wykazu nadal odbija się od
    # rzeczywistego rebuild propozycji i kontroli kolizji wszystkich wheels.
    proposal["packages"].append(dependency_module.inspect_wheel(path, origin))
    Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
    approval["proposal_sha256"] = dependency_module.digest(dependency_module.canonical(proposal))
    with pytest.raises(ValueError, match="WHEEL_MATERIALIZATION_COLLISION"):
        env["build"](inputs, approval, output)
    assert not output.exists()


def test_extractor_revalidates_proposal_omissions_before_writes(reviewed_wheel_staging) -> None:
    _, _, output, path, proposal, _ = reviewed_wheel_staging
    package = copy.deepcopy(proposal["packages"][0])
    package["omitted_members"].pop()
    with pytest.raises(ValueError, match="WHEEL_PLAN_CHANGED_AFTER_VERIFICATION"):
        env["extract_wheel"](path.read_bytes(), output, package)
    assert not output.exists()


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


@pytest.mark.parametrize("change", ["replace", "missing", "extra"])
def test_abi_analysis_cannot_reseal_changed_materialization(staging, monkeypatch, change) -> None:
    inputs, approval, output, _ = staging
    namespace = env["build"].__globals__
    original_helper = namespace["helper"]

    def changing_helper(name):
        tools = original_helper(name)
        if not name.endswith("abi.py"):
            return tools

        def diagnose(*args):
            result = tools["diagnose"](*args)
            controller = output / "packages/controller.py"
            if change == "replace":
                controller.write_bytes(b"x" * controller.stat().st_size)
            elif change == "missing":
                controller.unlink()
            else:
                (output / "packages/unapproved.py").write_bytes(
                    b"raise RuntimeError('must not run')"
                )
            return result

        return {"diagnose": diagnose}

    monkeypatch.setitem(namespace, "helper", changing_helper)
    with pytest.raises(ValueError, match="MATERIALIZED_INVENTORY_CHANGED_BEFORE_RECEIPT"):
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
