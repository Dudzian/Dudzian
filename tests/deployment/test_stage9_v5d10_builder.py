"""Syntetyczne wejścia V5d10. Nie uruchamiają żadnego z przygotowanych plików."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from scripts import stage9_v5d4_closure as closure
from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d10_builder as builder
from scripts import stage9_v5d10_contract as contract
from tests.deployment.test_stage9_v5d4_dependencies import wheel

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="Syntetyczny profil rootless POSIX."),
]


def put(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def document(path: Path, value: dict[str, Any]) -> str:
    return contract.digest(put(path, contract.canonical(value) + b"\n").read_bytes())


def make_wheels(
    wheelhouse: Path,
    *,
    foreign_member: bool = False,
    extra_native: bool = False,
    alternate: tuple[str, str] | None = None,
) -> dict[str, Any]:
    wheelhouse.mkdir()
    native = b"\x7fELF synthetic non-executable fixture"
    definitions = [
        ("cffi", "2.0.0", ("pycparser>=3",), {builder.NATIVE_WHEELS["cffi"]: native}),
        (
            "cryptography",
            "50.0.2",
            ("cffi>=2",),
            {
                builder.NATIVE_WHEELS["cryptography"]: native,
                **({"foreign.py": b"foreign code"} if foreign_member else {}),
                **({"cryptography/unexpected.so": native} if extra_native else {}),
            },
        ),
        ("defusedxml", "0.7.1", (), {}),
        ("pycparser", "3.0", (), {}),
    ]
    origins = {}
    for name, version, requirements, entries in definitions:
        if alternate is not None and name == alternate[0]:
            version = alternate[1]
        path, origin = wheel(wheelhouse, name, version, requires=requirements, files=entries)
        origins[path.name] = origin
    return origins


@pytest.fixture
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The synthetic cloud runner can expose root-owned mounts through an
    # unprivileged namespace as uid 65534. Model the physical Linux identities
    # at the two system anchors, while retaining fixture-directory metadata.
    if Path("/").lstat().st_uid == 65534:
        original_lstat = Path.lstat

        def physical_anchor(path):
            observed = original_lstat(path)
            if str(path) in {"/", "/tmp"} and observed.st_uid == 65534:
                fields = list(observed)
                fields[stat.ST_UID] = 0
                return os.stat_result(fields)
            return observed

        monkeypatch.setattr(Path, "lstat", physical_anchor)

    repository = tmp_path / "repo"
    sources = {
        "scripts/__init__.py": b"",
        "scripts/stage9_observer_kvm_preflight_v5d1.py": (
            b"import cryptography\nimport defusedxml\nimport importlib\n"
            b"def preflight():\n    importlib.import_module('libvirt')\n"
            b"def verifier():\n    importlib.import_module('zstandard')\n"
        ),
    }
    for name, data in sources.items():
        put(repository / name, data)
    put(
        repository / "pyproject.toml",
        b'[project]\nrequires-python=">=3.11"\n'
        b'dependencies=["cryptography>=50,<51", "PySide6>=6"]\n'
        b'[project.optional-dependencies]\nobserver-kvm-preflight=["defusedxml==0.7.1"]\n',
    )
    wheelhouse = tmp_path / "wheels"
    origins = make_wheels(wheelhouse)
    proposal = builder.propose(repository, wheelhouse, origins)
    proposal_sha = contract.digest(contract.canonical(proposal))
    output = tmp_path / "pristine"
    host = {
        "python_full_version": "3.12.3",
        "implementation": "CPython",
        "machine": "x86_64",
        "system": "Linux",
        "soabi": "cpython-312-x86_64-linux-gnu",
        "interpreter_sha256": contract.digest(b"synthetic interpreter"),
        "distribution": "ubuntu",
        "distribution_version": "24.04",
        "glibc": "glibc 2.39",
    }
    monkeypatch.setattr(builder, "build_host", lambda: host)
    base_root = tmp_path / "stdlib"
    extension = put(base_root / "math.cpython-312-x86_64-linux-gnu.so", b"\x7fELF synthetic stdlib")
    base = closure.base_runtime_inventory([base_root])
    host_tool = builder.HOST_TOOL
    host_tool_bytes = b"\x7fELF synthetic qemu"
    host_tools = {host_tool: contract.digest(host_tool_bytes)}
    monkeypatch.setattr(builder, "base_and_tools", lambda _: (base, host_tools))
    bindings = {}
    bindings_bytes = {
        "libvirt.py": b"import libvirtmod\nraise RuntimeError('must never be imported')\n",
        "libvirtmod.cpython-312-x86_64-linux-gnu.so": b"\x7fELF synthetic binding",
    }
    for name, data in bindings_bytes.items():
        bindings[name] = {
            "path": f"/usr/lib/python3/dist-packages/{name}",
            "sha256": contract.digest(data),
        }
    monkeypatch.setattr(builder, "capture_bindings", lambda _: bindings_bytes)
    venv_files = {
        "venv/bin/python3.12": (b"synthetic interpreter", 0o755),
        "venv/pyvenv.cfg": (b"include-system-site-packages = false\n", 0o600),
    }

    class FakeVenv:
        def __init__(self, **kwargs):
            assert kwargs == {"system_site_packages": False, "with_pip": False, "symlinks": False}

        def create(self, target):
            for name, (data, mode) in venv_files.items():
                path = put(target.parent / name, data)
                path.chmod(mode)
            (target / "lib").mkdir()
            (target / "lib64").symlink_to("lib", target_is_directory=True)

    monkeypatch.setattr(builder.venv, "EnvBuilder", FakeVenv)
    package_root = output / "packages"
    abi_imports = {
        "libvirt": "libvirt.py",
        "libvirtmod": "libvirtmod.cpython-312-x86_64-linux-gnu.so",
        "cryptography": "cryptography/__init__.py",
        "defusedxml": "defusedxml/__init__.py",
        "cffi": "cffi/__init__.py",
        "_cffi_backend": builder.NATIVE_WHEELS["cffi"],
        "pycparser": "pycparser/__init__.py",
    }
    request = {
        "profile": "stage9-v5d4-static-abi-v1",
        "required_imports": {name: str(package_root / rel) for name, rel in abi_imports.items()},
        "import_roots": [str(package_root)],
        "libvirt_module": str(package_root / abi_imports["libvirt"]),
        "libvirt_extension": str(package_root / abi_imports["libvirtmod"]),
        "interpreter": str(output / "venv/bin/python3.12"),
        "pyvenv_cfg": str(output / "venv/pyvenv.cfg"),
        "patchlevel_header": str(put(tmp_path / "patchlevel.h", b"header fixture")),
        "sysconfigdata": str(put(tmp_path / "_sysconfigdata.py", b"sysconfig fixture")),
        "additional_extensions": [
            str(package_root / name) for name in builder.NATIVE_WHEELS.values()
        ]
        + [str(extension), host_tool],
        "library_map": {"sys-fixture": str(extension)},
    }
    expectations = {
        "profile": "stage9-v5d4-static-expectations-v1",
        "review_reference": "synthetic fixture",
        "sha256": {
            request["patchlevel_header"]: contract.digest(b"header fixture"),
            request["sysconfigdata"]: contract.digest(b"sysconfig fixture"),
            request["interpreter"]: host["interpreter_sha256"],
            request["pyvenv_cfg"]: contract.digest(venv_files["venv/pyvenv.cfg"][0]),
            **base["files"],
            **host_tools,
        },
    }
    checked = []
    real_tool = builder.tool

    def read_for_test(path):
        if str(path) == host_tool:
            return host_tool_bytes
        return contract.secure_read(path)

    def synthetic_diagnose(actual_request, actual_expectations):
        checked.append((actual_request, actual_expectations))
        assert actual_request == request and actual_expectations == expectations
        assert Path(request["interpreter"]).read_bytes() == b"synthetic interpreter"
        return {
            "status": "STATIC_ABI_MATCH",
            "elf_measurements": {
                name: contract.digest(read_for_test(Path(name)))
                for name in [
                    request["interpreter"],
                    request["libvirt_extension"],
                    *request["additional_extensions"],
                ]
            },
        }

    def trusted_tool(name):
        if name == "stage9_v5d10_abi.py":
            return {"diagnose": synthetic_diagnose}
        original = real_tool(name)
        if name == "stage9_v5d10_contract.py":
            return {**original, "secure_read": read_for_test}
        return original

    monkeypatch.setattr(builder, "tool", trusted_tool)
    files = {
        name: {"sha256": contract.digest(data), "mode": mode}
        for name, (data, mode) in venv_files.items()
    }
    for package in proposal["packages"]:
        for row in package["materialized_members"].values():
            files[f"packages/{row['path']}"] = {"sha256": row["sha256"], "mode": 0o600}
    for name, data in (sources | bindings_bytes).items():
        files[f"packages/{name}"] = {"sha256": contract.digest(data), "mode": 0o600}
    scripts_root = Path(builder.__file__).parent
    for local, filename in {
        "contract.py": "stage9_v5d10_contract.py",
        "launch.py": "stage9_v5d10_launch.py",
        "runtime-audit.py": "stage9_v5d4_closure.py",
    }.items():
        files[local] = {
            "sha256": contract.digest((scripts_root / filename).read_bytes()),
            "mode": 0o600,
        }
    folders = {p.as_posix() for name in files for p in Path(name).parents if p.parts}
    plan = {"files": files, "directories": {name: 0o700 for name in folders | {".", "venv/lib"}}}
    inventory = closure.inventory(repository)
    review = {
        "profile": "stage9-v5d10-observer-preflight-dynamic-review-v1",
        "entrypoint": contract.ENTRYPOINT,
        "source_closure_sha256": contract.digest(deps.canonical(inventory)),
        "proposal_sha256": proposal_sha,
        "python_dynamic_targets": ["libvirt"],
        "excluded_python_dynamic_targets": ["zstandard"],
        "native_execution": "REQUIRES_SEPARATELY_AUTHORIZED_KERNEL_BOUNDARY",
    }
    inputs = {
        "profile": contract.INPUTS,
        "repository": str(repository),
        "wheelhouse": str(wheelhouse),
        "origins": str(tmp_path / "origins.json"),
        "proposal": str(tmp_path / "proposal.json"),
        "dynamic_review": str(tmp_path / "dynamic-review.json"),
        "abi_request": str(tmp_path / "abi-request.json"),
        "abi_expectations": str(tmp_path / "abi-expectations.json"),
    }
    document(Path(inputs["origins"]), origins)
    document(Path(inputs["proposal"]), proposal)
    approval = {
        "profile": contract.APPROVAL,
        "host": host,
        "proposal_sha256": proposal_sha,
        "wheel_native": builder.native_inventory(proposal),
        "code": {row["path"]: row["sha256"] for row in inventory["sources"]},
        "dynamic_review_sha256": document(Path(inputs["dynamic_review"]), review),
        "bindings": bindings,
        "base_runtime": base,
        "host_tools": host_tools,
        "abi_request_sha256": document(Path(inputs["abi_request"]), request),
        "abi_expectations_sha256": document(Path(inputs["abi_expectations"]), expectations),
        "file_plan_sha256": contract.plan_sha256(plan),
        "wheel_omissions": [],
        "wheel_shared_members": [],
        "wheel_header_omissions": [],
        "wheel_artifact_omissions": [],
    }
    return inputs, approval, output, plan, checked


def test_synthetic_builder_emits_exact_unexecuted_receipt(staging) -> None:
    inputs, approval, output, plan, checked = staging
    approval_raw = (json.dumps(approval, indent=2) + "\n").encode()
    receipt = builder.build(inputs, approval, output, approval_raw=approval_raw)
    assert contract.inventory(output, exclude_receipt=True) == plan
    assert (output / "ENVIRONMENT.json").read_bytes() == contract.canonical(receipt) + b"\n"
    assert stat.S_IMODE((output / "ENVIRONMENT.json").stat().st_mode) == 0o600
    assert receipt["authorization"] == "NOT_AUTHORIZED"
    assert receipt["native_load"] == "NOT_EXECUTED"
    assert receipt["profile"] == contract.RECEIPT
    assert receipt["approval_sha256"] == contract.digest(approval_raw)
    assert len(checked) == 1
    assert not (output / "venv/lib64").exists()


def test_receipt_binding_refuses_raw_approval_for_different_content(staging) -> None:
    inputs, approval, output, _, checked = staging
    raw = contract.canonical({**approval, "file_plan_sha256": "0" * 64})
    with pytest.raises(ValueError, match="OBSERVER_APPROVAL_BYTES_MISMATCH"):
        builder.build(inputs, approval, output, approval_raw=raw)
    assert not checked
    assert not output.exists()


def test_receipt_reader_refuses_interrupted_publication(staging, monkeypatch) -> None:
    inputs, approval, output, _, checked = staging
    real_unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        if path.name == ".ENVIRONMENT.json.pending":
            raise OSError("stopped before removing private name")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(OSError, match="stopped"):
        builder.build(inputs, approval, output)
    assert len(checked) == 1
    with pytest.raises(ValueError, match="SECURE_READ_OBJECT_REJECTED"):
        contract.secure_read(output / "ENVIRONMENT.json")


@pytest.mark.parametrize(
    "field", ["proposal_sha256", "dynamic_review_sha256", "abi_expectations_sha256"]
)
def test_digest_mismatch_fails_before_writing(staging, field) -> None:
    inputs, approval, output, _, checked = staging
    approval[field] = "0" * 64
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked


@pytest.mark.parametrize("kind", ["proposal", "approval", "inputs"])
def test_foreign_profile_refusal_even_with_matching_hash(staging, kind) -> None:
    inputs, approval, output, _, checked = staging
    if kind == "proposal":
        path = Path(inputs["proposal"])
        proposal = json.loads(path.read_bytes())
        proposal["profile"] = deps.V5D7_PROPOSAL
        document(path, proposal)
        approval["proposal_sha256"] = contract.digest(contract.canonical(proposal))
    elif kind == "approval":
        approval["profile"] = "stage9-v5d4-approval-v1"
    else:
        inputs["profile"] = "stage9-v5d10-offline-deb-inventory-inputs-v1"
    with pytest.raises(ValueError):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked


def test_omitting_elf_refused_with_recalculated_request_digest(staging) -> None:
    inputs, approval, output, _, checked = staging
    path = Path(inputs["abi_request"])
    request = json.loads(path.read_bytes())
    request["additional_extensions"].pop()
    approval["abi_request_sha256"] = document(path, request)
    with pytest.raises(ValueError, match="OBSERVER_ABI_NATIVE_INVENTORY"):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked


def test_external_pin_change_refused_before_writing(staging) -> None:
    inputs, approval, output, _, checked = staging
    request = json.loads(Path(inputs["abi_request"]).read_bytes())
    Path(request["patchlevel_header"]).write_bytes(b"changed header")
    with pytest.raises(ValueError, match="OBSERVER_EXTERNAL_PIN_MISMATCH"):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked


def test_abi_candidate_without_independent_review_refused_before_writing(staging) -> None:
    inputs, approval, output, _, checked = staging
    candidate = Path(inputs["abi_expectations"])
    expectations = json.loads(candidate.read_bytes())
    expectations["review_reference"] = None
    approval["abi_expectations_sha256"] = document(candidate, expectations)
    with pytest.raises(ValueError, match="EXTERNAL_EXPECTATIONS_REQUIRED"):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked


def test_file_plan_mismatch_has_no_receipt(staging) -> None:
    inputs, approval, output, _, checked = staging
    approval["file_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="OBSERVER_FILE_PLAN_MISMATCH"):
        builder.build(inputs, approval, output)
    assert output.is_dir()
    assert not (output / "ENVIRONMENT.json").exists()
    assert not checked
    with pytest.raises(ValueError, match="OUTPUT_ALREADY_EXISTS"):
        builder.build(inputs, approval, output)


def test_foreign_import_member_refused_in_exact_four_wheels(staging, tmp_path) -> None:
    inputs, _, _, _, _ = staging
    wheelhouse = tmp_path / "foreign"
    with pytest.raises(ValueError, match="OBSERVER_WHEEL_FOREIGN_MEMBER"):
        builder.propose(
            Path(inputs["repository"]), wheelhouse, make_wheels(wheelhouse, foreign_member=True)
        )


def test_foreign_elf_member_refused_in_an_allowed_package(staging, tmp_path) -> None:
    inputs, _, _, _, _ = staging
    wheelhouse = tmp_path / "foreign-native"
    with pytest.raises(ValueError, match="OBSERVER_WHEEL_NATIVE_INVENTORY"):
        builder.propose(
            Path(inputs["repository"]), wheelhouse, make_wheels(wheelhouse, extra_native=True)
        )


@pytest.mark.parametrize(
    "alternate",
    [
        ("cffi", "2.0.1"),
        ("cryptography", "50.0.3"),
        ("defusedxml", "0.7.2"),
        ("pycparser", "3.1"),
    ],
)
def test_other_versions_cannot_supply_profile_a(staging, tmp_path, alternate) -> None:
    inputs, _, _, _, _ = staging
    wheelhouse = tmp_path / "other-version"
    with pytest.raises(ValueError, match="OBSERVER_WHEEL_VERSION"):
        builder.propose(
            Path(inputs["repository"]), wheelhouse, make_wheels(wheelhouse, alternate=alternate)
        )


def test_extra_wheel_refused_before_staging(staging) -> None:
    inputs, approval, output, _, checked = staging
    wheel(Path(inputs["wheelhouse"]), "zstandard")
    with pytest.raises(ValueError, match="OBSERVER_WHEELHOUSE_INVENTORY"):
        builder.build(inputs, approval, output)
    assert not output.exists()
    assert not checked
