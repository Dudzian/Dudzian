"""Przygotuj wyłącznie profil Observer preflight, bez wykonania jego kodu.

Narzędzie oraz sąsiednie helpery muszą pochodzić z uprzednio przejrzanej kopii.
Oczekiwany skrót aprobaty musi pochodzić z niezależnego kanału recenzenta.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import stat
import sys
import venv
import zipfile
from pathlib import Path
from typing import Any, cast

PROPOSAL_PROFILE = "stage9-v5d10-observer-preflight-proposal-v1"
ROOT_DISTRIBUTIONS = {"cryptography", "defusedxml"}
DISTRIBUTIONS = {
    "cffi": "2.0.0",
    "cryptography": "50.0.2",
    "defusedxml": "0.7.1",
    "pycparser": "3.0",
}
NATIVE_WHEELS = {
    "cffi": "_cffi_backend.cpython-312-x86_64-linux-gnu.so",
    "cryptography": "cryptography/hazmat/bindings/_rust.abi3.so",
}
BINDING_FILES = {"libvirt.py", "libvirtmod.cpython-312-x86_64-linux-gnu.so"}
HOST_TOOL = "/usr/bin/qemu-img"


class Refusal(ValueError):
    """Niekompletne lub odrzucone wejście."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def tool(filename: str) -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name(filename)))


def absolute(value: Any) -> Path:
    require(type(value) is str and bool(value), "ABSOLUTE_PATH_REQUIRED")
    result = Path(value)
    require(result.anchor == "/" and value == os.path.abspath(value), "ABSOLUTE_PATH_REQUIRED")
    require(
        not any(p.is_symlink() for p in (result, *result.parents)),
        "SYMLINK_PATH_REFUSED",
    )
    return result


def propose(repository: Path, wheelhouse: Path, origins: dict[str, Any]) -> dict[str, Any]:
    """Recompute RECORD, metadata closure and the complete minimal wheel manifest."""
    deps = tool("stage9_v5d4_dependencies.py")
    raw = deps["read_regular"](absolute(str(repository)) / "pyproject.toml", 8 * 1024 * 1024)
    roots = []
    for requirement in deps["closure_roots"](raw):
        name, extras, _, marker = deps["requirement"](requirement)
        if name in ROOT_DISTRIBUTIONS:
            require(not extras and not marker, "OBSERVER_ROOT_REQUIREMENT_CHANGED")
            roots.append(requirement)
    require(
        len(roots) == len(ROOT_DISTRIBUTIONS)
        and {deps["requirement"](value)[0] for value in roots} == ROOT_DISTRIBUTIONS,
        "OBSERVER_ROOT_REQUIREMENT_CHANGED",
    )
    wheelhouse = absolute(str(wheelhouse))
    require(wheelhouse.is_dir() and type(origins) is dict, "OBSERVER_WHEELHOUSE_REQUIRED")
    paths = sorted(wheelhouse.iterdir())
    require(
        len(paths) == len(DISTRIBUTIONS)
        and all(path.suffix == ".whl" for path in paths)
        and {path.name for path in paths} == set(origins),
        "OBSERVER_WHEELHOUSE_INVENTORY",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
        package = deps["inspect_wheel"](path, origins[path.name])
        name = package["name"]
        require(name in DISTRIBUTIONS and name not in candidates, "OBSERVER_WHEEL_DISTRIBUTION")
        require(package["version"] == DISTRIBUTIONS[name], "OBSERVER_WHEEL_VERSION")
        native = {NATIVE_WHEELS[name]} if name in NATIVE_WHEELS else set()
        require(
            set(package["native_members"]) == native
            and all(package["materialized_members"][member]["path"] == member for member in native),
            "OBSERVER_WHEEL_NATIVE_INVENTORY",
        )
        require(
            not package["omitted_members"] and not package["header_omissions"],
            "OBSERVER_WHEEL_OMISSION_REFUSED",
        )
        owners = {name, f"{name}-{package['version']}.dist-info", *native}
        require(
            all(
                Path(record["path"]).parts[0] in owners
                and Path(record["path"]).suffix not in {".zip", ".whl", ".egg"}
                for record in package["materialized_members"].values()
            ),
            "OBSERVER_WHEEL_FOREIGN_MEMBER",
        )
        candidates[name] = package
    packages, edges = deps["resolve_dependency_closure"](sorted(roots), candidates)
    deps["validate_materialization"](packages)
    return {
        "profile": PROPOSAL_PROFILE,
        "status": "PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL",
        "authorization": "NOT_AUTHORIZED",
        "target": deps["TARGET"],
        "pyproject_sha256": deps["digest"](raw),
        "root_requirements": sorted(roots),
        "packages": packages,
        "edges": edges,
        "wheel_accounting": {
            "record": "EXACT_UPSTREAM_RECORD_AND_MATERIALIZATION",
            "sharing": "FORBIDDEN",
            "omissions": "FORBIDDEN",
        },
    }


def native_inventory(proposal: dict[str, Any]) -> dict[str, str]:
    return {
        f"{package['filename']}:{member}": digest
        for package in proposal["packages"]
        for member, digest in package["native_members"].items()
    }


def build_host() -> dict[str, str]:
    require(
        sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
        "ISOLATED_BUILD_REQUIRED",
    )
    require(sys.platform == "linux", "UBUNTU_LINUX_HOST_REQUIRED")
    geteuid = getattr(os, "geteuid", None)
    require(callable(geteuid) and geteuid() != 0, "UNPRIVILEGED_ACCOUNT_REQUIRED")
    environment = tool("stage9_v5d4_environment.py")
    host = cast(dict[str, str], environment["host_contract"]())
    require(
        {key: value for key, value in host.items() if key != "interpreter_sha256"}
        == {
            "python_full_version": "3.12.3",
            "implementation": "CPython",
            "machine": "x86_64",
            "system": "Linux",
            "soabi": "cpython-312-x86_64-linux-gnu",
            "distribution": "ubuntu",
            "distribution_version": "24.04",
            "glibc": "glibc 2.39",
        },
        "OBSERVER_HOST_PROFILE_MISMATCH",
    )
    return host


def validate_output(output: Path) -> None:
    output = absolute(str(output))
    require(
        not any(
            output.is_relative_to(root)
            for root in map(Path, ("/opt", "/etc", "/usr", "/boot", "/dev", "/proc", "/sys"))
        ),
        "SYSTEM_OUTPUT_REFUSED",
    )
    require(not output.exists(), "OUTPUT_ALREADY_EXISTS")
    uid = getattr(os, "geteuid")()
    for parent in output.parents:
        info = parent.lstat()
        require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid in {uid, 0}
            and (not info.st_mode & 0o022 or bool(info.st_mode & stat.S_ISVTX)),
            "OUTPUT_ANCESTOR_REFUSED",
        )
    require(output.parent.stat().st_uid == uid, "OUTPUT_PARENT_NOT_OWNED")


def capture_sources(repository: Path, approval: dict[str, Any]) -> tuple[dict[str, bytes], str]:
    closure_tools = tool("stage9_v5d4_closure.py")
    dependencies = tool("stage9_v5d4_dependencies.py")
    environment = tool("stage9_v5d4_environment.py")
    closure = closure_tools["inventory"](repository)
    require(
        closure["static_graph_closed"]
        and set(closure["preflight_external_static_roots"]) == ROOT_DISTRIBUTIONS
        and set(closure["preflight_dynamic_targets"]) == {"libvirt", "zstandard"},
        "OBSERVER_CODE_CLOSURE_MISMATCH",
    )
    sources = {row["path"]: row["sha256"] for row in closure["sources"]}
    require(sources == approval["code"], "OBSERVER_CODE_INVENTORY_MISMATCH")
    contents = {
        name: environment["read"](repository / environment["relative"](name)) for name in sources
    }
    require(
        {name: environment["sha"](raw) for name, raw in contents.items()} == sources,
        "OBSERVER_SOURCE_CHANGED",
    )
    return contents, cast(str, environment["sha"](dependencies["canonical"](closure)))


def capture_bindings(approval: dict[str, Any]) -> dict[str, bytes]:
    env = tool("stage9_v5d4_environment.py")
    bindings = approval["bindings"]
    require(type(bindings) is dict and set(bindings) == BINDING_FILES, "OBSERVER_BINDING_INVENTORY")
    contents = {}
    for name, pin in bindings.items():
        require(
            type(pin) is dict and set(pin) == {"path", "sha256"},
            "OBSERVER_BINDING_PIN_SCHEMA",
        )
        path = absolute(pin["path"])
        require(path.name == name, "OBSERVER_BINDING_PIN_PATH")
        raw = env["read"](path)
        require(env["sha"](raw) == pin["sha256"], "OBSERVER_BINDING_PIN_MISMATCH")
        contents[name] = raw
    return contents


def base_and_tools(approval: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    env = tool("stage9_v5d4_environment.py")
    base = approval["base_runtime"]
    roots = [absolute(name) for name in base["roots"]]
    require(
        tool("stage9_v5d4_closure.py")["base_runtime_inventory"](roots) == base,
        "OBSERVER_BASE_RUNTIME_MISMATCH",
    )
    for entry in sys.path:
        name = absolute(entry)
        require(
            not name.exists() or any(name.is_relative_to(root) for root in roots),
            "OBSERVER_BASE_IMPORT_PATH_UNCOVERED",
        )
    host_tools = approval["host_tools"]
    require(
        type(host_tools) is dict and set(host_tools) == {HOST_TOOL},
        "OBSERVER_HOST_TOOL_INVENTORY",
    )
    require(
        env["sha"](env["read"](Path(HOST_TOOL))) == host_tools[HOST_TOOL],
        "OBSERVER_HOST_TOOL_CHANGED",
    )
    return base, host_tools


def pinned_document(path: Path, expected: str, reason: str) -> dict[str, Any]:
    env = tool("stage9_v5d4_environment.py")
    raw = env["read"](path)
    require(env["sha"](raw) == expected, reason)
    value = env["parse_document"](raw)
    require(type(value) is dict, "OBSERVER_JSON_OBJECT_REQUIRED")
    return cast(dict[str, Any], value)


def check_abi_paths(
    request: dict[str, Any],
    output: Path,
    base: dict[str, Any],
    host_tools: dict[str, str],
) -> set[str]:
    packages = output / "packages"
    imports = {
        "libvirt": "libvirt.py",
        "libvirtmod": "libvirtmod.cpython-312-x86_64-linux-gnu.so",
        "cryptography": "cryptography/__init__.py",
        "defusedxml": "defusedxml/__init__.py",
        "cffi": "cffi/__init__.py",
        "_cffi_backend": NATIVE_WHEELS["cffi"],
        "pycparser": "pycparser/__init__.py",
    }
    paths = {name: str(packages / member) for name, member in imports.items()}
    require(
        set(request)
        == {
            "profile",
            "interpreter",
            "pyvenv_cfg",
            "import_roots",
            "libvirt_module",
            "libvirt_extension",
            "required_imports",
            "additional_extensions",
            "patchlevel_header",
            "sysconfigdata",
            "library_map",
        }
        and request["profile"] == "stage9-v5d4-static-abi-v1"
        and request["interpreter"] == str(output / "venv/bin/python3.12")
        and request["pyvenv_cfg"] == str(output / "venv/pyvenv.cfg")
        and request["import_roots"] == [str(packages)]
        and request["libvirt_module"] == paths["libvirt"]
        and request["libvirt_extension"] == paths["libvirtmod"]
        and request["required_imports"] == paths,
        "OBSERVER_ABI_PATH_PLAN_MISMATCH",
    )
    additional = (
        {str(packages / name) for name in NATIVE_WHEELS.values()}
        | set(base["native"])
        | set(host_tools)
    )
    declared = request["additional_extensions"]
    require(
        type(declared) is list and len(declared) == len(additional) and set(declared) == additional,
        "OBSERVER_ABI_NATIVE_INVENTORY",
    )
    return additional | {paths["libvirtmod"], request["interpreter"]}


def external_pins(
    output: Path, expectations: dict[str, Any], base: dict[str, Any], host_tools: dict[str, str]
) -> dict[str, str]:
    contract = tool("stage9_v5d10_contract.py")
    names = {**base["files"], **host_tools}
    for name, digest in expectations["sha256"].items():
        path = absolute(name)
        if not path.is_relative_to(output):
            require(names.get(name, digest) == digest, "OBSERVER_EXTERNAL_PIN_CONFLICT")
            names[name] = digest
    for name, digest in names.items():
        require(
            contract["digest"](contract["secure_read"](absolute(name))) == digest,
            "OBSERVER_EXTERNAL_PIN_MISMATCH",
        )
    return names


def check_dynamic_review(review: dict[str, Any], closure_sha: str, proposal_sha: str) -> None:
    require(
        review
        == {
            "profile": "stage9-v5d10-observer-preflight-dynamic-review-v1",
            "entrypoint": "scripts.stage9_observer_kvm_preflight_v5d1",
            "source_closure_sha256": closure_sha,
            "proposal_sha256": proposal_sha,
            "python_dynamic_targets": ["libvirt"],
            "excluded_python_dynamic_targets": ["zstandard"],
            "native_execution": "REQUIRES_SEPARATELY_AUTHORIZED_KERNEL_BOUNDARY",
        },
        "OBSERVER_DYNAMIC_REVIEW_SCOPE_MISMATCH",
    )


def build(
    inputs: dict[str, Any],
    approval: dict[str, Any],
    output: Path,
    *,
    approval_raw: bytes | None = None,
) -> dict[str, Any]:
    host = build_host()
    contract = tool("stage9_v5d10_contract.py")
    env = tool("stage9_v5d4_environment.py")
    contract["validate_inputs"](inputs)
    contract["validate_approval"](approval)
    canonical_approval = contract["canonical"](approval)
    if approval_raw is not None:
        require(
            contract["canonical"](contract["loads"](approval_raw)) == canonical_approval,
            "OBSERVER_APPROVAL_BYTES_MISMATCH",
        )
    approval_digest = contract["digest"](
        canonical_approval if approval_raw is None else approval_raw
    )
    require(host == approval["host"], "OBSERVER_HOST_MISMATCH")
    validate_output(output)
    repository = absolute(inputs["repository"])
    wheelhouse = absolute(inputs["wheelhouse"])
    origins = env["document"](absolute(inputs["origins"]))
    proposal = env["document"](absolute(inputs["proposal"]))
    rebuilt = propose(repository, wheelhouse, origins)
    require(
        contract["canonical"](proposal) == contract["canonical"](rebuilt),
        "OBSERVER_PROPOSAL_CONTENT_MISMATCH",
    )
    proposal_sha = contract["digest"](contract["canonical"](rebuilt))
    require(proposal_sha == approval["proposal_sha256"], "OBSERVER_PROPOSAL_DIGEST_MISMATCH")
    require(
        native_inventory(rebuilt) == approval["wheel_native"], "OBSERVER_NATIVE_APPROVAL_MISMATCH"
    )
    source_bytes, closure_sha = capture_sources(repository, approval)
    review = pinned_document(
        absolute(inputs["dynamic_review"]),
        approval["dynamic_review_sha256"],
        "OBSERVER_DYNAMIC_REVIEW_DIGEST_MISMATCH",
    )
    check_dynamic_review(review, closure_sha, proposal_sha)
    binding_bytes = capture_bindings(approval)
    base, host_tools = base_and_tools(approval)
    request = pinned_document(
        absolute(inputs["abi_request"]),
        approval["abi_request_sha256"],
        "OBSERVER_ABI_REQUEST_DIGEST_MISMATCH",
    )
    expectations = pinned_document(
        absolute(inputs["abi_expectations"]),
        approval["abi_expectations_sha256"],
        "OBSERVER_ABI_EXPECTATIONS_DIGEST_MISMATCH",
    )
    tool("stage9_v5d4_abi.py")["_pins"](expectations)
    expected_native = check_abi_paths(request, output, base, host_tools)
    external = external_pins(output, expectations, base, host_tools)
    snapshots, planned_packages, shared = env["prepare_wheel_set"](
        rebuilt["packages"], wheelhouse, None
    )
    require(not shared["members"], "OBSERVER_SHARED_WHEEL_MEMBER")
    parent_dirs = {p.as_posix() for name in planned_packages for p in Path(name).parents if p.parts}
    packages = {**source_bytes, **binding_bytes}
    require(len(packages) == len(source_bytes) + len(binding_bytes), "OBSERVER_PACKAGE_COLLISION")
    for name, raw in packages.items():
        env["reserve_path"](planned_packages, parent_dirs, name, env["sha"](raw))
    launcher_files = {
        "contract.py": "stage9_v5d10_contract.py",
        "launch.py": "stage9_v5d10_launch.py",
        "runtime-audit.py": "stage9_v5d4_closure.py",
    }
    launch_bytes = {
        target: contract["secure_read"](Path(__file__).resolve().with_name(source))
        for target, source in launcher_files.items()
    }

    # The output is single-use. An interrupted build has no sealed receipt.
    validate_output(output)
    previous_umask = os.umask(0o077)
    try:
        output.mkdir(mode=0o700)
        runtime = output / "venv"
        venv.EnvBuilder(system_site_packages=False, with_pip=False, symlinks=False).create(runtime)
        alias = runtime / "lib64"
        require(
            alias.is_symlink() and alias.readlink() == Path("lib"), "OBSERVER_VENV_ALIAS_MISMATCH"
        )
        alias.unlink()
        package_root = output / "packages"
        package_root.mkdir(mode=0o700)
        env["materialize_wheel_set"](snapshots, package_root, shared)
        for name, raw in sorted(packages.items()):
            env["write_new"](package_root, name, raw)
        env["verify_materialized"](package_root, planned_packages)
        for name, raw in launch_bytes.items():
            env["write_new"](output, name, raw)
        plan = contract["inventory"](output)
        require(
            contract["plan_sha256"](plan) == approval["file_plan_sha256"],
            "OBSERVER_FILE_PLAN_MISMATCH",
        )
        for name, digest in expectations["sha256"].items():
            path = Path(name)
            if path.is_relative_to(output):
                local = path.relative_to(output).as_posix()
                require(
                    plan["files"].get(local, {}).get("sha256") == digest,
                    "OBSERVER_STAGE_PIN_MISMATCH",
                )
        abi = tool("stage9_v5d10_abi.py")["diagnose"](request, expectations)
        require(
            abi["status"] == "STATIC_ABI_MATCH" and expected_native <= set(abi["elf_measurements"]),
            "OBSERVER_ABI_NOT_COMPLETE",
        )
        require(contract["inventory"](output) == plan, "OBSERVER_TREE_CHANGED_AFTER_ABI")
        external_pins(output, expectations, base, host_tools)
        receipt = {
            "profile": contract["RECEIPT"],
            "root": str(output),
            "entrypoint": contract["ENTRYPOINT"],
            "approval_sha256": approval_digest,
            "proposal_sha256": proposal_sha,
            "abi_request_sha256": approval["abi_request_sha256"],
            "abi_expectations_sha256": approval["abi_expectations_sha256"],
            "dynamic_review_sha256": approval["dynamic_review_sha256"],
            "file_plan_sha256": approval["file_plan_sha256"],
            **plan,
            "host": host,
            "base_runtime": base,
            "external_files": external,
            "authorization": "NOT_AUTHORIZED",
            "native_load": "NOT_EXECUTED",
        }
        contract["validate_receipt"](receipt)
        pending = ".ENVIRONMENT.json.pending"
        env["write_new"](output, pending, contract["canonical"](receipt) + b"\n")
        # An interrupted publication leaves no final path or a two-link file.
        # The receipt reader refuses the latter before interpreting any data.
        os.link(output / pending, output / contract["RECEIPT_NAME"], follow_symlinks=False)
        (output / pending).unlink()
    finally:
        os.umask(previous_umask)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    candidate = commands.add_parser("propose")
    candidate.add_argument("--repository", type=Path, required=True)
    candidate.add_argument("--wheelhouse", type=Path, required=True)
    candidate.add_argument("--origins", type=Path, required=True)
    staging = commands.add_parser("build")
    staging.add_argument("--inputs", type=Path, required=True)
    staging.add_argument("--approval", type=Path, required=True)
    staging.add_argument("--approval-sha256", required=True)
    staging.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(
            sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
            "ISOLATED_BUILD_REQUIRED",
        )
        contract = tool("stage9_v5d10_contract.py")
        if args.mode == "propose":
            result = propose(
                args.repository,
                args.wheelhouse,
                contract["loads"](contract["secure_read"](absolute(str(args.origins)))),
            )
        else:
            approval_raw = contract["secure_read"](absolute(str(args.approval)))
            require(
                contract["digest"](approval_raw) == args.approval_sha256,
                "INDEPENDENT_APPROVAL_DIGEST_MISMATCH",
            )
            result = build(
                contract["loads"](contract["secure_read"](absolute(str(args.inputs)))),
                contract["loads"](approval_raw),
                args.output,
                approval_raw=approval_raw,
            )
        sys.stdout.buffer.write(contract["canonical"](result) + b"\n")
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        reason = str(exc) if type(exc).__name__ == "Refusal" else type(exc).__name__
        print(json.dumps({"status": "REFUSED", "reason": reason}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
