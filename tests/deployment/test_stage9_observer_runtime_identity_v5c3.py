"""Synthetic byte-parser mechanics, never exact-target execution evidence."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from typing import Any

import pytest

from deployment import stage9_observer_runtime_identity_v5c3 as runtime
from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_pyc_reconstruction_v5c2 import (
    PycReconstructionPolicyV5c2,
    ReconstructionLibraryV5c2,
    ReconstructionPackageV5c2,
)
from deployment.stage9_observer_pyc_v5c1 import SourceBindingV5c1

pytestmark = [pytest.mark.security, pytest.mark.unit]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _elf(
    *,
    needed: tuple[str, ...] = (),
    interpreter: str | None = "/lib/ld.so",
    rpath: str | None = None,
    runpath: str | None = None,
    extra_tag: int | None = None,
) -> bytes:
    """Produce an inert ELF header model; these bytes are never executable."""
    raw = bytearray(1536)
    strings = bytearray(b"\x00")
    tags: list[tuple[int, int]] = []
    for name in needed:
        tags.append((1, len(strings)))
        strings.extend(name.encode("ascii") + b"\x00")
    for tag, value in ((15, rpath), (29, runpath)):
        if value is not None:
            tags.append((tag, len(strings)))
            strings.extend(value.encode("ascii") + b"\x00")
    tags.extend(((5, 0x400400), (10, len(strings))))
    if extra_tag is not None:
        tags.append((extra_tag, 0))
    tags.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *tag) for tag in tags)
    note = struct.pack("<III", 4, 20, 3) + b"GNU\x00" + b"\x12" * 20
    segments = [
        (1, 4, 0, 0x400000, 0, len(raw), len(raw), 4096),
        (2, 4, 0x200, 0x400200, 0, len(dynamic), len(dynamic), 8),
        (4, 4, 0x300, 0x400300, 0, len(note), len(note), 4),
    ]
    if interpreter is not None:
        interp = interpreter.encode("ascii") + b"\x00"
        raw[0x380 : 0x380 + len(interp)] = interp
        segments.append((3, 4, 0x380, 0x400380, 0, len(interp), len(interp), 1))
    ident = b"\x7fELF\x02\x01\x01" + b"\x00" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH", raw, 0, ident, 2, 62, 1, 0, 64, 0, 0, 64, 56, len(segments), 0, 0, 0
    )
    for i, segment in enumerate(segments):
        struct.pack_into("<IIQQQQQQ", raw, 64 + i * 56, *segment)
    raw[0x200 : 0x200 + len(dynamic)] = dynamic
    raw[0x300 : 0x300 + len(note)] = note
    raw[0x400 : 0x400 + len(strings)] = strings
    return bytes(raw)


def test_static_elf_measurements_preserve_order_and_full_hash() -> None:
    raw = _elf(needed=("libm.so.6", "libc.so.6"))
    measured = runtime.inspect_elf_v5c3(raw)
    assert measured.sha256 == _sha(raw)
    assert measured.build_id == "12" * 20
    assert measured.needed == ("libm.so.6", "libc.so.6")
    assert measured.interpreter_path == "/lib/ld.so"
    assert measured.rpath == measured.runpath == ()
    changed = raw[:-1] + b"x"
    assert runtime.inspect_elf_v5c3(changed).sha256 != measured.sha256


@pytest.mark.parametrize("field", ["rpath", "runpath"])
def test_search_metadata_is_measured_not_executed(field: str) -> None:
    measured = runtime.inspect_elf_v5c3(_elf(**{field: "/untrusted/lib"}))
    assert getattr(measured, field) == ("/untrusted/lib",)


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0, b"NOPE"),
        (4, b"\x01"),
        (5, b"\x02"),
        (18, struct.pack("<H", 183)),
        (32, struct.pack("<Q", 2**63)),
        (54, struct.pack("<H", 55)),
        (56, struct.pack("<H", 257)),
        (64 + 32, struct.pack("<Q", 2**63)),
        (0x200 + 8, struct.pack("<Q", 2**63)),
        (0x300, struct.pack("<I", 2**31)),
    ],
)
def test_malformed_and_out_of_bounds_elf_is_refused(offset: int, replacement: bytes) -> None:
    raw = bytearray(_elf(needed=("libc.so.6",)))
    raw[offset : offset + len(replacement)] = replacement
    with pytest.raises(ContractRefusal):
        runtime.inspect_elf_v5c3(bytes(raw))


@pytest.mark.parametrize("raw", [b"", b"\x7fELF", b"x" * 64])
def test_truncated_headers_are_refused(raw: bytes) -> None:
    with pytest.raises(ContractRefusal):
        runtime.inspect_elf_v5c3(raw)


@pytest.mark.parametrize("tag", [0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF])
def test_additional_implicit_loader_features_refused(tag: int) -> None:
    with pytest.raises(ContractRefusal, match="ELF_EXTRA_LOADER"):
        runtime.inspect_elf_v5c3(_elf(extra_tag=tag))


def test_missing_build_id_refused() -> None:
    raw = bytearray(_elf())
    raw[0x30C:0x310] = b"BAD\x00"
    with pytest.raises(ContractRefusal, match="ELF_BUILD_ID"):
        runtime.inspect_elf_v5c3(bytes(raw))


@pytest.mark.parametrize("token", ["/tmp/evil.so", "a:b.so", "same.so"])
def test_needed_path_injection_and_duplicate_tokens_refused(token: str) -> None:
    needed = (token, token) if token == "same.so" else (token,)
    with pytest.raises(ContractRefusal):
        runtime.inspect_elf_v5c3(_elf(needed=needed))


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    """Monkeypatched synthetic identities test policy mechanics ONLY."""
    interpreter = _elf(needed=("libc.so.6",))
    library = _elf(interpreter=None)
    monkeypatch.setattr(runtime, "TARGET_INTERPRETER_SHA256", _sha(interpreter))
    monkeypatch.setattr(runtime, "TARGET_BUILD_ID", "12" * 20)
    snapshot = _sha(b"synthetic snapshot")
    package = ReconstructionPackageV5c2(
        "python3.12-minimal", "3.12.3-1ubuntu0.17", "amd64", _sha(b"deb"), snapshot
    )
    source = SourceBindingV5c1(
        "source",
        "fixture",
        "1",
        "amd64",
        snapshot,
        _sha(b"srcdeb"),
        10,
        "./module.py",
        _sha(b"record"),
        _sha(b"x=1\n"),
        4,
    )
    pins = (
        runtime.RuntimeFilePinV5c3(
            "/usr/bin/python3.12", _sha(interpreter), "12" * 20, "interpreter"
        ),
        runtime.RuntimeFilePinV5c3("/lib/ld.so", _sha(library), "12" * 20, "loader"),
        runtime.RuntimeFilePinV5c3("/lib/libc.so.6", _sha(library), "12" * 20, "library"),
        runtime.RuntimeFilePinV5c3("/stage9/tool/compile_worker.py", _sha(b"tool"), None, "tool"),
        runtime.RuntimeFilePinV5c3(
            "/usr/lib/python3.12/py_compile.py", _sha(b"compiler"), None, "stdlib"
        ),
        runtime.RuntimeFilePinV5c3("/etc/ld.so.preload", None, None, "configuration"),
        runtime.RuntimeFilePinV5c3("/etc/ld.so.cache", None, None, "configuration"),
        runtime.RuntimeFilePinV5c3("/etc/ld.so.conf", None, None, "configuration"),
        runtime.RuntimeFilePinV5c3("/etc/ssl/openssl.cnf", None, None, "configuration"),
    )
    inventory = runtime.RuntimeInventoryV5c3(
        pins,
        (("/usr/bin/python3.12", "libc.so.6", "/lib/libc.so.6"),),
        ("/etc/ld.so.preload", "/etc/ld.so.cache", "/etc/ld.so.conf"),
        ("/usr/lib/python3.12",),
        ("/etc/ssl/openssl.cnf",),
    )
    captured = tuple(
        runtime.CapturedRuntimeFileV5c3(pin.path, data)
        for pin, data in zip(
            pins,
            (interpreter, library, library, b"tool", b"compiler", None, None, None, None),
            strict=True,
        )
    )
    policy = PycReconstructionPolicyV5c2(
        _sha(interpreter),
        "12" * 20,
        package,
        tuple(
            ReconstructionLibraryV5c2(pin.path, pin.sha256 or "", pin.build_id or "", package)
            for pin in pins[1:3]
        ),
        source,
        "CPython-3.12.3",
        "pep263-utf8-default-v1",
        "/module.py",
        0,
        "checked_hash",
        ("/usr/bin/python3.12", "-I", "-S"),
        (),
        ("no-network",),
        "image",
        _sha(b"img"),
        _sha(b"expected"),
        _sha(b"pyc"),
    )
    return inventory, captured, policy


def test_inventory_matches_exact_captured_bytes_under_synthetic_target_model(model: Any) -> None:
    measured = runtime.validate_runtime_identity_v5c3(*model)
    assert len(measured) == 9
    assert measured[0].needed == ("libc.so.6",)
    assert measured[-1].sha256 is None


def test_target_identity_is_hard_pinned(model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from deployment.stage9_observer_pyc_source_hash_v5c2 import TARGET_INTERPRETER_SHA256

    monkeypatch.setattr(runtime, "TARGET_INTERPRETER_SHA256", TARGET_INTERPRETER_SHA256)
    with pytest.raises(ContractRefusal, match="WRONG_INTERPRETER"):
        runtime.validate_runtime_identity_v5c3(*model)


@pytest.mark.parametrize(
    "name",
    ["LD_PRELOAD", "LD_LIBRARY_PATH", "OPENSSL_CONF", "PYTHONPATH", "GLIBC_TUNABLES", "GCONV_PATH"],
)
def test_loading_environment_overrides_refused(model: Any, name: str) -> None:
    with pytest.raises(ContractRefusal, match="LOADER_ENVIRONMENT_REJECTED"):
        runtime.validate_runtime_identity_v5c3(*model, environment=((name, "unexpected"),))


def test_changed_missing_or_reordered_runtime_bytes_refused(model: Any) -> None:
    inventory, captured, policy = model
    for replacements in (
        (replace(captured[0], data=captured[0].data + b"x"),) + captured[1:],
        captured[:-1],
        captured[::-1],
    ):
        with pytest.raises(ContractRefusal):
            runtime.validate_runtime_identity_v5c3(inventory, replacements, policy)


def test_incomplete_resolutions_and_unknown_loader_refused(model: Any) -> None:
    inventory, captured, policy = model
    with pytest.raises(ContractRefusal, match="RUNTIME_LIBRARY_MISSING"):
        runtime.validate_runtime_identity_v5c3(
            replace(inventory, needed_resolutions=()), captured, policy
        )
    pins = (replace(inventory.files[0], build_id="34" * 20),) + inventory.files[1:]
    with pytest.raises(ContractRefusal, match="WRONG_INTERPRETER"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), captured, policy)


def test_preload_file_nonempty_refused_even_when_pinned(model: Any) -> None:
    inventory, captured, policy = model
    pins = (
        inventory.files[:5]
        + (replace(inventory.files[5], sha256=_sha(b"evil.so\n")),)
        + inventory.files[6:]
    )
    data = captured[:5] + (replace(captured[5], data=b"evil.so\n"),) + captured[6:]
    with pytest.raises(ContractRefusal, match="PRELOAD_REJECTED"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), data, policy)


def test_wrong_snapshot_and_unknown_import_path_refused(model: Any) -> None:
    inventory, captured, policy = model
    changed = replace(
        policy, interpreter_package=replace(policy.interpreter_package, snapshot_sha256="ab" * 32)
    )
    with pytest.raises(ContractRefusal, match="RUNTIME_PACKAGE_MISMATCH"):
        runtime.validate_runtime_identity_v5c3(inventory, captured, changed)
    with pytest.raises(ContractRefusal):
        runtime.validate_runtime_identity_v5c3(
            replace(inventory, import_paths=("/unknown",)), captured, policy
        )


def test_no_host_resolution_or_execution_is_used(
    model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import subprocess

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("host runtime operation forbidden")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert runtime.validate_runtime_identity_v5c3(*model)


def test_kernel_loader_alias_is_bound_to_captured_canonical_path(model: Any) -> None:
    inventory, captured, policy = model
    canonical = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
    pins = (
        inventory.files[0],
        replace(inventory.files[1], path=canonical, aliases=("/lib/ld.so",)),
    ) + inventory.files[2:]
    data = (captured[0], replace(captured[1], path=canonical, aliases=("/lib/ld.so",))) + captured[
        2:
    ]
    libraries = (
        replace(policy.required_libraries[0], path=canonical),
        policy.required_libraries[1],
    )
    inventory = replace(inventory, files=pins)
    policy = replace(policy, required_libraries=libraries)
    assert (
        runtime.validate_runtime_identity_v5c3(inventory, data, policy)[0].interpreter_path
        == "/lib/ld.so"
    )
    unbound = (data[0], replace(data[1], aliases=())) + data[2:]
    with pytest.raises(ContractRefusal):
        runtime.validate_runtime_identity_v5c3(inventory, unbound, policy)


def test_unresolved_kernel_loader_refused(model: Any) -> None:
    inventory, captured, policy = model
    canonical = "/usr/lib/unknown-loader.so"
    pins = (inventory.files[0], replace(inventory.files[1], path=canonical)) + inventory.files[2:]
    data = (captured[0], replace(captured[1], path=canonical)) + captured[2:]
    libraries = (
        replace(policy.required_libraries[0], path=canonical),
        policy.required_libraries[1],
    )
    with pytest.raises(ContractRefusal, match="UNKNOWN_LOADER"):
        runtime.validate_runtime_identity_v5c3(
            replace(inventory, files=pins), data, replace(policy, required_libraries=libraries)
        )


def test_native_stdlib_dependencies_are_measured_and_required(model: Any) -> None:
    inventory, captured, policy = model
    path = "/usr/lib/python3.12/lib-dynload/_hashlib.so"
    native = _elf(needed=("libc.so.6",), interpreter=None)
    pins = inventory.files + (runtime.RuntimeFilePinV5c3(path, _sha(native), "12" * 20, "stdlib"),)
    data = captured + (runtime.CapturedRuntimeFileV5c3(path, native),)
    resolutions = inventory.needed_resolutions + ((path, "libc.so.6", "/lib/libc.so.6"),)
    measured = runtime.validate_runtime_identity_v5c3(
        replace(inventory, files=pins, needed_resolutions=resolutions), data, policy
    )
    assert measured[-1].needed == ("libc.so.6",)
    assert measured[-1].build_id == "12" * 20
    with pytest.raises(ContractRefusal, match="RUNTIME_LIBRARY_MISSING"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), data, policy)
    wrong_id = pins[:-1] + (replace(pins[-1], build_id=None),)
    with pytest.raises(ContractRefusal, match="RUNTIME_BUILD_ID_MISMATCH"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=wrong_id), data, policy)


def test_native_stdlib_rpath_is_refused(model: Any) -> None:
    inventory, captured, policy = model
    path = "/usr/lib/python3.12/lib-dynload/_hashlib.so"
    native = _elf(rpath="/untrusted", interpreter=None)
    pins = inventory.files + (runtime.RuntimeFilePinV5c3(path, _sha(native), "12" * 20, "stdlib"),)
    data = captured + (runtime.CapturedRuntimeFileV5c3(path, native),)
    with pytest.raises(ContractRefusal, match="RUNTIME_SEARCH_PATH_REJECTED"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), data, policy)


@pytest.mark.parametrize(
    ("index", "reason"), [(0, "WRONG_INTERPRETER_PATH"), (3, "WRONG_TOOL_PATH")]
)
def test_process_executable_and_tool_paths_are_fixed(model: Any, index: int, reason: str) -> None:
    inventory, captured, policy = model
    pins = (
        inventory.files[:index]
        + (replace(inventory.files[index], path="/wrong/path"),)
        + inventory.files[index + 1 :]
    )
    data = (
        captured[:index] + (replace(captured[index], path="/wrong/path"),) + captured[index + 1 :]
    )
    resolutions = inventory.needed_resolutions
    if index == 0:
        resolutions = (("/wrong/path", "libc.so.6", "/lib/libc.so.6"),)
    with pytest.raises(ContractRefusal, match=reason):
        runtime.validate_runtime_identity_v5c3(
            replace(inventory, files=pins, needed_resolutions=resolutions), data, policy
        )


def test_missing_loader_and_provider_configuration_pins_refused(model: Any) -> None:
    inventory, captured, policy = model
    for config in (
        replace(inventory, loader_configuration_paths=("/etc/ld.so.preload",)),
        replace(inventory, provider_configuration_paths=()),
    ):
        with pytest.raises(ContractRefusal, match="CONFIGURATION_MISSING"):
            runtime.validate_runtime_identity_v5c3(config, captured, policy)


def test_decoy_compiler_source_does_not_cover_actual_stdlib(model: Any) -> None:
    inventory, captured, policy = model
    path = "/usr/lib/python3.12/decoy/py_compile.py"
    pins = inventory.files[:4] + (replace(inventory.files[4], path=path),) + inventory.files[5:]
    data = captured[:4] + (replace(captured[4], path=path),) + captured[5:]
    with pytest.raises(ContractRefusal):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), data, policy)


def test_native_stdlib_invalid_elf_is_not_treated_as_opaque_source(model: Any) -> None:
    inventory, captured, policy = model
    path = "/usr/lib/python3.12/lib-dynload/_hashlib.so"
    native = b"invalid ELF data"
    pins = inventory.files + (runtime.RuntimeFilePinV5c3(path, _sha(native), None, "stdlib"),)
    data = captured + (runtime.CapturedRuntimeFileV5c3(path, native),)
    with pytest.raises(ContractRefusal, match="ELF_REJECTED"):
        runtime.validate_runtime_identity_v5c3(replace(inventory, files=pins), data, policy)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("files", []),
        ("files", (None,) * 5),
        ("needed_resolutions", (("/usr/bin/python3.12",),)),
        ("needed_resolutions", (("/usr/bin/python3.12", "../evil", "/lib/libc.so.6"),)),
        ("needed_resolutions", (("/absent", "libc.so.6", "/lib/libc.so.6"),)),
        ("loader_configuration_paths", ["/etc/ld.so.preload"]),
        ("import_paths", ("/usr/lib/../untrusted",)),
        ("import_paths", ("/usr/lib/\ud800",)),
        ("provider_configuration_paths", ("/etc/absent",)),
    ],
)
def test_inventory_structure_rejects_nested_and_unbounded_shapes_before_json(
    model: Any, field: str, value: Any
) -> None:
    with pytest.raises(runtime.RuntimeIdentityRefusalV5c3):
        runtime.validate_runtime_inventory_structure_v5c3(replace(model[0], **{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [("path", []), ("sha256", []), ("build_id", []), ("role", []), ("aliases", ["/alias"])],
)
def test_inventory_structure_checks_pin_scalar_types(model: Any, field: str, value: Any) -> None:
    inventory = model[0]
    pins = (replace(inventory.files[0], **{field: value}),) + inventory.files[1:]
    with pytest.raises(runtime.RuntimeIdentityRefusalV5c3):
        runtime.validate_runtime_inventory_structure_v5c3(replace(inventory, files=pins))


def test_runtime_manifest_has_aggregate_string_budget(model: Any) -> None:
    inventory = model[0]
    padding = tuple(
        runtime.RuntimeFilePinV5c3("/" + "x" * 4080 + str(index), None, None, "configuration")
        for index in range(300)
    )
    with pytest.raises(runtime.RuntimeIdentityRefusalV5c3, match="MANIFEST_LIMIT_EXCEEDED"):
        runtime.validate_runtime_inventory_structure_v5c3(
            replace(inventory, files=inventory.files + padding)
        )
