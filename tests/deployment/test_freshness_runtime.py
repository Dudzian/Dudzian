"""Negatywne granice podpisu, inwentarza i publikacji runtime.

Dane jednostkowe modelują filesystem, a podpisy Ed25519 są rzeczywiste.
Rzeczywiste importy wheels testuje osobny test buildera na Ubuntu.
"""

import hashlib
import json
import os
import runpy
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from deployment import freshness_runtime as runtime, freshness_runtime_bootstrap as bootstrap

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Kontrakt filesystem Linux")
REVISION = "a" * 40
ENVIRONMENT = "ubuntu2404-cpython312-test"


def write(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "host"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    source = root / "source"
    source.mkdir(mode=0o755)
    source.chmod(0o755)
    packages = {"cryptography": "50.0.2", "psycopg": "3.3.6"}
    lock_lines = []
    for name, version in packages.items():
        wheel = f"{name}-{version}-py3-none-any.whl"
        content = f"wheel {name}".encode()
        write(source / "wheelhouse" / wheel, content)
        lock_lines.append(f"{name}=={version} --hash=sha256:{hashlib.sha256(content).hexdigest()}")
        write(
            source / "lib/python3.12/site-packages" / f"{name}-{version}.dist-info/METADATA",
            f"Name: {name}\nVersion: {version}\n".encode(),
        )
    write(source / "requirements.lock", ("\n".join(lock_lines) + "\n").encode())
    write(
        source / "pyvenv.cfg",
        b"home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\n",
    )
    (source / "bin").mkdir()
    (source / "bin/python3.12").symlink_to("/usr/bin/python3.12")
    (source / "lib64").symlink_to("lib")
    external = []
    for name in (
        "/usr/bin/python3.12",
        "/usr/lib/python3.12/site.py",
        "/usr/lib/libpq.so.5",
        "/usr/lib/libssl.so.3",
        "/usr/lib/libcrypto.so.3",
        "/etc/ld.so.cache",
    ):
        path = root / name[1:]
        write(path, ("trusted:" + name).encode(), 0o755 if name.endswith("python3.12") else 0o644)
        external.append(path)
    external.append(root / "usr/lib/python3.12")
    key = Ed25519PrivateKey.generate()
    authority = root / "etc/cryptohunter/freshness-runtime-authority.pub"
    write(authority, key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
    manifest = runtime.collect_runtime_manifest(
        source,
        environment_id=ENVIRONMENT,
        release_revision=REVISION,
        interpreter={
            "implementation": "CPython",
            "version": "3.12.3",
            "abi": "cpython-312-x86_64-linux-gnu",
            "executable": "bin/python3.12",
            "base_executable": "/usr/bin/python3.12",
            "provenance": "Ubuntu dpkg python3.12 3.12.3-1ubuntu0",
        },
        packages=packages,
        external_paths=tuple(external),
        lock_sha256=hashlib.sha256((source / "requirements.lock").read_bytes()).hexdigest(),
    )
    manifest["external_files"] = {
        "/" + str(Path(name).relative_to(root)): value
        for name, value in manifest["external_files"].items()
    }
    config = runtime.RuntimeConfig(
        source,
        root / "manifest.json",
        root / "manifest.sig",
        authority,
        ENVIRONMENT,
        REVISION,
        os.getuid(),
        root,
        False,
    )

    def sign(data=None):
        raw = json.dumps(manifest if data is None else data, sort_keys=True).encode()
        write(config.manifest, raw)
        write(config.signature, key.sign(raw))

    sign()
    # Zarządzany executor ma /usr/bin/openssl należący do nobody. Jedynie ten
    # zewnętrzny trust anchor modelujemy; sign/verify nadal wykonuje prawdziwy OpenSSL.
    monkeypatch.setattr(bootstrap, "_openssl", lambda: "/usr/bin/openssl")
    return config, manifest, sign, key


def test_real_signature_inventory_and_symlink_to_directory(candidate):
    config, manifest, _, _ = candidate
    verified = runtime.verify_runtime(config)
    assert verified.environment_id == ENVIRONMENT
    assert verified.executable == runtime.RUNTIME_BASE / "releases" / ENVIRONMENT / "bin/python3.12"
    assert verified.base_executable == Path("/usr/bin/python3.12")
    assert verified.packages == manifest["packages"]


@pytest.mark.parametrize("path_alias", ["runtime", "trusted-key", "root"])
def test_authority_boundary_rejects_parent_traversal(candidate, path_alias):
    config, manifest, sign, _ = candidate
    original_runtime = config.runtime
    if path_alias in {"runtime", "trusted-key"}:
        inside = config.runtime / "authority.pub"
        write(inside, config.trusted_key.read_bytes())
        manifest["files"] = runtime._tree(config.runtime)
        sign()
        config = replace(config, trusted_key=inside)
    if path_alias == "runtime":
        config = replace(config, runtime=config.runtime / ".." / config.runtime.name)
    elif path_alias == "trusted-key":
        config = replace(
            config,
            trusted_key=config.root / "etc" / ".." / config.runtime.name / "authority.pub",
        )
    else:
        original_root = config.root
        alias = original_root / ".." / original_root.name
        config = replace(
            config,
            root=alias,
            runtime=alias / config.runtime.relative_to(original_root),
            manifest=alias / config.manifest.relative_to(original_root),
            signature=alias / config.signature.relative_to(original_root),
            trusted_key=alias / config.trusted_key.relative_to(original_root),
        )
    assert config.runtime.resolve() == original_runtime.resolve()
    with pytest.raises(runtime.RuntimeTrustError, match="ścieżka wejściowa runtime"):
        runtime.verify_runtime(config)


def test_external_symlink_requires_signed_target_inventory(candidate):
    config, manifest, sign, _ = candidate
    source = config.root / "usr/lib/python3.12/site.py"
    target = config.root / "etc/python3.12/site.py"
    write(target, source.read_bytes())
    source.unlink()
    source.symlink_to("../../../etc/python3.12/site.py")
    manifest["external_files"]["/usr/lib/python3.12/site.py"] = runtime._record(source)
    manifest["external_files"]["/etc/python3.12/site.py"] = runtime._record(target)
    sign()
    runtime.verify_runtime(config)

    del manifest["external_files"]["/etc/python3.12/site.py"]
    sign()
    with pytest.raises(runtime.RuntimeTrustError, match="poza zatwierdzone materiały"):
        runtime.verify_runtime(config)


def external_symlink_chain(candidate, length=2, *, absolute=False):
    """Model libpython Noble, bez dereferencji absolutnych linków poza testowy root."""
    config, manifest, sign, _ = candidate
    names = [
        "/usr/lib/python3.12/config-3.12-x86_64-linux-gnu/libpython3.12.so",
        "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1",
        *(f"/usr/lib/x86_64-linux-gnu/libpython3.12.so.link{index}" for index in range(length - 2)),
    ]
    final = "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"
    write(config.root / final[1:], b"signed libpython bytes")
    targets = []
    for name, destination in zip(names, [*names[1:], final], strict=True):
        target = destination if absolute else os.path.relpath(destination, str(Path(name).parent))
        targets.append(target)
        path = config.root / name[1:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    for name in [*names, final]:
        path = config.root / name[1:]
        for item in (path, *path.parents):
            if item == config.root:
                break
            manifest["external_files"]["/" + item.relative_to(config.root).as_posix()] = (
                runtime._record(item)
            )
    sign()
    return names, targets, final


@pytest.mark.parametrize("length", [2, 3, 5])
@pytest.mark.parametrize("absolute", [False, True])
def test_complete_signed_external_symlink_chains_pass(candidate, length, absolute):
    names, targets, final = external_symlink_chain(candidate, length, absolute=absolute)
    config, manifest, _, _ = candidate
    verified = runtime.verify_runtime(config)
    assert verified.manifest == manifest
    assert runtime._resolved(names[0], verified.manifest, config) == Path(final)
    for name, target in zip(names, targets, strict=True):
        assert verified.manifest["external_files"][name]["kind"] == "symlink"
        assert verified.manifest["external_files"][name]["target"] == target
    if not absolute:
        assert targets[0] == "../../x86_64-linux-gnu/libpython3.12.so.1"


def test_signed_usrmerge_directory_symlink_chains_pass(candidate):
    config, manifest, sign, _ = candidate
    for name in ("/usr/sbin", "/usr/lib"):
        path = config.root / name[1:]
        path.mkdir(parents=True, exist_ok=True)
        manifest["external_files"][name] = runtime._record(path)
    for name, target in (
        ("/bin", "usr/bin"),
        ("/sbin", "usr/sbin"),
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
        ("/usr/lib64", "lib"),
    ):
        path = config.root / name[1:]
        path.symlink_to(target)
        manifest["external_files"][name] = runtime._record(path)
    manifest["external_files"]["/usr/bin"] = runtime._record(config.root / "usr/bin")
    sign()
    verified = runtime.verify_runtime(config)
    assert runtime._resolved("/lib64", verified.manifest, config) == Path("/usr/lib")


@pytest.mark.parametrize(
    "attack,reason",
    [
        ("missing-intermediate", "poza zatwierdzone materiały"),
        ("missing-target", "Nie można ustanowić integralności runtime"),
        ("changed-content", "Niezgodna integralność pliku runtime"),
        ("changed-type", "Niechroniony plik runtime"),
        ("changed-mode", "Niezgodne uprawnienia runtime"),
        ("changed-link", "Niezgodny symlink runtime"),
        ("unapproved-target", "poza zatwierdzone materiały"),
        ("cycle", "Cykl symlinków runtime"),
    ],
)
def test_external_symlink_chain_refuses_incomplete_inventory_and_tampering(
    candidate, attack, reason
):
    names, _, final = external_symlink_chain(candidate, 3)
    config, manifest, sign, _ = candidate
    # Każdy scenariusz wychodzi od kandydata przechodzącego pełną weryfikację.
    runtime.verify_runtime(config)
    target = config.root / final[1:]
    intermediate = config.root / names[1][1:]
    if attack == "missing-intermediate":
        del manifest["external_files"][names[1]]
        sign()
    elif attack == "missing-target":
        target.unlink()
    elif attack == "changed-content":
        original = target.read_bytes()
        target.write_bytes(b"!" * len(original))
    elif attack == "changed-type":
        target.unlink()
        target.mkdir(mode=0o644)
        target.chmod(0o644)
    elif attack == "changed-mode":
        target.chmod(0o600)
    elif attack == "changed-link":
        intermediate.unlink()
        intermediate.symlink_to(final)
    elif attack == "unapproved-target":
        unapproved = "/usr/lib/unapproved-libpython.so"
        write(config.root / unapproved[1:], target.read_bytes())
        intermediate.unlink()
        intermediate.symlink_to(unapproved)
        manifest["external_files"][names[1]] = runtime._record(intermediate)
        sign()
    elif attack == "cycle":
        intermediate.unlink()
        intermediate.symlink_to(names[0])
        manifest["external_files"][names[1]] = runtime._record(intermediate)
        sign()
    with pytest.raises(runtime.RuntimeTrustError, match=reason):
        runtime.verify_runtime(config)


@pytest.mark.parametrize(
    "attack",
    [
        "hash",
        "signature",
        "key",
        "extra-package",
        "pth",
        "pyc",
        "pyvenv",
        "symlink",
        "permissions",
        "interpreter",
        "stdlib",
        "extra-stdlib",
        "zip",
        "preload",
        "cache",
        "missing-wheel",
        "missing-metadata",
        "environment",
        "revision",
        "abi",
        "user-site",
        "unknown-field",
        "duplicate-field",
        "lock",
        "extra-wheel",
        "missing-native",
        "key-in-bundle",
    ],
)
def test_refuse_tampering(candidate, attack):
    config, manifest, sign, _ = candidate
    if attack == "hash":
        write(config.runtime / "requirements.lock", b"changed")
    elif attack == "signature":
        write(config.signature, b"!" * 64)
    elif attack == "key":
        write(config.trusted_key, b"!" * 32)
    elif attack == "extra-package":
        write(config.runtime / "lib/python3.12/site-packages/rogue.py", b"raise SystemExit")
    elif attack == "pth":
        write(config.runtime / "lib/python3.12/site-packages/rogue.pth", b"import evil")
        manifest["files"] = runtime._tree(config.runtime)
        sign()
    elif attack == "pyc":
        write(config.runtime / "lib/python3.12/site-packages/rogue.pyc", b"untrusted")
        manifest["files"] = runtime._tree(config.runtime)
        sign()
    elif attack == "pyvenv":
        write(config.runtime / "pyvenv.cfg", b"include-system-site-packages = true\n")
    elif attack == "symlink":
        (config.runtime / "bin/python3.12").unlink()
        (config.runtime / "bin/python3.12").symlink_to("/usr/bin/python3")
    elif attack == "permissions":
        (config.runtime / "requirements.lock").chmod(0o666)
    elif attack == "interpreter":
        write(config.root / "usr/bin/python3.12", b"upgraded interpreter", 0o755)
    elif attack == "stdlib":
        write(config.root / "usr/lib/python3.12/site.py", b"changed")
    elif attack == "extra-stdlib":
        write(config.root / "usr/lib/python3.12/evil.py", b"unexpected")
    elif attack == "zip":
        write(config.root / "usr/lib/python312.zip", b"unexpected")
    elif attack == "preload":
        write(config.root / "etc/ld.so.preload", b"/evil.so")
    elif attack == "cache":
        write(config.root / "etc/ld.so.cache", b"changed")
    elif attack == "missing-wheel":
        next((config.runtime / "wheelhouse").iterdir()).unlink()
    elif attack == "missing-metadata":
        next(config.runtime.rglob("METADATA")).unlink()
        manifest["files"] = runtime._tree(config.runtime)
        sign()
    elif attack == "environment":
        config = replace(config, expected_environment_id="other")
    elif attack == "revision":
        config = replace(config, expected_revision="b" * 40)
    elif attack == "abi":
        manifest["interpreter"]["abi"] = "cpython-311-x86_64-linux-gnu"
        sign()
    elif attack == "user-site":
        write(
            config.runtime / "pyvenv.cfg",
            b"home = /usr/bin\ninclude-system-site-packages = true\nversion = 3.12.3\n",
        )
        manifest["files"] = runtime._tree(config.runtime)
        sign()
    elif attack == "unknown-field":
        manifest["optional-bypass"] = True
        sign()
    elif attack == "duplicate-field":
        raw = config.manifest.read_bytes().replace(b'"schema": 1', b'"schema": 1, "schema": 1')
        write(config.manifest, raw)
        write(config.signature, candidate[3].sign(raw))
    elif attack == "lock":
        write(config.runtime / "requirements.lock", b"--extra-index-url https://evil.invalid\n")
        manifest["files"] = runtime._tree(config.runtime)
        manifest["lock_sha256"] = manifest["files"]["requirements.lock"]["sha256"]
        sign()
    elif attack == "extra-wheel":
        write(config.runtime / "wheelhouse/evil-1.0-py3-none-any.whl", b"extra")
        manifest["files"] = runtime._tree(config.runtime)
        sign()
    elif attack == "missing-native":
        del manifest["external_files"]["/usr/lib/libpq.so.5"]
        sign()
    elif attack == "key-in-bundle":
        inside = config.runtime / "authority.pub"
        write(inside, config.trusted_key.read_bytes())
        config = replace(config, trusted_key=inside)
    with pytest.raises((runtime.RuntimeTrustError, OSError)):
        runtime.verify_runtime(config)


def prepare_publication(config):
    base = config.root / str(runtime.RUNTIME_BASE)[1:]
    base.mkdir(parents=True)
    return base


def test_publication_idempotent_keeps_previous_release(candidate):
    config, _, _, _ = candidate
    base = prepare_publication(config)
    previous = base / "releases/previous"
    previous.mkdir(parents=True)
    write(previous / "preserved", b"previous")
    first = runtime.publish_runtime(config)
    second = runtime.publish_runtime(config)
    assert first == second
    assert (previous / "preserved").read_bytes() == b"previous"
    assert not (base / f"{ENVIRONMENT}.journal").exists()
    destination = config.root / first.executable.relative_to("/")
    assert destination.is_symlink()


def test_publication_matches_modes_with_restrictive_umask(candidate):
    config, _, _, _ = candidate
    prepare_publication(config)
    previous = os.umask(0o077)
    try:
        result = runtime.publish_runtime(config)
    finally:
        os.umask(previous)
    installed = config.root / result.executable.relative_to("/").parent.parent
    assert stat.S_IMODE((installed / "requirements.lock").stat().st_mode) == 0o644


def test_partial_publication_refuses_retry(candidate):
    config, _, _, _ = candidate
    base = prepare_publication(config)
    write(base / f"{ENVIRONMENT}.journal", b"interrupted", 0o600)
    with pytest.raises(runtime.RuntimeTrustError):
        runtime.publish_runtime(config)
    assert (base / f"{ENVIRONMENT}.journal").read_bytes() == b"interrupted"


def test_crash_after_rename_blocks_installed_binding(candidate, monkeypatch):
    config, _, _, _ = candidate
    base = prepare_publication(config)
    original = runtime._fsync_directory

    def interrupted(path):
        original(path)
        if path == base / "releases":
            raise OSError("interrupted after atomic publication")

    monkeypatch.setattr(runtime, "_fsync_directory", interrupted)
    with pytest.raises(OSError):
        runtime.publish_runtime(config)
    assert (base / "releases" / ENVIRONMENT).exists()
    assert (base / f"{ENVIRONMENT}.journal").exists()
    binding = {
        "schema": 1,
        "environment_id": ENVIRONMENT,
        "source_revision": REVISION,
        "manifest_sha256": hashlib.sha256(config.manifest.read_bytes()).hexdigest(),
        "executable": str(runtime.RUNTIME_BASE / "releases" / ENVIRONMENT / "bin/python3.12"),
        "base_executable": "/usr/bin/python3.12",
        "base_executable_sha256": "a" * 64,
    }
    write(config.root / str(runtime.BINDING)[1:], json.dumps(binding).encode())
    with pytest.raises(runtime.RuntimeTrustError, match="Niedokończona"):
        runtime.verify_installed_runtime(
            root=config.root, owner_uid=os.getuid(), check_platform=False
        )


def test_bootstrap_real_ed25519_invalid_inputs(monkeypatch):
    monkeypatch.setattr(bootstrap, "_openssl", lambda: "/usr/bin/openssl")
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    raw = b"independently approved runtime"
    signature = key.sign(raw)
    bootstrap.verify_ed25519(raw, signature, public)
    for bad_raw, bad_signature, bad_key in (
        (raw + b"!", signature, public),
        (raw, signature[:-1], public),
        (raw, signature, public[:-1]),
    ):
        with pytest.raises(bootstrap.RuntimeTrustError):
            bootstrap.verify_ed25519(bad_raw, bad_signature, bad_key)


def test_platform_mismatch_refuses(candidate, monkeypatch):
    config, _, _, _ = candidate
    monkeypatch.setattr(
        runtime.platform, "freedesktop_os_release", lambda: {"ID": "debian", "VERSION_ID": "13"}
    )
    with pytest.raises(runtime.RuntimeTrustError):
        runtime.verify_runtime(replace(config, check_platform=True))


def test_bootstrap_refuses_unprotected_system_executable(monkeypatch):
    original = Path.lstat

    def substituted(path):
        info = original(path)
        fields = list(info)
        fields[4] = 12345 if path == Path("/usr/bin/openssl") else 0
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", substituted)
    with pytest.raises(bootstrap.RuntimeTrustError, match="Niechroniony"):
        bootstrap._openssl()


def installed_binding(config, verified):
    binding = {
        "schema": 1,
        "environment_id": verified.environment_id,
        "source_revision": verified.revision,
        "manifest_sha256": verified.manifest_sha256,
        "executable": str(verified.executable),
        "base_executable": str(verified.base_executable),
        "base_executable_sha256": verified.base_executable_sha256,
    }
    write(config.root / str(runtime.BINDING)[1:], json.dumps(binding).encode())
    return binding


def test_installed_binding_full_real_signature(candidate):
    config, _, _, _ = candidate
    prepare_publication(config)
    verified = runtime.publish_runtime(config)
    installed_binding(config, verified)
    assert (
        runtime.verify_installed_runtime(
            root=config.root, owner_uid=os.getuid(), check_platform=False
        )
        == verified
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "b" * 64),
        ("source_revision", "b" * 40),
        ("base_executable_sha256", "b" * 64),
        ("executable", "/usr/bin/python3.12"),
        ("base_executable", "/usr/bin/python3"),
        ("schema", True),
        ("environment_id", "../traversal"),
    ],
)
def test_installed_binding_mismatch_refuses(candidate, field, value):
    config, _, _, _ = candidate
    prepare_publication(config)
    verified = runtime.publish_runtime(config)
    binding = installed_binding(config, verified)
    binding[field] = value
    write(config.root / str(runtime.BINDING)[1:], json.dumps(binding).encode())
    with pytest.raises(runtime.RuntimeTrustError):
        runtime.verify_installed_runtime(
            root=config.root, owner_uid=os.getuid(), check_platform=False
        )


def test_bootstrap_cli_rejects_unrecognized_operation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bootstrap", "self-approve"])
    assert bootstrap.main() == 1
    assert "Odmowa" in capsys.readouterr().out


def test_bootstrap_cli_refuses_unprotected_core(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bootstrap", "verify-installed"])
    original = Path.lstat

    def substituted(path):
        info = original(path)
        fields = list(info)
        fields[4] = 12345
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", substituted)
    assert bootstrap.main() == 1
    assert "Odmowa" in capsys.readouterr().out


@pytest.fixture
def protected_bootstrap_metadata(monkeypatch):
    """Model tylko metadanych uprzednio zatwierdzonej kopii bootstrapu."""
    original = Path.lstat

    def protected(path):
        fields = list(original(path))
        fields[stat.ST_UID] = 0
        fields[stat.ST_MODE] &= ~0o022
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", protected)
    monkeypatch.setattr(sys, "argv", ["bootstrap", "verify-installed"])
    return protected


@pytest.mark.parametrize("filename", ["freshness_runtime.py", "freshness_application.py"])
@pytest.mark.parametrize(
    "attack", ["owner", "writable", "symlink", "hardlink", "directory", "missing"]
)
def test_bootstrap_checks_both_helpers_before_executing_code(
    protected_bootstrap_metadata, monkeypatch, capsys, filename, attack
):
    executed = []

    def substituted(path):
        if path.name != filename:
            return protected_bootstrap_metadata(path)
        if attack == "missing":
            raise FileNotFoundError(str(path))
        fields = list(protected_bootstrap_metadata(path))
        if attack == "owner":
            fields[stat.ST_UID] = 12345
        elif attack == "writable":
            fields[stat.ST_MODE] |= 0o022
        elif attack == "symlink":
            fields[stat.ST_MODE] = stat.S_IFLNK | 0o644
        elif attack == "hardlink":
            fields[stat.ST_NLINK] = 2
        elif attack == "directory":
            fields[stat.ST_MODE] = stat.S_IFDIR | 0o755
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", substituted)
    monkeypatch.setattr(runpy, "run_path", lambda *args, **kwargs: executed.append(args))
    assert bootstrap.main() == 1
    assert "Odmowa" in capsys.readouterr().out
    assert executed == []


@pytest.mark.parametrize("application_failure", [False, True])
def test_bootstrap_requires_application_verification_before_success(
    protected_bootstrap_metadata, monkeypatch, capsys, application_failure
):
    verified = []

    def verify_runtime():
        verified.append("runtime")

    def verify_application():
        verified.append("application")
        if application_failure:
            raise bootstrap.RuntimeTrustError("unapproved application bytes")

    def load(path, **_kwargs):
        if Path(path).name == "freshness_runtime.py":
            return {"verify_installed_runtime": verify_runtime}
        assert verified == ["runtime"]
        assert Path(path).name == "freshness_application.py"
        return {"verify_installed_application": verify_application}

    monkeypatch.setattr(runpy, "run_path", load)
    assert bootstrap.main() == int(application_failure)
    output = capsys.readouterr().out
    assert ("Odmowa" in output) == application_failure
    assert ("Integralność podpisanego runtime" in output) != application_failure
    assert verified == ["runtime", "application"]


def test_bootstrap_refuses_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(bootstrap.RuntimeTrustError, match="Linux"):
        bootstrap.verify_ed25519(b"payload", b"a" * 64, b"b" * 32)


def test_bootstrap_refuses_missing_memfd(monkeypatch):
    monkeypatch.delattr(os, "memfd_create")
    with pytest.raises(bootstrap.RuntimeTrustError, match="memfd"):
        bootstrap.verify_ed25519(b"payload", b"a" * 64, b"b" * 32)
