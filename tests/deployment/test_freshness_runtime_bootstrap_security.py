"""Bootstrap OpenSSL: stały RFC8032 wektor, bez generowania kluczy/podpisów.

W executorze modelowany jest wyłącznie root ownership systemowego OpenSSL.
Proces OpenSSL i jego odmowy wykonują się rzeczywiście. Próby nie kwalifikują
pochodzenia bibliotek systemowych ani hosta Ubuntu.
"""

import builtins
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

if sys.platform == "linux":
    import fcntl

from deployment import freshness_runtime_bootstrap as bootstrap

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Bootstrap Linux/memfd")
# RFC 8032 section 7.1, TEST 2 (wyłącznie publiczne, wcześniej opublikowane dane).
MESSAGE = bytes.fromhex("72")
PUBLIC_KEY = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
SIGNATURE = bytes.fromhex(
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
)


@pytest.fixture
def system_openssl(monkeypatch):
    # Nie zmieniamy uprawnień hosta. Test nie dowodzi ochrony trust anchor.
    monkeypatch.setattr(bootstrap, "_openssl", lambda: "/usr/bin/openssl")


def test_fixed_vector_and_sealed_explicit_contract(system_openssl, monkeypatch):
    run = subprocess.run
    observed = []

    def checked_run(arguments, **kwargs):
        config = kwargs["env"]["OPENSSL_CONF"]
        fd = int(config.rsplit("/", 1)[1])
        assert os.pread(fd, 65536, 0) == bootstrap.OPENSSL_CONFIGURATION
        assert fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) == 0x000F
        for descriptor in kwargs["pass_fds"]:
            with pytest.raises(PermissionError):
                os.write(descriptor, b"unauthorized change")
            with pytest.raises(PermissionError):
                os.ftruncate(descriptor, 0)
        assert kwargs["env"]["OPENSSL_MODULES"] == config
        assert kwargs["env"]["OPENSSL_CONF_INCLUDE"] == config
        assert kwargs["env"]["OPENSSL_ENGINES"] == config
        assert arguments[arguments.index("-provider") + 1] == "default"
        assert arguments[arguments.index("-provider-path") + 1] == config
        assert arguments[arguments.index("-propquery") + 1] == "provider=default"
        assert set(kwargs["env"]) == {
            "PATH",
            "LC_ALL",
            "OPENSSL_CONF",
            "OPENSSL_CONF_INCLUDE",
            "OPENSSL_MODULES",
            "OPENSSL_ENGINES",
        }
        result = run(arguments, **kwargs)
        observed.append(result.returncode)
        return result

    monkeypatch.setattr(bootstrap.subprocess, "run", checked_run)
    bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert observed == [0]


@pytest.mark.parametrize("attack", ["configuration", "include", "provider"])
def test_substituted_contract_refuses_before_execution(monkeypatch, attack):
    def forbidden(*args, **kwargs):
        raise AssertionError("Nie wolno uruchomić OpenSSL dla zmienionego kontraktu")

    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    if attack == "provider":
        monkeypatch.setattr(bootstrap, "OPENSSL_PROVIDER", "unknown-unapproved-provider")
    else:
        monkeypatch.setattr(
            bootstrap,
            "OPENSSL_CONFIGURATION",
            b".include /unapproved/openssl.cnf\n"
            if attack == "include"
            else b"openssl_conf = attacker\n[attacker]\nengines = arbitrary\n",
        )
    with pytest.raises(bootstrap.RuntimeTrustError, match="Niezatwierdzona"):
        bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)


def test_hostile_environment_config_and_provider_are_never_opened(
    system_openssl, tmp_path, monkeypatch
):
    config = tmp_path / "hostile-openssl.cnf"
    included = tmp_path / "included.cnf"
    provider = tmp_path / "unapproved-provider.so"
    included.write_bytes(b"invalid OpenSSL configuration, must never be read\n")
    provider.write_bytes(b"non-executable sentinel, must never be opened\n")
    config.write_text(
        f".include {included}\nopenssl_conf = hostile\n[hostile]\nproviders = providers\n"
        f"[providers]\nunapproved = injected\n[injected]\nmodule = {provider}\nactivate = 1\n"
    )
    monkeypatch.setenv("OPENSSL_CONF", str(config))
    monkeypatch.setenv("OPENSSL_CONF_INCLUDE", str(tmp_path))
    monkeypatch.setenv("OPENSSL_MODULES", str(tmp_path))
    monkeypatch.setenv("OPENSSL_ENGINES", str(tmp_path))
    monkeypatch.setenv("LD_PRELOAD", str(provider))
    trace = tmp_path / "opens.trace"
    run = subprocess.run
    observed = []

    def traced_run(arguments, **kwargs):
        result = run(
            ["/usr/bin/strace", "-e", "trace=openat,openat2", "-o", str(trace), *arguments],
            **kwargs,
        )
        observed.append(result.returncode)
        return result

    monkeypatch.setattr(bootstrap.subprocess, "run", traced_run)
    bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert observed == [0]
    actual = trace.read_text()
    assert "+++ exited with 0 +++" in actual
    assert "/proc/self/fd/" in actual
    assert all(str(path) not in actual for path in (config, included, provider))
    assert "ossl-modules" not in actual
    if evidence := os.environ.get("STAGE9_SAFE_OUTPUT"):
        (Path(evidence) / "bootstrap-opens.trace").write_text(actual)


@pytest.mark.parametrize("attack", ["message", "signature", "public-key"])
def test_real_verification_failure(system_openssl, monkeypatch, attack):
    raw, signature, key = MESSAGE, SIGNATURE, PUBLIC_KEY
    if attack == "message":
        raw += b"!"
    elif attack == "signature":
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    else:
        key = bytes([key[0] ^ 1]) + key[1:]
    run = subprocess.run
    observed = []

    def checked_run(arguments, **kwargs):
        result = run(arguments, **kwargs)
        observed.append(result.returncode)
        return result

    monkeypatch.setattr(bootstrap.subprocess, "run", checked_run)
    with pytest.raises(bootstrap.RuntimeTrustError, match="Nieprawidłowy podpis"):
        bootstrap.verify_ed25519(raw, signature, key)
    assert len(observed) == 1 and observed[0] != 0


def test_no_cryptography_import_before_or_during_verification(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise AssertionError("Biblioteka payloadu przed ustanowieniem zaufania")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = runpy.run_path(str(Path(bootstrap.__file__)))
    verifier = namespace["verify_ed25519"]
    verifier.__globals__["_openssl"] = lambda: "/usr/bin/openssl"
    verifier(MESSAGE, SIGNATURE, PUBLIC_KEY)


def test_failure_precedes_runtime_manifest_interpretation(system_openssl, tmp_path, monkeypatch):
    from deployment import freshness_runtime as runtime

    root = tmp_path / "host"
    root.mkdir()
    payload = root / "runtime"
    payload.mkdir()
    manifest = root / "manifest.json"
    signature = root / "manifest.sig"
    key = root / "authority.pub"
    # Publiczny wektor podpisuje 0x72, nigdy poniższy niezaufany dokument.
    manifest.write_bytes(b'{"untrusted": "must not be parsed"}')
    signature.write_bytes(SIGNATURE)
    key.write_bytes(PUBLIC_KEY)
    config = runtime.RuntimeConfig(
        payload, manifest, signature, key, "test-only", "a" * 40, os.getuid(), root, False
    )

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("Interpretacja manifestu przed prawidłowym podpisem")

    monkeypatch.setattr(runtime, "_json", forbidden_parse)
    with pytest.raises(bootstrap.RuntimeTrustError, match="Nieprawidłowy podpis"):
        runtime.verify_runtime(config)


def test_descriptor_cleanup_when_openssl_fails(system_openssl, monkeypatch):
    created = []
    sealed = bootstrap._sealed_input

    def remember(value):
        fd = sealed(value)
        created.append(fd)
        return fd

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("openssl", 15)

    monkeypatch.setattr(bootstrap, "_sealed_input", remember)
    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    with pytest.raises(bootstrap.RuntimeTrustError, match="niedostępny"):
        bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert len(created) == 4
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize(
    "attack",
    [
        "owner",
        "writable",
        "symlink",
        "hardlink",
        "directory",
        "fifo",
        "not-executable",
        "ancestor-owner",
        "ancestor-writable",
        "ancestor-symlink",
        "ancestor-file",
    ],
)
def test_openssl_metadata_refused_before_execution_and_closes_inputs(monkeypatch, attack):
    import stat

    executable = Path("/usr/bin/openssl")
    original_lstat = Path.lstat
    sealed = bootstrap._sealed_input
    descriptors = []

    def metadata(path):
        fields = list(original_lstat(path))
        fields[stat.ST_UID] = 0
        fields[stat.ST_MODE] &= ~0o022
        if path == executable:
            fields[stat.ST_MODE] = stat.S_IFREG | 0o755
            fields[stat.ST_NLINK] = 1
            if attack == "owner":
                fields[stat.ST_UID] = 1001
            elif attack == "writable":
                fields[stat.ST_MODE] |= 0o020
            elif attack == "symlink":
                fields[stat.ST_MODE] = stat.S_IFLNK | 0o755
            elif attack == "hardlink":
                fields[stat.ST_NLINK] = 2
            elif attack == "directory":
                fields[stat.ST_MODE] = stat.S_IFDIR | 0o755
            elif attack == "fifo":
                fields[stat.ST_MODE] = stat.S_IFIFO | 0o755
            elif attack == "not-executable":
                fields[stat.ST_MODE] &= ~stat.S_IXUSR
        elif path == executable.parent:
            if attack == "ancestor-owner":
                fields[stat.ST_UID] = 1001
            elif attack == "ancestor-writable":
                fields[stat.ST_MODE] |= 0o002
            elif attack == "ancestor-symlink":
                fields[stat.ST_MODE] = stat.S_IFLNK | 0o755
            elif attack == "ancestor-file":
                fields[stat.ST_MODE] = stat.S_IFREG | 0o755
        return os.stat_result(fields)

    def remember(value):
        fd = sealed(value)
        descriptors.append(fd)
        return fd

    def forbidden(*args, **kwargs):
        raise AssertionError("Unapproved OpenSSL object must never execute")

    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(bootstrap, "_sealed_input", remember)
    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    with pytest.raises(bootstrap.RuntimeTrustError, match="Niechroniony"):
        bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert len(descriptors) == 4
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("operation", ["write-error", "zero-write", "seek", "seal"])
def test_memfd_failure_closes_current_and_prior_inputs_without_execution(monkeypatch, operation):
    create = os.memfd_create
    write = os.write
    seek = os.lseek
    seal = fcntl.fcntl
    descriptors = []

    def remember(*args, **kwargs):
        fd = create(*args, **kwargs)
        descriptors.append(fd)
        return fd

    def fail_write(fd, value):
        if len(descriptors) == 2:
            if operation == "zero-write":
                return 0
            raise OSError("controlled write refusal")
        return write(fd, value)

    def fail_seek(*args):
        if len(descriptors) == 2:
            raise OSError("controlled seek refusal")
        return seek(*args)

    def fail_seal(*args):
        if len(descriptors) == 2:
            raise OSError("controlled sealing refusal")
        return seal(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsealed input must never reach OpenSSL")

    monkeypatch.setattr(os, "memfd_create", remember)
    if operation in {"write-error", "zero-write"}:
        monkeypatch.setattr(os, "write", fail_write)
    elif operation == "seek":
        monkeypatch.setattr(os, "lseek", fail_seek)
    else:
        monkeypatch.setattr(fcntl, "fcntl", fail_seal)
    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    with pytest.raises(bootstrap.RuntimeTrustError):
        bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_memfd_allocation_failure_closes_prior_inputs_without_execution(monkeypatch):
    create = os.memfd_create
    descriptors = []

    def fail_second(*args, **kwargs):
        if descriptors:
            raise OSError("controlled allocation refusal")
        fd = create(*args, **kwargs)
        descriptors.append(fd)
        return fd

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete inputs must never reach OpenSSL")

    monkeypatch.setattr(os, "memfd_create", fail_second)
    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    with pytest.raises(bootstrap.RuntimeTrustError):
        bootstrap.verify_ed25519(MESSAGE, SIGNATURE, PUBLIC_KEY)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
