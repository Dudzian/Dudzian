"""Rzeczywiste podpisy i bajty aplikacji; granica runtime jest modelem.

Test dodatni potwierdza zakres aplikacji i WITHHELD. Nie kwalifikuje hosta,
interpretera ani pełnego PRODUCTION_LOCAL przez zastąpioną granicę runtime.
"""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from deployment import freshness_application as application
from deployment import freshness_runtime_bootstrap as bootstrap
from deployment import freshness_verifier_entrypoint as launcher

REPOSITORY = Path(__file__).resolve().parents[2]
REVISION = "a" * 40


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in (path.parent, *path.parent.parents):
        if directory.name == "host":
            directory.chmod(0o755)
            break
        directory.chmod(0o755)
    path.write_bytes(content)
    path.chmod(0o644)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Odczytowy kontrakt instalacji Linux")
    root = tmp_path / "host"
    root.mkdir(mode=0o755)
    installation = root / application.INSTALLATION.relative_to("/")
    files = []
    for name in sorted(application.REQUIRED_FILES):
        content = (REPOSITORY / name).read_bytes()
        write(installation / name, content)
        files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "mode": 0o644,
            }
        )
    runtime = SimpleNamespace(
        environment_id="test-only",
        manifest_sha256="b" * 64,
        revision=REVISION,
        executable=Path("/opt/cryptohunter/freshness-runtime/releases/test-only/bin/python3.12"),
        base_executable=Path("/usr/bin/python3.12"),
        base_executable_sha256="c" * 64,
    )
    binding = {
        "environment_id": runtime.environment_id,
        "manifest_sha256": runtime.manifest_sha256,
        "source_revision": runtime.revision,
        "executable": str(runtime.executable),
        "base_executable": str(runtime.base_executable),
        "base_executable_sha256": runtime.base_executable_sha256,
    }
    manifest = {
        "schema_version": 2,
        "scope": application.SCOPE,
        "source_revision": REVISION,
        "files": files,
        "runtime": binding,
    }
    directory = root / application.APPROVAL.relative_to("/")
    key = Ed25519PrivateKey.generate()
    write(
        directory / "authority.pub", key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    )

    def sign():
        raw = json.dumps(manifest, sort_keys=True).encode()
        write(directory / "manifest.json", raw)
        write(directory / "manifest.sig", key.sign(raw))
        write(
            directory / "binding.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": application.SCOPE,
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_revision": manifest["source_revision"],
                    "runtime": manifest["runtime"],
                }
            ).encode(),
        )

    sign()
    # Zewnętrzny OpenSSL jest uruchamiany naprawdę, lecz executor nie ma root-owned /usr/bin.
    monkeypatch.setattr(bootstrap, "_openssl", lambda: "/usr/bin/openssl")
    monkeypatch.setattr(
        application,
        "_helpers",
        lambda: {
            "verify_ed25519": bootstrap.verify_ed25519,
            "verify_installed_runtime": lambda **_arguments: runtime,
        },
    )
    return SimpleNamespace(
        root=root,
        installation=installation,
        directory=directory,
        manifest=manifest,
        runtime=runtime,
        sign=sign,
        key=key,
    )


def verify(candidate):
    return application.verify_installed_application(root=candidate.root, owner_uid=os.getuid())


def test_real_application_signature_preserves_explicit_minimal_scope(installed):
    result = verify(installed)
    assert result["qualification_profile"] == "SIGNED_FRESHNESS_SUBSTRATE"
    assert result["scope"] == application.SCOPE
    assert result["production_qualification"] == "WITHHELD"
    assert result["source_revision"] == REVISION


@pytest.mark.parametrize(
    "attack",
    [
        "sha256",
        "modified-reviewed-signed",
        "missing",
        "extra",
        "extra-directory",
        "unapproved-revision",
        "wrong-manifest",
        "signature",
        "full-scope",
        "binding",
        "runtime",
        "symlink",
        "hardlink",
        "writable",
        "directory-mode",
        "missing-binding",
        "duplicate-record",
        "traversal",
        "invalid-mode",
        "invalid-size",
        "invalid-hash",
        "invalid-files",
        "missing-required",
        "unexpected-field",
        "bool-schema",
        "invalid-revision",
        "invalid-name",
        "duplicate-json",
        "key-length",
        "signature-length",
        "approval-extra",
        "signed-cache",
        "signed-git",
        "signed-deep",
        "wrong-owner",
    ],
)
def test_application_refuses_source_metadata_and_scope_substitutions(installed, attack):
    target = installed.installation / "bot_core/freshness_verifier_service.py"
    resign = True
    if attack == "sha256":
        target.write_bytes(target.read_bytes() + b"\n# modified after installation\n")
    elif attack == "modified-reviewed-signed":
        target.write_bytes(target.read_bytes() + b"\n# independently signed but unreviewed\n")
        entry = next(
            item
            for item in installed.manifest["files"]
            if item["path"].endswith("freshness_verifier_service.py")
        )
        entry.update(
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(), size=target.stat().st_size
        )
    elif attack == "missing":
        target.unlink()
    elif attack == "extra":
        write(installed.installation / "bot_core/unsigned.py", b"raise RuntimeError('unsigned')")
    elif attack == "extra-directory":
        (installed.installation / "bot_core/unsigned-empty").mkdir()
    elif attack == "unapproved-revision":
        installed.manifest["source_revision"] = "d" * 40
    elif attack == "wrong-manifest":
        installed.manifest["files"][0]["sha256"] = "0" * 64
    elif attack == "signature":
        write(installed.directory / "manifest.sig", b"0" * 64)
        resign = False
    elif attack == "full-scope":
        installed.manifest["scope"] = "PRODUCTION_LOCAL"
    elif attack == "binding":
        data = json.loads((installed.directory / "binding.json").read_bytes())
        data["source_revision"] = "d" * 40
        write(installed.directory / "binding.json", json.dumps(data).encode())
        resign = False
    elif attack == "runtime":
        installed.runtime.manifest_sha256 = "d" * 64
    elif attack in {"symlink", "hardlink"}:
        outside = installed.root / "outside.py"
        target.rename(outside)
        if attack == "symlink":
            target.symlink_to(outside)
        else:
            target.hardlink_to(outside)
    elif attack == "writable":
        target.chmod(0o666)
    elif attack == "directory-mode":
        target.parent.chmod(0o700)
    elif attack == "missing-binding":
        (installed.directory / "binding.json").unlink()
        resign = False
    elif attack == "duplicate-record":
        installed.manifest["files"].append(installed.manifest["files"][0])
    elif attack == "traversal":
        installed.manifest["files"][0]["path"] = "bot_core/../unsigned.py"
    elif attack == "invalid-mode":
        installed.manifest["files"][0]["mode"] = 0o666
    elif attack == "invalid-size":
        installed.manifest["files"][0]["size"] = -1
    elif attack == "invalid-hash":
        installed.manifest["files"][0]["sha256"] = "invalid"
    elif attack == "invalid-files":
        installed.manifest["files"] = {}
    elif attack == "missing-required":
        installed.manifest["files"].pop()
    elif attack == "unexpected-field":
        installed.manifest["optional-bypass"] = True
    elif attack == "bool-schema":
        installed.manifest["schema_version"] = True
    elif attack == "invalid-revision":
        installed.manifest["source_revision"] = "D" * 40
    elif attack == "invalid-name":
        installed.manifest["files"][0]["path"] = "bot_core\\unsigned.py"
    elif attack == "duplicate-json":
        raw = (
            (installed.directory / "manifest.json")
            .read_bytes()
            .replace(b'"schema_version": 2', b'"schema_version": 2, "schema_version": 2')
        )
        write(installed.directory / "manifest.json", raw)
        write(installed.directory / "manifest.sig", installed.key.sign(raw))
        resign = False
    elif attack == "key-length":
        write(installed.directory / "authority.pub", b"a" * 31)
    elif attack == "signature-length":
        write(installed.directory / "manifest.sig", b"a" * 63)
        resign = False
    elif attack == "approval-extra":
        write(installed.directory / "unapproved.json", b"{}")
    elif attack in {"signed-cache", "signed-git", "signed-deep"}:
        name = {
            "signed-cache": "bot_core/unsafe.pyc",
            "signed-git": "bot_core/.git/config",
            "signed-deep": "bot_core/" + "nested/" * 32 + "unsafe.py",
        }[attack]
        content = b"signed but outside the service import contract"
        write(installed.installation / name, content)
        installed.manifest["files"].append(
            {
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "mode": 0o644,
            }
        )
    elif attack == "wrong-owner":
        with pytest.raises(application.ApplicationTrustError):
            application.verify_installed_application(root=installed.root, owner_uid=os.getuid() + 1)
        return
    if resign:
        installed.sign()
    with pytest.raises(application.ApplicationTrustError):
        verify(installed)


def test_application_detects_change_during_read(installed, monkeypatch):
    original = application.os.read
    changed = False

    def mutate(fd, size):
        nonlocal changed
        raw = original(fd, size)
        if not changed:
            changed = True
            with (installed.directory / "manifest.json").open("ab") as stream:
                stream.write(b" ")
        return raw

    monkeypatch.setattr(application.os, "read", mutate)
    with pytest.raises(application.ApplicationTrustError, match="podczas odczytu"):
        verify(installed)


def test_application_rejects_relative_root_and_wrong_platform(monkeypatch):
    with pytest.raises(application.ApplicationTrustError):
        application.verify_installed_application(root=Path("relative"))
    monkeypatch.setattr(application.sys, "platform", "win32")
    with pytest.raises(application.ApplicationTrustError, match="natywnego Linux"):
        application.verify_installed_application()


@pytest.mark.parametrize("limit", ["MAX_FILE", "MAX_TOTAL", "MAX_NODES"])
def test_application_payload_size_limits_refuse(installed, monkeypatch, limit):
    monkeypatch.setattr(application, limit, 1)
    with pytest.raises(application.ApplicationTrustError):
        verify(installed)


def test_package_helpers_use_external_signature_bootstrap():
    helpers = application._helpers()
    assert helpers["verify_ed25519"] is bootstrap.verify_ed25519
    assert callable(helpers["verify_installed_runtime"])


def test_isolated_helper_loads_only_protected_runtime_siblings(monkeypatch):
    sentinel = object()
    loaded = []
    monkeypatch.setattr(application, "__package__", "")
    monkeypatch.setattr(
        application,
        "_protected",
        lambda *_args: SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_nlink=1),
    )

    def load(path):
        loaded.append(Path(path).name)
        return {"verify_ed25519": sentinel, "verify_installed_runtime": sentinel}

    monkeypatch.setattr(application.runpy, "run_path", load)
    assert application._helpers() == {
        "verify_ed25519": sentinel,
        "verify_installed_runtime": sentinel,
    }
    assert loaded == ["freshness_runtime_bootstrap.py", "freshness_runtime.py"]


def test_application_refuses_unavailable_runtime(installed, monkeypatch):
    def unavailable(**_arguments):
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(
        application,
        "_helpers",
        lambda: {
            "verify_ed25519": bootstrap.verify_ed25519,
            "verify_installed_runtime": unavailable,
        },
    )
    with pytest.raises(application.ApplicationTrustError):
        verify(installed)


def test_application_refuses_nonobject_signed_manifest(installed):
    raw = b"[]"
    write(installed.directory / "manifest.json", raw)
    write(installed.directory / "manifest.sig", installed.key.sign(raw))
    with pytest.raises(application.ApplicationTrustError, match="dokument"):
        verify(installed)


def test_launcher_refuses_before_import_on_post_install_application_substitution(monkeypatch):
    monkeypatch.setattr(launcher, "_check_installation", lambda: None)
    for name in list(sys.modules):
        if name == "bot_core" or name.startswith("bot_core."):
            monkeypatch.delitem(sys.modules, name)
    imports = []
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: imports.append(name))

    def refuse():
        raise application.ApplicationTrustError("podmieniony plik po instalacji")

    monkeypatch.setattr(launcher, "_verify_application", refuse)
    with pytest.raises(application.ApplicationTrustError, match="po instalacji"):
        launcher.main()
    assert imports == []


def test_launcher_refuses_unprotected_application_bootstrap(monkeypatch):
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFREG | 0o666, st_uid=0, st_nlink=1),
    )
    with pytest.raises(PermissionError, match="Niechroniony bootstrap"):
        launcher._verify_application()


@pytest.mark.parametrize("mismatch", ["path", "inode", "sha256"])
def test_launcher_refuses_wrong_running_interpreter(monkeypatch, mismatch):
    content = b"model running interpreter"
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0, st_nlink=1),
    )
    # Model the Linux procfs boundary on every test OS. This checks the refusal
    # itself on Windows/macOS without requiring their host to expose /proc.
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path: SimpleNamespace(
            st_dev=1, st_ino=2 if mismatch == "inode" and path == Path("/proc/self/exe") else 1
        ),
    )
    monkeypatch.setattr(Path, "read_bytes", lambda _path: content)
    monkeypatch.setattr(
        launcher.runpy,
        "run_path",
        lambda *_args, **_kwargs: {
            "verify_installed_application": lambda: {
                "runtime": {
                    "executable": "/unapproved/python" if mismatch == "path" else sys.executable,
                    "base_executable": "/approved/python",
                    "base_executable_sha256": (
                        "0" * 64 if mismatch == "sha256" else hashlib.sha256(content).hexdigest()
                    ),
                }
            }
        },
    )
    with pytest.raises(PermissionError, match="interpreter"):
        launcher._verify_application()
