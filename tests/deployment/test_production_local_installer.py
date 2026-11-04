"""Hermetyczne odmowy i publikacja instalatora; bez kwalifikacji fizycznej."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass, replace
from importlib import _bootstrap_external as bytecode
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment import production_local_installer as installer

REPOSITORY = Path(__file__).resolve().parents[2]
REVISION = "a" * 40
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Hermetyczny test natywnych mechanizmów instalatora Linux",
)
PACKAGES = frozenset(
    {
        "python3",
        "python3-venv",
        "postgresql-16",
        "postgresql-client-16",
        "postgresql-common",
        "systemd",
    }
)


def _snapshot(root: Path) -> dict[str, tuple[int, bytes | str]]:
    """Treść i tryb; odczyt preflight nie może utworzyć nawet pliku blokady."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            value = os.readlink(path)
        elif stat.S_ISREG(metadata.st_mode):
            value = path.read_bytes()
        else:
            value = b""
        result[relative] = (metadata.st_mode, value)
    return result


@dataclass
class Context:
    config: installer.Config
    host: installer.HostState
    key: Ed25519PrivateKey
    document: dict

    def sign(self) -> None:
        content = json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()
        self.config.manifest.write_bytes(content)
        self.config.signature.write_bytes(self.key.sign(content))

    def preflight(self):
        return installer.preflight(self.config, host=self.host)

    def install(self, checkpoint: Callable[[str], None] | None = None):
        return installer.install(self.config, host=self.host, checkpoint=checkpoint)


@pytest.fixture
def context(tmp_path: Path, monkeypatch) -> Context:
    root = tmp_path / "host"
    for relative in (
        "usr/lib",
        "etc/systemd/system",
        "etc/postgresql/16/main/conf.d",
        "var/lib",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        directory.chmod(0o755)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    runtime = SimpleNamespace(
        environment_id="test-runtime",
        manifest_sha256="b" * 64,
        revision=REVISION,
        executable=Path("/opt/cryptohunter/freshness-runtime/releases/test-runtime/bin/python3.12"),
        base_executable=Path("/usr/bin/python3.12"),
        base_executable_sha256="c" * 64,
        packages={"cryptography": "50.0.2", "psycopg": "3.3.6"},
    )
    # Te testy modelują tylko granicę already-verified runtime. Rzeczywiste
    # podpisy/inwentarz/publikację i importy sprawdzają osobne testy runtime.
    from deployment import freshness_runtime_bootstrap

    # Obraz executora ma systemowy OpenSSL należący do nobody. Modelujemy
    # tylko tę granicę własności; podpis Ed25519 nadal jest rzeczywiście weryfikowany.
    monkeypatch.setattr(freshness_runtime_bootstrap, "_openssl", lambda: "/usr/bin/openssl")
    original_helpers = installer._runtime_helpers

    def helpers():
        return {**original_helpers(), "verify_installed_runtime": lambda **_options: runtime}

    monkeypatch.setattr(installer, "_runtime_helpers", helpers)
    paths = set(installer.REQUIRED_FILES)
    paths.add("bot_core/installer_probe.sh")
    files = []
    for relative in sorted(paths):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        original = REPOSITORY / relative
        content = original.read_bytes() if original.is_file() else b"# synthetic fixture\n"
        if relative == "deployment/systemd/cryptohunter-freshness-verifier.service":
            content = content.replace(
                b"@FRESHNESS_RUNTIME_PYTHON@", str(runtime.executable).encode()
            )
            content = content.replace(
                b"ExecStart=/usr/bin/python3 ", f"ExecStart={runtime.executable} ".encode()
            )
        if relative.endswith("installer_probe.sh"):
            content = b"#!/bin/sh\nexit 0\n"
        mode = 0o755 if relative.endswith("installer_probe.sh") else 0o644
        path.write_bytes(content)
        path.chmod(mode)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "mode": mode,
            }
        )
    key = Ed25519PrivateKey.generate()
    trusted_key = root / "etc/installer-release.pub"
    trusted_key.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    trusted_key.chmod(0o644)
    config = installer.Config(
        bundle=bundle,
        manifest=tmp_path / "release.json",
        signature=tmp_path / "release.sig",
        trusted_key=trusted_key,
        expected_revision=REVISION,
        root=root,
        owner_uid=os.getuid(),
    )
    users = (
        installer.User(
            "os_freshness_crypto_verifier", 901, 801, "/nonexistent", "/usr/sbin/nologin"
        ),
        installer.User("os_freshness_runtime", 902, 802, "/nonexistent", "/usr/sbin/nologin"),
    )
    groups = (
        installer.Group("os_freshness_crypto_verifier", 801, ()),
        installer.Group("os_freshness_runtime", 802, ()),
        installer.Group("freshness_verifier_ipc", 803, tuple(user.name for user in users)),
        installer.Group("postgres", os.getgid() or 1, ()),
    )
    host = installer.HostState(True, PACKAGES, True, True, users, groups)
    result = Context(
        config,
        host,
        key,
        {
            "schema_version": 2,
            "scope": "cryptohunter.freshness.substrate.v2",
            "runtime": {
                "environment_id": runtime.environment_id,
                "manifest_sha256": runtime.manifest_sha256,
                "source_revision": runtime.revision,
                "executable": str(runtime.executable),
                "base_executable": str(runtime.base_executable),
                "base_executable_sha256": runtime.base_executable_sha256,
            },
            "source_revision": REVISION,
            "files": files,
        },
    )
    result.sign()
    return result


def _refuses(context: Context, code: int) -> installer.InstallerError:
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as error:
        context.preflight()
    assert error.value.code == code
    assert _snapshot(context.config.root) == before
    return error.value


def test_positive_preflight_is_readonly_even_for_unprivileged_test_identity(context: Context):
    before = _snapshot(context.config.root)
    plan = context.preflight()
    assert plan.state == "fresh"
    assert plan.manifest_sha256 == hashlib.sha256(context.config.manifest.read_bytes()).hexdigest()
    assert _snapshot(context.config.root) == before
    assert not (context.config.root / "usr/lib/cryptohunter").exists()


def test_postgresql_package_without_required_cluster_refuses_before_first_change(context: Context):
    context.host = replace(context.host, cluster_online=False)
    events = []
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as error:
        context.install(events.append)
    assert error.value.code == 21
    assert events == []
    assert _snapshot(context.config.root) == before


@pytest.mark.parametrize("missing", sorted(PACKAGES))
def test_required_package_must_be_present(context: Context, missing: str):
    context.host = replace(context.host, packages=context.host.packages - {missing})
    _refuses(context, 20)


@pytest.mark.parametrize("field", ["systemd", "python_ready"])
def test_service_manager_and_isolated_interpreter_are_required(context: Context, field: str):
    context.host = replace(context.host, **{field: False})
    _refuses(context, 20)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda users: (replace(users[0], uid=0), users[1]),
        lambda users: (replace(users[0], uid=users[1].uid), users[1]),
        lambda users: (replace(users[0], uid=1001), users[1]),
        lambda users: (replace(users[0], shell="/bin/bash"), users[1]),
        lambda users: (replace(users[0], home="/home/operator"), users[1]),
        lambda users: (*users, installer.User("foreign", users[0].uid, 999, "/", "/bin/false")),
        lambda users: users[:1],
    ],
)
def test_user_name_uid_and_identity_conflicts_refuse(context: Context, mutate):
    context.host = replace(context.host, users=mutate(context.host.users))
    _refuses(context, 22)


@pytest.mark.parametrize("attack", ["foreign-member", "missing-member", "duplicate-gid", "missing"])
def test_system_group_collisions_and_ipc_membership_refuse(context: Context, attack: str):
    groups = list(context.host.groups)
    if attack == "foreign-member":
        groups[2] = replace(groups[2], members=(*groups[2].members, "foreign"))
    elif attack == "missing-member":
        groups[2] = replace(groups[2], members=groups[2].members[:1])
    elif attack == "duplicate-gid":
        groups.append(installer.Group("foreign", groups[2].gid, ()))
    else:
        groups.pop(2)
    context.host = replace(context.host, groups=tuple(groups))
    _refuses(context, 22)


def test_group_writable_protected_parent_refuses_without_repair(context: Context):
    parent = context.config.root / "usr/lib"
    parent.chmod(0o775)
    _refuses(context, 23)
    assert stat.S_IMODE(parent.stat().st_mode) == 0o775


def test_tampered_artifact_refuses_despite_original_valid_signature(context: Context):
    path = context.config.bundle / "bot_core/freshness_verifier_service.py"
    path.write_bytes(path.read_bytes() + b"\n# tampered\n")
    _refuses(context, 12)


@pytest.mark.parametrize(
    "relative",
    [
        "bot_core/linux_service_ipc.py",
        "bot_core/linux_service_postgresql.py",
        "bot_core/production_signer_ipc.py",
        "bot_core/freshness_verifier_service.py",
        "bot_core/freshness_semantic_verifier.py",
    ],
)
def test_new_signature_does_not_approve_changed_reviewed_source(context, relative):
    file = context.config.bundle / relative
    content = file.read_bytes() + b"\n# independently unreviewed revision\n"
    file.write_bytes(content)
    record = next(item for item in context.document["files"] if item["path"] == relative)
    record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    context.sign()
    _refuses(context, 12)


def test_missing_required_file_cannot_be_authorized_by_shorter_signed_inventory(context):
    relative = "bot_core/freshness_semantic_verifier.py"
    (context.config.bundle / relative).unlink()
    context.document["files"] = [
        item for item in context.document["files"] if item["path"] != relative
    ]
    context.sign()
    _refuses(context, 12)


def test_wrong_release_key_cannot_authorize_bundle(context: Context):
    context.key = Ed25519PrivateKey.generate()
    context.sign()
    _refuses(context, 11)


def test_manifest_bytes_are_signed_without_line_ending_normalization(context: Context):
    context.config.manifest.write_bytes(context.config.manifest.read_bytes() + b"\r\n")
    _refuses(context, 11)


def test_approved_revision_must_match_independent_expected_revision(context: Context):
    context.config = replace(context.config, expected_revision="b" * 40)
    _refuses(context, 10)


@pytest.mark.parametrize("relative", ["/etc/passwd", "../outside", "bot_core/../outside"])
def test_signed_unsafe_inventory_paths_are_still_rejected(context: Context, relative: str):
    context.document["files"][0]["path"] = relative
    context.sign()
    _refuses(context, 23)


def test_setuid_mode_cannot_be_authorized_by_manifest(context: Context):
    context.document["files"][0]["mode"] = 0o4755
    context.sign()
    _refuses(context, 10)


def test_unlisted_payload_file_is_not_implicitly_installed(context: Context):
    (context.config.bundle / "bot_core/unreviewed.py").write_text(
        "raise RuntimeError('unreviewed')\n"
    )
    _refuses(context, 12)


@pytest.mark.parametrize("attack", ["artifact-symlink", "parent-symlink", "hardlink"])
def test_source_links_are_rejected_even_when_file_bytes_match(context: Context, attack: str):
    source = context.config.bundle / "bot_core/freshness_verifier_service.py"
    outside = context.config.bundle.parent / "outside.py"
    if attack == "parent-symlink":
        real = context.config.bundle / "bot_core"
        moved = context.config.bundle.parent / "real-bot-core"
        real.rename(moved)
        real.symlink_to(moved, target_is_directory=True)
    else:
        source.rename(outside)
        if attack == "artifact-symlink":
            source.symlink_to(outside)
        else:
            source.hardlink_to(outside)
    _refuses(context, 12 if attack == "hardlink" else 23)


def test_symlink_target_never_touches_foreign_data(context: Context):
    foreign = context.config.root / "foreign"
    foreign.mkdir()
    sentinel = foreign / "keep.dat"
    sentinel.write_bytes(b"DO NOT DELETE")
    (context.config.root / "usr/lib/cryptohunter").symlink_to(foreign, target_is_directory=True)
    error = _refuses(context, 24)
    assert error.reason
    assert sentinel.read_bytes() == b"DO NOT DELETE"


def test_existing_unmanaged_installation_is_preserved_and_not_adopted(context: Context):
    old = context.config.root / "usr/lib/cryptohunter"
    old.mkdir()
    saved = old / "existing-data.bin"
    saved.write_bytes(b"existing deployment and data")
    _refuses(context, 24)
    assert saved.read_bytes() == b"existing deployment and data"


def test_partial_existing_configuration_refuses_without_overwriting(context: Context):
    config = context.config.root / "etc/systemd/system/cryptohunter-freshness-verifier.service"
    config.write_bytes(b"operator-owned previous unit")
    _refuses(context, 24)
    assert config.read_bytes() == b"operator-owned previous unit"


def test_install_verifies_exact_payload_and_preserves_executable_modes(context: Context):
    plan = context.install()
    assert plan.state == "installed"
    target = context.config.root / "usr/lib/cryptohunter"
    for specification in context.document["files"]:
        installed = target / specification["path"]
        assert installed.is_file() and not installed.is_symlink()
        assert (
            installed.read_bytes() == (context.config.bundle / specification["path"]).read_bytes()
        )
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == specification["sha256"]
        assert stat.S_IMODE(installed.stat().st_mode) == specification["mode"]
    assert not (context.config.root / "etc/systemd/system/multi-user.target.wants").exists()


def test_install_retains_exact_signed_application_approval_separate_from_private_journal(context):
    plan = context.install()
    metadata = context.config.root / installer.APPLICATION
    assert stat.S_IMODE(metadata.stat().st_mode) == 0o755
    expected = {
        "manifest.json": context.config.manifest.read_bytes(),
        "manifest.sig": context.config.signature.read_bytes(),
        "authority.pub": context.config.trusted_key.read_bytes(),
        "binding.json": installer._application_binding(plan),
    }
    assert set(path.name for path in metadata.iterdir()) == set(expected)
    for name, content in expected.items():
        path = metadata / name
        assert path.read_bytes() == content
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
    binding = json.loads(expected["binding.json"])
    assert binding["scope"] == installer.SCOPE
    assert binding["manifest_sha256"] == plan.manifest_sha256
    assert binding["runtime"] == plan.runtime
    assert stat.S_IMODE((context.config.root / installer.STATE).stat().st_mode) == 0o700


def test_retained_installer_approval_verifies_only_signed_application_scope(context, monkeypatch):
    from deployment import freshness_application

    context.install()
    # Wyłącznie model already-verified runtime z fixture. Podpis aplikacji,
    # zachowane metadane i cały inwentarz są weryfikowane rzeczywiście.
    runtime_helpers = installer._runtime_helpers()
    monkeypatch.setattr(
        freshness_application,
        "_helpers",
        lambda: {
            "verify_ed25519": runtime_helpers["verify_ed25519"],
            "verify_installed_runtime": runtime_helpers["verify_installed_runtime"],
        },
    )
    proof = freshness_application.verify_installed_application(
        root=context.config.root, owner_uid=context.config.owner_uid
    )
    assert proof["scope"] == installer.SCOPE
    assert proof["qualification_profile"] == "SIGNED_FRESHNESS_SUBSTRATE"
    assert proof["production_qualification"] == "WITHHELD"
    assert (
        proof["manifest_sha256"] == hashlib.sha256(context.config.manifest.read_bytes()).hexdigest()
    )
    assert proof["source_revision"] == REVISION


def test_existing_application_approval_is_not_silently_adopted_or_overwritten(context):
    metadata = context.config.root / installer.APPLICATION
    metadata.mkdir(parents=True)
    metadata.parent.chmod(0o755)
    metadata.chmod(0o755)
    sentinel = metadata / "manifest.json"
    sentinel.write_bytes(b"administrator-owned earlier approval")
    _refuses(context, 24)
    assert sentinel.read_bytes() == b"administrator-owned earlier approval"


@pytest.mark.parametrize("name", ["manifest.json", "manifest.sig", "authority.pub", "binding.json"])
def test_retained_application_approval_drift_prevents_idempotent_adoption(context, name):
    context.install()
    metadata = context.config.root / installer.APPLICATION / name
    metadata.write_bytes(metadata.read_bytes() + b"unapproved")
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as error:
        context.install()
    assert error.value.code == 24
    assert _snapshot(context.config.root) == before


def test_application_binding_is_not_published_after_failed_metadata_write(context, monkeypatch):
    original = installer._Tree.write_new

    def fail_signature(tree, path, data, mode, gid=None):
        if path == f"{installer.APPLICATION}/manifest.sig":
            raise OSError("synthetic interrupted approval persistence")
        return original(tree, path, data, mode, gid)

    monkeypatch.setattr(installer._Tree, "write_new", fail_signature)
    with pytest.raises(installer.InstallerError) as error:
        context.install()
    assert error.value.code == 30
    assert not (context.config.root / installer.APPLICATION / "binding.json").exists()
    with pytest.raises(installer.InstallerError) as retry:
        context.install()
    assert retry.value.code == 24


def test_reinstall_of_identical_complete_state_is_readonly(context: Context):
    context.install()
    before = _snapshot(context.config.root)
    events = []
    plan = context.install(events.append)
    assert plan.state == "installed"
    assert events == []
    assert _snapshot(context.config.root) == before


@pytest.mark.parametrize("phase", ["staged", "code-published"])
def test_interruption_retains_recovery_evidence_and_refuses_automatic_retry(
    context: Context, phase: str
):
    def stop(stage: str) -> None:
        if stage == phase:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(installer.InstallerError) as error:
        context.install(stop)
    assert error.value.code == 30
    after_failure = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as retry:
        context.install()
    assert retry.value.code == 24
    assert _snapshot(context.config.root) == after_failure
    if phase == "staged":
        assert not (context.config.root / "usr/lib/cryptohunter").exists()


@pytest.mark.parametrize("attack", ["bytes", "mode", "extra-file", "symlink"])
def test_installed_integrity_drift_is_detected_without_automatic_repair(
    context: Context, attack: str
):
    context.install()
    installed = context.config.root / "usr/lib/cryptohunter/bot_core/freshness_verifier_service.py"
    if attack == "bytes":
        installed.write_bytes(b"tampered installed code")
    elif attack == "mode":
        installed.chmod(0o666)
    elif attack == "extra-file":
        (installed.parent / "unexpected.py").write_bytes(b"unreviewed")
    else:
        content = installed.read_bytes()
        installed.unlink()
        outside = context.config.root / "outside.py"
        outside.write_bytes(content)
        installed.symlink_to(outside)
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError):
        context.install()
    assert _snapshot(context.config.root) == before


def test_install_rechecks_artifact_modified_after_successful_preflight(context: Context):
    context.preflight()
    (context.config.bundle / "bot_core/freshness_verifier_service.py").write_bytes(b"changed later")
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as error:
        context.install()
    assert error.value.code == 12
    assert _snapshot(context.config.root) == before


def test_source_changed_during_staging_is_refused_before_code_publication(context):
    def change_source(phase):
        if phase == "staged":
            source = context.config.bundle / "bot_core/freshness_verifier_service.py"
            source.write_bytes(source.read_bytes() + b"\n# changed after internal preflight\n")

    with pytest.raises(installer.InstallerError) as error:
        context.install(change_source)
    assert error.value.code == 12
    assert not (context.config.root / installer.CODE).exists()
    assert not (context.config.root / installer.APPLICATION / "binding.json").exists()


def test_published_systemd_postgresql_and_ipc_contracts_keep_service_separation(context: Context):
    context.install()
    root = context.config.root
    unit = (root / "etc/systemd/system/cryptohunter-freshness-verifier.service").read_text()
    for exact in (
        "User=os_freshness_crypto_verifier\n",
        "Group=freshness_verifier_ipc\n",
        "RuntimeDirectory=cryptohunter/freshness\n",
        "RuntimeDirectoryMode=0750\n",
        "NoNewPrivileges=yes\n",
        "ProtectSystem=strict\n",
        "RestrictAddressFamilies=AF_UNIX\n",
        "CapabilityBoundingSet=\n",
        "AmbientCapabilities=\n",
    ):
        assert exact in unit
    postgresql = root / "etc/cryptohunter/postgresql"
    for name in ("pg_hba.conf", "pg_ident.conf"):
        assert (postgresql / name).read_bytes() == (
            REPOSITORY / "deployment/postgresql" / name
        ).read_bytes()
    hba = (postgresql / "pg_hba.conf").read_text()
    assert "local freshness_gate freshness_crypto_verifier peer map=freshness_authority_map" in hba
    assert "local freshness_gate freshness_runtime peer map=freshness_authority_map" in hba
    assert "local freshness_gate all reject" in hba
    assert "local all all reject" in hba
    assert " trust" not in hba
    configuration = root / "etc/postgresql/16/main/conf.d/cryptohunter-production-local.conf"
    assert (
        configuration.read_bytes()
        == (REPOSITORY / "deployment/postgresql/production-local.conf").read_bytes()
    )
    assert "listen_addresses = ''" in configuration.read_text()


def test_isolated_launcher_ignores_hostile_cwd_and_pythonpath_without_running_package_init(
    tmp_path,
):
    import subprocess

    installation = tmp_path / "installed"
    package = installation / "bot_core"
    package.mkdir(parents=True)
    marker = tmp_path / "exact-service-ran"
    (package / "__init__.py").write_text("raise AssertionError('package init executed')\n")
    (package / "freshness_verifier_service.py").write_text(
        "from pathlib import Path\n"
        f"def main():\n    Path({str(marker)!r}).write_text('fixed installation')\n"
    )
    service = package / "freshness_verifier_service.py"
    cache = (
        package / "__pycache__" / f"freshness_verifier_service.{sys.implementation.cache_tag}.pyc"
    )
    cache.parent.mkdir()
    service_metadata = service.stat()
    cached_sentinel = bytecode._code_to_timestamp_pyc(
        compile("raise AssertionError('SERVICE_PYC_SENTINEL_EXECUTED')", str(service), "exec"),
        mtime=int(service_metadata.st_mtime),
        source_size=service_metadata.st_size,
    )
    cache.write_bytes(cached_sentinel)
    hostile = tmp_path / "hostile"
    (hostile / "bot_core").mkdir(parents=True)
    (hostile / "bot_core/__init__.py").write_text("raise AssertionError('hostile import')\n")
    source = REPOSITORY / "deployment/freshness_verifier_entrypoint.py"
    program = (
        "import importlib.util\n"
        "from pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('verified_launcher', {str(source)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"module.INSTALLATION = Path({str(installation)!r})\n"
        # Wyłącznie rootless test importów: ochronę metadanych sprawdza osobny test.
        "module._check_installation = lambda: None\n"
        "module._verify_application = lambda: None\n"
        "raise SystemExit(module.main())\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", program],
        cwd=hostile,
        env={**os.environ, "PYTHONPATH": str(hostile)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "fixed installation"
    assert list(installation.rglob("__pycache__")) == [cache.parent]
    assert cache.read_bytes() == cached_sentinel
    assert "SERVICE_PYC_SENTINEL_EXECUTED" not in result.stderr


def test_launcher_rejects_unprotected_installation_before_importing_service(tmp_path, monkeypatch):
    from deployment import freshness_verifier_entrypoint as launcher

    installation = tmp_path / "installed"
    (installation / "bot_core").mkdir(parents=True)
    (installation / "deployment").mkdir()
    (installation / "bot_core").chmod(0o777)
    monkeypatch.setattr(launcher, "INSTALLATION", installation)
    with pytest.raises(PermissionError, match="Niechroniona"):
        launcher._check_installation()


def test_launcher_refuses_a_preloaded_bot_core_namespace(monkeypatch):
    from types import ModuleType

    from deployment import freshness_verifier_entrypoint as launcher

    monkeypatch.setattr(launcher, "_check_installation", lambda: None)
    monkeypatch.setitem(sys.modules, "bot_core", ModuleType("bot_core"))
    with pytest.raises(PermissionError, match="przed punktem wejścia"):
        launcher.main()


@pytest.fixture
def host_probe(monkeypatch):
    import grp
    import pwd as account_database
    from types import SimpleNamespace

    from deployment import production_local_installer_host as probe

    queries = []
    responses = {
        "cluster": (
            "16 main 5432 online postgres /var/lib/postgresql/16/main /var/log/postgresql.log"
        ),
        "unit": "loaded",
        "include": "include_dir = 'conf.d'",
        "python": "",
        "manager": "257",
        "active": None,
        "verifier_unit": "LoadState=not-found\nFragmentPath=\nDropInPaths=",
        **{package: "ii" for package in PACKAGES},
    }
    monkeypatch.setattr(probe.sys, "platform", "linux")
    users = [
        SimpleNamespace(
            pw_name=name,
            pw_uid=uid,
            pw_gid=gid,
            pw_dir="/nonexistent",
            pw_shell="/usr/sbin/nologin",
        )
        for name, uid, gid in [
            ("os_freshness_crypto_verifier", 901, 801),
            ("os_freshness_runtime", 902, 802),
        ]
    ]
    groups = [
        SimpleNamespace(gr_name="os_freshness_crypto_verifier", gr_gid=801, gr_mem=[]),
        SimpleNamespace(gr_name="os_freshness_runtime", gr_gid=802, gr_mem=[]),
        SimpleNamespace(
            gr_name="freshness_verifier_ipc", gr_gid=803, gr_mem=[user.pw_name for user in users]
        ),
        SimpleNamespace(gr_name="postgres", gr_gid=os.getgid() or 1, gr_mem=[]),
    ]
    monkeypatch.setattr(account_database, "getpwall", lambda: users)
    monkeypatch.setattr(
        account_database,
        "getpwnam",
        lambda name: next(item for item in users if item.pw_name == name),
    )
    monkeypatch.setattr(
        account_database, "getpwuid", lambda uid: next(item for item in users if item.pw_uid == uid)
    )
    monkeypatch.setattr(grp, "getgrall", lambda: groups)
    monkeypatch.setattr(
        grp, "getgrnam", lambda name: next(item for item in groups if item.gr_name == name)
    )
    monkeypatch.setattr(
        grp, "getgrgid", lambda gid: next(item for item in groups if item.gr_gid == gid)
    )
    monkeypatch.setattr(probe.os, "getgrouplist", lambda _name, gid: [gid, 803])
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        return "systemd\n" if path == Path("/proc/1/comm") else original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def query(arguments):
        queries.append(arguments)
        if arguments[0].endswith("dpkg-query"):
            return responses[arguments[-1]]
        if arguments[0].endswith("pg_lsclusters"):
            return responses["cluster"]
        if arguments[0].endswith("pg_conftool"):
            return responses["include"]
        if arguments[0].endswith(("python3", "python3.12")):
            return responses["python"]
        if "is-active" in arguments:
            return responses["active"]
        if "postgresql@16-main.service" in arguments:
            return responses["unit"]
        if "--property=Version" in arguments:
            return responses["manager"]
        if "--property=LoadState,FragmentPath,DropInPaths" in arguments:
            return responses["verifier_unit"]
        raise AssertionError(f"unexpected host command: {arguments}")

    monkeypatch.setattr(probe, "_query", query)
    return probe, responses, queries


def test_native_host_probe_queries_only_readonly_commands(host_probe):
    probe, _responses, queries = host_probe
    state = probe.inspect_host(
        SimpleNamespace(
            executable=Path("/approved/bin/python3.12"), packages={"cryptography": "50.0.2"}
        )
    )
    assert state.systemd and state.cluster_online and state.python_ready
    assert state.packages == PACKAGES
    assert tuple(user.name for user in state.users) == probe.USERS
    assert len(state.groups) == 4
    assert state.nss_ready
    assert state.unit_state == "not-found"
    assert state.unit_fragment == "" and state.unit_dropins == ()
    assert not state.verifier_active
    assert queries
    assert not any(
        operation in {"start", "restart", "reload", "enable", "create", "drop"}
        for arguments in queries
        for operation in arguments
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "Ver Cluster Port Status Owner Data directory Log file",
        "15 main 5432 online postgres /data /log",
        "16 foreign 5432 online postgres /data /log",
        "16 main 5433 online postgres /data /log",
        "16 main 5432 down postgres /data /log",
        "16 main 5432 online foreign /data /log",
    ],
)
def test_native_host_probe_never_infers_cluster_from_package_presence(host_probe, response):
    probe, responses, _queries = host_probe
    responses["cluster"] = response
    state = probe.inspect_host()
    assert state.packages == PACKAGES
    assert not state.cluster_online


@pytest.mark.parametrize("field", ["unit", "include", "python", "manager"])
def test_native_host_probe_inaccessible_dependencies_remain_unready(host_probe, field):
    probe, responses, _queries = host_probe
    responses[field] = None
    state = probe.inspect_host()
    if field in {"unit", "include"}:
        assert not state.cluster_online
    elif field == "python":
        assert not state.python_ready
    else:
        assert not state.systemd


def test_native_host_probe_reports_running_verifier_and_missing_package(host_probe):
    probe, responses, _queries = host_probe
    responses["active"] = "active"
    responses["postgresql-16"] = None
    state = probe.inspect_host()
    assert state.verifier_active
    assert "postgresql-16" not in state.packages


def test_host_probe_refuses_unsupported_platform_without_posix_queries(host_probe, monkeypatch):
    probe, _responses, queries = host_probe
    monkeypatch.setattr(probe.sys, "platform", "win32")
    state = probe.inspect_host()
    assert state == probe.HostState(False, frozenset(), False, False, (), ())
    assert queries == []


def test_host_probe_fails_closed_when_native_metadata_is_unreadable(host_probe, monkeypatch):
    probe, _responses, queries = host_probe

    def refuse(*args, **kwargs):
        raise PermissionError("synthetic unreadable procfs")

    monkeypatch.setattr(Path, "read_text", refuse)
    assert probe.inspect_host() == probe.HostState(False, frozenset(), False, False, (), ())
    assert queries == []


@pytest.mark.parametrize("returncode", [0, 1])
def test_host_query_has_bounded_sanitized_subprocess_boundary(monkeypatch, returncode):
    from types import SimpleNamespace

    from deployment import production_local_installer_host as probe

    observed = []

    def run(arguments, **options):
        observed.append((arguments, options))
        return SimpleNamespace(returncode=returncode, stdout="  observed\n")

    monkeypatch.setattr(probe.subprocess, "run", run)
    result = probe._query(["/usr/bin/systemctl", "show"])
    assert result == ("observed" if returncode == 0 else None)
    arguments, options = observed[0]
    assert arguments == ["/usr/bin/systemctl", "show"]
    assert options["timeout"] == 15
    assert options["cwd"] == "/"
    assert "shell" not in options or options["shell"] is False
    assert options["stderr"] == probe.subprocess.DEVNULL
    assert set(options["env"]) == {"PATH", "LANG", "LC_ALL"}
    assert options["env"]["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin"


@pytest.mark.parametrize("failure", ["os", "timeout", "unicode"])
def test_host_query_does_not_promote_timeout_or_malformed_output_to_success(monkeypatch, failure):
    from deployment import production_local_installer_host as probe

    def run(*args, **kwargs):
        if failure == "os":
            raise FileNotFoundError("missing native utility")
        if failure == "timeout":
            raise probe.subprocess.TimeoutExpired("query", 15)
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "malformed output")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe._query(["/usr/bin/systemctl", "show"]) is None


def test_installed_writable_empty_directory_cannot_hide_from_file_inventory(context: Context):
    context.install()
    injected = context.config.root / "usr/lib/cryptohunter/bot_core/injected"
    injected.mkdir()
    injected.chmod(0o777)
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError):
        context.install()
    assert _snapshot(context.config.root) == before


def test_unsigned_empty_payload_directory_is_not_implicitly_adopted(context: Context):
    (context.config.bundle / "bot_core/unsigned-empty").mkdir()
    _refuses(context, 12)


def test_launcher_constructs_only_fixed_namespace_and_imports_exact_service(monkeypatch):
    from types import SimpleNamespace

    from deployment import freshness_verifier_entrypoint as launcher

    original_paths = list(sys.path)
    imports, calls = [], []
    with monkeypatch.context() as patch:
        patch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
        patch.setattr(launcher, "_check_installation", lambda: None)
        patch.setattr(launcher, "_verify_application", lambda: None)
        for name in list(sys.modules):
            if name == "bot_core" or name.startswith("bot_core."):
                patch.delitem(sys.modules, name)
        # Przywróć także namespace dodany przez main(), po zakończeniu testu.
        patch.setitem(sys.modules, "bot_core", None)
        patch.delitem(sys.modules, "bot_core")

        def import_exact(name):
            imports.append(name)
            return SimpleNamespace(main=lambda: calls.append("main"))

        patch.setattr(launcher.importlib, "import_module", import_exact)
        assert launcher.main() == 0
        package = sys.modules["bot_core"]
        assert package.__path__ == ["/usr/lib/cryptohunter/bot_core"]
        assert package.__spec__.submodule_search_locations == package.__path__
    assert imports == ["bot_core.freshness_verifier_service"]
    assert calls == ["main"]
    assert sys.path == original_paths


def test_launcher_checks_its_own_exact_source_path_after_directory_guards(monkeypatch):
    from types import SimpleNamespace

    from deployment import freshness_verifier_entrypoint as launcher

    metadata = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", lambda _path: metadata)
        patch.setattr(launcher, "INSTALLATION", REPOSITORY)
        launcher._check_installation()
        patch.setattr(launcher, "INSTALLATION", Path("/a/different/installation"))
        with pytest.raises(PermissionError, match="Niezgodna ścieżka"):
            launcher._check_installation()


def _arguments(context: Context, action: str) -> list[str]:
    return [
        action,
        "--bundle",
        str(context.config.bundle),
        "--manifest",
        str(context.config.manifest),
        "--signature",
        str(context.config.signature),
        "--trusted-key",
        str(context.config.trusted_key),
        "--expected-revision",
        REVISION,
    ]


@pytest.mark.parametrize("action,result", [("--preflight", "READY"), ("--install", "INSTALLED")])
def test_cli_reports_scoped_success_without_qualifying_product(
    context, monkeypatch, capsys, action, result
):
    monkeypatch.setattr(installer, "Config", lambda *_args: context.config)
    monkeypatch.setattr(installer, "inspect_host", lambda *_args: context.host)
    assert installer.main(_arguments(context, action)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == result
    assert report["scope"] == "cryptohunter.freshness.substrate.v2"
    assert report["production_qualification"] == "WITHHELD"
    assert "LINUX_PRODUCTION_READY" not in report


def test_cli_exposes_safe_failure_code_without_host_stderr_or_secret(context, monkeypatch, capsys):
    monkeypatch.setattr(installer, "Config", lambda *_args: context.config)
    monkeypatch.setattr(
        installer, "inspect_host", lambda *_args: replace(context.host, cluster_online=False)
    )
    before = _snapshot(context.config.root)
    assert installer.main(_arguments(context, "--preflight")) == 21
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "REFUSED" and report["code"] == 21
    assert report["production_qualification"] == "WITHHELD"
    assert _snapshot(context.config.root) == before


def test_cli_interrupt_returns_explicit_failure_without_traceback(context, monkeypatch, capsys):
    monkeypatch.setattr(installer, "Config", lambda *_args: context.config)

    def interrupt(_config):
        raise KeyboardInterrupt

    monkeypatch.setattr(installer, "preflight", interrupt)
    assert installer.main(_arguments(context, "--preflight")) == 30
    assert json.loads(capsys.readouterr().out)["code"] == 30


@pytest.mark.parametrize("extra", [["--root", "/tmp"], ["--owner-uid", "1000"]])
def test_cli_does_not_expose_rootless_test_overrides(context, extra):
    with pytest.raises(SystemExit) as error:
        installer.main([*_arguments(context, "--install"), *extra])
    assert error.value.code == 2


def test_duplicate_manifest_fields_are_rejected_after_valid_signature(context):
    raw = context.config.manifest.read_bytes().replace(
        b'"schema_version":2', b'"schema_version":2,"schema_version":2', 1
    )
    context.config.manifest.write_bytes(raw)
    context.config.signature.write_bytes(context.key.sign(raw))
    _refuses(context, 10)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update(schema_version=True),
        lambda item: item.update(scope="cryptohunter.linux.release.v1"),
        lambda item: item.update(source_revision="A" * 40),
        lambda item: item.update(files=[]),
        lambda item: item.update(unexpected="not in signed contract"),
        lambda item: item["files"].append(dict(item["files"][0])),
        lambda item: item["files"][0].update(size=True),
        lambda item: item["files"][0].update(sha256="A" * 64),
        lambda item: item["files"][0].update(mode=True),
        lambda item: item["files"][0].update(unexpected="field"),
    ],
)
def test_signed_malformed_manifest_still_fails_closed(context, mutate):
    mutate(context.document)
    context.sign()
    _refuses(context, 10)


def test_complete_signed_inventory_must_contain_required_service_closure(context):
    removed = context.document["files"].pop(0)
    (context.config.bundle / removed["path"]).unlink()
    context.sign()
    _refuses(context, 12)


def test_prepared_trusted_key_must_keep_protected_permissions(context):
    context.config.trusted_key.chmod(0o666)
    _refuses(context, 23)


def test_fifo_in_payload_cannot_block_or_enter_installation(context):
    os.mkfifo(context.config.bundle / "bot_core/unexpected-fifo")
    _refuses(context, 12)


@pytest.mark.parametrize("state", ["loaded", "masked", "error", "unknown"])
def test_preexisting_or_unobservable_verifier_unit_refuses_fresh_install(context, state):
    context.host = replace(context.host, unit_state=state)
    _refuses(context, 24)


def test_preexisting_loaded_vendor_unit_refuses_even_when_inactive(context):
    context.host = replace(
        context.host,
        verifier_active=False,
        unit_state="loaded",
        unit_fragment="/usr/lib/systemd/system/cryptohunter-freshness-verifier.service",
    )
    _refuses(context, 24)


def test_unit_dropin_cannot_override_service_isolation(context):
    dropin = context.config.root / "etc/systemd/system/cryptohunter-freshness-verifier.service.d"
    dropin.mkdir()
    (dropin / "override.conf").write_text("[Service]\nUser=root\n")
    _refuses(context, 24)


def test_unobservable_nss_identity_refuses_even_with_consistent_enumeration(context):
    context.host = replace(context.host, nss_ready=False)
    _refuses(context, 22)


@pytest.mark.parametrize("attack", ["postgres-root", "postgres-alias", "foreign-primary-ipc"])
def test_postgresql_group_and_hidden_primary_ipc_membership_conflicts_refuse(context, attack):
    if attack == "postgres-root":
        groups = (*context.host.groups[:-1], replace(context.host.groups[-1], gid=0))
        context.host = replace(context.host, groups=groups)
    elif attack == "postgres-alias":
        groups = (*context.host.groups, installer.Group("alias", context.host.groups[-1].gid, ()))
        context.host = replace(context.host, groups=groups)
    else:
        users = (*context.host.users, installer.User("foreign", 903, 803, "/", "/bin/false"))
        context.host = replace(context.host, users=users)
    _refuses(context, 22)


@pytest.mark.parametrize("relative", ["usr", "etc/cryptohunter", "etc/cryptohunter/postgresql"])
def test_unreadable_existing_parent_is_rejected_without_chmod(context, relative):
    parent = context.config.root / relative
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o700)
    _refuses(context, 23)
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("response", [None, "", "LoadState=not-found", "malformed response"])
def test_missing_unit_observation_is_unknown_and_not_assumed_absent(host_probe, response):
    probe, responses, _queries = host_probe
    responses["verifier_unit"] = response
    assert probe.inspect_host().unit_state == "unknown"


def test_native_unit_observation_retains_vendor_fragment_and_dropins(host_probe):
    probe, responses, _queries = host_probe
    responses["verifier_unit"] = (
        "LoadState=loaded\n"
        "FragmentPath=/usr/lib/systemd/system/cryptohunter-freshness-verifier.service\n"
        "DropInPaths=/etc/systemd/system/cryptohunter-freshness-verifier.service.d/override.conf"
    )
    state = probe.inspect_host()
    assert state.unit_state == "loaded"
    assert state.unit_fragment.startswith("/usr/lib/systemd/system/")
    assert state.unit_dropins == (
        "/etc/systemd/system/cryptohunter-freshness-verifier.service.d/override.conf",
    )


@pytest.mark.parametrize("attack", ["name-missing", "reverse-mismatch", "hidden-group"])
def test_native_nss_direct_lookup_must_match_enumeration(host_probe, monkeypatch, attack):
    import pwd as account_database
    from types import SimpleNamespace

    probe, _responses, _queries = host_probe
    if attack == "name-missing":

        def missing(_name):
            raise KeyError("missing prepared user")

        monkeypatch.setattr(account_database, "getpwnam", missing)
    elif attack == "reverse-mismatch":
        monkeypatch.setattr(
            account_database, "getpwuid", lambda _uid: SimpleNamespace(pw_name="foreign")
        )
    else:
        monkeypatch.setattr(probe.os, "getgrouplist", lambda _name, gid: [gid, 803, 999])
    assert not probe.inspect_host().nss_ready


def test_isolated_installer_bootstrap_ignores_matching_timestamp_helper_bytecode(tmp_path):
    import shutil
    import subprocess

    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    script = bootstrap / "production_local_installer.py"
    helper = bootstrap / "production_local_installer_host.py"
    shutil.copyfile(REPOSITORY / "deployment" / script.name, script)
    shutil.copyfile(REPOSITORY / "deployment" / helper.name, helper)
    cache = helper.parent / "__pycache__" / f"{helper.stem}.{sys.implementation.cache_tag}.pyc"
    cache.parent.mkdir()
    metadata = helper.stat()
    sentinel = compile("raise RuntimeError('PYC_SENTINEL_EXECUTED')", str(helper), "exec")
    cache.write_bytes(
        bytecode._code_to_timestamp_pyc(
            sentinel, mtime=int(metadata.st_mtime), source_size=metadata.st_size
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--preflight" in result.stdout
    assert "PYC_SENTINEL_EXECUTED" not in result.stderr


@pytest.mark.parametrize("reader", ["input", "protected-tree"])
def test_fifo_inputs_refuse_without_waiting_for_writer(context, reader):
    import subprocess

    fifo = context.config.root / "etc/input-fifo"
    os.mkfifo(fifo)
    script = REPOSITORY / "deployment/production_local_installer.py"
    if reader == "input":
        operation = f"module['_read_input'](Path({str(fifo)!r}), 64)"
    else:
        operation = (
            f"tree = module['_Tree'](Path({str(context.config.root)!r}), {os.getuid()})\n"
            "tree.read('etc/input-fifo')"
        )
    program = (
        "import runpy\nfrom pathlib import Path\n"
        f"module = runpy.run_path({str(script)!r}, run_name='fifo_reader_test')\n"
        "try:\n"
        + "\n".join("    " + line for line in operation.splitlines())
        + "\nexcept module['InstallerError'] as exc:\n    print(exc.code)\n"
        "else:\n    raise AssertionError('FIFO was accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", program],
        text=True,
        capture_output=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) in {12, 23}


@pytest.mark.parametrize("returncode", [0, 1, 4])
def test_host_query_accepts_explicit_not_found_unit_observation_only(monkeypatch, returncode):
    from types import SimpleNamespace

    from deployment import production_local_installer_host as probe

    output = "LoadState=not-found\nFragmentPath=\nDropInPaths=\n"
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode, stdout=output),
    )
    assert probe._query(probe.UNIT_QUERY) == output.strip()


@pytest.mark.parametrize(
    "output",
    [
        "",
        "LoadState=not-found",
        "LoadState=loaded\nFragmentPath=\nDropInPaths=",
        "LoadState=error\nFragmentPath=\nDropInPaths=",
    ],
)
def test_failed_unit_query_cannot_convert_partial_or_loaded_output_to_absence(monkeypatch, output):
    from types import SimpleNamespace

    from deployment import production_local_installer_host as probe

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=4, stdout=output),
    )
    assert probe._query(probe.UNIT_QUERY) is None


def test_not_found_special_case_is_restricted_to_exact_unit_query(monkeypatch):
    from types import SimpleNamespace

    from deployment import production_local_installer_host as probe

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=4, stdout="LoadState=not-found\nFragmentPath=\nDropInPaths="
        ),
    )
    assert probe._query(["/usr/bin/systemctl", "show", "foreign.service"]) is None


def test_extra_recovery_state_file_prevents_idempotent_adoption(context):
    context.install()
    state = context.config.root / "var/lib/cryptohunter-installer"
    (state / "journal.next").write_bytes(b"unfinished previous transition")
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError):
        context.install()
    assert _snapshot(context.config.root) == before


def test_host_probe_without_verified_runtime_never_uses_another_python(host_probe):
    probe, _responses, queries = host_probe
    assert not probe.inspect_host().python_ready
    assert not any("python" in arguments[0] for arguments in queries)


@pytest.mark.parametrize(
    "field",
    [
        "environment_id",
        "manifest_sha256",
        "executable",
        "base_executable",
        "base_executable_sha256",
    ],
)
def test_runtime_binding_drift_refuses_before_mutation(context, field):
    context.document["runtime"][field] += "-changed"
    context.sign()
    _refuses(context, 25)


def test_legacy_manifest_does_not_authorize_new_runtime(context):
    context.document.pop("runtime")
    context.document.update(schema_version=1, scope="cryptohunter.freshness.substrate.v1")
    context.sign()
    _refuses(context, 10)


def test_untrusted_runtime_refuses_before_host_probe(context, monkeypatch):
    original = installer._runtime_helpers

    def fail(**_options):
        raise RuntimeError("synthetic invalid independent signature")

    def helpers():
        return {**original(), "verify_installed_runtime": fail}

    monkeypatch.setattr(installer, "_runtime_helpers", helpers)
    monkeypatch.setattr(
        installer, "inspect_host", lambda *_args: pytest.fail("untrusted Python probe")
    )
    before = _snapshot(context.config.root)
    with pytest.raises(installer.InstallerError) as error:
        installer.preflight(context.config)
    assert error.value.code == 25
    assert _snapshot(context.config.root) == before


def test_signed_unit_cannot_choose_another_interpreter(context):
    path = "deployment/systemd/cryptohunter-freshness-verifier.service"
    file = context.config.bundle / path
    content = file.read_bytes().replace(
        context.document["runtime"]["executable"].encode(), b"/usr/bin/python3"
    )
    file.write_bytes(content)
    record = next(item for item in context.document["files"] if item["path"] == path)
    record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    context.sign()
    _refuses(context, 25)


def test_signed_unit_requires_independent_bootstrap_before_runtime(context):
    path = "deployment/systemd/cryptohunter-freshness-verifier.service"
    file = context.config.bundle / path
    content = (
        b"\n".join(
            line for line in file.read_bytes().splitlines() if not line.startswith(b"ExecStartPre=")
        )
        + b"\n"
    )
    file.write_bytes(content)
    record = next(item for item in context.document["files"] if item["path"] == path)
    record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    context.sign()
    _refuses(context, 25)


@pytest.mark.parametrize(
    "attack",
    [
        "bootstrap-continuation",
        "start-continuation",
        "bootstrap-continuation-reset",
        "start-continuation-reset",
        "wrong-section",
        "indented-reset",
        "space-before-equals",
        "foreign-user",
        "weak-sandbox",
        "crlf",
        "duplicate-service-section",
    ],
)
def test_signed_unit_must_match_complete_reviewed_contract_before_mutation(
    context, monkeypatch, attack
):
    path = "deployment/systemd/cryptohunter-freshness-verifier.service"
    file = context.config.bundle / path
    content = file.read_bytes()
    if attack == "bootstrap-continuation":
        content = content.replace(b"ExecStartPre=", b"Environment=UNTRUSTED=1 \\\nExecStartPre=")
    elif attack == "start-continuation":
        content = content.replace(b"ExecStart=", b"Environment=UNTRUSTED=1 \\\nExecStart=")
    elif attack == "bootstrap-continuation-reset":
        content = content.replace(b"[Install]\n", b"ExecStartPre\\\n=\n\n[Install]\n")
    elif attack == "start-continuation-reset":
        content = content.replace(
            b"[Install]\n", b"ExecStart\\\n=\nExecStart\\\n=/bin/false\n\n[Install]\n"
        )
    elif attack == "wrong-section":
        content = content.replace(b"[Service]\n", b"[Unit]\n")
    elif attack == "indented-reset":
        content = content.replace(
            b"[Install]\n", b" ExecStart=\n ExecStart=/bin/false\n\n[Install]\n"
        )
    elif attack == "space-before-equals":
        content = content.replace(
            b"[Install]\n", b"ExecStart =\nExecStart =/bin/false\n\n[Install]\n"
        )
    elif attack == "foreign-user":
        content = content.replace(b"User=os_freshness_crypto_verifier\n", b"User=root\n")
    elif attack == "weak-sandbox":
        content = content.replace(b"NoNewPrivileges=yes\n", b"NoNewPrivileges=no\n")
    elif attack == "crlf":
        content = content.replace(b"\n", b"\r\n")
    elif attack == "duplicate-service-section":
        content = content.replace(
            b"[Install]\n", b"[Service]\n Environment=FOREIGN=1\n\n[Install]\n"
        )
    file.write_bytes(content)
    record = next(item for item in context.document["files"] if item["path"] == path)
    record.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    # Podpis i hashe są poprawne: sprawdzamy niezależny kontrakt jednostki,
    # a nie odrzucenie uszkodzonego pakietu przed odczytem treści systemd.
    context.sign()
    before = _snapshot(context.config.root)

    def mutation_forbidden(*_args, **_kwargs):
        pytest.fail("Mutacja przed sprawdzeniem pełnego kontraktu jednostki")

    monkeypatch.setattr(installer._Tree, "write_new", mutation_forbidden)
    monkeypatch.setattr(installer, "_publish", mutation_forbidden)
    for operation in (context.preflight, context.install):
        with pytest.raises(installer.InstallerError) as error:
            operation()
        assert error.value.code == 25
        assert _snapshot(context.config.root) == before


def test_trusted_unit_template_matches_reviewed_repository_bytes():
    from deployment.freshness_runtime import UNIT_TEMPLATE

    assert (
        UNIT_TEMPLATE
        == (REPOSITORY / "deployment/systemd/cryptohunter-freshness-verifier.service").read_bytes()
    )


@pytest.mark.parametrize(
    "executable",
    [
        "/usr/bin/python3",
        "/opt/cryptohunter/freshness-runtime/releases/test-runtime/bin/python3",
        "/opt/cryptohunter/freshness-runtime/releases/test-runtime//bin/python3.12",
        "/opt/cryptohunter/freshness-runtime/releases/../bin/python3.12",
        "/opt/cryptohunter/freshness-runtime/releases/test-runtime/bin/python3.12 ",
        "/opt/cryptohunter/freshness-runtime/releases/test-runtime/bin/python3.12\nUser=root",
        "/opt/cryptohunter/freshness-runtime/releases/TEST/bin/python3.12",
    ],
)
def test_trusted_unit_template_rejects_noncanonical_runtime_path(executable):
    from deployment.freshness_runtime import render_runtime_unit

    with pytest.raises(RuntimeError, match="Niekanoniczna"):
        render_runtime_unit(executable)


def test_direct_script_helper_exposes_same_complete_unit_validator():
    import subprocess

    program = (
        "import runpy; "
        f"m=runpy.run_path({str(REPOSITORY / 'deployment/production_local_installer.py')!r}); "
        "h=m['_runtime_helpers'](); "
        f"unit={str(REPOSITORY / 'deployment/systemd/cryptohunter-freshness-verifier.service')!r}; "
        "exe='/opt/cryptohunter/freshness-runtime/releases/test-runtime/bin/python3.12'; "
        "content=open(unit,'rb').read().replace(b'@FRESHNESS_RUNTIME_PYTHON@',exe.encode()); "
        "assert h['validate_runtime_unit'](content,exe)==content; "
        "print('COMPLETE_UNIT_CONTRACT=PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "COMPLETE_UNIT_CONTRACT=PASS"


def test_isolated_installer_loads_reviewed_application_contract_without_package_imports():
    import subprocess

    program = (
        "import runpy, sys; "
        f"m=runpy.run_path({str(REPOSITORY / 'deployment/production_local_installer.py')!r}); "
        "digests=m['_application_helpers']()['REVIEWED_SUBSTRATE_DIGESTS']; "
        "assert len(digests)==5; "
        "assert set(digests)<=m['REQUIRED_FILES']; "
        "assert 'bot_core' not in sys.modules; "
        "assert 'deployment' not in sys.modules; "
        "print('ISOLATED_APPLICATION_CONTRACT=PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ISOLATED_APPLICATION_CONTRACT=PASS"
