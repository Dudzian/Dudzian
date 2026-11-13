"""Revision equality gates only: in-memory models, never approvals or signatures.

No test constructs a signed manifest, writes an approval, runs an installer,
executes candidate code or builds a runtime. Positive schema/equality checks
are models at the already authenticated boundary, not runtime qualification.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from deployment import freshness_application as application
from deployment import freshness_runtime as runtime
from deployment import production_local_installer as installer
from scripts import prepare_stage9_vm_bundle as exporter

OLD = "fc15b98298c5430286dd2206ec7eeb8f20e2497b"
REVIEWED = "5e0646a533d071b3f4a1c62d8d9836ece9081c95"
ENVIRONMENT = "revision-contract-model"
REPOSITORY = Path(__file__).resolve().parents[2]
# Hash of the public source blob at OLD, read independently with git show.
# A historical comparison value only; never a new runtime/application pin.
OLD_BOOTSTRAP_SOURCE_SHA256 = "f19cb38167e078048fef181e56412957a53b7aef14b878cb9159c8ac8a48bc68"


def runtime_model(revision: str) -> runtime.VerifiedRuntime:
    # Model a Linux runtime regardless of the OS executing the test.
    release_root = PurePosixPath("/opt/cryptohunter/freshness-runtime/releases")
    return runtime.VerifiedRuntime(
        ENVIRONMENT,
        revision,
        "b" * 64,
        cast(Path, release_root / ENVIRONMENT / "bin/python3.12"),
        cast(Path, PurePosixPath("/usr/bin/python3.12")),
        "c" * 64,
        "c" * 64,
        {"cryptography": "50.0.2", "psycopg": "3.3.6"},
        {},
    )


def binding_model(revision: str) -> dict[str, str]:
    model = runtime_model(revision)
    return {
        "environment_id": model.environment_id,
        "manifest_sha256": model.manifest_sha256,
        "source_revision": revision,
        "executable": str(model.executable),
        "base_executable": str(model.base_executable),
        "base_executable_sha256": model.base_executable_sha256,
    }


@pytest.mark.parametrize("revision", (OLD, REVIEWED))
def test_runtime_model_uses_canonical_linux_paths_on_every_host(revision: str) -> None:
    model = runtime_model(revision)
    assert str(model.executable) == (
        f"/opt/cryptohunter/freshness-runtime/releases/{ENVIRONMENT}/bin/python3.12"
    )
    assert str(model.base_executable) == "/usr/bin/python3.12"


def schema_model(revision: str) -> dict:
    def file_record() -> dict:
        return {"kind": "file", "mode": 0o644, "size": 0, "sha256": "d" * 64}

    return {
        "schema": 1,
        "scope": runtime.SCOPE,
        "environment_id": ENVIRONMENT,
        "release_revision": revision,
        "target": {"os": "ubuntu", "version": "24.04", "architecture": "amd64"},
        "installation_path": str(runtime.RUNTIME_BASE / "releases" / ENVIRONMENT),
        "interpreter": {
            "implementation": "CPython",
            "version": "3.12.3",
            "abi": "cpython-312-x86_64-linux-gnu",
            "executable": "bin/python3.12",
            "base_executable": "/usr/bin/python3.12",
            "provenance": "Ubuntu dpkg python3.12 model-only",
        },
        "packages": {"cryptography": "50.0.2", "psycopg": "3.3.6"},
        "files": {
            "bin/python3.12": {"kind": "symlink", "mode": 0o777, "target": "/usr/bin/python3.12"},
            "pyvenv.cfg": file_record(),
            "requirements.lock": file_record(),
            "wheelhouse": {"kind": "directory", "mode": 0o755},
            "wheelhouse/model.whl": file_record(),
        },
        "external_files": {"/usr/bin/python3.12": file_record()},
        "lock_sha256": "d" * 64,
    }


@pytest.mark.parametrize("manifest_revision,expected", itertools.product((OLD, REVIEWED), repeat=2))
def test_runtime_schema_requires_exact_revision(manifest_revision, expected):
    config = runtime.RuntimeConfig(
        Path("/model/runtime"),
        Path("/model/manifest"),
        Path("/model/signature"),
        Path("/model/authority"),
        ENVIRONMENT,
        expected,
    )
    if manifest_revision == expected:
        assert runtime._schema(schema_model(manifest_revision), config) is None
    else:
        with pytest.raises(RuntimeError, match="Niezgodna tożsamość lub rewizja runtime"):
            runtime._schema(schema_model(manifest_revision), config)


@pytest.mark.parametrize("binding_revision,expected", itertools.product((OLD, REVIEWED), repeat=2))
def test_bundle_export_refuses_relabelled_runtime_binding(monkeypatch, binding_revision, expected):
    data = {"schema": 1, **binding_model(binding_revision)}
    monkeypatch.setattr(exporter, "_read", lambda path: json.dumps(data).encode())
    if binding_revision == expected:
        assert exporter._binding(Path("/model/not-a-real-binding"), expected) == binding_model(
            expected
        )
    else:
        with pytest.raises(exporter.BundleError, match="BINDING_VALUES_REFUSED"):
            exporter._binding(Path("/model/not-a-real-binding"), expected)


@pytest.mark.parametrize("field", ("executable", "base_executable"))
def test_bundle_export_refuses_windows_separators(monkeypatch, field: str) -> None:
    binding = binding_model(REVIEWED)
    binding[field] = binding[field].replace("/", "\\")
    monkeypatch.setattr(
        exporter, "_read", lambda path: json.dumps({"schema": 1, **binding}).encode()
    )
    with pytest.raises(exporter.BundleError, match="BINDING_VALUES_REFUSED"):
        exporter._binding(Path("/model/not-a-real-binding"), REVIEWED)


def test_runtime_unit_refuses_windows_separator_path() -> None:
    executable = binding_model(REVIEWED)["executable"].replace("/", "\\")
    with pytest.raises(RuntimeError, match="Niekanoniczna ścieżka interpretera"):
        runtime.render_runtime_unit(executable)


@pytest.mark.parametrize(
    "application_revision,binding_revision,observed_revision",
    itertools.product((OLD, REVIEWED), repeat=3),
)
def test_application_requires_all_three_revision_identities(
    application_revision,
    binding_revision,
    observed_revision,
):
    manifest_model = {
        "source_revision": application_revision,
        "runtime": binding_model(binding_revision),
    }
    observed = runtime_model(observed_revision)
    if application_revision == binding_revision == observed_revision:
        assert application._runtime_binding(observed, manifest_model) == binding_model(
            observed_revision
        )
    else:
        with pytest.raises(application.ApplicationTrustError, match="Runtime nie odpowiada"):
            application._runtime_binding(observed, manifest_model)


@pytest.mark.parametrize(
    "expected,binding_revision,observed_revision", itertools.product((OLD, REVIEWED), repeat=3)
)
def test_installer_refuses_runtime_binding_or_approved_revision_mix(
    monkeypatch,
    expected,
    binding_revision,
    observed_revision,
):
    observed = runtime_model(observed_revision)
    unit = runtime.render_runtime_unit(str(observed.executable))
    artifact = installer.Artifact(
        "deployment/systemd/cryptohunter-freshness-verifier.service",
        hashlib.sha256(unit).hexdigest(),
        len(unit),
        0o644,
        unit,
    )
    helpers = {
        "verify_installed_runtime": lambda **kwargs: observed,
        "validate_runtime_unit": runtime.validate_runtime_unit,
    }
    monkeypatch.setattr(installer, "_runtime_helpers", lambda: helpers)
    config = installer.Config(
        Path("/model/bundle"),
        Path("/model/manifest"),
        Path("/model/signature"),
        Path("/model/authority"),
        expected,
    )
    if expected == binding_revision == observed_revision:
        assert (
            installer._verified_runtime(config, binding_model(binding_revision), (artifact,))
            is observed
        )
    else:
        with pytest.raises(installer.InstallerError, match="Runtime nie odpowiada") as error:
            installer._verified_runtime(config, binding_model(binding_revision), (artifact,))
        assert error.value.code == 25


@pytest.mark.skipif(os.name != "posix", reason="Rootless source read uses POSIX descriptors")
def test_bootstrap_anchor_reads_current_exact_source_without_using_payload(tmp_path):
    # Repository source is read as data. No original artifact,
    # VM, filesystem owner or permission is changed.
    config = installer.Config(
        tmp_path / "absent-bundle",
        tmp_path / "absent-manifest",
        tmp_path / "absent-signature",
        tmp_path / "absent-authority",
        REVIEWED,
        owner_uid=os.getuid(),
    )
    anchors = installer._trusted_bootstrap(config)
    assert set(anchors) == set(installer.TRUSTED_BOOTSTRAP_FILES)
    for name, content in anchors.items():
        assert content == (REPOSITORY / name).read_bytes()
    assert hashlib.sha256(anchors["deployment/freshness_runtime_bootstrap.py"]).hexdigest() != (
        OLD_BOOTSTRAP_SOURCE_SHA256
    )
    assert not list(tmp_path.iterdir())
