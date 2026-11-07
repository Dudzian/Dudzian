"""Lock i rzeczywisty opt-in build Noble; bez kont, usług i klastrów hosta."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from deployment import build_freshness_runtime as builder


def test_catalog_freezes_complete_runtime_and_build_tool_closure():
    entries = builder._catalog()
    assert len(entries) == 7
    assert {e["name"]: e["version"] for e in entries if e["role"] == "runtime"} == builder.VERSIONS
    lock = builder.LOCK.read_text()
    for entry in entries:
        if entry["role"] == "runtime":
            assert f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}" in lock
        assert entry["url"].endswith(entry["filename"])
    assert "--break-system-packages" not in lock


def test_incomplete_wheelhouse_and_extra_package_refused(tmp_path):
    with pytest.raises(builder.BuildError, match="Niekompletny"):
        builder.verify_wheelhouse(tmp_path)
    (tmp_path / "foreign-1-py3-none-any.whl").write_bytes(b"foreign")
    with pytest.raises(builder.BuildError, match="dodatkowy pakiet"):
        builder.verify_wheelhouse(tmp_path)


def test_catalog_rejects_changed_version_and_source(tmp_path):
    data = json.loads(builder.CATALOG.read_text())
    path = tmp_path / "catalog.json"
    data["wheels"][0]["version"] = "0.0.0"
    path.write_text(json.dumps(data))
    with pytest.raises(builder.BuildError):
        builder._catalog(path)
    data = json.loads(builder.CATALOG.read_text())
    data["wheels"][0]["url"] = "https://attacker.invalid/a.whl"
    path.write_text(json.dumps(data))
    with pytest.raises(builder.BuildError):
        builder._catalog(path)


@pytest.mark.parametrize(
    "member", ["foreign.pth", "sitecustomize.py", "usercustomize.py", "../escape", "/escape"]
)
def test_foreign_import_hook_and_path_traversal_refused(tmp_path, member):
    path = tmp_path / "candidate.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(member, b"import attacker")
    entry = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(builder.BuildError, match="Niedozwolony plik"):
        builder._wheel(path, entry)


def test_bad_hash_refused_before_archive_interpretation(tmp_path):
    path = tmp_path / "candidate.whl"
    path.write_bytes(b"not an archive")
    with pytest.raises(builder.BuildError, match="hash"):
        builder._wheel(path, {"size": path.stat().st_size, "sha256": "0" * 64})


def test_symlink_wheel_refused(tmp_path):
    real = tmp_path / "real.whl"
    real.write_bytes(b"untrusted")
    path = tmp_path / "alias.whl"
    try:
        path.symlink_to(real)
    except OSError:
        pytest.skip("Platforma bez uprawnień symlink")
    with pytest.raises(builder.BuildError, match="metadane"):
        builder._wheel(
            path,
            {"size": real.stat().st_size, "sha256": hashlib.sha256(real.read_bytes()).hexdigest()},
        )


def test_builder_rejects_unapproved_platform_before_destination_write(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.platform, "freedesktop_os_release", lambda: {"ID": "debian"})
    destination = tmp_path / "runtime"
    with pytest.raises(builder.BuildError, match="systemowego CPython"):
        builder.build_runtime(
            tmp_path, destination, environment_id="noble-v1", release_revision="a" * 40
        )
    assert not destination.exists()


@pytest.fixture
def readonly_build_inputs(tmp_path, monkeypatch):
    """Prawdziwe archiwa i walidacja; bez instalacji, subprocessów i inspekcji hosta."""
    wheelhouse = tmp_path / "readonly-wheelhouse"
    wheelhouse.mkdir()
    catalog = tmp_path / "catalog.json"
    data = json.loads(builder.CATALOG.read_text())
    for entry in data["wheels"]:
        path = wheelhouse / entry["filename"]
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{entry['name']}/__init__.py", b"# synthetic wheel\n")
        entry["size"] = path.stat().st_size
        entry["sha256"] = builder._digest(path)
        path.chmod(0o400)
    catalog.write_text(json.dumps(data))
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "# Ubuntu 24.04 amd64 / CPython 3.12. Exact runtime closure; no sources.\n"
        + "\n".join(
            f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}"
            for entry in data["wheels"]
            if entry["role"] == "runtime"
        )
        + "\n"
    )
    read_catalog = builder._catalog
    monkeypatch.setattr(builder, "_catalog", lambda: read_catalog(catalog))
    monkeypatch.setattr(builder, "CATALOG", catalog)
    monkeypatch.setattr(builder, "LOCK", lock)
    monkeypatch.setattr(builder, "BASE_PYTHON", Path(sys.executable).resolve())
    monkeypatch.setattr(builder, "_check_platform", lambda: "test-provenance")
    monkeypatch.setattr(builder, "_external_inventory", lambda destination, probe: ())

    class IsolatedVenv:
        def __init__(self, *, with_pip, symlinks):
            assert with_pip is False and symlinks is True

        def create(self, destination):
            (destination / "bin").mkdir(parents=True)
            (destination / "bin/python3.12").symlink_to(builder.BASE_PYTHON)

    monkeypatch.setattr(builder.venv, "EnvBuilder", IsolatedVenv)

    def run(arguments):
        if arguments[-1] == builder._PROBE:
            return json.dumps(
                {
                    "packages": builder.VERSIONS,
                    "abi": "cpython-312-x86_64-linux-gnu",
                    "python": "3.12",
                }
            )
        assert "--no-index" in arguments and "--require-hashes" in arguments
        assert arguments[arguments.index("--find-links") + 1] == str(wheelhouse)
        return ""

    monkeypatch.setattr(builder, "_run", run)
    wheelhouse.chmod(0o500)
    try:
        yield wheelhouse, data["wheels"]
    finally:
        # Przywróć możliwość sprzątania wyłącznie testowego katalogu źródłowego.
        wheelhouse.chmod(0o700)


@pytest.mark.skipif(os.name != "posix", reason="Regresja uprawnień katalogów POSIX")
def test_build_accepts_readonly_wheelhouse_without_changing_inputs(tmp_path, readonly_build_inputs):
    wheelhouse, entries = readonly_build_inputs
    original_bytes = {path.name: path.read_bytes() for path in wheelhouse.iterdir()}
    destination = tmp_path / "runtime"
    manifest = builder.build_runtime(
        wheelhouse, destination, environment_id="readonly-test", release_revision="a" * 40
    )
    assert stat.S_IMODE(wheelhouse.stat().st_mode) == 0o500
    assert {path.name: path.read_bytes() for path in wheelhouse.iterdir()} == original_bytes
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 for path in wheelhouse.iterdir())
    assert {path.name for path in (destination / "wheelhouse").iterdir()} == {
        entry["filename"] for entry in entries if entry["role"] == "runtime"
    }
    assert {path.name for path in (destination / "build-tools").iterdir()} == {
        entry["filename"] for entry in entries if entry["role"] == "build-tool"
    }
    for entry in entries:
        directory = "build-tools" if entry["role"] == "build-tool" else "wheelhouse"
        relative = f"{directory}/{entry['filename']}"
        assert (destination / relative).read_bytes() == original_bytes[entry["filename"]]
        assert builder._digest(destination / relative) == entry["sha256"]
        assert manifest["files"][relative]["sha256"] == entry["sha256"]
    assert (destination / "requirements.lock").read_bytes() == builder.LOCK.read_bytes()
    assert manifest["lock_sha256"] == builder._digest(builder.LOCK)
    for path in (destination, *destination.rglob("*")):
        if not path.is_symlink():
            assert stat.S_IMODE(path.stat().st_mode) == (0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize("tampered", ["wheel", "lock"])
def test_readonly_wheelhouse_integrity_failure_precedes_destination_write(
    tmp_path, readonly_build_inputs, tampered
):
    wheelhouse, entries = readonly_build_inputs
    if tampered == "wheel":
        path = wheelhouse / entries[0]["filename"]
        path.chmod(0o600)
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        path.chmod(0o400)
        expected_error = "hash"
    else:
        builder.LOCK.write_text(builder.LOCK.read_text() + "# changed lock\n")
        expected_error = "Lock"
    destination = tmp_path / "runtime"
    with pytest.raises(builder.BuildError, match=expected_error):
        builder.build_runtime(
            wheelhouse, destination, environment_id="readonly-test", release_revision="a" * 40
        )
    assert not destination.exists()


def test_real_noble_runtime_build_offline_and_imports(tmp_path):
    supplied = os.environ.get("FRESHNESS_RUNTIME_WHEELHOUSE")
    if not supplied:
        pytest.skip("Wymaga jawnego FRESHNESS_RUNTIME_WHEELHOUSE na jednorazowym Ubuntu 24.04")
    base_python = os.environ.get("FRESHNESS_RUNTIME_BASE_PYTHON", "/usr/bin/python3.12")
    wheelhouse = Path(supplied).resolve()
    builder.verify_wheelhouse(wheelhouse)
    runtime = tmp_path / "runtime"
    manifest = tmp_path / "unsigned.json"
    # Builder ma własne środowisko subprocessów; wejściowe user-site i PYTHONPATH
    # nie mogą dostać się do venv ani do rzeczywistego importu.
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "sitecustomize.py").write_text("raise RuntimeError('foreign site')")
    env = dict(os.environ, PYTHONPATH=str(hostile), PYTHONUSERBASE=str(hostile))
    result = subprocess.run(
        [
            base_python,
            "-I",
            "-s",
            "-B",
            str(builder.HERE / "build_freshness_runtime.py"),
            "--wheelhouse",
            str(wheelhouse),
            "--destination",
            str(runtime),
            "--environment-id",
            "noble-cp312-test-v1",
            "--release-revision",
            "a" * 40,
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(manifest.read_text())
    assert data["packages"] == builder.VERSIONS
    assert data["interpreter"]["base_executable"] == "/usr/bin/python3.12"
    assert data["interpreter"]["abi"] == "cpython-312-x86_64-linux-gnu"
    assert "/usr/lib/python3.12/site.py" in data["external_files"]
    assert "/usr/bin/python3.12" in data["external_files"]
    assert any("libssl" in name for name in data["external_files"])
    assert not list(runtime.rglob("*.pth"))
    assert not list(runtime.rglob("*.pyc"))
    assert (runtime / "bin/python3.12").resolve() == Path(base_python).resolve()
    probe = json.loads((runtime / "build-probe.json").read_text())
    assert probe["libpq"] >= 160000
    assert probe["cryptography_openssl"].startswith("OpenSSL ")
    assert set(probe["origins"]) == {"cryptography", "psycopg", "psycopg.pq"}
    again = subprocess.run(result.args, capture_output=True, text=True, env=env, timeout=30)
    assert again.returncode != 0
    # Błąd hasha i nadmiarowy wheel są kontrolowane na rzeczywistych artefaktach.
    copied = tmp_path / "wheelhouse"
    shutil.copytree(wheelhouse, copied)
    entry = builder._catalog()[0]
    wheel = copied / entry["filename"]
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(builder.BuildError, match="hash"):
        builder.verify_wheelhouse(copied)
