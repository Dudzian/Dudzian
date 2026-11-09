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
from deployment import freshness_runtime as runtime


@pytest.fixture
def external_host(tmp_path, monkeypatch):
    """Prawdziwy filesystem; podmieniamy tylko ścieżki hosta i wynik ldd."""
    if sys.platform != "linux":
        pytest.skip("Kontrakt filesystem Linux")
    host = tmp_path / "host"
    destination = tmp_path / "runtime"
    destination.mkdir()
    for name in ("usr/bin", "usr/sbin", "usr/lib/python3.12", "etc"):
        (host / name).mkdir(parents=True, exist_ok=True)
    (host / "usr/bin/python3.12").write_bytes(b"interpreter")
    (host / "usr/lib/python3.12/site.py").write_bytes(b"stdlib")
    (host / "etc/ld.so.cache").write_bytes(b"loader cache")

    def isolated_path(value):
        path = Path(value)
        if path.parts[:2] in {
            ("/", "etc"),
            ("/", "lib"),
            ("/", "lib64"),
            ("/", "bin"),
            ("/", "sbin"),
        }:
            return host / path.relative_to("/")
        return path

    monkeypatch.setattr(builder, "Path", isolated_path)
    monkeypatch.setattr(builder, "BASE_PYTHON", host / "usr/bin/python3.12")
    dependencies = set()
    monkeypatch.setattr(builder, "_native_dependencies", lambda roots: dependencies)
    return host, destination, {"base_prefix": str(host / "usr")}, dependencies


@pytest.mark.parametrize("length", [2, 3, 5, 31])
@pytest.mark.parametrize("absolute", [False, True])
def test_external_inventory_closes_every_literal_symlink(external_host, length, absolute):
    host, destination, probe, _ = external_host
    first = host / "usr/lib/python3.12/config-3.12-x86_64-linux-gnu/libpython3.12.so"
    first.parent.mkdir()
    library = host / "usr/lib/x86_64-linux-gnu"
    library.mkdir()
    links = [first, library / "libpython3.12.so.1"]
    links.extend(library / f"libpython3.12.so.link{i}" for i in range(length - 2))
    final = library / "libpython3.12.so.1.0"
    final.write_bytes(b"trusted libpython")
    targets = {}
    for link, target in zip(links, [*links[1:], final], strict=True):
        literal = str(target) if absolute else os.path.relpath(target, link.parent)
        link.symlink_to(literal)
        targets[link] = literal

    inventory = builder._external_inventory(destination, probe)
    assert inventory == tuple(sorted(set(inventory)))
    assert inventory == builder._external_inventory(destination, probe)
    assert set(links) | {final} <= set(inventory)
    assert all(
        parent in inventory
        for path in [*links, final]
        for parent in path.parents
        if parent != Path("/")
    )
    assert Path("/") not in inventory
    assert not absolute or Path(targets[first]).is_absolute()
    if not absolute:
        assert targets[first] == "../../x86_64-linux-gnu/libpython3.12.so.1"
    manifest = runtime.collect_runtime_manifest(
        destination,
        environment_id="test-closure",
        release_revision="a" * 40,
        interpreter={},
        packages={},
        external_paths=inventory,
        lock_sha256="0" * 64,
    )
    for link, literal in targets.items():
        assert manifest["external_files"][str(link)] == {
            "kind": "symlink",
            "mode": 0o777,
            "target": literal,
        }
    assert (
        manifest["external_files"][str(final)]["sha256"]
        == hashlib.sha256(final.read_bytes()).hexdigest()
    )


def test_external_inventory_closes_usrmerge_parents_and_directory_targets(external_host):
    host, destination, probe, dependencies = external_host
    for name, target in (
        ("lib", "usr/lib"),
        ("bin", "usr/bin"),
        ("sbin", "usr/sbin"),
        ("lib64", "usr/lib64"),
    ):
        (host / name).symlink_to(target)
    (host / "usr/lib64").symlink_to("lib")
    library = host / "usr/lib/libnative.so"
    library.symlink_to("libnative.so.1")
    (host / "usr/lib/libnative.so.1").write_bytes(b"native dependency")
    dependencies.add(host / "lib/libnative.so")
    (host / "usr/lib/python3.12/config").symlink_to("../native-config")
    (host / "usr/lib/native-config").mkdir()
    (host / "usr/lib/native-config/unrelated").write_bytes(b"outside stdlib traversal")

    inventory = builder._external_inventory(destination, probe)
    assert {
        *(
            host / name
            for name in (
                "lib",
                "lib64",
                "bin",
                "sbin",
                "usr/lib",
                "usr/lib64",
                "usr/bin",
                "usr/sbin",
            )
        ),
        library,
        host / "usr/lib/libnative.so.1",
        host / "usr/lib/python3.12/config",
        host / "usr/lib/native-config",
    } <= set(inventory)
    assert host / "lib/libnative.so" not in inventory
    assert host / "usr/lib/native-config/unrelated" not in inventory
    assert runtime._record(host / "lib")["target"] == "usr/lib"


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "cycle",
        "long-chain",
        "special",
        "file-parent",
        "aliased-target",
        "dotdot-through-link",
    ],
)
def test_external_inventory_refuses_invalid_symlink_closure(external_host, failure):
    host, destination, probe, _ = external_host
    link = host / "usr/lib/python3.12/link"
    if failure == "missing":
        link.symlink_to("missing")
    elif failure == "cycle":
        link.symlink_to("second")
        link.with_name("second").symlink_to("link")
    elif failure == "long-chain":
        for index in range(32):
            (link.parent / f"link{index}").symlink_to(f"link{index + 1}")
        (link.parent / "link32").write_bytes(b"final")
    elif failure == "special":
        os.mkfifo(link)
    elif failure == "file-parent":
        link.symlink_to("site.py/../site.py")
    elif failure == "aliased-target":
        (host / "lib").symlink_to("usr/lib")
        link.symlink_to("../../../lib/python3.12/site.py")
    else:
        physical = host / "usr/lib/physical"
        (physical / "directory").mkdir(parents=True)
        (physical / "site.py").write_bytes(b"physical endpoint")
        (link.parent / "directory").symlink_to("../physical/directory")
        link.symlink_to("directory/../site.py")
    with pytest.raises((builder.BuildError, OSError)):
        builder._external_inventory(destination, probe)


@pytest.mark.parametrize("name", ["ld.so.preload", "ld.so.conf", "ld.so.conf.d", "python312.zip"])
def test_external_inventory_refuses_dangling_loader_inputs(external_host, name):
    host, destination, probe, _ = external_host
    parent = host / ("usr/lib" if name == "python312.zip" else "etc")
    (parent / name).symlink_to("missing")
    with pytest.raises((builder.BuildError, OSError)):
        builder._external_inventory(destination, probe)


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


@pytest.fixture(params=[False, True], ids=["writable", "readonly-0500-0400"])
def native_wheelhouse(tmp_path, request):
    """Osobne, zatwierdzone wejście każdej rzeczywistej próby, bez mocków."""
    supplied = os.environ.get("FRESHNESS_RUNTIME_WHEELHOUSE")
    if not supplied:
        if os.environ.get("FRESHNESS_RUNTIME_REQUIRE_NATIVE") == "1":
            pytest.fail("Wymagany natywny build CI nie otrzymał FRESHNESS_RUNTIME_WHEELHOUSE")
        pytest.skip("Wymaga jawnego FRESHNESS_RUNTIME_WHEELHOUSE na jednorazowym Ubuntu 24.04")
    assert sys.platform == "linux" and os.geteuid() != 0, "Natywny build wymaga Linux bez roota"
    source = Path(supplied).resolve()
    entries = builder.verify_wheelhouse(source)
    wheelhouse = tmp_path / "input-wheelhouse"
    shutil.copytree(source, wheelhouse)
    readonly = request.param
    for entry in entries:
        (wheelhouse / entry["filename"]).chmod(0o400 if readonly else 0o644)
    wheelhouse.chmod(0o500 if readonly else 0o755)
    try:
        yield wheelhouse, entries, readonly
    finally:
        # Dopiero po obserwacji niezmienionych wejść umożliwiamy cleanup tmp_path.
        wheelhouse.chmod(0o700)


def test_real_noble_runtime_build_offline_and_imports(tmp_path, native_wheelhouse):
    wheelhouse, entries, readonly = native_wheelhouse
    base_python = os.environ.get("FRESHNESS_RUNTIME_BASE_PYTHON", "/usr/bin/python3.12")
    builder.verify_wheelhouse(wheelhouse)
    input_mode = 0o500 if readonly else 0o755
    wheel_mode = 0o400 if readonly else 0o644
    assert stat.S_IMODE(wheelhouse.stat().st_mode) == input_mode
    assert all(stat.S_IMODE(path.stat().st_mode) == wheel_mode for path in wheelhouse.iterdir())
    expected_hashes = {entry["filename"]: entry["sha256"] for entry in entries}
    assert {path.name: builder._digest(path) for path in wheelhouse.iterdir()} == expected_hashes
    base_digest = builder._digest(Path(base_python))
    runtime = tmp_path / "runtime"
    manifest = tmp_path / "unsigned.json"
    assert not runtime.exists() and not manifest.exists()
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
    assert data["environment_id"] == "noble-cp312-test-v1"
    assert data["release_revision"] == "a" * 40
    assert data["interpreter"]["base_executable"] == "/usr/bin/python3.12"
    assert data["interpreter"]["abi"] == "cpython-312-x86_64-linux-gnu"
    assert data["lock_sha256"] == builder._digest(builder.LOCK)
    assert (runtime / "requirements.lock").read_bytes() == builder.LOCK.read_bytes()
    assert (runtime / "wheelhouse.json").read_bytes() == builder.CATALOG.read_bytes()
    assert "/usr/lib/python3.12/site.py" in data["external_files"]
    assert "/usr/bin/python3.12" in data["external_files"]
    assert any("libssl" in name for name in data["external_files"])
    for name, record in data["external_files"].items():
        if record["kind"] == "symlink":
            target = Path(record["target"])
            if not target.is_absolute():
                target = Path(name).parent / target
            assert os.path.normpath(str(target)) in data["external_files"]
    assert not list(runtime.rglob("*.pth"))
    assert not list(runtime.rglob("*.pyc"))
    assert (runtime / "bin/python3.12").resolve() == Path(base_python).resolve()
    probe = json.loads((runtime / "build-probe.json").read_text())
    assert probe["packages"] == builder.VERSIONS
    assert probe["abi"] == "cpython-312-x86_64-linux-gnu"
    assert probe["libpq"] >= 160000
    assert probe["libpq_build"] >= 160000
    assert probe["cryptography_openssl"].startswith("OpenSSL ")
    assert set(probe["origins"]) == {"cryptography", "psycopg", "psycopg.pq"}
    # Ponownie wykonaj rzeczywiste importy, sign/verify Ed25519 i assert
    # psycopg.pq.__impl__ == "binary" przez wygenerowany interpreter venv.
    observed = subprocess.run(
        [str(runtime / "bin/python3.12"), "-I", "-s", "-B", "-c", builder._PROBE],
        capture_output=True,
        text=True,
        env=builder._ENV,
        timeout=30,
    )
    assert observed.returncode == 0, observed.stderr
    assert json.loads(observed.stdout) == probe
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o755
    actual_files = {}
    for path in runtime.rglob("*"):
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            record = {"kind": "symlink", "mode": mode, "target": os.readlink(path)}
        elif path.is_dir():
            assert mode == 0o755
            record = {"kind": "directory", "mode": mode}
        else:
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and mode == 0o644
            record = {
                "kind": "file",
                "mode": mode,
                "size": info.st_size,
                "sha256": builder._digest(path),
            }
        actual_files[path.relative_to(runtime).as_posix()] = record
    assert data["files"] == actual_files
    assert {path.name for path in (runtime / "wheelhouse").iterdir()} == {
        entry["filename"] for entry in entries if entry["role"] == "runtime"
    }
    assert {path.name for path in (runtime / "build-tools").iterdir()} == {
        entry["filename"] for entry in entries if entry["role"] == "build-tool"
    }
    for entry in entries:
        directory = "build-tools" if entry["role"] == "build-tool" else "wheelhouse"
        assert builder._digest(runtime / directory / entry["filename"]) == entry["sha256"]
    assert stat.S_IMODE(wheelhouse.stat().st_mode) == input_mode
    assert all(stat.S_IMODE(path.stat().st_mode) == wheel_mode for path in wheelhouse.iterdir())
    assert {path.name: builder._digest(path) for path in wheelhouse.iterdir()} == expected_hashes
    builder.verify_wheelhouse(wheelhouse)
    assert builder._digest(Path(base_python)) == base_digest
    assert data["external_files"]["/usr/bin/python3.12"]["sha256"] == base_digest
    # Powtórne wskazanie istniejącego wyniku ma odmówić nadpisania kandydata.
    again = subprocess.run(result.args, capture_output=True, text=True, env=env, timeout=30)
    assert again.returncode != 0
    # Błąd hasha i nadmiarowy wheel są kontrolowane na rzeczywistych artefaktach.
    copied = tmp_path / "wheelhouse"
    shutil.copytree(wheelhouse, copied)
    copied.chmod(0o700)
    entry = builder._catalog()[0]
    wheel = copied / entry["filename"]
    wheel.chmod(0o600)
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(builder.BuildError, match="hash"):
        builder.verify_wheelhouse(copied)
