"""Istniejący bytecode nadal jest materiałem wykonywalnym przy -I -s -B."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from deployment import build_freshness_runtime as builder
from deployment import freshness_runtime as runtime


@pytest.fixture(autouse=True)
def local_bytecode_cache(monkeypatch):
    """Syntetyczny cache pozostaje w tmp_path także po testach bootstrapu."""
    # Bootstrap celowo kieruje cache do /dev/null; ten stan procesu pytest
    # nie może zmieniać miejsca przygotowania plików wejściowych tej próby.
    monkeypatch.setattr(sys, "pycache_prefix", None)


@pytest.mark.parametrize("kind", ["timestamp", "unchecked-hash", "sourceless"])
def test_isolated_no_write_bytecode_still_executes_existing_pyc(tmp_path, kind):
    source = tmp_path / "retained_bytecode.py"
    source.write_text("VALUE = 'bytecode'\n")
    original = source.stat()
    mode = (
        py_compile.PycInvalidationMode.UNCHECKED_HASH
        if kind == "unchecked-hash"
        else py_compile.PycInvalidationMode.TIMESTAMP
    )
    cfile = str(source.with_suffix(".pyc")) if kind == "sourceless" else None
    cache = Path(py_compile.compile(str(source), cfile=cfile, doraise=True, invalidation_mode=mode))
    if kind == "sourceless":
        source.unlink()
    else:
        # Poprawny timestamp/rozmiar albo unchecked hash pozwala wykonać
        # istniejący bytecode pomimo odmiennych bajtów pliku źródłowego.
        source.write_text("VALUE = 'source__'\n")
        assert source.stat().st_size == original.st_size
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-s",
            "-B",
            "-c",
            "import json,sys; sys.path.insert(0,sys.argv[1]); "
            "import retained_bytecode; "
            "assert sys.flags.isolated and sys.flags.no_user_site "
            "and sys.flags.dont_write_bytecode; "
            "print(json.dumps([retained_bytecode.VALUE, retained_bytecode.__cached__]))",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["bytecode", str(cache)]
    assert {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


@pytest.mark.skipif(sys.platform != "linux", reason="Kontrakt filesystem Linux")
def test_external_inventory_keeps_bytecode_hashes_and_native_dependencies(tmp_path, monkeypatch):
    host = tmp_path / "host"
    destination = tmp_path / "candidate"
    stdlib = host / "usr/lib/python3.12"
    for directory in (host / "usr/bin", host / "etc", stdlib / "__pycache__", destination):
        directory.mkdir(parents=True, exist_ok=True)
    interpreter = host / "usr/bin/python3.12"
    interpreter.write_bytes(b"interpreter")
    (host / "etc/ld.so.cache").write_bytes(b"loader cache")
    source = stdlib / "retained.py"
    source.write_text("VALUE = 1\n")
    cached = Path(py_compile.compile(str(source), doraise=True))
    sourceless = stdlib / "legacy.pyc"
    py_compile.compile(str(source), cfile=str(sourceless), doraise=True)
    native = stdlib / "_native.so"
    native.write_bytes(b"stdlib native extension")
    bundled = destination / "_bundled.so"
    bundled.write_bytes(b"wheel native extension")
    versioned = destination / "libbundled.so.1"
    versioned.write_bytes(b"wheel native library")
    dependency = host / "usr/lib/libdependency.so.1"
    dependency.write_bytes(b"external native library")
    link = dependency.with_name("libdependency.so")
    link.symlink_to(dependency.name)

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

    native_roots = []

    def dependencies(roots):
        native_roots.extend(roots)
        return {link}

    monkeypatch.setattr(builder, "Path", isolated_path)
    monkeypatch.setattr(builder, "BASE_PYTHON", interpreter)
    monkeypatch.setattr(builder, "_native_dependencies", dependencies)
    inventory = builder._external_inventory(destination, {"base_prefix": str(host / "usr")})
    assert set(native_roots) == {interpreter, native, bundled, versioned}
    assert {source, cached, cached.parent, sourceless, native, link, dependency} <= set(inventory)
    manifest = runtime.collect_runtime_manifest(
        destination,
        environment_id="bytecode-inventory-test",
        release_revision="a" * 40,
        interpreter={},
        packages={},
        external_paths=inventory,
        lock_sha256="0" * 64,
    )
    for path in (cached, sourceless, native, dependency):
        entry = manifest["external_files"][str(path)]
        assert entry["kind"] == "file"
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["external_files"][str(link)]["target"] == dependency.name
