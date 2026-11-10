"""CSV RECORD pozostaje kompletny po usunięciu skryptów w nowym buildzie."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
from pathlib import Path

import pytest

from deployment import build_freshness_runtime as builder


def write_record(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def file_row(name: str, content: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
    return [name, "sha256=" + digest, str(len(content))]


def installation(destination: Path):
    site = destination / "lib/python3.12/site-packages"
    package = site / "cffi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# biblioteka\n")
    # Nazwa wymagająca cytowania CSV sprawdza parser i zapis rzeczywistego CSV.
    (package / "dane,polskie.txt").write_bytes("żółć".encode())
    metadata = site / "cffi-2.1.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_bytes(b"Name: cffi\nVersion: 2.1.1\n")
    binary = destination / "bin"
    binary.mkdir()
    script = binary / "cffi-gen-src"
    script.write_bytes(f"#!{destination}/bin/python3.12\n# skrypt konsolowy\n".encode())
    (binary / "activate").write_bytes(b"# aktywacja venv nie jest plikiem pakietu\n")
    record = metadata / "RECORD"
    rows = [
        file_row("cffi/__init__.py", (package / "__init__.py").read_bytes()),
        file_row("cffi/dane,polskie.txt", (package / "dane,polskie.txt").read_bytes()),
        file_row("cffi-2.1.1.dist-info/METADATA", (metadata / "METADATA").read_bytes()),
        file_row("../../../bin/cffi-gen-src", script.read_bytes()),
        ["cffi-2.1.1.dist-info/RECORD", "", ""],
    ]
    write_record(record, rows)
    return destination, site, record, rows, script


@pytest.fixture
def installed(tmp_path):
    return installation(tmp_path / "nowy-kandydat")


def test_cleanup_removes_only_deleted_script_rows_and_preserves_integrity_metadata(installed):
    destination, site, record, rows, script = installed
    before = {path: path.read_bytes() for path in site.rglob("*") if path.is_file()}
    assert builder._installed_records(destination) == {record: rows}
    builder._remove_installation_scripts(destination)
    assert not script.exists()
    assert not list((destination / "bin").iterdir())
    expected = [row for row in rows if row[0] != "../../../bin/cffi-gen-src"]
    assert builder._installed_records(destination) == {record: expected}
    assert expected[-1] == ["cffi-2.1.1.dist-info/RECORD", "", ""]
    assert all(path.read_bytes() == content for path, content in before.items() if path != record)
    assert b'"cffi/dane,polskie.txt"' in record.read_bytes()


def test_cleanup_is_independent_of_staging_name_and_length(tmp_path):
    inventories = []
    script_sizes = []
    for name in ("a", "clean-runtime-with-a-much-longer-staging-name"):
        destination, site, _, _, script = installation(tmp_path / name)
        script_sizes.append(script.stat().st_size)
        builder._remove_installation_scripts(destination)
        inventories.append(
            {
                path.relative_to(site).as_posix(): builder._digest(path)
                for path in site.rglob("*")
                if path.is_file()
            }
        )
    assert script_sizes[0] != script_sizes[1]
    assert inventories[0] == inventories[1]


@pytest.mark.parametrize("field", [1, 2])
def test_optional_hash_or_size_fields_remain_legal_and_unchanged(installed, field):
    destination, _, record, rows, _ = installed
    rows[0][field] = ""
    write_record(record, rows)
    builder._remove_installation_scripts(destination)
    assert builder._installed_records(destination)[record][0] == rows[0]


@pytest.mark.parametrize(
    "attack",
    [
        "hash",
        "hash-algorithm",
        "size",
        "size-nonnumeric",
        "size-negative",
        "size-leading-zero",
        "duplicate",
        "missing-self",
        "self-hash",
        "self-size",
        "columns",
        "empty-path",
        "absolute-path",
        "escape",
        "backslash",
        "noncanonical",
        "missing-package-file",
        "missing-script",
        "missing-record",
        "unrecorded-file",
        "unrecorded-metadata",
        "directory-row",
        "invalid-csv",
        "invalid-utf8",
    ],
)
def test_damaged_record_refuses_before_any_script_is_removed(installed, attack):
    destination, site, record, rows, script = installed
    if attack == "hash":
        rows[0][1] = "sha256=" + "A" * 43
    elif attack == "hash-algorithm":
        rows[0][1] = "md5=" + "A" * 22
    elif attack == "size":
        rows[0][2] = str(int(rows[0][2]) + 1)
    elif attack == "size-nonnumeric":
        rows[0][2] = "nan"
    elif attack == "size-negative":
        rows[0][2] = "-1"
    elif attack == "size-leading-zero":
        rows[0][2] = "00"
    elif attack == "duplicate":
        rows.insert(1, rows[0][:])
    elif attack == "missing-self":
        rows.pop()
    elif attack == "self-hash":
        rows[-1][1] = "sha256=" + "A" * 43
    elif attack == "self-size":
        rows[-1][2] = "0"
    elif attack == "columns":
        rows[0].append("nadmiarowe pole")
    elif attack in {"empty-path", "absolute-path", "escape", "backslash", "noncanonical"}:
        rows[0][0] = {
            "empty-path": "",
            "absolute-path": "/etc/passwd",
            "escape": "../../../../poza-runtime",
            "backslash": "cffi\\__init__.py",
            "noncanonical": "cffi/./__init__.py",
        }[attack]
    elif attack == "missing-package-file":
        (site / rows[0][0]).unlink()
    elif attack == "missing-script":
        script.unlink()
    elif attack == "unrecorded-file":
        (site / "cffi/obcy.py").write_bytes(b"# obcy plik\n")
    elif attack == "unrecorded-metadata":
        rows.pop(2)
    elif attack == "directory-row":
        rows.insert(0, ["cffi", "", ""])
    write_record(record, rows)
    if attack == "missing-record":
        record.unlink()
    elif attack == "invalid-csv":
        record.write_text('"niedomknięty cytat\n', encoding="utf-8")
    elif attack == "invalid-utf8":
        record.write_bytes(b"\xff\xfe")
    original_record = record.read_bytes() if record.exists() else None
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert (destination / "bin/activate").is_file()
    assert script.exists() == (attack != "missing-script")
    assert (record.read_bytes() if record.exists() else None) == original_record


@pytest.mark.parametrize("attack", ["file-symlink", "parent-symlink", "bin-symlink", "hardlink"])
def test_record_and_bin_reject_aliased_files(installed, attack):
    destination, site, _, _, _ = installed
    target = site / "cffi/__init__.py"
    outside = destination / "poza-pakietem.py"
    outside.write_bytes(target.read_bytes())
    try:
        if attack == "file-symlink":
            target.unlink()
            target.symlink_to(outside)
        elif attack == "parent-symlink":
            package = site / "cffi"
            package.rename(site / "przeniesione-cffi")
            package.symlink_to("przeniesione-cffi", target_is_directory=True)
        elif attack == "bin-symlink":
            (destination / "bin/narzedzie-buildowe").symlink_to(outside)
        else:
            target.unlink()
            os.link(outside, target)
    except OSError:
        pytest.skip("Platforma bez możliwości utworzenia wymaganego dowiązania")
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert (destination / "bin/activate").exists()


def test_empty_installation_refuses(tmp_path):
    site = tmp_path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    with pytest.raises(builder.BuildError, match="Brak metadanych RECORD"):
        builder._installed_records(tmp_path)


def test_unchanged_record_bytes_are_not_rewritten(installed):
    destination, _, record, rows, script = installed
    rows = [row for row in rows if row[0] != "../../../bin/cffi-gen-src"]
    script.unlink()
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\r\n").writerows(rows)
    original = record.read_bytes()
    builder._remove_installation_scripts(destination)
    assert record.read_bytes() == original
