"""CSV RECORD pozostaje kompletny po usunięciu skryptów w nowym buildzie."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import subprocess
import sys
from importlib import metadata as installed_metadata
from pathlib import Path

import pytest

from deployment import build_freshness_runtime as builder


def write_record(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def file_row(name: str, content: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
    return [name, "sha256=" + digest, str(len(content))]


def add_distribution(site: Path, name: str, version: str, directory: str | None = None):
    """Poprawne, rozłączne metadane, aby testować semantykę dystrybucji."""
    metadata = site / (directory or f"{name.replace('-', '_')}-{version}.dist-info")
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    record = metadata / "RECORD"
    rows = [
        file_row(
            (metadata / "METADATA").relative_to(site).as_posix(),
            (metadata / "METADATA").read_bytes(),
        ),
        [record.relative_to(site).as_posix(), "", ""],
    ]
    write_record(record, rows)
    return record, rows


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
    for name, version in builder.VERSIONS.items():
        if name != "cffi":
            add_distribution(site, name, version)
    return destination, site, record, rows, script


@pytest.fixture
def installed(tmp_path):
    return installation(tmp_path / "nowy-kandydat")


def test_cleanup_removes_only_deleted_script_rows_and_preserves_integrity_metadata(installed):
    destination, site, record, rows, script = installed
    before = {path: path.read_bytes() for path in site.rglob("*") if path.is_file()}
    installed_records = builder._installed_records(destination)
    assert len(installed_records) == 6
    assert installed_records[record] == rows
    builder._remove_installation_scripts(destination)
    assert not script.exists()
    assert not list((destination / "bin").iterdir())
    expected = [row for row in rows if row[0] != "../../../bin/cffi-gen-src"]
    installed_records = builder._installed_records(destination)
    assert len(installed_records) == 6
    assert installed_records[record] == expected
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
    with pytest.raises(builder.BuildError):
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


def snapshot_files(destination: Path):
    return {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_unknown_bin_file_refuses_before_removing_any_installed_script(installed):
    destination, _, _, _, script = installed
    unknown = destination / "bin/unknown-executable"
    unknown.write_bytes(b"#!/bin/sh\nexit 0\n")
    unknown.chmod(0o755)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert unknown.is_file() and script.is_file()
    assert (destination / "bin/activate").is_file()
    assert snapshot_files(destination) == before


@pytest.mark.parametrize(
    "directory",
    ["CFFI-2.1.1.DIST-INFO", "cffi_duplicate-2.1.1.DiSt-InFo", "cffi-2.1.1.egg-info"],
)
def test_other_metadata_layout_cannot_hide_in_an_approved_record(installed, directory):
    destination, site, record, rows, _ = installed
    # Sprawdź alias na rzeczywistym systemie plików, niezależnie od platformy.
    # Gdy różnica liter nie tworzy osobnego katalogu, zachowaj dodatkową
    # dystrybucję cffi i wielki sufiks, używając rozłącznej nazwy katalogu.
    if directory == "CFFI-2.1.1.DIST-INFO" and (site / directory).exists():
        directory = "cffi_duplicate-2.1.1.DIST-INFO"
    extra_record, extra_rows = add_distribution(site, "cffi", "2.1.1", directory)
    if directory.endswith(".egg-info"):
        package_info = extra_record.parent / "PKG-INFO"
        (extra_record.parent / "METADATA").rename(package_info)
        extra_rows = [
            file_row(package_info.relative_to(site).as_posix(), package_info.read_bytes()),
            [extra_record.relative_to(site).as_posix(), "", ""],
        ]
        write_record(extra_record, extra_rows)
    visible = list(installed_metadata.distributions(path=[str(site)]))
    assert len(visible) == 7
    assert sum(distribution.metadata["Name"] == "cffi" for distribution in visible) == 2
    # Wszystkie pliki dodatkowej dystrybucji są ujęte w poprawnym CSV: samo
    # pokrycie plików nie dowodzi, że istnieje tylko sześć dystrybucji.
    write_record(record, [*rows, *extra_rows])
    before = snapshot_files(destination)
    if directory.endswith(".egg-info"):
        expected_error = "Niedozwolone metadane egg-info runtime"
    elif directory.startswith("cffi_duplicate-"):
        expected_error = "Powtórzona dystrybucja runtime"
    else:
        # Kolejność dwóch nazw różniących się literami zależy także od Path.
        expected_error = "Niekanoniczny katalog dystrybucji runtime|Powtórzona dystrybucja runtime"
    with pytest.raises(builder.BuildError, match=expected_error):
        builder._remove_installation_scripts(destination)
    assert snapshot_files(destination) == before


def test_other_metadata_layout_on_case_insensitive_filesystem(installed, monkeypatch):
    _, site, _, _, _ = installed
    directory = "CFFI-2.1.1.DIST-INFO"
    original_exists = Path.exists
    original_mkdir = Path.mkdir

    def case_insensitive_exists(path):
        if path.parent == site and path.name.casefold() == directory.casefold():
            return True
        return original_exists(path)

    def case_insensitive_mkdir(path, *args, **kwargs):
        if path.parent == site and any(
            child.name.casefold() == path.name.casefold() for child in site.iterdir()
        ):
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    # Modeluj kolizję także na Linux, ale wykonaj cały scenariusz odmowy
    # na rzeczywistych plikach, bez zastępowania walidacji buildera.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", case_insensitive_exists)
        patch.setattr(Path, "mkdir", case_insensitive_mkdir)
        test_other_metadata_layout_cannot_hide_in_an_approved_record(installed, directory)
    assert (site / "cffi_duplicate-2.1.1.DIST-INFO").is_dir()
    assert len(list(installed_metadata.distributions(path=[str(site)]))) == 7


@pytest.mark.parametrize(
    "script_name", ["activate", "activate.csh", "activate.fish", "Activate.ps1"]
)
def test_known_venv_activation_scripts_are_removed(installed, script_name):
    destination, _, _, _, script = installed
    (destination / "bin" / script_name).write_bytes(b"# znany skrypt aktywacji systemowego venv\n")
    builder._remove_installation_scripts(destination)
    assert not script.exists()
    assert not list((destination / "bin").iterdir())


def test_unknown_bin_script_cannot_gain_trust_from_record(installed):
    destination, _, record, rows, _ = installed
    script = destination / "bin/unknown-executable"
    script.write_bytes(b"# dodatkowy skrypt\n")
    rows.insert(0, file_row("../../../bin/unknown-executable", script.read_bytes()))
    write_record(record, rows)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert snapshot_files(destination) == before


def test_expected_wheel_script_requires_its_owning_distribution_record(installed):
    destination, site, record, rows, script = installed
    script_row = next(row for row in rows if row[0] == "../../../bin/cffi-gen-src")
    write_record(record, [row for row in rows if row is not script_row])
    other = site / "pycparser-3.12.dist-info/RECORD"
    with other.open(newline="", encoding="utf-8") as stream:
        other_rows = list(csv.reader(stream))
    write_record(other, [script_row, *other_rows])
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before


@pytest.mark.parametrize(
    "name,version,directory",
    [
        ("unapproved", "1.0", "unapproved-1.0.dist-info"),
        ("cffi", "2.1.1", "cffi_clone-2.1.1.dist-info"),
        ("typing.extensions", "4.16.0", "typing.extensions-4.16.0.dist-info"),
    ],
    ids=["dodatkowa-dystrybucja", "duplikat-nazwy-i-wersji", "duplikat-po-normalizacji"],
)
def test_extra_or_duplicate_distribution_refuses_before_cleanup(
    installed, name, version, directory
):
    destination, site, _, _, script = installed
    add_distribution(site, name, version, directory)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before


def test_missing_expected_distribution_refuses_before_cleanup(installed):
    destination, site, _, _, script = installed
    distribution = site / "pycparser-3.12.dist-info"
    for path in distribution.iterdir():
        path.unlink()
    distribution.rmdir()
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before


@pytest.mark.parametrize(
    "metadata",
    [
        "Name: unexpected\nVersion: 2.1.1\n",
        "Name: cffi!\nVersion: 2.1.1\n",
        "Name: cffi\nVersion: 0.0.0\n",
        "Version: 2.1.1\n",
        "Name: cffi\n",
        "Name: cffi\nName: cffi\nVersion: 2.1.1\n",
        "Name: cffi\nVersion: 2.1.1\nVersion: 2.1.1\n",
    ],
    ids=[
        "inna-nazwa",
        "niedozwolona-skladnia-nazwy",
        "inna-wersja",
        "brak-nazwy",
        "brak-wersji",
        "dwie-nazwy",
        "dwie-wersje",
    ],
)
def test_distribution_metadata_contract_is_checked_even_with_valid_record(installed, metadata):
    destination, site, record, rows, script = installed
    metadata_path = record.parent / "METADATA"
    metadata_path.write_text(metadata, encoding="utf-8")
    metadata_name = metadata_path.relative_to(site).as_posix()
    rows = [
        file_row(metadata_name, metadata_path.read_bytes()) if row[0] == metadata_name else row
        for row in rows
    ]
    write_record(record, rows)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before


def test_distribution_names_are_normalized_before_exact_set_comparison(installed):
    destination, site, _, _, _ = installed
    record = site / "typing_extensions-4.16.0.dist-info/RECORD"
    metadata = record.parent / "METADATA"
    metadata.write_text("Name: Typing.Extensions\nVersion: 4.16.0\n", encoding="utf-8")
    metadata_name = metadata.relative_to(site).as_posix()
    with record.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    write_record(
        record,
        [
            file_row(metadata_name, metadata.read_bytes()) if row[0] == metadata_name else row
            for row in rows
        ],
    )
    assert len(builder._installed_records(destination)) == 6
    builder._remove_installation_scripts(destination)
    assert len(builder._installed_records(destination)) == 6


def test_wheel_record_cannot_claim_venv_activation_script(installed):
    destination, _, record, rows, _ = installed
    script = destination / "bin/activate"
    rows.insert(0, file_row("../../../bin/activate", script.read_bytes()))
    write_record(record, rows)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert snapshot_files(destination) == before


def test_runtime_probe_rejects_duplicate_distribution_instead_of_overwriting_it():
    # Ten test bada enumerację metadanych. Przy globalnym runnerze modelujemy
    # tylko granicę venv; rzeczywistą izolację sprawdzają pełne buildy Noble.
    prelude = """
import importlib.metadata as metadata, sys
if sys.prefix == sys.base_prefix:
    sys.base_prefix = sys.prefix + "-testowa-baza"
distribution = metadata.distribution("cffi")
metadata.distributions = lambda: [distribution, distribution]
"""
    result = subprocess.run(
        [sys.executable, "-I", "-s", "-B", "-c", prelude + builder._PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "AssertionError: Powtorzona dystrybucja runtime" in result.stderr
    assert result.stdout == ""


def test_distribution_must_record_its_own_metadata_before_cleanup(installed):
    destination, site, record, rows, script = installed
    metadata_name = (record.parent / "METADATA").relative_to(site).as_posix()
    metadata_row = next(row for row in rows if row[0] == metadata_name)
    write_record(record, [row for row in rows if row is not metadata_row])
    other = site / "pycparser-3.12.dist-info/RECORD"
    with other.open(newline="", encoding="utf-8") as stream:
        other_rows = list(csv.reader(stream))
    write_record(other, [metadata_row, *other_rows])
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before


def test_distribution_directory_alias_is_refused_even_with_consistent_record(installed):
    destination, site, record, rows, script = installed
    original_name = record.parent.name
    alias = site / "cffi_alias-2.1.1.dist-info"
    record.parent.rename(alias)
    record = alias / "RECORD"
    updated_rows = []
    for row in rows:
        name = row[0].replace(original_name + "/", alias.name + "/", 1)
        if name == record.relative_to(site).as_posix():
            updated_rows.append([name, "", ""])
        else:
            updated_rows.append(file_row(name, (site / name).read_bytes()))
    write_record(record, updated_rows)
    before = snapshot_files(destination)
    with pytest.raises(builder.BuildError):
        builder._remove_installation_scripts(destination)
    assert script.is_file()
    assert snapshot_files(destination) == before
