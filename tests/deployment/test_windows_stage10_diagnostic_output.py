"""Zapis diagnostyki Stage 10 w izolowanych katalogach; bez dowodów fizycznych."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import windows_stage10_diagnostic_output as output_io
from deployment import windows_stage10_proof_validation as validation


NATIVE_POSIX = pytest.mark.skipif(
    os.name == "nt", reason="Natywne dir_fd POSIX; Windows ma osobne testy"
)


@pytest.fixture
def approved_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Root przestrzeni nazw sandboxa może mieć zmapowany UID 65534. Modelujemy
    # wyłącznie właściciela tych przodków; właściwy katalog testu zachowuje
    # rzeczywiste kontrole UID/mode i wszystkie natywne operacje dir_fd.
    # Na zwykłym Linux/macOS CI (root UID 0) ten adapter jest nieaktywny.
    if os.name != "nt":
        namespace_owner = Path("/").stat().st_uid
        if namespace_owner not in {0, os.geteuid()}:
            qualify = output_io._qualify_directory

            def qualified_namespace(info: os.stat_result, *, final: bool) -> None:
                if not final and info.st_uid == namespace_owner:
                    fields = {
                        name: getattr(info, name) for name in dir(info) if name.startswith("st_")
                    }
                    fields["st_uid"] = 0
                    info = SimpleNamespace(**fields)
                qualify(info, final=final)

            monkeypatch.setattr(output_io, "_qualify_directory", qualified_namespace)
    root = tmp_path.resolve() / "approved"
    root.mkdir(mode=0o700)
    return root


def _reports(root: Path) -> list[Path]:
    return list(root.glob("*.json"))


def _assert_blocked(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["report_kind"] == "windows-stage10-proof-blockers"
    assert report["status"] == "BLOCKED"
    assert report["native_qualification_performed"] is False
    assert all(
        proof["status"] == "BLOCKED"
        for result in report["results"].values()
        for proof in result["proofs"].values()
    )
    return report


@NATIVE_POSIX
def test_two_attempts_publish_distinct_new_reports_with_private_permissions(
    approved_directory: Path,
) -> None:
    first = validation.write_blocked_report(approved_directory=approved_directory)
    second = validation.write_blocked_report(approved_directory=approved_directory)
    assert first != second
    assert first.parent == second.parent == approved_directory
    assert first.name.startswith("stage10-blocked-") and first.suffix == ".json"
    assert second.name.startswith("stage10-blocked-") and second.suffix == ".json"
    assert _assert_blocked(first)["attempt_id"] != _assert_blocked(second)["attempt_id"]
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(second.stat().st_mode) == 0o600
    assert first.stat().st_uid == os.geteuid()
    assert not list(approved_directory.glob("*.partial"))


@NATIVE_POSIX
def test_explicit_new_output_is_published_only_inside_approved_directory(
    approved_directory: Path,
) -> None:
    output = approved_directory / "diagnostics.json"
    assert validation.write_blocked_report(output, approved_directory=approved_directory) == output
    _assert_blocked(output)
    assert tuple(approved_directory.iterdir()) == (output,)


@pytest.mark.parametrize(
    "name", ["receipt.json", "contract.json", "installer-manifest.json", "old.json"]
)
def test_existing_artifact_collision_preserves_every_byte(
    approved_directory: Path, name: str
) -> None:
    output = approved_directory / name
    original = b'{"status":"PASS","attempt_id":"historical","protected":true}'
    output.write_bytes(original)
    with pytest.raises((output_io.DiagnosticOutputError, FileExistsError)):
        validation.write_blocked_report(output, approved_directory=approved_directory)
    assert output.read_bytes() == original
    assert tuple(approved_directory.iterdir()) == (output,)


@pytest.mark.parametrize("name", ["receipt.json", "contract.json", "installer-manifest.json"])
def test_even_nonexistent_protected_input_name_is_not_an_output(
    approved_directory: Path, name: str
) -> None:
    output = approved_directory / name
    with pytest.raises(output_io.DiagnosticOutputError):
        output_io.publish_diagnostic(
            output,
            b"diagnostyka",
            approved_directory=approved_directory,
            protected_inputs=(output,),
        )
    assert not tuple(approved_directory.iterdir())


def test_output_cannot_select_its_own_approved_parent(
    tmp_path: Path, approved_directory: Path
) -> None:
    output = tmp_path / "elsewhere.json"
    with pytest.raises(output_io.DiagnosticOutputError):
        output_io.publish_diagnostic(output, b"diagnostyka", approved_directory=approved_directory)
    assert not output.exists()
    assert not tuple(approved_directory.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "CON.json",
        "PRN.json",
        "AUX.json",
        "NUL.json",
        "COM1.json",
        "LPT9.json",
        "bad:stream.json",
        "trailing.json ",
        "trailing.json.",
        "bad\x00.json",
    ],
)
def test_reserved_or_redirecting_filename_refuses_before_creating_any_file(
    approved_directory: Path, name: str
) -> None:
    with pytest.raises((output_io.DiagnosticOutputError, ValueError)):
        output_io.publish_diagnostic(
            approved_directory / name, b"diagnostyka", approved_directory=approved_directory
        )
    assert not tuple(approved_directory.iterdir())


@NATIVE_POSIX
def test_missing_approved_directory_is_not_created(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    with pytest.raises((output_io.DiagnosticOutputError, FileNotFoundError)):
        validation.write_blocked_report(approved_directory=absent)
    assert not absent.exists()


@NATIVE_POSIX
@pytest.mark.parametrize("mode", [0o777, 0o770, 0o707])
def test_untrusted_directory_write_permissions_are_refused_without_repair(
    approved_directory: Path, mode: int
) -> None:
    approved_directory.chmod(mode)
    with pytest.raises(output_io.DiagnosticOutputError):
        validation.write_blocked_report(approved_directory=approved_directory)
    assert stat.S_IMODE(approved_directory.stat().st_mode) == mode
    assert not tuple(approved_directory.iterdir())


@NATIVE_POSIX
def test_foreign_directory_owner_is_refused_without_chown(
    approved_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fstat = output_io.os.fstat
    identity = approved_directory.stat()

    def different_owner(descriptor: int) -> Any:
        actual = original_fstat(descriptor)
        if (actual.st_dev, actual.st_ino) != (identity.st_dev, identity.st_ino):
            return actual
        attributes = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        attributes["st_uid"] = os.geteuid() + 1
        return SimpleNamespace(**attributes)

    with monkeypatch.context() as isolated:
        isolated.setattr(output_io.os, "fstat", different_owner)
        with pytest.raises(output_io.DiagnosticOutputError):
            validation.write_blocked_report(approved_directory=approved_directory)
    assert approved_directory.stat().st_uid == os.geteuid()
    assert not tuple(approved_directory.iterdir())


@NATIVE_POSIX
@pytest.mark.parametrize("at_ancestor", [False, True])
def test_directory_symlink_and_redirecting_ancestor_are_not_followed(
    tmp_path: Path, at_ancestor: bool
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    link = tmp_path / "redirect"
    link.symlink_to(actual, target_is_directory=True)
    root = link
    if at_ancestor:
        (actual / "reports").mkdir(mode=0o700)
        root = link / "reports"
    with pytest.raises((output_io.DiagnosticOutputError, OSError)):
        validation.write_blocked_report(approved_directory=root)
    assert not list(actual.rglob("*.json"))
    assert not list(actual.rglob("*.partial"))


@NATIVE_POSIX
@pytest.mark.parametrize("hardlink", [False, True])
def test_output_link_cannot_delete_or_modify_its_protected_target(
    tmp_path: Path, approved_directory: Path, hardlink: bool
) -> None:
    target = tmp_path / "protected-receipt.json"
    original = b"chronione dane wejscia"
    target.write_bytes(original)
    output = approved_directory / "diagnostics.json"
    if hardlink:
        os.link(target, output)
    else:
        output.symlink_to(target)
    with pytest.raises((output_io.DiagnosticOutputError, FileExistsError)):
        validation.write_blocked_report(output, approved_directory=approved_directory)
    assert target.read_bytes() == original
    assert output.read_bytes() == original
    assert tuple(approved_directory.iterdir()) == (output,)


@NATIVE_POSIX
@pytest.mark.parametrize("failure_at", ["write", "fsync", "publish"])
def test_io_failure_preserves_existing_data_and_publishes_no_partial_json(
    approved_directory: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    old = approved_directory / "old.json"
    original = b'{"status":"PASS","attempt_id":"previous"}'
    old.write_bytes(original)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("syntetyczny blad IO")

    owner, name = {
        "write": (output_io.os, "write"),
        "fsync": (output_io.os, "fsync"),
        "publish": (output_io, "_rename_noreplace"),
    }[failure_at]
    monkeypatch.setattr(owner, name, fail)
    with pytest.raises((output_io.DiagnosticOutputError, OSError)):
        validation.write_blocked_report(
            approved_directory / "new.json", approved_directory=approved_directory
        )
    assert old.read_bytes() == original
    assert _reports(approved_directory) == [old]
    assert all(path.suffix == ".partial" for path in approved_directory.iterdir() if path != old)


@NATIVE_POSIX
def test_interruption_during_write_never_publishes_final_report_or_deletes_inputs(
    approved_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = approved_directory / "receipt.json"
    original = b"zachowaj dane wejscia"
    protected.write_bytes(original)

    def interrupted(descriptor: int, raw: bytes) -> None:
        os.write(descriptor, raw[:17])
        raise KeyboardInterrupt("syntetyczne przerwanie w polowie zapisu")

    monkeypatch.setattr(output_io, "_write_descriptor", interrupted)
    with pytest.raises(KeyboardInterrupt):
        validation.write_blocked_report(
            approved_directory / "new.json", approved_directory=approved_directory
        )
    assert protected.read_bytes() == original
    assert _reports(approved_directory) == [protected]
    partials = list(approved_directory.glob("*.partial"))
    assert len(partials) == 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(partials[0].read_text(encoding="utf-8"))


@NATIVE_POSIX
@pytest.mark.parametrize("symlink", [False, True])
def test_directory_swap_between_validation_and_publication_cannot_redirect_writes(
    tmp_path: Path,
    approved_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink: bool,
) -> None:
    retained = tmp_path / "retained-original"
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    protected = attacker / "new.json"
    original = b"niezalezne dane poza zatwierdzonym katalogiem"
    protected.write_bytes(original)
    write = output_io._write_descriptor

    def swap_after_write(descriptor: int, raw: bytes) -> None:
        write(descriptor, raw)
        approved_directory.rename(retained)
        if symlink:
            approved_directory.symlink_to(attacker, target_is_directory=True)
        else:
            approved_directory.mkdir(mode=0o700)

    monkeypatch.setattr(output_io, "_write_descriptor", swap_after_write)
    with pytest.raises(output_io.DiagnosticOutputError):
        validation.write_blocked_report(
            approved_directory / "new.json", approved_directory=approved_directory
        )
    assert protected.read_bytes() == original
    assert tuple(attacker.iterdir()) == (protected,)
    assert not (retained / "new.json").exists()
    assert len(list(retained.glob("*.partial"))) == 1
    if not symlink:
        assert not tuple(approved_directory.iterdir())


@NATIVE_POSIX
def test_collision_winning_race_at_publication_is_preserved(
    approved_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = approved_directory / "new.json"
    winner = b'{"status":"PASS","attempt_id":"foreign"}'
    rename = output_io._rename_noreplace

    def concurrent_winner(directory: int, source: str, destination: str) -> None:
        output.write_bytes(winner)
        rename(directory, source, destination)

    monkeypatch.setattr(output_io, "_rename_noreplace", concurrent_winner)
    with pytest.raises((output_io.DiagnosticOutputError, FileExistsError)):
        validation.write_blocked_report(output, approved_directory=approved_directory)
    assert output.read_bytes() == winner
    assert len(list(approved_directory.glob("*.partial"))) == 1
