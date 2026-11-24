"""Synthetic XML refusal tests shared by the non-qualifying JUnit readers."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from scripts.quality import check_s9_fv_01_junit as gate
from scripts.quality import run_stage9_safe_validation as safe


_REQUIRED = frozenset({("fixture", "synthetic")})
_NORMAL = '<testsuite tests="1"><testcase classname="fixture" name="synthetic"/></testsuite>'
_DTD = '<!DOCTYPE testsuite [<!ENTITY name "synthetic">]><testsuite tests="1"><testcase classname="fixture" name="&name;"/></testsuite>'


def _accepted(consumer: str, report: Path) -> bool:
    if consumer == "gate":
        return not gate.validate_report(report, _REQUIRED).blocked
    return safe.junit_result(report)["status"] == "PASS"


@pytest.mark.parametrize("consumer", ["gate", "safe"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_dtd_is_refused_in_all_document_encodings(
    tmp_path: Path, consumer: str, encoding: str
) -> None:
    report = tmp_path / "synthetic.xml"
    report.write_bytes(_DTD.encode(encoding))
    assert not _accepted(consumer, report)


@pytest.mark.parametrize("consumer", ["gate", "safe"])
@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_plain_utf16_keeps_existing_compatibility(
    tmp_path: Path, consumer: str, encoding: str
) -> None:
    report = tmp_path / "synthetic.xml"
    report.write_bytes(_NORMAL.encode(encoding))
    # Existing readers accept BOMless UTF16; unsupported UTF32 still refuses.
    assert _accepted(consumer, report) == encoding.startswith("utf-16")


@pytest.mark.parametrize("consumer", ["gate", "safe"])
@pytest.mark.parametrize("bom", [False, True])
def test_normal_pytest_utf8_junit_is_accepted(tmp_path: Path, consumer: str, bom: bool) -> None:
    report = tmp_path / "normal.xml"
    text = '<?xml version="1.0" encoding="utf-8"?>' + _NORMAL
    report.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    assert _accepted(consumer, report)


@pytest.mark.parametrize("consumer", ["gate", "safe"])
def test_report_size_is_bounded_before_xml_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    module = gate if consumer == "gate" else safe
    monkeypatch.setattr(module, "_MAX_REPORT_BYTES", 256, raising=False)
    report = tmp_path / "large.xml"
    report.write_text(
        _NORMAL.replace("<testcase", "<!--" + "X" * 256 + "--><testcase"), encoding="utf-8"
    )
    assert not _accepted(consumer, report)


@pytest.mark.parametrize("consumer", ["gate", "safe"])
@pytest.mark.parametrize(
    "raw", [b"", b"invalid", b"\xffinvalid", b"<testsuite>", b"\0<testsuite/>"]
)
def test_malformed_or_non_utf8_report_cannot_pass(
    tmp_path: Path, consumer: str, raw: bytes
) -> None:
    report = tmp_path / "bad.xml"
    report.write_bytes(raw)
    assert not _accepted(consumer, report)


@pytest.mark.parametrize("consumer", ["gate", "safe"])
def test_report_io_error_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    report = tmp_path / "unreadable.xml"
    report.write_text(_NORMAL)
    original_open = builtins.open
    original_path_open = Path.open

    def deny_builtin(name, *args, **kwargs):
        if Path(name) == report:
            raise PermissionError("synthetic report refusal")
        return original_open(name, *args, **kwargs)

    def deny_path(path, *args, **kwargs):
        if path == report:
            raise PermissionError("synthetic report refusal")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_builtin)
    monkeypatch.setattr(Path, "open", deny_path)
    assert not _accepted(consumer, report)


@pytest.mark.parametrize("consumer", ["gate", "safe"])
def test_missing_report_never_passes(tmp_path: Path, consumer: str) -> None:
    assert not _accepted(consumer, tmp_path / "absent.xml")
