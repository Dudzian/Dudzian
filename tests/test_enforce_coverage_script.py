from __future__ import annotations

from pathlib import Path

import pytest

from scripts import enforce_coverage


@pytest.fixture()
def coverage_file(tmp_path: Path) -> Path:
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        """
        <coverage line-rate="0.92">
          <packages>
            <package name="bot_core.strategies" line-rate="0.95" />
            <package name="bot_core.runtime" line-rate="0.89" />
          </packages>
        </coverage>
        """.strip(),
        encoding="utf-8",
    )
    return xml


def test_script_passes_with_default_threshold(
    coverage_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = enforce_coverage.main(
        [
            "--coverage-file",
            str(coverage_file),
            "--minimum",
            "80",
            "--package",
            "bot_core.strategies=85",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Pokrycie całkowite" in out
    assert "Pakiet bot_core.strategies" in out


def test_script_fails_when_threshold_not_met(coverage_file: Path) -> None:
    exit_code = enforce_coverage.main(["--coverage-file", str(coverage_file), "--minimum", "95"])
    assert exit_code == 1


def test_script_errors_on_missing_package(coverage_file: Path) -> None:
    exit_code = enforce_coverage.main(
        ["--coverage-file", str(coverage_file), "--package", "missing.module"]
    )
    assert exit_code == 1


def test_script_rejects_doctype_xml(tmp_path: Path) -> None:
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        '<!DOCTYPE coverage [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<coverage line-rate="1.0"><packages/></coverage>',
        encoding="utf-8",
    )
    assert enforce_coverage.main(["--coverage-file", str(xml)]) == 2


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize(
    "declaration",
    [
        "<!DOCTYPE coverage>",
        '<!DOCTYPE coverage SYSTEM "https://invalid.example/coverage.dtd">',
        '<!DOCTYPE coverage [<!ENTITY rate "1.0">]>',
        '<!DOCTYPE coverage [<!ENTITY external SYSTEM "file:///nonexistent-audit-fixture">]>',
        '<!DOCTYPE coverage [<!ENTITY a "1"><!ENTITY rate "&a;.0">]>',
    ],
)
def test_script_rejects_dtd_in_every_supported_encoding(
    tmp_path: Path, encoding: str, declaration: str
) -> None:
    report = tmp_path / "coverage.xml"
    report.write_bytes(
        (declaration + '<coverage line-rate="1.0"><packages/></coverage>').encode(encoding)
    )
    assert enforce_coverage.main(["--coverage-file", str(report)]) == 2


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"])
def test_script_preserves_valid_encoded_xml(tmp_path: Path, encoding: str) -> None:
    report = tmp_path / "coverage.xml"
    report.write_bytes('<coverage line-rate="1.0"><packages/></coverage>'.encode(encoding))
    assert enforce_coverage.main(["--coverage-file", str(report)]) == 0


@pytest.mark.parametrize(
    "document",
    [
        b"",
        b"<coverage",
        b'<coverage line-rate="1.0"><packages/>',
        b'<!DOCTYPE coverage [<!ENTITY rate "1.0">',
        b'<?xml version="1.0" encoding="utf-16"?><coverage line-rate="1.0"/>',
        b'<?xml version="1.0" encoding="unknown-audit-encoding"?><coverage line-rate="1.0"/>',
        b'<?xml version="1.0" encoding="shift_jis"?><coverage line-rate="1.0"/>',
        '<coverage line-rate="1.0"/>'.encode("utf-32"),
    ],
)
def test_script_rejects_incomplete_or_invalid_xml(tmp_path: Path, document: bytes) -> None:
    report = tmp_path / "coverage.xml"
    report.write_bytes(document)
    assert enforce_coverage.main(["--coverage-file", str(report)]) == 2


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be"])
def test_script_never_expands_internal_entities_into_accepted_coverage(
    tmp_path: Path, encoding: str
) -> None:
    report = tmp_path / "coverage.xml"
    report.write_bytes(
        (
            '<!DOCTYPE coverage [<!ENTITY a "1"><!ENTITY rate "&a;.0">]>'
            '<coverage line-rate="&rate;"><packages/></coverage>'
        ).encode(encoding)
    )
    assert enforce_coverage.main(["--coverage-file", str(report)]) == 2


def test_script_bounds_actual_xml_read_even_if_size_metadata_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "coverage.xml"
    report.write_bytes(b'<coverage line-rate="1.0"><packages/>' + b" " * 128 + b"</coverage>")
    monkeypatch.setattr(enforce_coverage, "MAX_COVERAGE_XML_BYTES", 64)
    original_stat = Path.stat

    def stale_stat(path: Path, **kwargs: object) -> object:
        metadata = original_stat(path, **kwargs)
        if path == report:
            values = list(metadata)
            values[6] = 1
            return type(metadata)(values)
        return metadata

    monkeypatch.setattr(Path, "stat", stale_stat)
    assert enforce_coverage.main(["--coverage-file", str(report)]) == 2


def test_script_errors_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.xml"
    exit_code = enforce_coverage.main(["--coverage-file", str(missing)])
    assert exit_code == 2
