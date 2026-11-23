from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from deployment import core_test_plan
from deployment.core_test_plan import (
    CorePlanError,
    canonical_plan_digest,
    execute_plan,
    load_manifest,
    main,
    marker_document,
    selectors_for_platform,
)
from deployment.platform_evidence import EvidenceProductionError, aggregate_core_markers
from deployment.host_identity import UnsupportedHostOSError, canonical_host_os


def test_manifest_is_complete_deterministic_and_points_to_tests() -> None:
    manifest = load_manifest()
    assert manifest["test_plan_id"] == "CORE_REQUIRED_SUITES_V1"
    assert manifest["required_platforms"] == ["Linux", "Windows", "macOS"]
    assert manifest["acceptance_policy"]["skips"] == 0
    assert all(
        suite["platform_requirement"] == "REQUIRED_ON_ALL_THREE" for suite in manifest["suites"]
    )
    selectors = selectors_for_platform(manifest, "Linux")
    assert selectors == selectors_for_platform(manifest, "Windows")
    assert selectors == selectors_for_platform(manifest, "macOS")
    assert all(Path(selector).is_file() for selector in selectors)
    assert len(canonical_plan_digest(manifest)) == 64


@pytest.mark.parametrize(
    ("native", "canonical"),
    [("Linux", "Linux"), ("Windows", "Windows"), ("Darwin", "macOS")],
)
def test_canonical_host_os_mapping(native: str, canonical: str) -> None:
    assert canonical_host_os(native) == canonical


def test_canonical_host_os_rejects_unknown() -> None:
    assert issubclass(UnsupportedHostOSError, OSError)
    with pytest.raises(UnsupportedHostOSError, match="unsupported host OS"):
        canonical_host_os("Plan9")


def test_core_plan_cli_reports_unsupported_host_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("deployment.host_identity.platform.system", lambda: "Plan9")

    result = main(
        [
            "--runner-os",
            "Linux",
            "--source-revision",
            "current",
            "--ci-run-id",
            "run-1",
            "--ci-provider",
            "https://github.com",
            "--output",
            str(tmp_path / "marker.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result != 0
    assert captured.out == ""
    assert captured.err == "unsupported host OS: 'Plan9'\n"
    assert "Traceback" not in captured.err
    assert not (tmp_path / "marker.json").exists()


def test_darwin_host_executes_canonical_macos_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deployment.host_identity.platform.system", lambda: "Darwin")

    def successful_plan(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        report_arg = next(value for value in command if value.startswith("--junitxml="))
        Path(report_arg.split("=", 1)[1]).write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    output = tmp_path / "core-macOS.json"
    execute_plan(
        manifest_path=Path("deployment/core_required_suites_v1.json"),
        runner_os="macOS",
        source_revision="current",
        ci_run_id="run-1",
        ci_provider="https://github.com",
        output=output,
        runner=successful_plan,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["runner_os"] == "macOS"

    with pytest.raises(CorePlanError, match="runner OS"):
        execute_plan(
            manifest_path=Path("deployment/core_required_suites_v1.json"),
            runner_os="Windows",
            source_revision="current",
            ci_run_id="run-1",
            ci_provider="https://github.com",
            output=tmp_path / "wrong.json",
            runner=successful_plan,
        )


def test_core_plan_rejects_doctype_xml_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deployment.host_identity.platform.system", lambda: "Linux")

    def malicious_plan(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        report_arg = next(value for value in command if value.startswith("--junitxml="))
        Path(report_arg.split("=", 1)[1]).write_text(
            '<!DOCTYPE testsuites [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(CorePlanError, match="unsafe XML declaration"):
        execute_plan(
            manifest_path=Path("deployment/core_required_suites_v1.json"),
            runner_os="Linux",
            source_revision="current",
            ci_run_id="run-1",
            ci_provider="https://github.com",
            output=tmp_path / "marker.json",
            runner=malicious_plan,
        )


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize(
    "declaration",
    [
        "<!DOCTYPE testsuites>",
        '<!DOCTYPE testsuites SYSTEM "https://invalid.example/junit.dtd">',
        '<!DOCTYPE testsuites [<!ENTITY skipped "0">]>',
        '<!DOCTYPE testsuites [<!ENTITY external SYSTEM "file:///nonexistent-audit-fixture">]>',
        '<!DOCTYPE testsuites [<!ENTITY a "0"><!ENTITY skipped "&a;">]>',
    ],
)
def test_core_plan_rejects_encoded_dtd_before_writing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str, declaration: str
) -> None:
    monkeypatch.setattr("deployment.host_identity.platform.system", lambda: "Linux")

    def malicious_plan(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        report_arg = next(value for value in command if value.startswith("--junitxml="))
        Path(report_arg.split("=", 1)[1]).write_bytes(
            (
                declaration
                + '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/>'
                + "</testsuites>"
            ).encode(encoding)
        )
        return subprocess.CompletedProcess(command, 0)

    output = tmp_path / "marker.json"
    with pytest.raises(CorePlanError, match="unsafe XML declaration"):
        execute_plan(
            manifest_path=Path("deployment/core_required_suites_v1.json"),
            runner_os="Linux",
            source_revision="current",
            ci_run_id="run-1",
            ci_provider="https://github.com",
            output=output,
            runner=malicious_plan,
        )
    assert not output.exists()


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"])
def test_core_report_preserves_valid_encoded_xml(tmp_path: Path, encoding: str) -> None:
    report = tmp_path / "core.xml"
    report.write_bytes('<testsuites><testsuite skipped="0"/></testsuites>'.encode(encoding))
    assert core_test_plan._parse_core_xml_report(report).tag == "testsuites"


@pytest.mark.parametrize(
    "document",
    [
        b"",
        b"<testsuites",
        b'<testsuites><testsuite skipped="0"/>',
        b'<!DOCTYPE testsuites [<!ENTITY skipped "0">',
        b'<?xml version="1.0" encoding="utf-16"?><testsuites/>',
        b'<?xml version="1.0" encoding="unknown-audit-encoding"?><testsuites/>',
        b'<?xml version="1.0" encoding="shift_jis"?><testsuites/>',
        "<testsuites/>".encode("utf-32"),
    ],
)
def test_core_report_rejects_incomplete_or_invalid_xml(tmp_path: Path, document: bytes) -> None:
    report = tmp_path / "core.xml"
    report.write_bytes(document)
    with pytest.raises(CorePlanError):
        core_test_plan._parse_core_xml_report(report)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be"])
def test_core_report_never_expands_internal_entities(tmp_path: Path, encoding: str) -> None:
    report = tmp_path / "core.xml"
    report.write_bytes(
        (
            '<!DOCTYPE testsuites [<!ENTITY a "0"><!ENTITY skipped "&a;">]>'
            '<testsuites><testsuite skipped="&skipped;"/></testsuites>'
        ).encode(encoding)
    )
    with pytest.raises(CorePlanError, match="unsafe XML declaration"):
        core_test_plan._parse_core_xml_report(report)


def test_core_report_bounds_actual_read_even_if_size_metadata_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "core.xml"
    report.write_bytes(b'<testsuites><testsuite skipped="0"/>' + b" " * 128 + b"</testsuites>")
    monkeypatch.setattr(core_test_plan, "MAX_CORE_XML_REPORT_BYTES", 64)
    original_stat = Path.stat

    def stale_stat(path: Path, **kwargs: object) -> object:
        metadata = original_stat(path, **kwargs)
        if path == report:
            values = list(metadata)
            values[6] = 1
            return type(metadata)(values)
        return metadata

    monkeypatch.setattr(Path, "stat", stale_stat)
    with pytest.raises(CorePlanError, match="size limit"):
        core_test_plan._parse_core_xml_report(report)


def test_smoke_subset_marker_cannot_be_aggregated_as_complete_core(
    tmp_path: Path,
) -> None:
    canonical = load_manifest()
    smoke = copy.deepcopy(canonical)
    smoke["suites"] = smoke["suites"][:1]
    markers = []
    for runner_os in ("Linux", "Windows", "macOS"):
        path = tmp_path / f"{runner_os}.json"
        path.write_text(
            json.dumps(marker_document(smoke, "current", runner_os, "run-1", "https://github.com")),
            encoding="utf-8",
        )
        markers.append(path)
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current",
            markers,
            tmp_path / "evidence.json",
            ci_run_id="run-1",
            ci_provider="https://github.com",
        )
    assert not (tmp_path / "evidence.json").exists()


def test_skip_or_failed_required_test_never_writes_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deployment.host_identity.platform.system", lambda: "Linux")

    def skipped_plan(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        report_arg = next(value for value in command if value.startswith("--junitxml="))
        Path(report_arg.split("=", 1)[1]).write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1"/></testsuites>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    output = tmp_path / "marker.json"
    with pytest.raises(CorePlanError, match="skip policy"):
        execute_plan(
            manifest_path=Path("deployment/core_required_suites_v1.json"),
            runner_os="Linux",
            source_revision="current",
            ci_run_id="run-1",
            ci_provider="https://github.com",
            output=output,
            runner=skipped_plan,
        )
    assert not output.exists()
