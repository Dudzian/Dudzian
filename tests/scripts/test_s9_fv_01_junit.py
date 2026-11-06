"""Bramka migracyjna: pełne JUnit albo BLOCKED, bez dowodów administracyjnych."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.quality import check_s9_fv_01_junit as gate

_CONTRACT = "tests.security.test_account_genesis_verified_transport_contract"
_UNIX = "tests.security.test_account_genesis_verified_transport_unix"
_LEGACY = "tests.security.test_freshness_runtime_ipc"
_REQUIRED = frozenset(
    {
        (_CONTRACT, "test_exact_candidate[proposer]"),
        (_UNIX, "test_half_close[once]"),
    }
)


def _case(classname: str, name: str, **attributes: str) -> ET.Element:
    return ET.Element("testcase", {"classname": classname, "name": name, **attributes})


def _passed_cases() -> list[ET.Element]:
    return [_case(*identity) for identity in sorted(_REQUIRED)]


def _report(
    tmp_path: Path,
    cases: list[ET.Element],
    *,
    root_tag: str = "testsuites",
    suite_attributes: dict[str, str] | None = None,
) -> Path:
    root = ET.Element(root_tag)
    suite = root if root_tag == "testsuite" else ET.SubElement(root, "testsuite")
    suite.attrib.update(suite_attributes or {})
    suite.extend(cases)
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _assert_blocked(report: Path, required=_REQUIRED, *, pytest_exit_code: int = 0) -> None:
    result = gate.validate_report(report, required, pytest_exit_code=pytest_exit_code)
    assert result.blocked, "Niepełny dowód wykonania nie może dać PASS"
    assert result.issues


@pytest.mark.parametrize("root_tag", ["testsuite", "testsuites"])
def test_complete_exact_cases_pass_in_standard_junit_forms(tmp_path, root_tag):
    report = _report(tmp_path, list(reversed(_passed_cases())), root_tag=root_tag)

    result = gate.validate_report(report, _REQUIRED)

    assert not result.blocked
    assert result.issues == ()
    assert result.required_cases == result.passed_cases == len(_REQUIRED)
    assert result.legacy_skipped_cases == 0


def test_complete_report_allows_legacy_skip_without_substituting_required_case(tmp_path):
    cases = _passed_cases()
    legacy = _case(_LEGACY, "test_real_extra_server_groups_are_denied_before_send")
    ET.SubElement(legacy, "skipped", {"message": "Konto nie ma dodatkowych grup"})
    cases.append(legacy)
    report = _report(
        tmp_path,
        cases,
        suite_attributes={"tests": "3", "skipped": "1", "errors": "0", "failures": "0"},
    )

    result = gate.validate_report(report, _REQUIRED)

    assert not result.blocked
    assert result.passed_cases == 2
    assert result.legacy_skipped_cases == 1


def test_skip_in_new_validator_tests_is_not_an_allowed_legacy_skip(tmp_path):
    cases = _passed_cases()
    skipped = _case("tests.scripts.test_s9_fv_01_junit", "test_validator")
    ET.SubElement(skipped, "skipped")
    cases.append(skipped)

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize("wrapper_tag", ["testsuite", "testsuites"])
def test_nested_suites_preserve_exact_required_identities(tmp_path, wrapper_tag):
    root = ET.Element("testsuites", {"tests": "2"})
    wrapper = ET.SubElement(root, wrapper_tag, {"tests": "2"})
    for case in _passed_cases():
        suite = ET.SubElement(wrapper, "testsuite", {"tests": "1"})
        suite.append(case)
    report = tmp_path / "nested.xml"
    ET.ElementTree(root).write(report, encoding="utf-8")

    result = gate.validate_report(report, _REQUIRED)

    assert not result.blocked
    assert result.passed_cases == 2


def test_full_reviewed_manifest_has_all_eighty_three_distinct_parameterized_cases():
    required = gate.load_required_cases(gate.DEFAULT_REQUIRED_CASES)
    counts = {
        suffix: sum(classname.endswith(suffix) for classname, _ in required)
        for suffix in ("_contract", "_unix", "_scope")
    }

    assert len(required) == 83
    assert counts == {"_contract": 46, "_unix": 33, "_scope": 4}
    assert any("[" in name for _, name in required)


def test_all_reviewed_cases_can_pass_as_one_exact_report(tmp_path):
    required = gate.load_required_cases(gate.DEFAULT_REQUIRED_CASES)
    report = _report(tmp_path, [_case(*identity) for identity in sorted(required)])

    result = gate.validate_report(report, required)

    assert not result.blocked
    assert result.passed_cases == 83


@pytest.mark.parametrize("missing", sorted(_REQUIRED))
def test_each_missing_case_blocks_even_with_many_legacy_passes(tmp_path, missing):
    cases = [_case(*identity) for identity in _REQUIRED if identity != missing]
    cases.extend(_case(_LEGACY, f"test_legacy_pass[{index}]") for index in range(100))

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize(
    "replacement",
    [
        (_CONTRACT, "test_exact_candidate[document]"),
        (_LEGACY, "test_exact_candidate[proposer]"),
        (_CONTRACT, "test_exact_candidate"),
    ],
    ids=["inny-parametr", "inna-klasa", "utracony-parametr"],
)
def test_similar_name_or_parameter_does_not_replace_exact_required_identity(tmp_path, replacement):
    cases = [_case(*identity) for identity in _REQUIRED if identity[0] != _CONTRACT]
    cases.append(_case(*replacement))

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize(
    "reason",
    [
        "SO_PEERPIDFD niedostępny, errno=1",
        "Brak możliwości bind socketu AF_UNIX",
        "Wymagane Linux AF_UNIX i fork",
    ],
    ids=["pidfd", "socket", "platforma"],
)
def test_skipped_mandatory_case_is_blocked_for_each_environment_limitation(tmp_path, reason):
    cases = _passed_cases()
    ET.SubElement(cases[0], "skipped", {"message": reason})

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_nonpassing_mandatory_outcome_blocks(tmp_path, outcome):
    cases = _passed_cases()
    ET.SubElement(cases[1], outcome).text = "Nie ma poprawnego dowodu wykonania"

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_legacy_failure_or_collection_error_is_not_an_allowed_skip(tmp_path, outcome):
    cases = _passed_cases()
    legacy = _case(_LEGACY, "test_legacy")
    ET.SubElement(legacy, outcome).text = "Błąd starszej regresji"
    cases.append(legacy)

    _assert_blocked(_report(tmp_path, cases))


def test_duplicate_required_result_never_counts_as_missing_case_execution(tmp_path):
    identity = sorted(_REQUIRED)[0]
    cases = [_case(*identity), _case(*identity)]

    _assert_blocked(_report(tmp_path, cases))


def test_duplicate_required_case_blocks_even_when_all_required_cases_are_present(tmp_path):
    cases = _passed_cases()
    cases.append(_case(*sorted(_REQUIRED)[0]))

    _assert_blocked(_report(tmp_path, cases))


def test_duplicate_identity_across_suites_is_still_blocked(tmp_path):
    root = ET.Element("testsuites")
    ET.SubElement(root, "testsuite").extend(_passed_cases())
    ET.SubElement(root, "testsuite").append(_case(*sorted(_REQUIRED)[0]))
    report = tmp_path / "duplicate-suites.xml"
    ET.ElementTree(root).write(report, encoding="utf-8")

    _assert_blocked(report)


def test_unknown_case_in_mandatory_module_requires_inventory_review(tmp_path):
    cases = _passed_cases()
    cases.append(_case(_UNIX, "test_unreviewed_extra_parameter[new]"))

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize(
    "attributes",
    [
        {"name": "test_without_class"},
        {"classname": _UNIX},
        {"classname": "", "name": "test_empty_class"},
        {"classname": _UNIX, "name": ""},
    ],
    ids=["brak-klasy", "brak-nazwy", "pusta-klasa", "pusta-nazwa"],
)
def test_incomplete_testcase_identity_is_blocked(tmp_path, attributes):
    cases = _passed_cases()
    cases.append(ET.Element("testcase", attributes))

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize(
    "attributes",
    [
        {"status": "notrun"},
        {"status": "disabled"},
        {"status": "skipped"},
        {"result": "skipped"},
    ],
)
def test_explicit_nonexecution_attribute_cannot_appear_as_pass(tmp_path, attributes):
    cases = _passed_cases()
    cases[0].attrib.update(attributes)

    _assert_blocked(_report(tmp_path, cases))


@pytest.mark.parametrize(
    "counters",
    [
        {"tests": "0"},
        {"tests": "3"},
        {"skipped": "1"},
        {"failures": "1"},
        {"errors": "1"},
        {"tests": "-1"},
        {"tests": "NaN"},
        {"tests": "2.0"},
    ],
)
def test_inconsistent_or_invalid_declared_counters_are_blocked(tmp_path, counters):
    _assert_blocked(_report(tmp_path, _passed_cases(), suite_attributes=counters))


def test_inconsistent_root_counter_cannot_hide_in_complete_child_suite(tmp_path):
    report = _report(tmp_path, _passed_cases(), suite_attributes={"tests": "2"})
    tree = ET.parse(report)
    tree.getroot().set("tests", "0")
    tree.write(report, encoding="utf-8")

    _assert_blocked(report)


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        b" \n\t",
        b"<testsuites><testsuite>",
        b'<testsuites><testsuite tests="0" /></testsuites>',
        b'<testsuites><testsuite tests="83" /></testsuites>',
        b"<not-junit />",
        b'<testsuites xmlns="urn:obcy-format"><testsuite /></testsuites>',
    ],
    ids=["pusty", "whitespace", "uciety", "zero", "same-liczniki", "obcy-root", "namespace"],
)
def test_absent_incomplete_or_wrong_format_report_cannot_pass(tmp_path, contents):
    report = tmp_path / "invalid.xml"
    report.write_bytes(contents)

    _assert_blocked(report)


def test_missing_report_is_blocked(tmp_path):
    _assert_blocked(tmp_path / "missing.xml")


def test_empty_required_set_never_creates_vacuous_pass(tmp_path):
    report = _report(tmp_path, [_case(_LEGACY, "test_legacy_pass")])

    _assert_blocked(report, frozenset())


@pytest.mark.parametrize("pytest_exit_code", [1, 2, 3, 4, 5])
def test_nonzero_pytest_exit_blocks_even_with_complete_old_report(tmp_path, pytest_exit_code):
    report = _report(tmp_path, _passed_cases())

    _assert_blocked(report, pytest_exit_code=pytest_exit_code)


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"schema_version": 1, "cases": []},
        {"schema_version": 2, "cases": []},
        {"schema_version": True, "cases": []},
        {"schema_version": 1, "cases": [{"name": "test_without_class"}]},
        {"schema_version": 1, "cases": [{"classname": _UNIX}]},
        {"schema_version": 1, "cases": [{"classname": _UNIX, "name": ""}]},
        {"schema_version": 1, "cases": [{"classname": None, "name": "test_invalid"}]},
        {
            "schema_version": 1,
            "cases": [{"classname": _UNIX, "name": "test_duplicate"}] * 2,
        },
    ],
)
def test_invalid_required_inventory_is_refused(tmp_path, manifest):
    path = tmp_path / "required.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        gate.load_required_cases(path)


def test_cli_prints_pass_only_after_complete_required_execution(tmp_path, monkeypatch, capsys):
    report = _report(tmp_path, _passed_cases())
    monkeypatch.setattr(gate, "load_required_cases", lambda: _REQUIRED)

    assert gate.main([str(report)]) == 0
    assert "S9-FV-01: PASS" in capsys.readouterr().out


@pytest.mark.parametrize("pytest_exit_code", ["0", "5"])
def test_cli_reports_blocked_for_skips_or_zero_selected_tests(
    tmp_path, monkeypatch, capsys, pytest_exit_code
):
    cases = _passed_cases()
    if pytest_exit_code == "0":
        ET.SubElement(cases[0], "skipped", {"message": "SO_PEERPIDFD niedostępny"})
    report = _report(tmp_path, cases)
    monkeypatch.setattr(gate, "load_required_cases", lambda: _REQUIRED)

    assert gate.main([str(report), "--pytest-exit-code", pytest_exit_code]) == 1
    output = capsys.readouterr().out
    assert "S9-FV-01: BLOCKED" in output
    assert "S9-FV-01: PASS" not in output


def test_cli_reports_blocked_for_unreadable_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "load_required_cases", lambda: _REQUIRED)

    assert gate.main([str(tmp_path / "missing.xml")]) == 1
    assert "S9-FV-01: BLOCKED" in capsys.readouterr().out
